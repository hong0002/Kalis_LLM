import argparse
import math
import json
from pathlib import Path

import yaml
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

from src.data import load_law_sft_dataset
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="학습된 결과 폴더 경로 (adapter_model.safetensors, config.yaml이 있는 폴더)",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="config.yaml",
        help="run_dir 안에서 사용할 config 파일 이름 (기본값: config.yaml)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="생성 시 최대 생성 토큰 수",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=-1,
        help="테스트 셋 일부만 생성하고 싶을 때 사용 (-1이면 전체)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    # 1) run_dir 안의 config 로드 (config_resolved.yaml이 있으면 그걸 우선 사용)
    cfg_path_resolved = run_dir / "config_resolved.yaml"
    cfg_path_simple = run_dir / args.config_name
    if cfg_path_resolved.exists():
        cfg_path = cfg_path_resolved
    else:
        cfg_path = cfg_path_simple

    print(f"[Eval] Loading config from: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]
    seed = int(cfg.get("seed", 42))

    set_seed(seed)

    # 2) test 데이터 로드
    test_path = dataset_cfg["test_path"]
    print(f"[Eval] Loading test dataset from: {test_path}")
    test_dataset = load_law_sft_dataset(test_path)
    print(f"[Eval] Test samples: {len(test_dataset)}")

    # 3) 학습된 LoRA 모델 로드 (run_dir 기준)
    print(f"[Eval] Loading fine-tuned model from: {run_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(run_dir),
        max_seq_length=int(model_cfg.get("max_seq_length", 2048)),
        dtype=None,
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
    )

    # 4) 학습 때와 동일한 포맷팅 함수 정의 (Teacher forcing용)
    def formatting_prompts_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # 정답까지 포함해서 teacher forcing
            )
            texts.append(text)
        return texts

    # 5) 평가 전용 SFTConfig (loss 계산용)
    eval_output_dir = run_dir / "eval_tmp"

    sft_config = SFTConfig(
        output_dir=str(eval_output_dir),
        per_device_eval_batch_size=1,
        max_seq_length=int(model_cfg.get("max_seq_length", 2048)),
        packing=bool(training_cfg.get("packing", False)),
        bf16=bool(training_cfg.get("bf16", True)),
        gradient_checkpointing=False,
        eval_strategy="no",   # ← 여기! eval_strategy가 아니라 evaluation_strategy
        report_to=[],               # 로깅 안 함
    )

    # 6) Trainer 생성 (evaluate로 loss 측정)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=test_dataset,
        eval_dataset=test_dataset,
        formatting_func=formatting_prompts_func,
        args=sft_config,
    )

    # 7) eval_loss / perplexity 계산
    print("[Eval] Running evaluation on test set (loss/perplexity)...")
    metrics = trainer.evaluate()

    results = dict(metrics)
    if "eval_loss" in metrics:
        try:
            ppl = math.exp(metrics["eval_loss"])
            results["perplexity"] = ppl
        except OverflowError:
            results["perplexity"] = None

    print("\n===== Evaluation Metrics =====")
    for k, v in results.items():
        print(f"{k}: {v}")

    # 8) 메트릭 저장
    metrics_path = run_dir / "eval_results_test.yaml"
    with metrics_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(results, f, allow_unicode=True)
    print(f"[Eval] Saved evaluation metrics to: {metrics_path}")

    # 9) 생성 결과 저장 (test_outputs.jsonl)
    print("[Eval] Generating outputs on test set...")

    # 생성 모드 설정
    FastLanguageModel.for_inference(model)

    # 몇 개만 샘플링해서 보고 싶으면 --sample 사용
    num_samples = len(test_dataset) if args.sample == -1 else min(args.sample, len(test_dataset))

    outputs_path = run_dir / "test_outputs.jsonl"
    with outputs_path.open("w", encoding="utf-8") as f_out, torch.no_grad():
        for idx in range(num_samples):
            example = test_dataset[idx]
            messages = example["messages"]

            # 1) gold answer (assistant) 추출
            gold_answer = ""
            for m in messages:
                if m.get("role") == "assistant":
                    gold_answer = m.get("content", "")
                    break

            # 2) 프롬프트용 메시지 (system/user 등만 남기기)
            prompt_messages = [m for m in messages if m.get("role") != "assistant"]

            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,   # assistant 답변을 새로 생성
            )

            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
            ).to(model.device)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )

            # 프롬프트 부분 이후만 잘라서 decode
            gen_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
            model_answer = tokenizer.decode(
                gen_ids,
                skip_special_tokens=True,
            ).strip()

            out_obj = {
                "index": idx,
                "id": example.get("id"),
                "question_messages": prompt_messages,
                "gold_answer": gold_answer,
                "model_answer": model_answer,
            }
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            if (idx + 1) % 20 == 0 or idx == num_samples - 1:
                print(f"[Eval] Generated {idx+1}/{num_samples}")

    print(f"[Eval] Saved generated outputs to: {outputs_path}")
    print("\n[Eval] Done.")


if __name__ == "__main__":
    main()
