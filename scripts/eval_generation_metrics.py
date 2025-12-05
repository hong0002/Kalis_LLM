import argparse
import json
from pathlib import Path

import torch
import yaml
from unsloth import FastLanguageModel
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from bert_score import score as bertscore_score   # 🔹 BERTScore
from nltk.translate.meteor_score import meteor_score   # 🔹 METEOR

from src.data import load_law_sft_dataset
from utils import set_seed


# ===== RAG 관련 설정 =====
RAG_INDEX_DIR = Path("datasets/processed/rag_index")
RAG_INDEX_PATH = RAG_INDEX_DIR / "rag_index.faiss"
RAG_META_PATH = RAG_INDEX_DIR / "rag_meta.jsonl"
RAG_EMBED_MODEL = "nlpai-lab/KURE-v1"


def load_rag_meta(meta_path: Path):
    metas = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas


def rag_retrieve(query: str, embed_model, index, metas, top_k: int = 5):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, top_k)

    scores = scores[0]
    idxs = idxs[0]

    results = []
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        if i < 0:
            continue
        m = metas[i]
        results.append(
            {
                "rank": rank,
                "score": float(s),
                "source_file": m["source_file"],
                "page": m["page"],
                "text": m["text"],
            }
        )
    return results


def build_rag_context_str(retrieved, max_chars: int = 1500) -> str:
    parts = []
    parts.append("다음은 관련 법령 및 유권해석에서 발췌한 내용입니다.\n")
    for r in retrieved:
        header = f"[{r['rank']}] {r['source_file']} p.{r['page']} (유사도: {r['score']:.3f})"
        parts.append(header)
        parts.append(r["text"].strip())
        parts.append("")

    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (이하 생략)"
    return text


def add_rag_to_messages(messages, context_str: str):
    """system 메시지에 [참고할 관련 법령 발췌]를 붙이는 방식."""
    new_messages = []
    added = False

    for i, m in enumerate(messages):
        if i == 0 and m.get("role") == "system":
            new_content = (
                m.get("content", "")
                + "\n\n[참고할 관련 법령 발췌]\n"
                + context_str
            )
            new_messages.append({"role": "system", "content": new_content})
            added = True
        else:
            new_messages.append(m)

    if not added:
        sys_msg = {
            "role": "system",
            "content": "[참고할 관련 법령 발췌]\n" + context_str,
        }
        new_messages = [sys_msg] + new_messages

    return new_messages


# ===== 기본 eval 스크립트 =====


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="학습된 결과 폴더 (config.yaml, adapter_model.safetensors 위치)",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="평가에 사용할 최대 샘플 수 (None이면 전체 사용)",
    )
    p.add_argument(
        "--output_file",
        type=str,
        default="gen_eval_results.jsonl",
        help="per-sample 결과를 저장할 파일 이름 (run_dir 아래)",
    )
    # 🔹 RAG 옵션들
    p.add_argument(
        "--use_rag",
        action="store_true",
        help="RAG 컨텍스트를 프롬프트에 추가해서 평가할지 여부",
    )
    p.add_argument(
        "--rag_top_k",
        type=int,
        default=5,
        help="RAG로 검색할 문단 개수",
    )
    p.add_argument(
        "--rag_max_context_chars",
        type=int,
        default=1500,
        help="RAG 컨텍스트 최대 글자 수",
    )
    # 🔹 BERTScore / METEOR 옵션 (필요하면 끄고 켤 수 있게)
    p.add_argument(
        "--use_bertscore",
        action="store_true",
        help="BERTScore를 계산할지 여부",
    )
    p.add_argument(
        "--bertscore_model_type",
        type=str,
        default="klue/bert-base",
        help="BERTScore에 사용할 HF 모델 이름 (예: klue/bert-base, xlm-roberta-large 등)",
    )
    p.add_argument(
        "--use_meteor",
        action="store_true",
        help="METEOR 점수를 계산할지 여부",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    # 1) config 로드
    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        cfg_path = run_dir / "config.yaml"

    print(f"[Eval-Gen] Loading config from: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # 2) test 데이터 로드
    test_path = dataset_cfg["test_path"]
    print(f"[Eval-Gen] Loading test dataset from: {test_path}")
    test_dataset = load_law_sft_dataset(test_path)
    print(f"[Eval-Gen] Total test samples: {len(test_dataset)}")

    if args.max_samples is not None:
        test_dataset = test_dataset.select(
            range(min(args.max_samples, len(test_dataset)))
        )
        print(f"[Eval-Gen] Using subset of {len(test_dataset)} samples for evaluation.")

    # 3) 학습된 LoRA 모델 로드
    print(f"[Eval-Gen] Loading fine-tuned model from: {run_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(run_dir),
        max_seq_length=int(model_cfg.get("max_seq_length", 2048)),
        dtype=None,
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
    )
    FastLanguageModel.for_inference(model)

    # 4) ROUGE scorer 준비
    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
    )

    # 5) RAG 리소스 로드 (옵션)
    if args.use_rag:
        print(f"[Eval-Gen] RAG mode ON. Loading index from: {RAG_INDEX_PATH}")
        index = faiss.read_index(str(RAG_INDEX_PATH))
        metas = load_rag_meta(RAG_META_PATH)
        print(f"[Eval-Gen] Loaded {len(metas)} RAG metadata.")

        print(f"[Eval-Gen] Loading embedding model: {RAG_EMBED_MODEL}")
        embed_model = SentenceTransformer(RAG_EMBED_MODEL)
    else:
        index = None
        metas = None
        embed_model = None

    # 6) per-sample 결과 수집
    refs = []
    hyps = []
    metrics_per_sample = []

    # 🔹 RAG 여부에 따라 파일 이름 자동 변경
    base_results_name = Path(args.output_file)  # 예: gen_eval_results.jsonl
    if base_results_name.suffix == "":
        base_results_name = base_results_name.with_suffix(".jsonl")

    suffix = "rag" if args.use_rag else "norag"
    results_filename = f"{base_results_name.stem}_{suffix}{base_results_name.suffix}"
    summary_filename = f"gen_eval_summary_{suffix}.json"

    output_path = run_dir / results_filename
    summary_path = run_dir / summary_filename

    fout = output_path.open("w", encoding="utf-8")

    print(f"[Eval-Gen] Start generating and evaluating...")
    print(f"[Eval-Gen] Per-sample results -> {output_path}")

    # 🔹 tqdm으로 진행 상황 표시
    for i, example in enumerate(
        tqdm(test_dataset, desc="[Eval-Gen] Samples", total=len(test_dataset))
    ):
        ex_id = example.get("id", f"sample_{i}")
        messages = example["messages"]

        # 마지막 assistant를 정답으로 사용
        if messages[-1]["role"] != "assistant":
            ref_idx = None
            for idx in range(len(messages) - 1, -1, -1):
                if messages[idx]["role"] == "assistant":
                    ref_idx = idx
                    break
            if ref_idx is None:
                print(f"[Warn] No assistant message found in {ex_id}, skip.")
                continue
        else:
            ref_idx = len(messages) - 1

        ref_answer = messages[ref_idx]["content"]

        # 프롬프트용 메시지는 ref 이전까지 사용
        prompt_messages = messages[:ref_idx]

        # 🔹 use_rag인 경우, user 질문으로 검색 → system에 컨텍스트 추가
        if args.use_rag:
            user_text = ""
            for m in prompt_messages:
                if m.get("role") == "user":
                    user_text = m.get("content", "").strip()
                    break

            if user_text:
                retrieved = rag_retrieve(
                    query=user_text,
                    embed_model=embed_model,
                    index=index,
                    metas=metas,
                    top_k=args.rag_top_k,
                )
                if retrieved:
                    ctx_str = build_rag_context_str(
                        retrieved,
                        max_chars=args.rag_max_context_chars,
                    )
                    prompt_messages = add_rag_to_messages(prompt_messages, ctx_str)

        # 프롬프트 생성
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )

        gen_ids = generated[0][inputs["input_ids"].shape[1]:]
        hyp_answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        refs.append(ref_answer)
        hyps.append(hyp_answer)

        # ROUGE per-sample
        r_scores = rouge.score(ref_answer, hyp_answer)
        sample_metric = {
            "id": ex_id,
            "ref": ref_answer,
            "hyp": hyp_answer,
            "rouge1": r_scores["rouge1"].fmeasure,
            "rouge2": r_scores["rouge2"].fmeasure,
            "rougeL": r_scores["rougeL"].fmeasure,
        }
        fout.write(json.dumps(sample_metric, ensure_ascii=False) + "\n")
        metrics_per_sample.append(sample_metric)

    fout.close()
    print(f"[Eval-Gen] Saved per-sample results to: {output_path}")

    # 7) corpus-level BLEU 계산
    bleu = corpus_bleu(hyps, [refs])

    # 8) 평균 ROUGE 계산
    avg_rouge1 = sum(m["rouge1"] for m in metrics_per_sample) / len(metrics_per_sample)
    avg_rouge2 = sum(m["rouge2"] for m in metrics_per_sample) / len(metrics_per_sample)
    avg_rougeL = sum(m["rougeL"] for m in metrics_per_sample) / len(metrics_per_sample)

    # 9) BERTScore / METEOR 계산 (옵션)
    bert_p = bert_r = bert_f = None
    meteor_avg = None

    if args.use_bertscore:
        print(f"[Eval-Gen] Computing BERTScore with model: {args.bertscore_model_type}")
        # lang='ko'는 다국어 모델일 때 optional, 한국어라서 같이 줘도 됨
        P, R, F1 = bertscore_score(
            hyps,
            refs,
            model_type=args.bertscore_model_type,
            lang="ko",
            verbose=True,
        )
        bert_p = float(P.mean())
        bert_r = float(R.mean())
        bert_f = float(F1.mean())

    if args.use_meteor:
        print("[Eval-Gen] Computing METEOR...")
        meteor_scores = []
        for ref, hyp in zip(refs, hyps):
            ref_tokens = ref.split()
            hyp_tokens = hyp.split()
            meteor_scores.append(meteor_score([ref_tokens], hyp_tokens))
        meteor_avg = float(sum(meteor_scores) / len(meteor_scores))

    # 10) 요약 지표 저장
    summary = {
        "num_samples": len(metrics_per_sample),
        "bleu_score": bleu.score,
        "rouge1_f": avg_rouge1,
        "rouge2_f": avg_rouge2,
        "rougeL_f": avg_rougeL,
        "use_rag": args.use_rag,
        "rag_top_k": args.rag_top_k if args.use_rag else None,
        "rag_max_context_chars": args.rag_max_context_chars if args.use_rag else None,
        "use_bertscore": args.use_bertscore,
        "bertscore_model_type": args.bertscore_model_type if args.use_bertscore else None,
        "bertscore_P": bert_p,
        "bertscore_R": bert_r,
        "bertscore_F1": bert_f,
        "use_meteor": args.use_meteor,
        "meteor": meteor_avg,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== Generation Evaluation Summary =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\n[Eval-Gen] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
