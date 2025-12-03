import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from datasets import DatasetDict

from src.models import build_kanana_lora_model_and_tokenizer
from src.data import load_law_sft_dataset
from src.training import create_sft_trainer
from utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/law_sft_kanana.yaml",
        help="학습 설정 yaml 경로",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) config 로드
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]
    seed = int(cfg.get("seed", 42))

    # 2) 런 디렉터리 생성 (현재 시간 + 모델 이름)
    base_output_dir = Path(training_cfg["output_dir"])
    model_name_short = model_cfg["name"].split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / f"{timestamp}_{model_name_short}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # training_cfg 의 output_dir 을 실제 런 폴더로 교체
    training_cfg["output_dir"] = str(run_dir)
    cfg["training"]["output_dir"] = str(run_dir)

    # ✅ TensorBoard 로그도 run_dir 아래에 저장되도록 설정
    tb_dir = run_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    training_cfg["logging_dir"] = str(tb_dir)
    cfg["training"]["logging_dir"] = str(tb_dir)

    # 3) config / 스크립트 백업
    shutil.copy2(args.config, run_dir / "config.yaml")
    this_script_path = Path(__file__).resolve()
    shutil.copy2(this_script_path, run_dir / "train_sft.py")
    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    print(f"[Run] Output directory: {run_dir}")

    # 4) 시드 고정
    set_seed(seed)

    # 5) train / valid 데이터 각각 로드
    train_dataset = load_law_sft_dataset(dataset_cfg["train_path"])
    eval_dataset = load_law_sft_dataset(dataset_cfg["valid_path"])

    # 7) 모델 + 토크나이저 로드 (Unsloth + LoRA)
    model, tokenizer = build_kanana_lora_model_and_tokenizer(
        model_cfg, lora_cfg, {**training_cfg, "seed": seed}
    )

    # 8) Trainer 생성 (TensorBoard + best model 설정 포함)
    trainer = create_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
    )

    # 9) 학습
    trainer.train()

    # 10) load_best_model_at_end=True 인 경우
    #     -> trainer.model 은 이미 best checkpoint 로 교체된 상태
    print("[Train] Saving (best) model & tokenizer...")
    trainer.model.save_pretrained(run_dir)
    trainer.tokenizer.save_pretrained(run_dir)
    print("[Train] Done.")


if __name__ == "__main__":
    main()
