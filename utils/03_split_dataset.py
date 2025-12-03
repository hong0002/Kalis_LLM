import json
import random
from pathlib import Path


def split_jsonl(
    input_path: str,
    train_path: str,
    valid_path: str,
    test_path: str,
    train_ratio: float = 0.9,
    valid_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
):
    """
    하나의 JSONL 파일(한 줄당 1 샘플)을
    train / valid / test 3개 JSONL로 나눔.

    - 비율은 기본 0.9 / 0.05 / 0.05
    - 비율 합이 1.0이 아니어도 자동 normalize 하도록 구현할 수도 있지만,
      여기서는 0.9+0.05+0.05 = 1.0이라고 가정.
    """
    input_path = Path(input_path)
    train_path = Path(train_path)
    valid_path = Path(valid_path)
    test_path = Path(test_path)

    print(f"[Split] Loading from: {input_path}")
    samples = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    n = len(samples)
    print(f"[Split] Total samples: {n}")

    # 셔플
    random.seed(seed)
    random.shuffle(samples)

    # 개수 계산
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    # 나머지는 test로
    n_test = n - n_train - n_valid

    train_samples = samples[:n_train]
    valid_samples = samples[n_train:n_train + n_valid]
    test_samples = samples[n_train + n_valid:]

    print(
        f"[Split] train={len(train_samples)}, "
        f"valid={len(valid_samples)}, test={len(test_samples)}"
    )

    # 디렉토리 생성
    train_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    # 저장 함수
    def write_jsonl(path: Path, data):
        with path.open("w", encoding="utf-8") as f:
            for obj in data:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[Split] Wrote {len(data)} samples to {path}")

    write_jsonl(train_path, train_samples)
    write_jsonl(valid_path, valid_samples)
    write_jsonl(test_path, test_samples)

    print("[Split] Done.")


if __name__ == "__main__":
    # 🔧 경로는 프로젝트 구조에 맞게 수정
    input_jsonl = "datasets/processed/merged_law_kalis_sft_cleaned.jsonl"

    train_jsonl = "datasets/processed/law_train.jsonl"
    valid_jsonl = "datasets/processed/law_valid.jsonl"
    test_jsonl  = "datasets/processed/law_test.jsonl"

    split_jsonl(
        input_path=input_jsonl,
        train_path=train_jsonl,
        valid_path=valid_jsonl,
        test_path=test_jsonl,
        train_ratio=0.9,
        valid_ratio=0.05,
        test_ratio=0.05,
        seed=42,
    )
