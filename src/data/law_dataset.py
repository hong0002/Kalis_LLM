from datasets import load_dataset
from pathlib import Path
from typing import Tuple, Any


def load_law_sft_dataset(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Train dataset not found: {path}")

    dataset = load_dataset(
        "json",
        data_files={"train": str(path)},
        split="train",
    )
    print(f"[Dataset] Loaded {len(dataset)} samples from {path}")
    return dataset