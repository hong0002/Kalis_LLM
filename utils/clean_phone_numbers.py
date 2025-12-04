'''
python utils/clean_phone_numbers.py --input datasets/processed/law_valid.jsonl --output datasets/processed/law_valid_nophone.jsonl

'''

import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm


# 전화번호 / 담당자 전체를 날리고 싶을 때:
#   예) "(업무담당 김효원 ☏ 055-771-8442)" 통째로 삭제
PAREN_PHONE_PATTERN = re.compile(r"\([^)]*?[☏☎]\s*[\d\-]+\)", re.UNICODE)

# 일반 전화번호 패턴 (02-123-4567, 055-771-8442, 010-1234-5678 등)
PHONE_PATTERN = re.compile(
    r"\b\d{2,4}-\d{3,4}-\d{4}\b"
)

def clean_text(text: str) -> str:
    if not text:
        return text

    # 1) "(업무담당 ... ☏ 055-771-8442)" 같은 괄호 덩어리 통째로 삭제
    text = PAREN_PHONE_PATTERN.sub("", text)

    # 2) 남아있는 전화번호 패턴은 마스킹
    text = PHONE_PATTERN.sub("***-****-****", text)

    # 3) 공백 정리
    text = " ".join(text.split())
    return text


def process_file(input_path: Path, output_path: Path):
    print(f"[Clean] Input : {input_path}")
    print(f"[Clean] Output: {output_path}")

    n_in = 0
    n_out = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc=f"Cleaning {input_path.name}"):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            n_in += 1

            messages = obj.get("messages", [])
            new_messages = []
            for m in messages:
                content = m.get("content", "")
                content_clean = clean_text(content)
                new_messages.append({
                    **m,
                    "content": content_clean,
                })

            obj["messages"] = new_messages
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[Clean] Done. {n_in} -> {n_out} samples")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="원본 SFT jsonl 경로 (예: datasets/processed/law_train.jsonl)",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="정제 후 저장할 jsonl 경로 (예: datasets/processed/law_train_nophone.jsonl)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    process_file(input_path, output_path)


if __name__ == "__main__":
    main()