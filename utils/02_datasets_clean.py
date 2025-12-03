import json
import re
from pathlib import Path


# -------------------------------
# 1. 텍스트 클리닝 함수
# -------------------------------

def clean_text(text: str) -> str:
    if text is None:
        return ""

    # 기본 공백/개행 정리
    text = text.replace("\u00a0", " ")  # non-breaking space -> 일반 공백
    text = text.replace("\t", " ")

    # 특수 따옴표를 일반 따옴표로
    text = text.replace("“", "\"").replace("”", "\"")
    text = text.replace("‘", "'").replace("’", "'")

    # 보기만 잡음인 특수기호들(원하면 계속 추가 가능)
    # ※, ■, ◆, ▲, △, ○, ●, ▶, ▷, ◀, ◁ 등
    text = re.sub(r"[■◆▲△▶▷◀◁○●※★☆◇]", " ", text)

    # 너무 많은 공백/개행 정리
    # 연속 개행 3줄 이상 -> 2줄로 축소
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 줄 내에서 연속 공백 축소
    text = re.sub(r"[ ]{2,}", " ", text)

    # 앞뒤 공백 제거
    text = text.strip()

    return text


# -------------------------------
# 2. 길이 필터 + 클리닝 파이프라인
# -------------------------------

def filter_and_clean_dataset(
    input_jsonl: str,
    output_jsonl: str,
    max_len: int = 1024,
):
    in_path = Path(input_jsonl)
    out_path = Path(output_jsonl)

    kept = 0
    dropped_len = 0
    dropped_err = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                dropped_err += 1
                continue

            msgs = obj.get("messages", [])
            if len(msgs) < 3:
                dropped_err += 1
                continue

            # system / user / assistant 텍스트 가져오기
            # (현재 구조: 0=system, 1=user, 2=assistant 라고 가정)
            for m in msgs:
                if "content" in m:
                    m["content"] = clean_text(m["content"])

            user_text = msgs[1].get("content", "")
            assistant_text = msgs[2].get("content", "")

            # 길이 필터 (질문 or 답변 중 하나라도 max_len 이상이면 제거)
            if len(user_text) >= max_len or len(assistant_text) >= max_len:
                dropped_len += 1
                continue

            # 통과한 샘플만 출력
            obj["messages"] = msgs
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print("=== 전처리 결과 ===")
    print(f"입력 파일: {in_path}")
    print(f"출력 파일: {out_path}")
    print(f"유지된 샘플 수   : {kept}")
    print(f"길이 초과로 제거 : {dropped_len}")
    print(f"파싱/구조 오류 제거: {dropped_err}")


if __name__ == "__main__":
    input_jsonl = "datasets/processed/merged_law_kalis_sft.jsonl"
    output_jsonl = "datasets/processed/merged_law_kalis_sft_cleaned.jsonl"

    filter_and_clean_dataset(input_jsonl, output_jsonl, max_len=1024)
