import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any


# =========================
# 1. MD 파일 파싱 함수
# =========================

def parse_law_md(md_path: Path) -> Dict[str, Any]:
    """
    국토부/행안부 법령해석 형태의 MD 파일에서
    - 제목 (첫 줄 # ... )
    - 질의요지 (## 【질의요지】 ~ ## 【회답】)
    - 회답 (## 【회답】 ~ 그 다음 섹션 or EOF)
    를 추출해서 dict로 반환.
    """
    text = md_path.read_text(encoding="utf-8")

    # 제목: 맨 첫 줄에 '# ' 로 시작하는 부분
    title_match = re.search(r"^#\s*(.+)", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else md_path.stem

    # 질의요지
    q_match = re.search(
        r"##\s*【질의요지】\s*(.*?)(?=##\s*【회답】)",
        text,
        re.DOTALL
    )

    # 회답
    a_match = re.search(
        r"##\s*【회답】\s*(.*?)(?=##\s*【중앙부처|$\Z)",
        text,
        re.DOTALL
    )

    question = q_match.group(1).strip() if q_match else ""
    answer = a_match.group(1).strip() if a_match else ""

    # 메타 정보 (안건번호, 해석일자 등 필요하면 추가 파싱 가능)
    # 대략적으로 “안건번호:” 같은 패턴을 추출
    case_no_match = re.search(r"안건번호[:：]\s*([^\n]+)", text)
    case_no = case_no_match.group(1).strip() if case_no_match else ""

    law_id_match = re.search(r"법령해석일련번호[:：]\s*([^\n]+)", text)
    law_id = law_id_match.group(1).strip() if law_id_match else ""

    date_match = re.search(r"해석일자[:：]\s*([^\n]+)", text)
    law_date = date_match.group(1).strip() if date_match else ""

    return {
        "source_type": "법령해석",
        "file_name": md_path.name,
        "title": title,
        "case_no": case_no,
        "law_id": law_id,
        "law_date": law_date,
        "question": question,
        "answer": answer,
        "raw_text": text,
    }


# =========================
# 2. 국토안전관리원 민원 JSONL 파싱
# =========================

def parse_qna_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """
    국토안전관리원 민원 답변 JSONL 파일을 로드.
    각 라인은 다음과 같은 형태라고 가정:
    {
        "title": "...",
        "category": "...",
        "question": "...",
        "answer": "...",
        "answer_person": "..."
    }
    """
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["source_type"] = "민원QnA"
            records.append(obj)
    return records


# =========================
# 3. SFT 학습용 예제 생성
#    (Kanana 1.5 8B instruct용 chat 형태)
# =========================

BASE_SYSTEM_PROMPT = (
    "당신은 국토교통부, 국토안전관리원 등에서 제공한 법령해석과 민원 답변을 "
    "기반으로 답변하는 법령 상담 AI입니다. 관련 법령과 기존 유권해석을 존중하면서도, "
    "질문자가 이해하기 쉽게 한국어로 설명해 주세요."
)


def build_example_from_law(law_item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    MD 법령해석 1개에서 chat 형식 학습 샘플 1개 생성.
    """
    title = law_item.get("title", "")
    question = law_item.get("question", "").strip()
    answer = law_item.get("answer", "").strip()

    if not question or not answer:
        # 질의나 회답이 비어있으면 사용하지 않도록
        raise ValueError("Empty question or answer in law item")

    # 유저에게 보여줄 프롬프트
    user_content = (
        f"[사례 유형] 법령해석 사례\n"
        f"[제목]\n{title}\n\n"
        f"[질의]\n{question}\n\n"
        f"위와 같은 질의에 대해, 관련 법령과 기존 해석을 참고하여 "
        f"전문가 답변을 작성해 주세요."
    )

    return {
        "id": f"law_{idx:06d}",
        "messages": [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ],
        "meta": {
            "source_type": "법령해석",
            "title": title,
            "case_no": law_item.get("case_no"),
            "law_id": law_item.get("law_id"),
            "law_date": law_item.get("law_date"),
            "file_name": law_item.get("file_name"),
        },
    }


def build_example_from_qna(qna_item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    국토안전관리원 민원 Q&A 1개에서 chat 형식 학습 샘플 1개 생성.
    """
    title = qna_item.get("title", "")
    category = qna_item.get("category", "")
    question = qna_item.get("question", "").strip()
    answer = qna_item.get("answer", "").strip()
    answer_person = qna_item.get("answer_person", "")

    if not question or not answer:
        raise ValueError("Empty question or answer in qna item")

    user_content = (
        f"[사례 유형] 민원 질의\n"
        f"[민원 제목]\n{title}\n\n"
        f"[분류]\n{category}\n\n"
        f"[질문]\n{question}\n\n"
        f"위 민원에 대해, 관련 법령과 지침을 고려하여 전문가 관점에서 답변해 주세요."
    )

    # 답변 마지막에 담당자 이름은 그대로 둬도 되고, 깔끔하게 잘라내도 됨
    assistant_content = answer.strip()

    return {
        "id": f"qna_{idx:06d}",
        "messages": [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "meta": {
            "source_type": "민원QnA",
            "title": title,
            "category": category,
            "answer_person": answer_person,
        },
    }


# =========================
# 4. 전체 파이프라인
# =========================

def build_merged_dataset(
    md_dir: str,
    qna_jsonl_path: str,
    output_jsonl_path: str,
):
    """
    - md_dir: 법령해석 md 파일들이 들어 있는 폴더
    - qna_jsonl_path: 민원 Q&A jsonl 파일 경로
    - output_jsonl_path: 통합된 학습 데이터(jsonl) 저장 경로
    """
    md_dir_path = Path(md_dir)
    qna_path = Path(qna_jsonl_path)
    out_path = Path(output_jsonl_path)

    # ---- 1) MD 파싱 ----
    law_items: List[Dict[str, Any]] = []
    for md_file in sorted(md_dir_path.glob("*.md")):
        try:
            item = parse_law_md(md_file)
            if item["question"] and item["answer"]:
                law_items.append(item)
        except Exception as e:
            print(f"[WARN] MD 파싱 실패: {md_file} -> {e}")

    print(f"법령해석 MD 문서 수: {len(law_items)}")

    # ---- 2) QnA JSONL 파싱 ----
    qna_items: List[Dict[str, Any]] = parse_qna_jsonl(qna_path)
    print(f"민원 QnA 수: {len(qna_items)}")

    # ---- 3) SFT 샘플로 변환 ----
    examples: List[Dict[str, Any]] = []

    idx = 0
    for law in law_items:
        try:
            ex = build_example_from_law(law, idx)
            examples.append(ex)
            idx += 1
        except Exception as e:
            print(f"[WARN] 법령해석 샘플 생성 실패: {law.get('file_name')} -> {e}")

    for qna in qna_items:
        try:
            ex = build_example_from_qna(qna, idx)
            examples.append(ex)
            idx += 1
        except Exception as e:
            print(f"[WARN] QnA 샘플 생성 실패: {qna.get('title')} -> {e}")

    print(f"최종 학습 샘플 수: {len(examples)}")

    # ---- 4) JSONL로 저장 ----
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"통합 학습 데이터 저장 완료: {out_path}")


# =========================
# 5. 실행 예시
# =========================

if __name__ == "__main__":
    """
    예시:
    md_dir = "/home/jihong/law_data/md"              # 지방자치단체 공중화장실 등 법령해석 md들
    qna_jsonl = "/home/jihong/law_data/qna.jsonl"    # 국토안전관리원 민원 Q&A 모음
    out_jsonl = "/home/jihong/law_data/merged_qa_law_chat.jsonl"
    """
    md_dir = "datasets/raws/clean"   
    qna_jsonl = "datasets/raws/merged_qna.jsonl"
    out_jsonl = "datasets/processed/merged_law_kalis_sft.jsonl"

    build_merged_dataset(md_dir, qna_jsonl, out_jsonl)
