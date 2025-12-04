import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer
from unsloth import FastLanguageModel

from utils import set_seed


INDEX_DIR = Path("datasets/processed/rag_index")
INDEX_PATH = INDEX_DIR / "rag_index.faiss"
META_PATH = INDEX_DIR / "rag_meta.jsonl"

EMBED_MODEL_NAME = "nlpai-lab/KURE-v1"  # build 때 쓴 모델과 같아야 함


def load_meta() -> list:
    metas = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas


def retrieve(query: str, embed_model, index, metas, top_k: int = 5):
    # 쿼리 임베딩
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, top_k)
    idxs = idxs[0]
    scores = scores[0]

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


def build_context(retrieved: list) -> str:
    """모델에게 줄 컨텍스트 문자열 생성"""
    lines = []
    lines.append("다음은 관련 법령 및 유권해석에서 발췌한 내용입니다.\n")
    for r in retrieved:
        header = f"[{r['rank']}] {r['source_file']} p.{r['page']} (유사도: {r['score']:.3f})"
        lines.append(header)
        lines.append(r["text"].strip())
        lines.append("")  # 빈 줄
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="학습된 LoRA 결과 폴더 (adapter_model.safetensors, config.yaml 위치)",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=5,
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    # 1) config 로드 (seed, 시스템 프롬프트 등에 사용)
    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        cfg_path = run_dir / "config.yaml"

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # 2) RAG 인덱스/메타/임베딩 모델 로드
    print(f"[RAG] Loading index from: {INDEX_PATH}")
    index = faiss.read_index(str(INDEX_PATH))
    metas = load_meta()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # 3) LoRA 모델 로드
    print(f"[RAG] Loading fine-tuned model from: {run_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(run_dir),
        max_seq_length=int(model_cfg.get("max_seq_length", 2048)),
        dtype=None,
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
    )
    FastLanguageModel.for_inference(model)

    # 4) 인터랙티브 루프
    print("\n[RAG] 국토 법령 RAG 챗봇입니다. 'exit' 입력 시 종료.\n")

    while True:
        query = input("질문> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("종료합니다.")
            break

        # (1) 관련 청크 검색
        retrieved = retrieve(query, embed_model, index, metas, top_k=args.top_k)
        context_str = build_context(retrieved)

        # (2) Chat 템플릿용 메시지 구성
        system_msg = {
            "role": "system",
            "content": (
                "당신은 국토교통부, 국토안전관리원 등에서 제공한 법령해석과 "
                "관련 법령 문서를 기반으로 상담하는 AI입니다. "
                "아래 제공된 법령 발췌를 우선적으로 참고하여, "
                "질문에 대해 한국어로 차분히 설명해 주세요. "
                "근거가 되는 조문이나 해석이 있다면 함께 밝혀 주세요."
            ),
        }
        user_msg = {
            "role": "user",
            "content": (
                f"질문:\n{query}\n\n"
                f"참고할 관련 법령 발췌 내용은 다음과 같습니다.\n\n{context_str}"
            ),
        }

        messages = [system_msg, user_msg]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )

        gen_ids = generated[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            gen_ids,
            skip_special_tokens=True,
        ).strip()

        print("\n=== 답변 ===")
        print(answer)
        print("\n=== 사용된 RAG 컨텍스트 요약 ===")
        for r in retrieved:
            print(f"- [{r['rank']}] {r['source_file']} p.{r['page']} (score={r['score']:.3f})")
        print()


if __name__ == "__main__":
    main()