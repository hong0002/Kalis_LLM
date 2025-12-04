import os
import json
from pathlib import Path
from typing import List, Dict

import fitz  # pymupdf
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


RAG_PDF_DIR = Path("datasets/processed/rag")
INDEX_DIR = Path("datasets/processed/rag_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "nlpai-lab/KURE-v1"  # 예시 한국어 임베딩 모델


def extract_text_from_pdf(pdf_path: Path) -> List[Dict]:
    """각 페이지 텍스트를 추출해서 [{'page': int, 'text': str}, ...] 형태로 반환"""
    doc = fitz.open(pdf_path)
    pages = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": page_idx + 1, "text": text})
    doc.close()
    return pages


def split_into_chunks(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """문자 단위로 chunk_size 길이로 자르고, overlap 만큼 겹치게 분할"""
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())  # 공백 정규화

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def main():
    pdf_files = sorted(RAG_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[Build] No PDF files found in {RAG_PDF_DIR}")
        return

    print(f"[Build] Found {len(pdf_files)} pdf files.")

    # 1) PDF → 청크 목록 수집
    all_chunks = []  # 청크 텍스트 리스트
    metas = []       # 각 청크에 대한 메타 정보

    for pdf_idx, pdf_path in enumerate(pdf_files):
        print(f"[Build] Processing ({pdf_idx+1}/{len(pdf_files)}): {pdf_path.name}")
        pages = extract_text_from_pdf(pdf_path)

        for page_info in pages:
            page = page_info["page"]
            text = page_info["text"]
            chunks = split_into_chunks(text, chunk_size=600, overlap=100)
            for ci, chunk in enumerate(chunks):
                meta = {
                    "source_file": pdf_path.name,
                    "source_path": str(pdf_path),
                    "page": page,
                    "chunk_id": f"{pdf_path.stem}_p{page}_c{ci}",
                    "text": chunk,
                }
                metas.append(meta)
                all_chunks.append(chunk)

    print(f"[Build] Total chunks: {len(all_chunks)}")

    # 2) 임베딩 모델 로드
    print(f"[Build] Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # 3) 청크 임베딩
    print("[Build] Computing embeddings...")
    embeddings = embed_model.encode(
        all_chunks,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    # 4) FAISS 인덱스 생성 (내적 기반)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    # 벡터 정규화(코사인 유사도 비슷하게 쓰고 싶으면)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # 5) 인덱스 & 메타데이터 저장
    index_path = INDEX_DIR / "rag_index.faiss"
    meta_path = INDEX_DIR / "rag_meta.jsonl"

    faiss.write_index(index, str(index_path))
    print(f"[Build] Saved FAISS index to: {index_path}")

    with meta_path.open("w", encoding="utf-8") as f:
        for meta in metas:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"[Build] Saved metadata to: {meta_path}")

    print("[Build] Done.")


if __name__ == "__main__":
    main()