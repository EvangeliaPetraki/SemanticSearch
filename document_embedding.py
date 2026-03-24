from __future__ import annotations

from pathlib import Path

import chromadb
import fitz
from sentence_transformers import SentenceTransformer

from SemanticSearch.config import (
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBED_MODEL,
    PDF_PATH,
)


def extract_text_from_pdf(pdf_path: Path) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text")
    return text


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def embed_text(
    pdf_path: Path | None = None,
    chroma_path: Path | None = None,
    embed_model: str | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
    collection_name: str | None = None,
) -> None:
    pdf_path = pdf_path or PDF_PATH
    chroma_path = chroma_path or CHROMA_PATH
    embed_model = embed_model or EMBED_MODEL
    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP
    collection_name = collection_name or COLLECTION_NAME

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    chroma_path.mkdir(parents=True, exist_ok=True)

    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size, overlap)
    print(f"Split text into {len(chunks)} chunks")

    print("Loading embedding model...")
    model = SentenceTransformer(embed_model)

    print("Creating Chroma collection...")
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    collection = chroma_client.get_or_create_collection(name=collection_name)

    print("Computing embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
    )
    print("Text embedded and stored successfully.")


if __name__ == "__main__":
    embed_text()
