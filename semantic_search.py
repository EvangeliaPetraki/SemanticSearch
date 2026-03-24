from __future__ import annotations

from pathlib import Path

import chromadb
from ollama import chat
from sentence_transformers import SentenceTransformer

from SemanticSearch.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBED_MODEL,
    OLLAMA_MODEL,
    SUMMARY_FILE,
    TOP_K,
)


def load_summary(summary_file: Path) -> str:
    if not summary_file.exists():
        return ""
    return summary_file.read_text(encoding="utf-8")


def open_collection(chroma_path: Path, collection_name: str):
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_collection(name=collection_name)


def build_prompt(question: str, article_summary: str, context: str) -> str:
    return f"""
You are analysing a study related to green and digital transition policy.
Below is a global summary of the document, followed by retrieved excerpts.

Question:
{question}

Article Summary:
{article_summary}

Context:
{context}

Provide a detailed, well-reasoned answer that connects the retrieved excerpts
to the overall themes of the document.

Answer:
""".strip()


def ask_question(
    query: str,
    *,
    chroma_path: Path | None = None,
    collection_name: str | None = None,
    embed_model: str | None = None,
    ollama_model: str | None = None,
    top_k: int | None = None,
    summary_file: Path | None = None,
) -> str:
    chroma_path = chroma_path or CHROMA_PATH
    collection_name = collection_name or COLLECTION_NAME
    embed_model = embed_model or EMBED_MODEL
    ollama_model = ollama_model or OLLAMA_MODEL
    top_k = top_k or TOP_K
    summary_file = summary_file or SUMMARY_FILE

    if not chroma_path.exists():
        raise FileNotFoundError(f"Chroma database not found: {chroma_path}")

    print("Loading Chroma collection and embedding model...")
    collection = open_collection(chroma_path, collection_name)
    embedder = SentenceTransformer(embed_model)
    article_summary = load_summary(summary_file)

    print("Searching relevant text...")
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    documents = results["documents"][0]

    if not documents:
        return "No relevant text found."

    print(f"Found {len(documents)} relevant chunks:")
    for i, doc in enumerate(documents, start=1):
        preview = doc[:500].replace("\n", " ")
        print(f"--- Chunk {i} ---")
        print(preview)
        print("...")

    context = "\n\n".join(documents)
    prompt = build_prompt(query, article_summary, context)

    print("Asking Ollama...")
    response = chat(model=ollama_model, messages=[{"role": "user", "content": prompt}])
    return response.message.content


def interactive_search() -> None:
    while True:
        query = input("\nEnter your question about the document:\n> ").strip()
        if not query:
            print("\nExiting semantic search. Goodbye!")
            break

        answer = ask_question(query)
        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":
    interactive_search()
