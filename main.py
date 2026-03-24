from __future__ import annotations

import argparse

from SemanticSearch.document_embedding import embed_text
from SemanticSearch.semantic_search import ask_question, interactive_search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SemanticSearch mini-project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("embed", help="Embed the configured PDF into Chroma")

    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("question", help="Question to ask about the document")

    subparsers.add_parser("chat", help="Start interactive question-answering")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "embed":
        embed_text()
    elif args.command == "ask":
        answer = ask_question(args.question)
        print("\nAnswer:\n")
        print(answer)
    elif args.command == "chat":
        interactive_search()


if __name__ == "__main__":
    main()
