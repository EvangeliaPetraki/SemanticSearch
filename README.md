# SemanticSearch

Small local project for:

- embedding one PDF into a Chroma database
- asking questions over the embedded document with Ollama

## Setup

Edit [`SemanticSearch/.env`](/c:/Users/Evangelia/Documents/ΕΥΑΓΓΕΛΙΑ%20ΕΓΓΡΑΦΑ/UoCrete/tasks/policy_retrieve_and_analysis/SemanticSearch/.env) and set:

- `SEMANTIC_SEARCH_PDF_PATH`
- `SEMANTIC_SEARCH_SUMMARY_FILE`
- `SEMANTIC_SEARCH_CHROMA_PATH`

## Run

Embed the document:

```powershell
python -m SemanticSearch.main embed
```

Ask one question:

```powershell
python -m SemanticSearch.main ask "What are the main transition challenges?"
```

Interactive mode:

```powershell
python -m SemanticSearch.main chat
```
