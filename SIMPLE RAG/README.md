# Simple RAG

A basic, **fully local** Retrieval-Augmented Generation (RAG) pipeline built with
[LlamaIndex](https://docs.llamaindex.ai/) and [Ollama](https://ollama.com/). Load
documents from a folder, embed them with a local model, and ask questions
grounded in that knowledge — no API keys, no cloud calls, no per-token cost.

Two ways to use it:

| Interface | File | What it's for |
|-----------|------|----------------|
| CLI       | `rag.py` | Ask one question or run an interactive Q&A loop |
| Notebook  | `rag_workflow.ipynb` | Step-by-step walkthrough of the pipeline |

## Features

- **100% local** — LLM and embeddings run through Ollama on your machine.
- **Persistent index** — builds once, loads fast on later runs (`storage/`).
- **Flexible input** — drop `.txt`, `.md`, `.pdf`, and other supported files into `data/`.
- **Source citations** — answers include the retrieved chunks and relevance scores.
- **Configurable** — models, folders, and retrieval settings via `config.py` or env vars.

## How RAG works here

```
data/ documents -> chunks -> embeddings (nomic-embed-text)
                -> vector index (persisted in storage/)
                -> retrieve top-k chunks -> LLM (gemma3:4b) -> answer
```

1. **Load** documents from `data/` with `SimpleDirectoryReader`.
2. **Chunk & embed** them with the `nomic-embed-text` model via Ollama.
3. **Index** the vectors in a `VectorStoreIndex` and persist to `storage/`.
4. **Retrieve** the most relevant chunks for a question.
5. **Generate** an answer with `gemma3:4b`, grounded in those chunks.

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/download) installed and running
- The two models used by this project:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Ask a single question:

```bash
python rag.py "What is Retrieval-Augmented Generation?"
```

Start an interactive Q&A loop:

```bash
python rag.py
```

Force the index to rebuild after you change the documents in `data/`:

```bash
python rag.py --rebuild
```

The first run reads `data/`, embeds the chunks, and saves the index to
`storage/`. Later runs load that saved index, so they start much faster. Use
`--rebuild` whenever you add or edit documents.

## Project structure

```
.
├── data/                       # source documents (.md/.txt/.pdf/...)
│   └── llamaindex_overview.md  # sample document to query
├── storage/                    # persisted vector index (auto-generated)
├── config.py                   # models, folders, and retrieval settings
├── rag.py                      # RAG pipeline / CLI
├── rag_workflow.ipynb          # step-by-step documented notebook
├── requirements.txt
└── README.md
```

## Configuration

Defaults live in `config.py` and can be overridden with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_LLM_MODEL` | `gemma3:4b` | Ollama chat model for generation |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `RAG_DATA_DIR` | `data` | Folder of source documents |
| `RAG_STORAGE_DIR` | `storage` | Where the index is persisted |
| `RAG_TOP_K` | `3` | Number of chunks retrieved per query |
| `RAG_REQUEST_TIMEOUT` | `120` | LLM request timeout (seconds) |

## Adding your own documents

Drop `.txt`, `.md`, `.pdf`, or other supported files into `data/`, then run
`python rag.py --rebuild` to index them. After that, ask away.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| "No documents found in `data/`" | Add at least one file to `data/` and run again. |
| Ollama connection errors | Start Ollama with `ollama serve` and confirm `http://localhost:11434` is reachable. |
| Model not found | Run `ollama pull gemma3:4b` and `ollama pull nomic-embed-text`. |
| Stale answers after editing docs | Rebuild the index with `python rag.py --rebuild`. |
| First run is slow | Embedding and indexing happen once; later runs load from `storage/`. |
