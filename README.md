# RAG Projects

A collection of Retrieval-Augmented Generation (RAG) tools. Each project lives in its own folder with its own dependencies and documentation.

## Projects

| Project | Description |
|---------|-------------|
| [SIMPLE RAG](SIMPLE%20RAG/README.md) | Fully local RAG with LlamaIndex and Ollama — no API keys, no cloud |
| [DOCUMENT CHAT RAG](DOCUMENT%20CHAT%20RAG/README.md) | Streamlit chat over your PDFs/TXT with FAISS + Llama via Ollama |

## Repository layout

```
RAGS/
├── SIMPLE RAG/           # Basic local RAG pipeline (LlamaIndex + Ollama)
├── DOCUMENT CHAT RAG/    # Streamlit document chat (LangChain + FAISS + Ollama)
└── README.md             # This file
```

Each subproject has its own `README.md`, `requirements.txt`, and `.gitignore`. Install and run from inside the project folder you care about.
