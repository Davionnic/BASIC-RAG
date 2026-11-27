# RAG Projects

A collection of Retrieval-Augmented Generation (RAG) tools. Each project lives in its own folder with its own dependencies and documentation.

## Projects

| Project | Description |
|---------|-------------|
| [SIMPLE RAG](SIMPLE%20RAG/README.md) | Fully local RAG with LlamaIndex and Ollama — no API keys, no cloud |
| [DOCUMENT CHAT RAG](DOCUMENT%20CHAT%20RAG/README.md) | Streamlit chat over your PDFs/TXT with FAISS + Llama via Ollama |
| [FAST RAG](FAST%20RAG/README.md) | Fast cloud RAG with SambaNova Llama 3.3, LlamaIndex, Qdrant, Streamlit |
| [GITHUB RAG](GITHUB%20RAG/README.md) | Chat with a GitHub repo via GitIngest + LlamaIndex, local (Ollama) or OpenAI |

## How it works

### SIMPLE RAG

```mermaid
flowchart TD
  data["Documents in data/"] --> reader[LlamaIndex SimpleDirectoryReader]
  reader --> index["VectorStoreIndex with Ollama nomic-embed-text"]
  index --> persist["Persist to storage/"]
  question[Question] --> query["Query engine, top 3 chunks"]
  persist --> query
  query --> llm["Ollama gemma3:4b"]
  llm --> answer["Answer plus source file names and scores"]
```

### DOCUMENT CHAT RAG

```mermaid
flowchart TD
  upload["Upload PDF, TXT, or MD"] --> load["PyPDFLoader or TextLoader"]
  load --> split["LangChain splitter, chunk 1000, overlap 150"]
  split --> faiss["FAISS with Ollama nomic-embed-text"]
  question[Question] --> retrieve["Retrieve top 4 chunks"]
  faiss --> retrieve
  retrieve --> llm["ChatOllama, default llama3.1:8b"]
  llm --> answer["Streamlit streams the answer and cites source file names"]
```

### FAST RAG

```mermaid
flowchart TD
  pdf["Upload a PDF"] --> reader[LlamaIndex SimpleDirectoryReader]
  reader --> embed["Embed with BAAI/bge-large-en-v1.5"]
  embed --> qdrant["Store in Qdrant, local or QDRANT_URL"]
  question[Question] --> retrieve["Retrieve top 2 passages"]
  qdrant --> retrieve
  retrieve --> llm["SambaNova Meta-Llama-3.3-70B-Instruct"]
  llm --> answer["Stream the answer in Streamlit"]
```

### GITHUB RAG

```mermaid
flowchart TD
  url["GitHub repo URL"] --> ingest[GitIngest markdown digest]
  ingest --> split["SentenceSplitter, chunk 1024, overlap 100"]
  split --> index["LlamaIndex index with BAAI/bge-small-en-v1.5"]
  question[Question] --> query["Query engine, top 4 chunks"]
  index --> query
  query --> llm["Ollama llama3.1:8b or OpenAI gpt-4o-mini"]
  llm --> answer["Stream the answer in Streamlit"]
```

## Repository layout

```
BASIC-RAG/
├── SIMPLE RAG/           # Basic local RAG pipeline (LlamaIndex + Ollama)
├── DOCUMENT CHAT RAG/    # Streamlit document chat (LangChain + FAISS + Ollama)
├── FAST RAG/             # SambaNova + LlamaIndex + Qdrant + Streamlit
├── GITHUB RAG/           # GitIngest + LlamaIndex chat over a GitHub repo
└── README.md             # This file
```

Each subproject has its own `README.md`, `requirements.txt`, and `.gitignore`. Install and run from inside the project folder you care about.
