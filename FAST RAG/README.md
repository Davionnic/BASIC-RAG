# Llama3.3-RAG application

The fastest stack to build a RAG application and chat with your docs:

- **SambaNova** — inference for Llama 3.3
- **LlamaIndex** — RAG orchestration
- **Qdrant** — vector store for embeddings
- **Streamlit** — UI

## Setup

### 1. SambaNova

Get an API key from [SambaNova Cloud](https://cloud.sambanova.ai) and copy `.env.example` to `.env`:

```bash
SAMBANOVA_API_KEY=<YOUR_SAMBANOVA_API_KEY>
```

### 2. Qdrant (optional)

By default the app uses **embedded Qdrant** (`./qdrant_storage`) — no Docker needed.

To use a Qdrant server instead, start Docker then set in `.env`:

```
QDRANT_URL=http://localhost:6333
```

```powershell
docker run -p 6333:6333 -p 6334:6334 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

### 3. Python dependencies

Python 3.11+ recommended.

```bash
pip install -r requirements.txt
```

### 4. Run

```powershell
cd "d:\PROJECTS\RAGS\FAST RAG"
python -m streamlit run app.py
```

On Windows, `streamlit` alone often fails if Python Scripts isn't on PATH — use `python -m streamlit` instead.

Upload a PDF in the sidebar, wait for indexing, then chat.
