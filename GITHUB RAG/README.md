# Chat with a GitHub repo — 100% local RAG

A Streamlit app that lets you chat with any GitHub repository. [GitIngest](https://github.com/cyclotruc/gitingest)
pulls the repo and flattens it into a single markdown digest, then [LlamaIndex](https://www.llamaindex.ai/)
indexes and retrieves over it so you can ask questions grounded in the actual code.

Runs fully offline with [Ollama](https://ollama.com) for both embeddings and generation — no API keys required.
An OpenAI backend is also supported if you'd rather use a hosted model.

## Setup

### 1. Python dependencies

Python 3.9+ (tested with 3.11.9).

```bash
pip install -r requirements.txt
```

<details>
<summary>Manual install (if you don't want to use requirements.txt)</summary>

```bash
pip install gitingest llama-index llama-index-llms-ollama llama-index-llms-openai llama-index-agent-openai llama-index-embeddings-huggingface streamlit pandas python-dotenv huggingface-hub
```
</details>

### 2. Ollama

Make sure the Ollama server is running and pull a model to chat with, e.g.:

```bash
ollama pull llama3.1:8b
```

### 3. OpenAI (optional)

To use OpenAI instead of a local model, copy `.env.example` to `.env` and set your key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run

```bash
streamlit run app_local.py
```

Paste a GitHub repo URL in the sidebar, wait for it to be ingested and indexed, then chat.
