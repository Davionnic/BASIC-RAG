"""Central configuration for the Simple RAG workflow.

Edit the values here (or set the matching environment variables) to point the
pipeline at different models, folders, or an Ollama server on another host.
"""

import os

# Ollama server endpoint. Ollama listens on port 11434 by default.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Chat model used to generate the final answer.
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "gemma3:4b")

# Embedding model used to turn text chunks into vectors.
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")

# Folder that holds the source documents to index.
DATA_DIR = os.getenv("RAG_DATA_DIR", "data")

# Folder where the built vector index is persisted so it can be reused.
STORAGE_DIR = os.getenv("RAG_STORAGE_DIR", "storage")

# How many of the most relevant chunks to retrieve for each question.
TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# Seconds to wait for the LLM before timing out (local models can be slow).
REQUEST_TIMEOUT = float(os.getenv("RAG_REQUEST_TIMEOUT", "120"))
