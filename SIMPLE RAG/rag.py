"""Simple RAG workflow with LlamaIndex + Ollama.

This script builds a Retrieval-Augmented Generation pipeline that runs fully
locally:

  documents (data/)  ->  chunks  ->  embeddings (nomic-embed-text via Ollama)
                     ->  vector index (persisted to storage/)
                     ->  retrieval + generation (gemma3:4b via Ollama)

Usage:
    python rag.py                       # interactive Q&A loop
    python rag.py "your question here"  # ask a single question and exit
    python rag.py --rebuild             # force re-indexing of the documents
"""

import argparse
import os
import sys

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

import config


def configure_models() -> None:
    """Point LlamaIndex at the local Ollama LLM and embedding model."""
    Settings.llm = Ollama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        request_timeout=config.REQUEST_TIMEOUT,
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


def build_index() -> VectorStoreIndex:
    """Load documents from DATA_DIR, embed them, and persist the index."""
    if not os.path.isdir(config.DATA_DIR) or not os.listdir(config.DATA_DIR):
        sys.exit(
            f"No documents found in '{config.DATA_DIR}/'. "
            "Add some .txt/.md/.pdf files and try again."
        )

    print(f"Reading documents from '{config.DATA_DIR}/' ...")
    documents = SimpleDirectoryReader(config.DATA_DIR).load_data()
    print(f"  Loaded {len(documents)} document(s).")

    print("Building vector index (embedding chunks) ...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    index.storage_context.persist(persist_dir=config.STORAGE_DIR)
    print(f"  Index persisted to '{config.STORAGE_DIR}/'.")
    return index


def load_index() -> VectorStoreIndex:
    """Load a previously persisted index from STORAGE_DIR."""
    print(f"Loading existing index from '{config.STORAGE_DIR}/' ...")
    storage_context = StorageContext.from_defaults(persist_dir=config.STORAGE_DIR)
    return load_index_from_storage(storage_context)


def get_index(rebuild: bool = False) -> VectorStoreIndex:
    """Return a ready-to-query index, building it only when needed."""
    index_exists = os.path.isdir(config.STORAGE_DIR) and os.listdir(config.STORAGE_DIR)
    if rebuild or not index_exists:
        return build_index()
    return load_index()


def answer(query_engine, question: str) -> None:
    """Run one question through the query engine and print answer + sources."""
    response = query_engine.query(question)

    print("\n" + "=" * 70)
    print(f"Q: {question}")
    print("-" * 70)
    print(f"A: {response}")

    sources = getattr(response, "source_nodes", []) or []
    if sources:
        print("-" * 70)
        print("Sources:")
        for i, node in enumerate(sources, start=1):
            file_name = node.metadata.get("file_name", "unknown")
            snippet = node.get_content().strip().replace("\n", " ")[:140]
            print(f"  [{i}] ({node.score:.3f}) {file_name}: {snippet}...")
    print("=" * 70 + "\n")


def interactive_loop(query_engine) -> None:
    """Prompt the user for questions until they type 'exit'."""
    print("\nRAG ready. Ask a question (type 'exit' or 'quit' to stop).\n")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        answer(query_engine, question)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple RAG with LlamaIndex + Ollama.")
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to ask. If omitted, an interactive loop starts.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-reading and re-indexing of the documents.",
    )
    args = parser.parse_args()

    configure_models()
    index = get_index(rebuild=args.rebuild)
    query_engine = index.as_query_engine(similarity_top_k=config.TOP_K)

    if args.question:
        answer(query_engine, " ".join(args.question))
    else:
        interactive_loop(query_engine)


if __name__ == "__main__":
    main()
