"""Chat with a GitHub repo.

GitIngest clones the repo and flattens it into a single markdown digest,
LlamaIndex chunks + embeds that digest (HuggingFace, always local) and
answers questions over it using either Ollama or OpenAI as the chat model.
"""

import os

import streamlit as st
from dotenv import load_dotenv
from gitingest import ingest
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

load_dotenv()

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def configure_models(backend: str, chat_model: str) -> None:
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    if backend == "OpenAI":
        Settings.llm = OpenAI(model=chat_model)
    else:
        Settings.llm = Ollama(model=chat_model, request_timeout=120)


def parse_patterns(raw: str):
    patterns = {p.strip() for p in raw.split(",") if p.strip()}
    return patterns or None


def build_index(repo_url: str, include_patterns: str, exclude_patterns: str):
    summary, _tree, content = ingest(
        repo_url,
        include_patterns=parse_patterns(include_patterns),
        exclude_patterns=parse_patterns(exclude_patterns),
    )
    if not content.strip():
        raise ValueError("GitIngest returned no content — check the URL and include/exclude patterns.")

    document = Document(text=content, metadata={"source": repo_url})
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    index = VectorStoreIndex.from_documents(
        [document], transformations=[splitter], show_progress=True
    )
    return index, summary


st.set_page_config(page_title="Chat with a GitHub repo", page_icon="🐙")
st.title("🐙 Chat with a GitHub repo")

with st.sidebar:
    st.header("Setup")
    backend = st.radio("Chat backend", ["Ollama (local)", "OpenAI"], index=0)
    if backend == "OpenAI":
        chat_model = st.text_input("Chat model (OpenAI)", value="gpt-4o-mini")
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("Set OPENAI_API_KEY in .env to use the OpenAI backend.")
    else:
        chat_model = st.text_input("Chat model (Ollama)", value="llama3.1:8b")

    st.divider()
    repo_url = st.text_input("GitHub repo URL", placeholder="https://github.com/owner/repo")
    include_patterns = st.text_input("Include patterns (comma-separated, optional)")
    exclude_patterns = st.text_input("Exclude patterns (comma-separated, optional)")

    if st.button("Ingest repo", disabled=not repo_url):
        with st.spinner("Cloning and ingesting repo with GitIngest..."):
            try:
                configure_models(backend, chat_model)
                index, summary = build_index(repo_url, include_patterns, exclude_patterns)
                st.session_state.index = index
                st.session_state.summary = summary
                st.session_state.messages = []
                st.success("Repo indexed. Ask away.")
            except Exception as e:  # surface ingest/auth/index errors to the user
                st.error(f"Could not ingest/index repo: {e}")

    st.caption("First run with Ollama: `ollama pull llama3.1:8b`")

st.session_state.setdefault("messages", [])

if "summary" in st.session_state:
    with st.expander("Repo summary (GitIngest)"):
        st.text(st.session_state.summary)

for role, text in st.session_state.messages:
    st.chat_message(role).write(text)

if prompt := st.chat_input("Ask a question about the repo"):
    if "index" not in st.session_state:
        st.warning("Enter a GitHub repo URL and click **Ingest repo** first.")
        st.stop()

    st.session_state.messages.append(("user", prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        try:
            query_engine = st.session_state.index.as_query_engine(
                streaming=True, similarity_top_k=4
            )
            response = query_engine.query(prompt)
            reply = st.write_stream(response.response_gen)
            st.session_state.messages.append(("assistant", reply))
        except Exception as e:
            st.error(f"Generation failed (is Ollama running / is OPENAI_API_KEY set?): {e}")
