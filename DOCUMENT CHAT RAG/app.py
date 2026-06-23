import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SYSTEM_PROMPT = (
    "You answer strictly from the provided context. "
    "If the answer is not in the context, say you don't know. "
    "Cite the source file names you used."
)


def load_docs(files):
    docs = []
    for f in files:
        suffix = "." + f.name.rsplit(".", 1)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.getvalue())
            path = tmp.name
        loader = PyPDFLoader(path) if suffix == ".pdf" else TextLoader(path, encoding="utf-8")
        for d in loader.load():
            d.metadata["source"] = f.name
            docs.append(d)
    return docs


def build_retriever(files, embed_model):
    docs = load_docs(files)
    if not docs:
        raise ValueError("No text could be extracted from the uploaded files.")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(docs)
    store = FAISS.from_documents(chunks, OllamaEmbeddings(model=embed_model))
    return store.as_retriever(search_kwargs={"k": 4})


def answer(question, retriever, chat_model):
    hits = retriever.invoke(question)
    context = "\n\n".join(f"[{d.metadata.get('source', '?')}]\n{d.page_content}" for d in hits)
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    for chunk in ChatOllama(model=chat_model, temperature=0).stream(messages):
        yield chunk.content


st.set_page_config(page_title="Chat with your docs", page_icon="📄")
st.title("📄 Chat with your docs — local Llama 3.3")

with st.sidebar:
    st.header("Setup")
    chat_model = st.text_input("Chat model (Ollama)", value="llama3.1:8b")
    embed_model = st.text_input("Embedding model (Ollama)", value="nomic-embed-text")
    files = st.file_uploader(
        "Upload PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if st.button("Process documents", disabled=not files):
        with st.spinner("Indexing..."):
            try:
                st.session_state.retriever = build_retriever(files, embed_model)
                st.session_state.messages = []
                st.success(f"Indexed {len(files)} file(s). Ask away.")
            except Exception as e:  # surface Ollama-down / parse errors to the user
                st.error(f"Could not index documents: {e}")
    st.caption("First run: `ollama pull llama3.3` and `ollama pull nomic-embed-text`")

st.session_state.setdefault("messages", [])

for role, text in st.session_state.messages:
    st.chat_message(role).write(text)

if prompt := st.chat_input("Ask a question about your documents"):
    if "retriever" not in st.session_state:
        st.warning("Upload documents and click **Process documents** first.")
        st.stop()
    st.session_state.messages.append(("user", prompt))
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(
                answer(prompt, st.session_state.retriever, chat_model)
            )
            st.session_state.messages.append(("assistant", reply))
        except Exception as e:
            st.error(f"Generation failed (is Ollama running?): {e}")
