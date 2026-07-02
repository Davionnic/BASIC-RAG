import base64
import gc
import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader

from rag_code import EmbedData, QdrantVDB_QB, RAG, Retriever

load_dotenv()

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
collection_name = "chat with docs"
batch_size = 32
query_engine = None


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


with st.sidebar:
    st.header("Add your documents!")
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get("file_cache", {}):
                    loader = SimpleDirectoryReader(
                        input_dir=temp_dir,
                        required_exts=[".pdf"],
                        recursive=True,
                    )
                    docs = loader.load_data()
                    documents = [doc.text for doc in docs]

                    embeddata = EmbedData(
                        embed_model_name="BAAI/bge-large-en-v1.5",
                        batch_size=batch_size,
                    )
                    embeddata.embed(documents)

                    qdrant_vdb = QdrantVDB_QB(
                        collection_name=collection_name,
                        batch_size=batch_size,
                        vector_dim=1024,
                    )
                    qdrant_vdb.define_client()
                    qdrant_vdb.create_collection()
                    qdrant_vdb.ingest_data(embeddata=embeddata)

                    retriever = Retriever(vector_db=qdrant_vdb, embeddata=embeddata)
                    query_engine = RAG(
                        retriever=retriever,
                        llm_name="Meta-Llama-3.3-70B-Instruct",
                    )
                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.header("Fastest RAG Stack with SambaNova and Llama-3.3")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

if "messages" not in st.session_state:
    reset_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    if not st.session_state.get("file_cache"):
        st.warning("Upload a PDF in the sidebar first.")
        st.stop()

    active_engine = next(iter(st.session_state.file_cache.values()))

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        streaming_response = active_engine.query(prompt)

        for chunk in streaming_response:
            if chunk.delta:
                full_response += chunk.delta
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
