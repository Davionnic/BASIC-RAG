import os

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.sambanovasystems import SambaNovaCloud
from qdrant_client import QdrantClient, models


def batch_iterate(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


class EmbedData:
    def __init__(self, embed_model_name="BAAI/bge-large-en-v1.5", batch_size=32):
        self.embed_model_name = embed_model_name
        self.embed_model = HuggingFaceEmbedding(
            model_name=self.embed_model_name,
            trust_remote_code=True,
            cache_folder="./hf_cache",
        )
        self.batch_size = batch_size
        self.embeddings = []

    def embed(self, contexts):
        self.contexts = contexts
        for batch_context in batch_iterate(contexts, self.batch_size):
            batch_embeddings = self.embed_model.get_text_embedding_batch(batch_context)
            self.embeddings.extend(batch_embeddings)


class QdrantVDB_QB:
    def __init__(self, collection_name, vector_dim=1024, batch_size=512):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_dim = vector_dim

    def define_client(self):
        url = os.getenv("QDRANT_URL")
        if url:
            self.client = QdrantClient(url=url, prefer_grpc=True)
        else:
            # ponytail: embedded mode — no Docker server required
            self.client = QdrantClient(path="./qdrant_storage")

    def create_collection(self):
        if self.client.collection_exists(collection_name=self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_dim,
                distance=models.Distance.DOT,
                on_disk=True,
            ),
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=5,
                indexing_threshold=0,
            ),
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True),
            ),
        )

    def ingest_data(self, embeddata):
        for batch_context, batch_embeddings in zip(
            batch_iterate(embeddata.contexts, self.batch_size),
            batch_iterate(embeddata.embeddings, self.batch_size),
        ):
            self.client.upload_collection(
                collection_name=self.collection_name,
                vectors=batch_embeddings,
                payload=[{"context": context} for context in batch_context],
            )

        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )


class Retriever:
    def __init__(self, vector_db, embeddata):
        self.vector_db = vector_db
        self.embeddata = embeddata

    def search(self, query):
        query_embedding = self.embeddata.embed_model.get_query_embedding(query)
        return self.vector_db.client.query_points(
            collection_name=self.vector_db.collection_name,
            query=query_embedding,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=2.0,
                )
            ),
            limit=2,
            timeout=1000,
        ).points


class RAG:
    def __init__(self, retriever, llm_name="Meta-Llama-3.3-70B-Instruct"):
        self.retriever = retriever
        self.llm_name = llm_name
        self.llm = SambaNovaCloud(
            model=self.llm_name,
            temperature=0.7,
            context_window=100000,
        )
        self.qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context}\n"
            "---------------------\n"
            "Given the context information above I want you to think step by step to answer "
            "the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
            "Query: {query}\n"
            "Answer: "
        )

    def generate_context(self, query):
        result = self.retriever.search(query)
        contexts = [entry.payload["context"] for entry in result[:2]]
        return "\n\n---\n\n".join(contexts)

    def query(self, query):
        context = self.generate_context(query=query)
        prompt = self.qa_prompt_tmpl_str.format(context=context, query=query)
        return self.llm.stream_complete(prompt)
