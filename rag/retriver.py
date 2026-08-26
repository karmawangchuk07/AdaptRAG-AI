import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from src.helper import download_hugging_face_embeddings

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "adaptrag_docling")

_embedding = download_hugging_face_embeddings()
_vectordb = None


def get_retriever(k=5):
    global _vectordb

    if _vectordb is None:
        print(f"[INFO] Connecting to Qdrant collection '{COLLECTION_NAME}'")
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        _vectordb = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=_embedding,
        )

    return _vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )