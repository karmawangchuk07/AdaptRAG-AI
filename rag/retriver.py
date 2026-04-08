from langchain_chroma import Chroma
from src.helper import download_hugging_face_embeddings

DB_PATH = "chroma_db"
DOCLING_DB_PATH = "chroma_db_docling"

USE_DOCLING = True  

_embedding = download_hugging_face_embeddings()
_vectordb = None


def get_baseline_retriever(k=5):
    global _vectordb

    if _vectordb is None:
        print("[INFO] Loading Baseline Chroma DB...")

        _vectordb = Chroma(
            persist_directory=DB_PATH,
            embedding_function=_embedding
        )

    return _vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def get_docling_retriever(k=5):
    print("[INFO] Loading Docling Chroma DB...")

    vectordb = Chroma(
        persist_directory=DOCLING_DB_PATH,
        embedding_function=_embedding
    )

    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def get_retriever(k=5):

    if USE_DOCLING:
        print("[INFO] Using DOCLING retriever")
        return get_docling_retriever(k)
    else:
        print("[INFO] Using BASELINE retriever")
        return get_baseline_retriever(k)