import os
from langchain_chroma import Chroma

from src.helper import (
    load_pdf_file,
    text_split,
    download_hugging_face_embeddings
)

DB_PATH = "chroma_db"


def create_vector_db():
    print("Creating new vector DB...")

    docs = load_pdf_file("data")

    for i, doc in enumerate(docs):
        doc.metadata["source"] = doc.metadata.get("source", f"doc_{i}")
        doc.metadata["chunk_id"] = i

    texts = text_split(docs)
    print(f" Total chunks: {len(texts)}")

    embedding = download_hugging_face_embeddings()

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=DB_PATH
    )

    vectordb.persist()

    print("Vector DB created successfully!")

    return vectordb


def load_vector_db():
    embedding = download_hugging_face_embeddings()

    if not os.path.exists(DB_PATH):
        # 👉 First time (or after deployment reset)
        return create_vector_db()

    print("Loading existing vector DB...")

    vectordb = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding
    )

    return vectordb