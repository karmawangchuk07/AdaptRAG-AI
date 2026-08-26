import os
import hashlib
import uuid
from typing import List

from langchain_chroma import Chroma
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from src.helper import (
    load_pdf_file,
    text_split,
    download_hugging_face_embeddings
)

DB_PATH = "chroma_db"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "adaptrag_docling")
EMBED_DIM = 384


def create_vector_db():
    print("Creating new baseline vector DB...")
    docs = load_pdf_file("data")
    for i, doc in enumerate(docs):
        doc.metadata["source"] = doc.metadata.get("source", f"doc_{i}")
        doc.metadata["chunk_id"] = i
    texts = text_split(docs)
    print(f"Total baseline chunks: {len(texts)}")
    embedding = download_hugging_face_embeddings()
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=DB_PATH
    )
    print("Baseline vector DB created successfully!")
    return vectordb


def load_baseline_db():
    embedding = download_hugging_face_embeddings()
    if not os.path.exists(DB_PATH):
        return create_vector_db()
    print("Loading existing baseline vector DB...")
    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embedding)
    return vectordb


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection(client: QdrantClient):
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        print(f"[INFO] Created Qdrant collection '{COLLECTION_NAME}'")


def chunk_id(text: str, source: str) -> str:
    h = hashlib.sha256(f"{source}::{text}".encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, h))


def docling_chunks(data_dir: str = "data") -> List[Document]:
    converter = DocumentConverter()
    chunker = HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2")
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    all_docs: List[Document] = []
    for fname in pdf_files:
        path = os.path.join(data_dir, fname)
        try:
            result = converter.convert(path)
        except Exception as e:
            print(f"[SKIP] {fname}: {e}")
            continue
        chunk_iter = chunker.chunk(dl_doc=result.document)
        for i, chunk in enumerate(chunk_iter):
            enriched = chunker.contextualize(chunk).strip()
            if len(enriched) > 50:
                all_docs.append(Document(
                    page_content=enriched,
                    metadata={"source": fname, "chunk_id": i}
                ))
    print(f"Total docling chunks parsed: {len(all_docs)}")
    return all_docs


def incremental_upsert(docs: List[Document], batch_size: int = 100):
    client = get_qdrant_client()
    ensure_collection(client)
    embedding = download_hugging_face_embeddings()
    ids = [chunk_id(d.page_content, d.metadata.get("source", "")) for d in docs]
    existing_ids = set()
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=batch,
            with_payload=False,
            with_vectors=False,
        )
        existing_ids.update(p.id for p in points)
    new_docs, new_ids = [], []
    for doc, id_ in zip(docs, ids):
        if id_ not in existing_ids:
            new_docs.append(doc)
            new_ids.append(id_)
    print(f"New chunks to embed: {len(new_docs)} (skipped {len(docs) - len(new_docs)} already ingested)")
    if not new_docs:
        print("Nothing new to ingest.")
        return
    vectordb = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding,
    )
    vectordb.add_documents(documents=new_docs, ids=new_ids)
    print(f"Upserted {len(new_docs)} chunks into '{COLLECTION_NAME}'")


def docling_db(data_dir: str = "data"):
    docs = docling_chunks(data_dir)
    incremental_upsert(docs)
    print("Docling ingestion complete.")
    return load_docling_db()


def load_docling_db():
    embedding = download_hugging_face_embeddings()
    client = get_qdrant_client()
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding,
    )


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "both"
    if target in ("baseline", "both"):
        create_vector_db()
    if target in ("docling", "both"):
        docling_db()
