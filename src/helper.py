from typing import List
import re

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def load_pdf_file(data: str) -> List[Document]:
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    for i, doc in enumerate(documents):
        doc.metadata["page"] = doc.metadata.get("page", i)

    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Keep only useful metadata + clean text
    """
    minimal_docs: List[Document] = []

    for doc in docs:
        src  = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", -1)

        text = clean_text(doc.page_content)

        if not text.strip():
            continue

        minimal_docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": src,
                    "page": page
                }
            )
        )

    return minimal_docs


def clean_text(text: str) -> str:
    """
    Clean PDF noise for better embeddings
    """
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)        
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  
    return text.strip()


def text_split(extracted_data: List[Document]) -> List[Document]:
    """
    High-quality chunking (MOST IMPORTANT for RAG)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,         
        chunk_overlap=100,      
        separators=[
            "\n\n", "\n", ".", "!", "?", ",", " "
        ]
    )

    chunks = splitter.split_documents(extracted_data)

    filtered_chunks = []
    for chunk in chunks:
        if len(chunk.page_content) > 100:  
            filtered_chunks.append(chunk)

    print(f"Total chunks after filtering: {len(filtered_chunks)}")

    return filtered_chunks


def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings