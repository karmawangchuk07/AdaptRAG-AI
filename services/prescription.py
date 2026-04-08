import io
import os
import re
from datetime import datetime

import fitz
import pytesseract
from PIL import Image

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.helper import download_hugging_face_embeddings


def extract_text_from_file(file) -> str:
    """Extract text from an uploaded PDF or image file."""
    filename = file.filename.lower()
    text = ""

    if filename.endswith(".pdf"):
        raw = file.read()
        pdf = fitz.open(stream=raw, filetype="pdf")

        for page in pdf:
            text += page.get_text()

        if not text.strip():
            pdf = fitz.open(stream=raw, filetype="pdf")
            for page in pdf:
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text += pytesseract.image_to_string(img)

    elif filename.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(file.stream)
        text = pytesseract.image_to_string(img)

    else:
        raise ValueError("Unsupported file type. Upload a PDF, PNG, or JPG.")

    return clean_text(text)


def clean_text(text: str) -> str:
    """Remove noise from OCR / PDFs"""
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'Page \d+.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'GEM.*', '', text)  
    return text.strip()


def save_prescription(user_id: str, text: str, filename: str, prescription_col):
    prescription_col.update_one(
        {"user_id": user_id},
        {"$set": {
            "text": text,
            "filename": filename,
            "uploaded_at": datetime.utcnow()
        }},
        upsert=True
    )

    if "diagnosis" in text.lower() or "impression" in text.lower():
        print("[INFO] Prescription likely contains diagnosis")


def store_prescription_in_vector_db(user_id: str, text: str):
    embedding = download_hugging_face_embeddings()

    db_path = f"chroma_rx/{user_id}"

    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    docs = [Document(page_content=text)]
    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=db_path
    )

    vectordb.persist()

    print(f"[Prescription RAG stored for user {user_id}]")
    print(f"[Chunks created: {len(chunks)}]")


def get_prescription_retriever(user_id: str):
    db_path = f"chroma_rx/{user_id}"

    if not os.path.exists(db_path):
        return None

    embedding = download_hugging_face_embeddings()

    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embedding
    )

    return vectordb.as_retriever(search_kwargs={"k": 3})


def clear_prescription(user_id: str, prescription_col):
    prescription_col.delete_one({"user_id": user_id})

    db_path = f"chroma_rx/{user_id}"
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)

    print(f"[Prescription cleared for {user_id}]")