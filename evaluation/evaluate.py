import sys
import os
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation_data import evaluation_data
from services.answer import generate_answer
from rag.retriver import get_retriever, get_docling_retriever

from langchain_groq import ChatGroq

import re

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

def get_answer_wrapper(query, retriever):
    return generate_answer(
        message=query,
        chat_history_text="",
        memory_text="",
        user_id="test_user",
        llm=llm,
        prescription_col=None,
        retriever=retriever
    )

def normalize(text):
    return re.sub(r"[^a-z0-9\s]", "", text.lower())

SYNONYMS = {
    "detect": ["identify", "find", "diagnose"],
    "safe": ["harmless", "no risk"],
    "no radiation": ["radiation free"],
    "imaging": ["scan", "visualization"],
    "tumor": ["mass"],
    "kidney": ["renal"],
    "blood flow": ["circulation"],
}

def keyword_match(answer, keyword):
    answer = normalize(answer)
    keyword = normalize(keyword)

    if keyword in answer:
        return True

    words = keyword.split()
    match_count = sum(1 for w in words if w in answer)

    if match_count >= len(words) * 0.6:
        return True

    if keyword in SYNONYMS:
        return any(syn in answer for syn in SYNONYMS[keyword])

    return False

def evaluate_answer(answer, keywords):
    matches = 0
    matched_keywords = []

    for kw in keywords:
        if keyword_match(answer, kw):
            matches += 1
            matched_keywords.append(kw)

    score = matches / len(keywords)

    if score >= 0.5:
        score += 0.1

    score = min(score, 1.0)
    return score, matched_keywords


def evaluate_system(retriever, label):
    print(f"\nEvaluating: {label}\n")

    total_score = 0
    total = len(evaluation_data)

    for i, item in enumerate(evaluation_data):
        q = item["question"]
        keywords = item["keywords"]

        print(f"\n Q{i+1}: {q}")

        try:
            answer = get_answer_wrapper(q, retriever)

            print(f"Answer: {answer[:200]}...")

            score, matched = evaluate_answer(answer, keywords)

            print(f"Matched Keywords: {matched}")
            print(f"Score: {score:.2f}")

            total_score += score

        except Exception as e:
            print(f"Error: {e}")

    accuracy = (total_score / total) * 100

    print(f"\n {label} Accuracy: {accuracy:.2f}%")

    return accuracy


def compare():
    baseline_retriever = get_retriever(k=5)
    docling_retriever = get_docling_retriever(k=5)

    baseline_score = evaluate_system(baseline_retriever, "Baseline")
    docling_score = evaluate_system(docling_retriever, "Docling")

    print("\n📊 FINAL COMPARISON")
    print(f"Baseline: {baseline_score:.2f}%")
    print(f"Docling: {docling_score:.2f}%")
    print(f"Improvement: {docling_score - baseline_score:.2f}%")


if __name__ == "__main__":
    compare()