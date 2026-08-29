from rag.retriver import get_retriever
from services.intent import detect_intent
from services.prescription import get_prescription_retriever
from sentence_transformers import CrossEncoder
from services.intent import classify_and_rewrite
from rag.hybrid_retriever import get_hybrid_retriever


PERSONAS = {
    "medical": (
        "You are a knowledgeable and caring cardiovascular health assistant. "
        "Your knowledge base covers cardiac and vascular conditions, including "
        "arrhythmias, heart failure, coronary artery disease, hypertension, "
        "valvular disease, and related medications and procedures. "
        "Use the provided medical context ONLY if it is directly relevant to the question. "
        "If the question falls outside cardiovascular medicine and the context "
        "isn't useful, answer normally using general medical knowledge, and be clear "
        "that it's outside your specialized focus area. "
        "Be accurate, cite specific mechanisms or guidelines when the context supports it, "
        "and avoid hallucination."
    ),
    "prescription": (
        "You are a helpful medical assistant reviewing the patient's prescription. "
        "Explain medicines clearly — purpose, dosage, and side effects. "
        "Do NOT override doctor's instructions."
    ),
    "emotional": (
        "You are a compassionate emotional support assistant. "
        "Respond with empathy and warmth."
    ),
    "general": (
        "You are a friendly and natural conversational assistant."
    ),
}

# Loaded once at import time so it's not reloaded on every call.
# Small (~80MB), runs fine on CPU.
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        print("[INFO] Loading cross-encoder reranker...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank_docs(query, docs, top_k=3):
    """Cross-encoder reranking: jointly scores (query, doc) pairs for relevance,
    rather than counting overlapping words. Falls back to original doc order
    if the model fails to load or score for any reason."""
    if not docs:
        return []

    try:
        model = _get_reranker()
        pairs = [[query, doc.page_content] for doc in docs]
        scores = model.predict(pairs)

        scored = list(zip(scores, docs))
        scored.sort(reverse=True, key=lambda x: x[0])

        return [doc for _, doc in scored[:top_k]]

    except Exception as e:
        print(f"[WARN] Reranker failed, falling back to original order: {e}")
        return docs[:top_k]


def improve_query(query, llm):
    try:
        prompt = f"Rewrite this for better medical search:\n{query}"
        return llm.invoke(prompt).content.strip()
    except:
        return query


def generate_answer(
    message: str,
    chat_history_text: str,
    memory_text: str,
    user_id: str,
    llm,
    prescription_col,
    retriever=None
) -> str:

    query = message.strip()

    intent, improved_query = classify_and_rewrite(query, llm)
    print(f"[Intent] {intent}")
    medical_context = ""

    if intent == "medical":
        retriever = retriever or get_hybrid_retriever(k=5)
        docs = retriever.invoke(improved_query)

        print("\n--- MEDICAL RETRIEVED DOCS ---")
        for i, d in enumerate(docs):
            print(f"[{i}] {d.page_content[:150]}")

        # 🔥 FILTER JUNK
        filtered_docs = [
            d for d in docs
            if len(d.page_content.strip()) > 100
            and not any(x in d.page_content.lower() for x in [
                "page", "copyright", "gale", "isbn",
                "prentice", "encyclopedia"
            ])
        ]

        ranked_docs = rerank_docs(query, filtered_docs)

        if ranked_docs:
            medical_context = "\n\n".join(
                f"[Context {i+1}]\n{d.page_content[:400]}"
                for i, d in enumerate(ranked_docs)
            )

    rx_context = ""

    if intent == "prescription":

        rx_retriever = get_prescription_retriever(user_id)

        if rx_retriever:
            rx_docs = rx_retriever.invoke(improved_query)

            print("\n--- PRESCRIPTION RETRIEVED DOCS ---")
            for i, d in enumerate(rx_docs):
                print(f"[{i}] {d.page_content[:150]}")

            filtered_rx_docs = [
                d for d in rx_docs
                if len(d.page_content.strip()) > 30
            ]

            ranked_rx_docs = rerank_docs(query, filtered_rx_docs)

            if ranked_rx_docs:
                rx_context = "\n\n".join(
                    f"[Prescription {i+1}]\n{d.page_content[:300]}"
                    for i, d in enumerate(ranked_rx_docs)
                )

    if intent == "prescription" and not rx_context:
        return (
            "I couldn't find any prescription data yet. "
            "Please upload your prescription and I'll help explain it."
        )

    sections = [
        PERSONAS.get(intent, PERSONAS["general"]),
        "",
        "STRICT INSTRUCTIONS:",
        "- Use context ONLY if directly relevant",
        "- If context is irrelevant, IGNORE it",
        "- If unsure, say you don't know",
        "- Do NOT hallucinate",
        "- Keep answer clear, structured, and natural",
        "",
        "Patient Information:",
        memory_text or "No patient data available",
        "",
        "Conversation History:",
        chat_history_text or "No previous conversation",
    ]

    if medical_context:
        sections += [
            "",
            "Relevant Medical Knowledge:",
            medical_context
        ]

    if rx_context:
        sections += [
            "",
            "Relevant Prescription Information:",
            rx_context
        ]

    sections += [
        "",
        f"User Question: {query}",
        "",
        "Answer:"
    ]

    prompt = "\n".join(sections)

    response = llm.invoke(prompt).content.strip()

    return response