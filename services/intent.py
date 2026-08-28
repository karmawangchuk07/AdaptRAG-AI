from typing import Literal, Optional
from pydantic import BaseModel, Field


class IntentClassification(BaseModel):
    """Structured classification of a patient message, plus an optionally
    rewritten version of the query optimized for medical document search."""

    intent: Literal["medical", "prescription", "emotional", "general"] = Field(
        description=(
            "medical: asking about symptoms, diseases, medicines, treatments, health conditions. "
            "prescription: asking about their uploaded prescription, specific medicines in it, dosage. "
            "emotional: expressing stress, anxiety, sadness, fear, loneliness, frustration, mental health. "
            "general: greetings, casual chat, lifestyle questions, anything not fitting above."
        )
    )
    rewritten_query: str = Field(
        description=(
            "The user's message rewritten for better medical document search "
            "(clearer, more specific medical terminology). If the message is "
            "already clear or not medical in nature, return it unchanged."
        )
    )


def classify_and_rewrite(message: str, llm) -> tuple[str, str]:
    """
    Single structured call that replaces the old detect_intent() + improve_query()
    pair. Returns (intent, rewritten_query). Falls back to ("general", original
    message) if the structured call fails for any reason, so callers never crash.
    """
    try:
        structured_llm = llm.with_structured_output(IntentClassification)
        prompt = (
            "Classify this patient message and rewrite it for medical search "
            f"if relevant.\n\nMessage: \"{message}\""
        )
        result: IntentClassification = structured_llm.invoke(prompt)
        return result.intent, result.rewritten_query or message

    except Exception as e:
        print(f"[WARN] classify_and_rewrite failed, falling back: {e}")
        return "general", message


# Kept for backward compatibility with any other code still importing this directly.
def detect_intent(message: str, llm) -> str:
    intent, _ = classify_and_rewrite(message, llm)
    return intent