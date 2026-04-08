def detect_intent(message: str, llm) -> str:
    """
    Classify the user message into one of four intents.
    Returns: 'medical' | 'prescription' | 'emotional' | 'general'
    """
    prompt = f"""Classify the following patient message into exactly ONE category:

medical      → asking about symptoms, diseases, medicines, treatments, health conditions
prescription → asking about their uploaded prescription, specific medicines in it, dosage
emotional    → expressing stress, anxiety, sadness, fear, loneliness, frustration, mental health
general      → greetings, casual chat, lifestyle questions, anything not fitting above

Message: "{message}"

Reply with ONLY one word from: medical, prescription, emotional, general"""

    result = llm.invoke(prompt).content.strip().lower()
    for intent in ("medical", "prescription", "emotional", "general"):
        if intent in result:
            return intent
    return "general"