import re
import json


def get_memory(user_id: str, memory_col) -> dict:
    doc = memory_col.find_one({"user_id": user_id}) or {}
    doc.pop("_id", None)
    doc.pop("user_id", None)
    return doc


def update_memory(user_id: str, user_message: str, current_memory: dict, memory_col, llm) -> dict:
    import re
    import json

    prompt = f"""
You are a STRICT memory extractor.

Current patient memory:
{json.dumps(current_memory, indent=2)}

User message:
"{user_message}"

RULES:
- Extract ONLY facts explicitly about the user
- DO NOT assume anything
- DO NOT extract diseases unless clearly stated

ONLY extract if user says things like:
- "I have diabetes"
- "I was diagnosed with asthma"
- "I am suffering from fever"
- "My condition is hypertension"

DO NOT extract:
- "What is diabetes?"
- "Tell me about cancer"
- "Explain fever"

Return ONLY JSON.
If nothing → return {{}}.
"""

    raw = llm.invoke(prompt).content.strip()

    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        new_facts = json.loads(match.group()) if match else {}
    except Exception:
        new_facts = {}

    
    msg = user_message.lower()

    if "sickness" in new_facts or "conditions" in new_facts:
        if not any(x in msg for x in [
            "i have", "i am suffering", "diagnosed with",
            "my condition", "i got", "i've been diagnosed"
        ]):
            new_facts.pop("sickness", None)
            new_facts.pop("conditions", None)

    
    if new_facts:
        for key, val in new_facts.items():
            existing = current_memory.get(key)

            if isinstance(val, list):
                current_memory[key] = list(set((existing or []) + val))
            else:
                current_memory[key] = val

        memory_col.update_one(
            {"user_id": user_id},
            {"$set": {**current_memory, "user_id": user_id}},
            upsert=True
        )

    return current_memory


def format_memory(memory: dict) -> str:
    if not memory:
        return "No patient information known yet."
    lines = []
    for k, v in memory.items():
        if isinstance(v, list):
            lines.append(f"- {k}: {', '.join(str(i) for i in v)}")
        else:
            lines.append(f"- {k}: {v}")
    return "\n".join(lines)