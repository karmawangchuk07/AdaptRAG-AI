from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage


def get_chat_history(user_id: str, chat_col) -> list:
    docs = list(chat_col.find({"user_id": user_id}).sort("_id", -1).limit(6))
    docs = docs[::-1]
    history = []
    for doc in docs:
        if doc["role"] == "user":
            history.append(HumanMessage(content=doc["content"]))
        else:
            history.append(AIMessage(content=doc["content"]))
    return history


def format_chat_history(history: list) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for msg in history:
        prefix = "Patient" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{prefix}: {msg.content}")
    return "\n".join(lines)


def store_messages(user_id: str, user_msg: str, bot_msg: str, chat_col):
    now = datetime.utcnow()
    chat_col.insert_many([
        {"user_id": user_id, "role": "user",     "content": user_msg, "created_at": now},
        {"user_id": user_id, "role": "assistant", "content": bot_msg,  "created_at": now},
    ])
    recent_ids = [
        d["_id"] for d in
        chat_col.find({"user_id": user_id}).sort("_id", -1).limit(30)
    ]
    chat_col.delete_many({"user_id": user_id, "_id": {"$nin": recent_ids}})