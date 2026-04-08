import os
import uuid
from datetime import datetime, timezone

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_groq import ChatGroq

from rag.retriver import get_retriever
from services.memory_manager import get_memory, update_memory, format_memory
from services.prescription import (
    extract_text_from_file,
    save_prescription,
    clear_prescription,
    store_prescription_in_vector_db
)
from services.answer import generate_answer



app = Flask(
    __name__,
    template_folder="frontend",
    static_folder="frontend"
)

CORS(app)

load_dotenv()

mongo_uri = os.getenv("MONGO_URI")
mongo_client = MongoClient(mongo_uri)

db = mongo_client.get_database("chatbot_db")

conv_col = db.get_collection("conversations")
msg_col = db.get_collection("messages")
memory_col = db.get_collection("user_memory")
prescription_col = db.get_collection("prescriptions")

print("[DEBUG] DB:", db.name)
print("[DEBUG] Collections:", db.list_collection_names())
print("[DEBUG] Using collection:", msg_col.name)


llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

retriever = get_retriever(k=5)
print("[DEBUG] Retriever ready")



def get_or_create_conversation(user_id, message, conversation_id, now):
    if conversation_id:
        return str(conversation_id)

    conversation_id = str(uuid.uuid4())

    conv_col.insert_one({
        "_id": conversation_id,
        "user_id": str(user_id),
        "title": message[:40],
        "created_at": now,
        "updated_at": now
    })

    print("[DEBUG] Created conversation:", conversation_id)

    return conversation_id


def get_chat_history(conversation_id, user_id):
    print("[DEBUG] Fetch history:", conversation_id, user_id)

    history_docs = list(
        msg_col.find({
            "conversation_id": str(conversation_id),
            "user_id": str(user_id)
        })
        .sort("_id", -1)
        .limit(6)
    )[::-1]

    return "\n".join(
        f"{'User' if d['role']=='user' else 'Assistant'}: {d['content']}"
        for d in history_docs
    ) or "No previous conversation."


def store_messages(conversation_id, user_id, user_msg, bot_msg, now):
    try:
        print("[DEBUG] Saving to:", msg_col.name)
        print("[DEBUG] conversation_id:", conversation_id)

        result = msg_col.insert_many([
            {
                "conversation_id": str(conversation_id),
                "user_id": str(user_id),
                "role": "user",
                "content": user_msg,
                "created_at": now
            },
            {
                "conversation_id": str(conversation_id),
                "user_id": str(user_id),
                "role": "assistant",
                "content": bot_msg,
                "created_at": now
            }
        ])

        print("[DEBUG] Inserted IDs:", result.inserted_ids)

        conv_col.update_one(
            {"_id": str(conversation_id)},
            {"$set": {"updated_at": now}}
        )

    except Exception as e:
        print("[DB ERROR]", e)



@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json or {}

        user_id = data.get("user_id")
        if not user_id:
            return jsonify({"error": "user_id required"}), 400

        message = data.get("message", "").strip()
        conversation_id = data.get("conversation_id")

        print("[DEBUG] Incoming:", message, user_id, conversation_id)

        if not message:
            return jsonify({"error": "Message required"}), 400

        now = datetime.now(timezone.utc)

        conversation_id = get_or_create_conversation(
            user_id, message, conversation_id, now
        )

        memory = get_memory(user_id, memory_col)
        memory = update_memory(user_id, message, memory, memory_col, llm)
        memory_text = format_memory(memory)

        chat_history_text = get_chat_history(conversation_id, user_id)

        try:
            response = generate_answer(
                message,
                chat_history_text,
                memory_text,
                user_id,
                llm,
                prescription_col,
                retriever=retriever
            )
        except Exception as e:
            print("[GEN ERROR]", e)
            response = "Sorry, something went wrong."

        store_messages(conversation_id, user_id, message, response, now)

        return jsonify({
            "response": response,
            "conversation_id": conversation_id,
            "user_id": user_id
        })

    except Exception as e:
        print("[CHAT ERROR]", e)
        return jsonify({"error": "Something went wrong"}), 500


@app.route("/conversations", methods=["GET"])
def get_conversations():
    user_id = request.args.get("user_id")

    convs = list(
        conv_col.find({"user_id": str(user_id)})
        .sort("updated_at", -1)
    )

    return jsonify([
        {"id": str(c["_id"]), "title": c["title"]}
        for c in convs
    ])


@app.route("/messages", methods=["GET"])
def get_messages():
    conversation_id = request.args.get("conversation_id")
    user_id = request.args.get("user_id")

    print("[DEBUG] Fetch messages:", conversation_id, user_id)

    msgs = list(
        msg_col.find({
            "conversation_id": str(conversation_id),
            "user_id": str(user_id)
        })
        .sort("_id", 1)
        .limit(50)
    )

    return jsonify([
        {"role": m["role"], "text": m["content"]}
        for m in msgs
    ])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)