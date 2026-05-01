
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from groq import Groq
import uuid
import os

from firebase_config import db
from pattern_engine import detect_patterns, infer_mode, build_system_prompt

app = FastAPI(title="Mitra Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq setup
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ─── Pydantic Models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    mode: str = "auto"

class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: int


# ─── Helper: Firestore session doc ────────────────────────────────────────────

def get_or_create_session(session_id: Optional[str]) -> tuple[str, dict]:
    if not session_id:
        session_id = str(uuid.uuid4())

    doc_ref = db.collection("sessions").document(session_id)
    doc = doc_ref.get()

    if doc.exists:
        return session_id, doc.to_dict()

    new_session = {
        "session_id": session_id,
        "patterns": {
            "emotional": 0,
            "venting": 0,
            "advice": 0,
            "solution": 0,
            "why": 0,
            "totalMsgs": 0,
        },
        "history": [],
        "mode_history": [],
        "created_at": __import__("datetime").datetime.utcnow().isoformat(),
    }
    doc_ref.set(new_session)
    return session_id, new_session


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    session_id, session = get_or_create_session(req.session_id)
    doc_ref = db.collection("sessions").document(session_id)

    updated_patterns = detect_patterns(req.message, session["patterns"])
    resolved_mode = infer_mode(req.mode, updated_patterns)
    system_prompt = build_system_prompt(resolved_mode, updated_patterns)

    history = session.get("history", [])
    history.append({"role": "user", "content": req.message})
    history = history[-20:]

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system_prompt}] + history,
            max_tokens=500,
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")

    history.append({"role": "assistant", "content": reply})
    updated_patterns["totalMsgs"] += 1

    doc_ref.update({
        "patterns": updated_patterns,
        "history": history,
        "mode_history": session.get("mode_history", []) + [resolved_mode],
    })

    message_id = str(uuid.uuid4())
    doc_ref.collection("messages").document(message_id).set({
        "user_msg": req.message,
        "bot_reply": reply,
        "mode": resolved_mode,
        "patterns_snapshot": updated_patterns,
        "feedback": 0,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
    })

    return {
        "session_id": session_id,
        "reply": reply,
        "mode_used": resolved_mode,
        "patterns": updated_patterns,
        "message_id": message_id,
    }


@app.get("/profile/{session_id}")
async def get_profile(session_id: str):
    doc = db.collection("sessions").document(session_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Session not found")

    data = doc.to_dict()
    patterns = data.get("patterns", {})
    mode_history = data.get("mode_history", [])
    dominant_mode = max(set(mode_history), key=mode_history.count) if mode_history else "auto"

    return {
        "session_id": session_id,
        "patterns": patterns,
        "dominant_mode": dominant_mode,
        "total_messages": patterns.get("totalMsgs", 0),
        "mode_breakdown": {
            m: mode_history.count(m) for m in ["friend", "guide", "therapist"]
        },
    }


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    if req.rating not in [1, -1]:
        raise HTTPException(status_code=400, detail="Rating must be 1 or -1")

    msg_ref = (
        db.collection("sessions")
        .document(req.session_id)
        .collection("messages")
        .document(req.message_id)
    )
    msg = msg_ref.get()
    if not msg.exists:
        raise HTTPException(status_code=404, detail="Message not found")

    msg_ref.update({"feedback": req.rating})
    return {"status": "ok", "message_id": req.message_id, "rating": req.rating}


@app.get("/health")
async def health():
    return {"status": "running", "service": "Mitra Chatbot API"}
