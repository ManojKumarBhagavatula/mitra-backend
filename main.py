import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth as firebase_auth
from groq import Groq
from pydantic import BaseModel, Field

from firebase_config import db

from pattern_engine import build_system_prompt, detect_patterns, infer_mode


app = FastAPI(title="Mitra Chatbot API")

# Set FRONTEND_ORIGIN to a comma-separated list of trusted browser origins.
# Local development defaults to the Create React App development server.
allowed_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("FRONTEND_ORIGIN", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

bearer_scheme = HTTPBearer(auto_error=False)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(min_length=1, max_length=10000)
    mode: str = "auto"


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: int


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    """Validate the Firebase ID token sent by the signed-in frontend."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        return firebase_auth.verify_id_token(credentials.credentials)
    except Exception:
        # Do not expose token verification details to the caller.
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_or_create_session(
    requested_session_id: Optional[str],
    user_uid: str,
) -> tuple[str, dict]:
    """Load a session owned by this Firebase user, or create a new one."""
    session_id = requested_session_id or str(uuid.uuid4())
    doc_ref = db.collection("sessions").document(session_id)
    doc = doc_ref.get()

    if doc.exists:
        session = doc.to_dict() or {}
        owner_uid = session.get("owner_uid")

        # Backward compatibility for the original first-session convention:
        # the frontend used the authenticated Firebase UID as the document ID.
        # Only that exact UID/document match may claim a legacy unowned record.
        if not owner_uid and session_id == user_uid:
            doc_ref.update({"owner_uid": user_uid})
            session["owner_uid"] = user_uid
            return session_id, session

        if owner_uid != user_uid:
            # Use 404 so callers cannot discover another user's session IDs.
            raise HTTPException(status_code=404, detail="Session not found")

        return session_id, session

    new_session = {
        "session_id": session_id,
        "owner_uid": user_uid,
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
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    doc_ref.set(new_session)
    return session_id, new_session


@app.post("/chat")
async def chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    user_uid = user["uid"]
    session_id, session = get_or_create_session(req.session_id, user_uid)
    doc_ref = db.collection("sessions").document(session_id)

    updated_patterns = detect_patterns(req.message, session.get("patterns", {}))
    resolved_mode = infer_mode(req.mode, updated_patterns)
    system_prompt = build_system_prompt(resolved_mode, updated_patterns)

    history = session.get("history", [])
    history.append({"role": "user", "content": req.message})
    history = history[-20:]

    try:
        response = groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            messages=[{"role": "system", "content": system_prompt}] + history,
            max_tokens=500,
        )
        reply = (response.choices[0].message.content or "").strip()
    except Exception:
        # Keep provider details and credentials out of API responses.
        raise HTTPException(
            status_code=502,
            detail="The response service is temporarily unavailable. Please try again.",
        )

    history.append({"role": "assistant", "content": reply})
    updated_patterns["totalMsgs"] = updated_patterns.get("totalMsgs", 0) + 1

    doc_ref.update({
        "patterns": updated_patterns,
        "history": history[-20:],
        "mode_history": session.get("mode_history", []) + [resolved_mode],
    })

    message_id = str(uuid.uuid4())
    doc_ref.collection("messages").document(message_id).set({
        "user_msg": req.message,
        "bot_reply": reply,
        "mode": resolved_mode,
        "patterns_snapshot": updated_patterns,
        "feedback": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "session_id": session_id,
        "reply": reply,
        "mode_used": resolved_mode,
        "patterns": updated_patterns,
        "message_id": message_id,
    }


@app.get("/profile/{session_id}")
async def get_profile(
    session_id: str,
    user: dict = Depends(get_current_user),
):
    doc = db.collection("sessions").document(session_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Session not found")

    data = doc.to_dict() or {}
    if data.get("owner_uid") != user["uid"]:
        raise HTTPException(status_code=404, detail="Session not found")

    patterns = data.get("patterns", {})
    mode_history = data.get("mode_history", [])
    dominant_mode = max(set(mode_history), key=mode_history.count) if mode_history else "auto"

    return {
        "session_id": session_id,
        "patterns": patterns,
        "dominant_mode": dominant_mode,
        "total_messages": patterns.get("totalMsgs", 0),
        "mode_breakdown": {
            mode: mode_history.count(mode)
            for mode in ["friend", "guide", "therapist"]
        },
    }


@app.post("/feedback")
async def feedback(
    req: FeedbackRequest,
    user: dict = Depends(get_current_user),
):
    if req.rating not in [1, -1]:
        raise HTTPException(status_code=400, detail="Rating must be 1 or -1")

    session_ref = db.collection("sessions").document(req.session_id)
    session_doc = session_ref.get()
    if not session_doc.exists:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = session_doc.to_dict() or {}
    if session_data.get("owner_uid") != user["uid"]:
        raise HTTPException(status_code=404, detail="Session not found")

    msg_ref = session_ref.collection("messages").document(req.message_id)
    msg = msg_ref.get()
    if not msg.exists:
        raise HTTPException(status_code=404, detail="Message not found")

    msg_ref.update({"feedback": req.rating})
    return {"status": "ok", "message_id": req.message_id, "rating": req.rating}


@app.get("/health")
async def health():
    # Public endpoint for hosting-provider health checks.
    return {"status": "running", "service": "Mitra Chatbot API"}
