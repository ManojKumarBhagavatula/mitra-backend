# Mitra — Adaptive Companion Chatbot 🤖

A FastAPI backend for an emotionally intelligent chatbot that adapts its response style
(Friend / Guide / Therapist) based on the user's messaging patterns — backed by
Firebase Firestore and Claude API.

---

## Project Structure

```
mitra-backend/
├── main.py              # FastAPI routes (/chat, /profile, /feedback, /health)
├── firebase_config.py   # Firebase initialization (local + Render)
├── pattern_engine.py    # Pattern detection, mode inference, prompt builder
├── requirements.txt
├── render.yaml          # One-click Render deployment config
├── .env.example         # Copy to .env for local dev
└── .gitignore
```

---

## Step 1 — Firebase Firestore Setup

1. Go to https://console.firebase.google.com
2. Click **"Add project"** → name it `mitra-chatbot` → Create
3. In the left sidebar → **Firestore Database** → **Create database**
   - Choose **"Start in test mode"** (good for dev; tighten rules before production)
   - Select any region (e.g. `asia-south1` for India)
4. Go to **Project Settings** (gear icon) → **Service Accounts** tab
5. Click **"Generate new private key"** → Download the JSON file
6. Rename it to `serviceAccountKey.json` and place it in this project folder

> ⚠️ Never commit `serviceAccountKey.json` to GitHub — it's in `.gitignore`

---

## Step 2 — Anthropic API Key

1. Go to https://console.anthropic.com
2. Create an API key
3. Copy it — you'll need it in the next step

---

## Step 3 — Local Development

```bash
# Clone your repo and go into the folder
cd mitra-backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env — fill in ANTHROPIC_API_KEY and FIREBASE_CREDENTIALS_PATH

# Run the server
uvicorn main:app --reload
```

Server will start at: http://localhost:8000
Interactive API docs: http://localhost:8000/docs

---

## Step 4 — API Endpoints

### POST /chat
Send a user message and get an adaptive response.

**Request:**
```json
{
  "session_id": null,
  "message": "I've been so stressed about work lately and don't know what to do",
  "mode": "auto"
}
```

**Response:**
```json
{
  "session_id": "abc-123-uuid",
  "reply": "That sounds really exhausting...",
  "mode_used": "therapist",
  "patterns": { "emotional": 1, "venting": 0, "advice": 0, "solution": 0, "why": 0, "totalMsgs": 1 },
  "message_id": "msg-uuid"
}
```

> Save the `session_id` from the first response — send it in all future messages to maintain continuity.

---

### GET /profile/{session_id}
Get the learned pattern profile for a session.

**Response:**
```json
{
  "session_id": "abc-123-uuid",
  "patterns": { "emotional": 3, "advice": 5, "solution": 2, ... },
  "dominant_mode": "guide",
  "total_messages": 10,
  "mode_breakdown": { "friend": 2, "guide": 6, "therapist": 2 }
}
```

---

### POST /feedback
Send thumbs up (1) or thumbs down (-1) for a message.

**Request:**
```json
{
  "session_id": "abc-123-uuid",
  "message_id": "msg-uuid",
  "rating": 1
}
```

---

### GET /health
Simple health check.

---

## Step 5 — Deploy to Render

1. Push this project to a GitHub repo (make sure `.env` and `serviceAccountKey.json` are gitignored)
2. Go to https://render.com → New → **Web Service** → Connect your repo
3. Render auto-detects `render.yaml` — click **Apply**
4. Go to **Environment** tab → Add these two variables:
   - `ANTHROPIC_API_KEY` → your Claude API key
   - `FIREBASE_CREDENTIALS_JSON` → open `serviceAccountKey.json`, copy ALL content, paste as value
5. Click **Deploy** — done!

Your API will be live at: `https://mitra-chatbot-api.onrender.com`

---

## Firestore Data Structure

```
sessions/
  {session_id}/
    patterns:       { emotional, venting, advice, solution, why, totalMsgs }
    history:        [ { role, content }, ... ]   ← last 20 messages
    mode_history:   [ "friend", "guide", ... ]
    created_at:     ISO timestamp

    messages/       ← sub-collection
      {message_id}/
        user_msg:          string
        bot_reply:         string
        mode:              string
        patterns_snapshot: object
        feedback:          0 | 1 | -1
        timestamp:         ISO string
```

---

## Phase 2 — ML Upgrade (your roadmap)

The `pattern_engine.py` is designed to be swapped out as you learn more ML:

| Phase | What to do |
|-------|-----------|
| Now   | Rule-based regex (current) |
| Next  | Fine-tune `distilbert-base-uncased` on labeled chat data for 5-class classification |
| Then  | Use feedback scores as training signal (reinforcement from human feedback) |
| Later | User embedding vectors → personalized few-shot prompting |

Train your classifier like this:
```python
from transformers import pipeline
classifier = pipeline("text-classification", model="your-finetuned-model")
scores = classifier("I feel so overwhelmed today")
# → [{"label": "emotional", "score": 0.91}, ...]
```
Then replace the regex block in `detect_patterns()` with classifier output.

---

## Testing Locally with curl

```bash
# First message (no session_id)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I feel so lost today", "mode": "auto"}'

# Follow-up (use session_id from above response)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_SESSION_ID", "message": "What should I do about it?", "mode": "auto"}'

# Check profile
curl http://localhost:8000/profile/YOUR_SESSION_ID
```

---

Built with FastAPI + Firebase Firestore + Claude API (Anthropic)
