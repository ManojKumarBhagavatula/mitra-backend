import re
from typing import Optional

# ─── Pattern Detection ────────────────────────────────────────────────────────
#
# This is the "ML" core of Mitra. Currently rule-based (regex), but designed
# so you can swap in a trained classifier (Phase 2) without changing the API.
#
# Each pattern maps to a response style:
#   emotional  → therapist mode (validate first, then gently explore)
#   venting    → friend mode (just listen, don't immediately fix)
#   advice     → guide mode (structured, actionable)
#   solution   → guide mode (step-by-step, concrete)
#   why        → blend of therapist + guide (explore root cause)
# ─────────────────────────────────────────────────────────────────────────────

import os

# Try to load ML model — falls back to regex if not available (e.g. Render deployment)
MODEL_PATH = './mitra-model'
classifier = None

try:
    from transformers import pipeline as hf_pipeline
    import torch
    if os.path.exists(MODEL_PATH):
        classifier = hf_pipeline(
            'text-classification',
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            device=-1
        )
        print("✅ ML model loaded")
    else:
        print("⚠️ Model not found — using regex fallback")
except ImportError:
    print("⚠️ Transformers not installed — using regex fallback")

PATTERNS = {
    "emotional": [
        r"\bi feel\b", r"\bi'm (sad|upset|angry|anxious|scared|lost|empty|hurt|broken|tired)\b",
        r"\bi hate (this|my|myself|everything)\b", r"\bit hurts\b", r"\bcrying\b",
        r"\boverwhelmed\b", r"\bstressed\b", r"\bcan't take\b", r"\bnot okay\b",
        r"\bfeel like\b", r"\bfeeling\b", r"\bdepressed\b", r"\bworried\b",
    ],
    "venting": [
        r"\bjust listen\b", r"\bneeded to (say|vent|talk)\b",
        r"\bso frustrated\b", r"\bugh\b", r"\bwhy does (everything|this)\b",
        r"\bnobody (understands|cares|listens)\b", r"\bi can't believe\b",
        r"\bthis is (so|too) (unfair|hard|much)\b",
    ],
    "advice": [
        r"\bwhat should i\b", r"\bhow (do|can|should) i\b", r"\bhelp me\b",
        r"\bwhat (can|do) i do\b", r"\badvice\b", r"\bsuggest\b",
        r"\bwhat would you\b", r"\bshould i\b", r"\bwhat if i\b",
    ],
    "solution": [
        r"\bplan\b", r"\bsteps?\b", r"\bstrategy\b", r"\baction\b",
        r"\bfix\b", r"\bsolve\b", r"\bresolve\b", r"\bhow to\b",
        r"\bgoal\b", r"\bdeadline\b", r"\bschedule\b", r"\btodo\b",
    ],
    "why": [
        r"\bwhy (do|does|am|is|can't|won't|didn't|don't)\b",
        r"\bwhat's wrong with\b", r"\bwhat causes\b",
        r"\bwhy (do|does) this happen\b", r"\bwhy me\b",
    ],
}

def detect_patterns(text: str, current: dict) -> dict:
    updated = dict(current)
    
    if classifier:
        # Use ML model
        result = classifier(text)[0]
        LABEL_TO_PATTERN = {
            'therapist': 'emotional',
            'friend':    'venting',
            'guide':     'advice',
        }
        pattern = LABEL_TO_PATTERN[result['label']]
        updated[pattern] = round(updated.get(pattern, 0) + result['score'], 3)
    else:
        # Regex fallback
        t = text.lower()
        for pattern_name, regexes in PATTERNS.items():
            for rx in regexes:
                if re.search(rx, t):
                    updated[pattern_name] = updated.get(pattern_name, 0) + 1
                    break
    
    return updated


# ─── Mode Inference ───────────────────────────────────────────────────────────

def infer_mode(requested_mode: str, patterns: dict) -> str:
    """
    If user has manually selected a mode, respect it.
    Otherwise, pick the best mode based on accumulated patterns.
    """
    if requested_mode != "auto":
        return requested_mode

    emotional = patterns.get("emotional", 0)
    venting   = patterns.get("venting", 0)
    advice    = patterns.get("advice", 0)
    solution  = patterns.get("solution", 0)
    total     = patterns.get("totalMsgs", 0)

    # Not enough data yet — default to friend
    if total < 3:
        return "friend"

    # Score each mode
    friend_score    = venting * 2 + emotional * 1
    therapist_score = emotional * 2 + venting * 1
    guide_score     = (advice + solution) * 2

    scores = {
        "friend": friend_score,
        "therapist": therapist_score,
        "guide": guide_score,
    }

    return max(scores, key=scores.get)


# ─── System Prompt Builder ────────────────────────────────────────────────────

MODE_INSTRUCTIONS = {
    "friend": (
        "Respond as a close, warm, real friend. Be casual and human — "
        "use natural conversational language. Always validate their feelings first. "
        "Then, if appropriate, offer a gentle perspective or encouragement. "
        "Never be preachy. Keep it short — 2-4 sentences unless they need more."
    ),
    "guide": (
        "Respond as a supportive life coach or mentor. Be structured and motivating. "
        "Acknowledge their situation briefly, then provide 1-2 clear, concrete action steps. "
        "Be direct without being cold. Use simple language, not jargon. "
        "End with something empowering."
    ),
    "therapist": (
        "Respond as a warm, skilled therapist. Reflect their feelings back to them accurately. "
        "Ask one thoughtful clarifying question to help them go deeper. "
        "Do NOT give direct advice — guide them to their own insights. "
        "Avoid clinical language. Feel human, not scripted."
    ),
}

def build_system_prompt(mode: str, patterns: dict) -> str:
    total = patterns.get("totalMsgs", 0)

    # Build user insight string
    insights = []
    if patterns.get("emotional", 0) > 1:
        insights.append("this user often expresses emotions openly")
    if patterns.get("venting", 0) > 0:
        insights.append("this user sometimes just needs to vent without being 'fixed'")
    if patterns.get("advice", 0) > 1:
        insights.append("this user frequently seeks concrete advice")
    if patterns.get("solution", 0) > 1:
        insights.append("this user is goal and solution oriented")
    if patterns.get("why", 0) > 1:
        insights.append("this user tends to question root causes — explore the 'why' with them")

    insight_str = (
        f"Based on {total} messages, you have learned: {'; '.join(insights)}."
        if insights else
        "This is an early session — be warm and open, learn their style."
    )

    return (
        "You are Mitra, a deeply empathetic and adaptive personal companion. "
        "You help people process their thoughts, emotions, and challenges. "
        "You are NOT a generic assistant — you are personal, present, and real.\n\n"
        f"Current mode: {mode.upper()}\n"
        f"{MODE_INSTRUCTIONS[mode]}\n\n"
        f"User profile: {insight_str}\n\n"
        "Rules:\n"
        "- Never give a list of bullet points as an emotional response — write naturally.\n"
        "- Never say 'I understand how you feel' — show it instead.\n"
        "- Keep responses 2-5 sentences unless the user clearly wants more depth.\n"
        "- Never break character or mention being an AI unless directly asked.\n"
        "- Always end on something that makes the user feel heard or empowered."
    )
