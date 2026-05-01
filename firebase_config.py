import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

# ─── Firebase Initialization ──────────────────────────────────────────────────
#
# Two ways to provide credentials:
#
# Option A (Local dev): Set FIREBASE_CREDENTIALS_PATH env var pointing to
#   your downloaded serviceAccountKey.json file.
#
# Option B (Render/production): Set FIREBASE_CREDENTIALS_JSON env var with
#   the full JSON content of serviceAccountKey.json as a single-line string.
#   (Render → Environment → Add env var → paste the JSON)
#
# ─────────────────────────────────────────────────────────────────────────────

def _init_firebase():
    if firebase_admin._apps:
        return  # Already initialized

    # Option B: JSON string in env (preferred for Render deployment)
    cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
    if cred_json:
        cred_dict = json.loads(cred_json)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        return

    # Option A: Local file path
    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        return

    raise RuntimeError(
        "Firebase credentials not found.\n"
        "Set FIREBASE_CREDENTIALS_JSON (production) or "
        "FIREBASE_CREDENTIALS_PATH (local dev)."
    )


_init_firebase()
db = firestore.client()
