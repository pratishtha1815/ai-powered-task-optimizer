"""
config.py
─────────────────────────────────────────────────────────────
Centralised configuration loader.
Reads values from .env and exposes them as typed constants.
Every other module imports from here — never from os.environ directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent / ".env")


# ── Paths ─────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
MODEL_DIR     = BASE_DIR / "models"
LOG_DIR       = BASE_DIR / "logs"

import cv2
DATABASE_URL  = os.getenv("DATABASE_URL", "sqlite:///data/task_optimizer.db")
MODEL_PATH    = BASE_DIR / os.getenv("MODEL_PATH", "models/emotion_cnn.h5")
HAAR_PATH     = os.getenv("HAAR_CASCADE_PATH",
                           os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

# ── CNN ───────────────────────────────────────────────────────
IMG_SIZE      = int(os.getenv("IMG_SIZE", 48))
EMOTION_LABELS = os.getenv(
    "EMOTION_LABELS", "Angry,Disgust,Fear,Happy,Neutral,Sad,Surprise"
).split(",")
EMOTION_THRESHOLD = float(os.getenv("EMOTION_THRESHOLD", 0.35))
EMOTION_BACKEND   = os.getenv("EMOTION_BACKEND", "retinaface")

# ── Sentiment ─────────────────────────────────────────────────
SENT_POS_THRESH = float(os.getenv("SENTIMENT_POSITIVE_THRESHOLD",  0.05))
SENT_NEG_THRESH = float(os.getenv("SENTIMENT_NEGATIVE_THRESHOLD", -0.05))

# ── Well-being Weights ────────────────────────────────────────
WEIGHT_EMOTION    = float(os.getenv("WEIGHT_EMOTION",    0.40))
WEIGHT_SENTIMENT  = float(os.getenv("WEIGHT_SENTIMENT",  0.30))
WEIGHT_VOICE      = float(os.getenv("WEIGHT_VOICE",      0.30))

# ── Voice Analyzer ───────────────────────────────────────────
TRANSCRIPTION_LANG = os.getenv("TRANSCRIPTION_LANGUAGE", "en-US")

# ── HR Alerts ─────────────────────────────────────────────────
HR_CONSECUTIVE_DAYS = int(os.getenv("HR_ALERT_CONSECUTIVE_DAYS", 3))
HR_THRESHOLD        = float(os.getenv("HR_ALERT_THRESHOLD", 0.35))

# ── Ensure required folders exist at import time ──────────────
for _dir in (DATA_DIR, MODEL_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
