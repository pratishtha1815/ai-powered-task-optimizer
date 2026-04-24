"""
modules/emotion_detector.py  —  MODULE 1
─────────────────────────────────────────────────────────────
Facial Emotion Recognition Engine.

Responsibilities
────────────────
• Load a trained CNN model (emotion_cnn.h5) once at startup
• Use OpenCV's Haar Cascade to detect faces in a BGR video frame
• Preprocess each detected face (grayscale → resize → normalise)
• Run CNN inference and return the dominant emotion + confidence
• Expose a helper that also draws annotated bounding boxes

Public API
──────────
    detector = EmotionDetector()
    result   = detector.predict_from_frame(bgr_frame)
        → EmotionResult(emotion, confidence, face_box, all_scores)

    annotated = detector.annotate_frame(bgr_frame, result)
        → BGR frame with box + label overlay
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

# ── Project imports ───────────────────────────────────────────
# ── DeepFace Import with Robust Error Handling ────────────────
DeepFace = None
try:
    with logger.contextualize(module="deepface_loader"):
        from deepface import DeepFace
except Exception as e:
    logger.error(f"❌ DeepFace loading failed (Environment Issue): {e}")

# Create logs directory if it doesn't exist
from config import (
    MODEL_PATH, HAAR_PATH, IMG_SIZE, EMOTION_LABELS, LOG_DIR,
    EMOTION_THRESHOLD, EMOTION_BACKEND
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(LOG_DIR / "emotion_detector.log", rotation="5 MB", level="DEBUG")


# ─────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────
@dataclass
class FaceBox:
    """Bounding box for a detected face (pixel coordinates)."""
    x: int
    y: int
    w: int
    h: int


@dataclass
class EmotionResult:
    """
    Full output of one inference pass on a single video frame.

    Attributes
    ----------
    emotion     : str   – dominant emotion label, e.g. "Happy"
    confidence  : float – probability of dominant emotion [0.0, 1.0]
    face_box    : FaceBox | None – bounding box of the detected face;
                  None when no face was found
    all_scores  : dict[str, float] – probability for every emotion class
    face_found  : bool – convenience flag
    """
    emotion:    str
    confidence: float
    face_box:   Optional[FaceBox]
    all_scores: dict = field(default_factory=dict)

    @property
    def face_found(self) -> bool:
        return self.face_box is not None

    # Numeric well-being weight: maps emotion → [0.0, 1.0]
    # Used downstream by the Task Matching Engine (Module 3).
    EMOTION_WEIGHTS: dict = field(default_factory=lambda: {
        "Happy":    1.00,
        "Neutral":  0.70,
        "Surprise": 0.55,
        "Sad":      0.35,
        "Fear":     0.25,
        "Angry":    0.15,
        "Disgust":  0.10,
    })

    def wellbeing_score(self) -> float:
        """
        Map the dominant emotion to a numeric well-being score in [0, 1].
        Falls back to 0.5 (neutral) if emotion is unknown.
        """
        weights = {
            "Happy":    1.00,
            "Neutral":  0.70,
            "Surprise": 0.55,
            "Sad":      0.35,
            "Fear":     0.25,
            "Angry":    0.15,
            "Disgust":  0.10,
        }
        base = weights.get(self.emotion, 0.50)
        # Blend with confidence: uncertain prediction → nudge toward neutral
        return round(base * self.confidence + 0.50 * (1 - self.confidence), 4)


# ─────────────────────────────────────────────────────────────
# Colour palette for bounding-box overlays (BGR)
# ─────────────────────────────────────────────────────────────
_BOX_COLOURS: dict[str, tuple] = {
    "Happy":    (0,   220,  80),   # green
    "Neutral":  (200, 200, 200),   # light grey
    "Surprise": (0,   165, 255),   # orange
    "Sad":      (255, 100,  50),   # blue-ish
    "Fear":     (128,   0, 128),   # purple
    "Angry":    (0,    0,  220),   # red
    "Disgust":  (0,   128, 255),   # amber
}
_DEFAULT_COLOUR = (180, 180, 180)


# ─────────────────────────────────────────────────────────────
# EmotionDetector — main class
# ─────────────────────────────────────────────────────────────
class EmotionDetector:
    """
    Singleton-friendly emotion detection engine.

    Parameters
    ----------
    model_path : Path | str
        Path to the saved Keras CNN weights (.h5 file).
        Defaults to config.MODEL_PATH.
    haar_path : Path | str
        Path to the Haar Cascade XML file.
        Defaults to config.HAAR_PATH.
    confidence_threshold : float
        Minimum CNN softmax probability to accept a prediction.
        Predictions below this return emotion="Neutral". Default 0.40.
    """

    def __init__(
        self,
        confidence_threshold: float = EMOTION_THRESHOLD,
        detector_backend: str       = EMOTION_BACKEND,
    ):
        self.threshold = confidence_threshold
        self.backend   = detector_backend
        # DeepFace will lazily load models internally on first call
        # but we can optionally warm it up to prevent UI lag.
        self._is_ready = False

    def warmup(self):
        """Pre-load model weights using a dummy image."""
        if self._is_ready:
            return
        try:
            logger.info("🎬 Warming up Emotion Engine (DeepFace)...")
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            # Run a dummy analysis (ignore results)
            DeepFace.analyze(dummy, actions=['emotion'], enforce_detection=False, silent=True)
            self._is_ready = True
            logger.info("✅ Emotion Engine ready.")
        except Exception as e:
            logger.warning(f"Emotion Engine warmup failed: {e}")

    def predict_from_frame(self, bgr_frame: np.ndarray) -> EmotionResult:
        """
        Full pipeline: frame → face detection → CNN → EmotionResult using DeepFace.
        """
        try:
            if DeepFace is None:
                raise ImportError("DeepFace is not installed")
            
            results = DeepFace.analyze(
                img_path=bgr_frame,
                actions=['emotion'],
                enforce_detection=True,
                detector_backend=self.backend,
                align=False,   # faster
                silent=True
            )
        except ValueError:
            # No face detected
            logger.debug(f"No face detected using {self.backend} backend.")
            return EmotionResult(
                emotion="No Face",
                confidence=0.0,
                face_box=None,
                all_scores={label: 0.0 for label in EMOTION_LABELS},
            )
        except ImportError:
            logger.error("DeepFace not installed! Cannot predict. Run: pip install deepface")
            return EmotionResult("Error", 0.0, None)
        except Exception as e:
            logger.error(f"Unexpected error in emotion detection: {e}")
            return EmotionResult("Error", 0.0, None)

        if isinstance(results, list):
            res = results[0]  # Take first/largest face
        else:
            res = results

        # DeepFace labels are lowercase: 'happy', 'sad', 'neutral', etc.
        raw_dom = res.get('dominant_emotion', 'neutral')
        top_emotion = raw_dom.capitalize()
        
        # DeepFace scores are given as percentages (0-100)
        emotions_dict = res.get('emotion', {})
        top_conf = float(emotions_dict.get(raw_dom, 0.0)) / 100.0

        all_scores = {
            k.capitalize(): round(float(v) / 100.0, 4)
            for k, v in emotions_dict.items()
        }

        region = res.get('region', {})
        x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
        face_box = FaceBox(x=x, y=y, w=w, h=h) if w > 0 and h > 0 else None

        # Threshold gate: low-confidence → Neutral
        if top_conf < self.threshold:
            top_emotion = "Neutral"
            logger.debug(f"Confidence {top_conf:.2f} < threshold {self.threshold}. Returning Neutral.")

        logger.debug(f"Detected: {top_emotion} ({top_conf:.2%})")
        return EmotionResult(
            emotion=top_emotion,
            confidence=round(top_conf, 4),
            face_box=face_box,
            all_scores=all_scores,
        )



    def predict_from_image_path(self, image_path: str | Path) -> EmotionResult:
        """
        Convenience wrapper — load an image file and run prediction.
        Useful for testing without a live webcam.
        """
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self.predict_from_frame(frame)

    # ── Annotation helper ─────────────────────────────────────
    def annotate_frame(
        self,
        bgr_frame: np.ndarray,
        result: EmotionResult,
    ) -> np.ndarray:
        """
        Draw bounding box + emotion label + confidence bar on the frame.

        Returns a copy of the frame with overlays (original is not mutated).
        """
        frame = bgr_frame.copy()

        if not result.face_found:
            cv2.putText(
                frame, "No Face Detected",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 220), 2, cv2.LINE_AA,
            )
            return frame

        fb    = result.face_box
        colour = _BOX_COLOURS.get(result.emotion, _DEFAULT_COLOUR)

        # ── Bounding box ──────────────────────────────────────
        cv2.rectangle(frame, (fb.x, fb.y),
                      (fb.x + fb.w, fb.y + fb.h), colour, 2)

        # ── Label background pill ─────────────────────────────
        label   = f"{result.emotion}  {result.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        lx, ly = fb.x, fb.y - 10
        cv2.rectangle(frame,
                      (lx, ly - th - 6), (lx + tw + 8, ly + 4),
                      colour, -1)
        cv2.putText(
            frame, label,
            (lx + 4, ly), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 0), 2, cv2.LINE_AA,
        )

        # ── Confidence bar ────────────────────────────────────
        bar_x, bar_y = fb.x, fb.y + fb.h + 8
        bar_w = int(fb.w * result.confidence)
        cv2.rectangle(frame,
                      (bar_x, bar_y), (bar_x + fb.w, bar_y + 6),
                      (60, 60, 60), -1)
        cv2.rectangle(frame,
                      (bar_x, bar_y), (bar_x + bar_w, bar_y + 6),
                      colour, -1)

        return frame

    # ── Live webcam demo (run this file directly) ─────────────
    def run_webcam_demo(self, camera_index: int = 0) -> None:
        """
        Open the default webcam and display annotated emotion predictions.
        Press  Q  to quit.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {camera_index}.")

        logger.info("Webcam demo started. Press Q to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera.")
                break

            result      = self.predict_from_frame(frame)
            annotated   = self.annotate_frame(frame, result)
            cv2.imshow("EmotionDetector — Live Demo", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam demo stopped.")


# ─────────────────────────────────────────────────────────────
# Quick standalone test  (python modules/emotion_detector.py)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = EmotionDetector()
    detector.run_webcam_demo()
