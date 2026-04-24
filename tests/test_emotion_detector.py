"""
tests/test_emotion_detector.py
─────────────────────────────────────────────────────────────
Unit tests for Module 1 — EmotionDetector.

Tests are designed to run WITHOUT a webcam by mocking DeepFace.
This lets you validate the entire pipeline logic independently.

Run with:
    pytest tests/test_emotion_detector.py -v
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.emotion_detector import EmotionDetector, EmotionResult, FaceBox
from config import EMOTION_LABELS


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def dummy_bgr_frame():
    """A 480×640×3 uint8 frame (solid grey — simulates webcam output)."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def mock_detector():
    """
    EmotionDetector with default config.
    """
    return EmotionDetector(confidence_threshold=0.40)


# ─────────────────────────────────────────────────────────────
# 1. EmotionResult dataclass
# ─────────────────────────────────────────────────────────────
class TestEmotionResult:

    def test_face_found_true_when_box_set(self):
        result = EmotionResult(
            emotion="Happy", confidence=0.90,
            face_box=FaceBox(10, 10, 80, 80)
        )
        assert result.face_found is True

    def test_face_found_false_when_no_box(self):
        result = EmotionResult(emotion="No Face", confidence=0.0, face_box=None)
        assert result.face_found is False

    def test_wellbeing_score_happy_high(self):
        """Happy at max confidence → score close to 1.0."""
        result = EmotionResult(
            emotion="Happy", confidence=1.0, face_box=FaceBox(0, 0, 48, 48)
        )
        assert result.wellbeing_score() >= 0.90

    def test_wellbeing_score_angry_low(self):
        """Angry at max confidence → score well below 0.5."""
        result = EmotionResult(
            emotion="Angry", confidence=1.0, face_box=FaceBox(0, 0, 48, 48)
        )
        assert result.wellbeing_score() < 0.50

    def test_wellbeing_score_low_confidence_nudges_to_neutral(self):
        """
        A very angry (0.15 base) prediction with zero confidence
        should return 0.50 (pure neutral fallback).
        """
        result = EmotionResult(
            emotion="Angry", confidence=0.0, face_box=FaceBox(0, 0, 48, 48)
        )
        assert result.wellbeing_score() == pytest.approx(0.50)

    def test_wellbeing_score_unknown_emotion(self):
        """Unknown emotion label → defaults to neutral (0.50 base)."""
        result = EmotionResult(
            emotion="Confused", confidence=1.0, face_box=FaceBox(0, 0, 48, 48)
        )
        assert result.wellbeing_score() == pytest.approx(0.50)

    def test_wellbeing_score_range(self):
        """Score must always be in [0, 1]."""
        for emotion in EMOTION_LABELS:
            for conf in [0.0, 0.5, 1.0]:
                result = EmotionResult(
                    emotion=emotion, confidence=conf,
                    face_box=FaceBox(0, 0, 48, 48)
                )
                score = result.wellbeing_score()
                assert 0.0 <= score <= 1.0, \
                    f"Score {score} out of range for {emotion}@{conf}"


# ─────────────────────────────────────────────────────────────
# 2. Full prediction pipeline (with mocked DeepFace)
# ─────────────────────────────────────────────────────────────
class TestPredictFromFrame:

    @patch("modules.emotion_detector.DeepFace.analyze")
    def test_returns_emotion_result(self, mock_analyze, mock_detector, dummy_bgr_frame):
        mock_analyze.return_value = [{
            "dominant_emotion": "happy",
            "emotion": {"happy": 95.0, "sad": 2.0, "neutral": 3.0},
            "region": {"x": 50, "y": 50, "w": 100, "h": 100}
        }]
        result = mock_detector.predict_from_frame(dummy_bgr_frame)
        assert isinstance(result, EmotionResult)

    @patch("modules.emotion_detector.DeepFace.analyze")
    def test_dominant_emotion_is_happy(self, mock_analyze, mock_detector, dummy_bgr_frame):
        mock_analyze.return_value = [{
            "dominant_emotion": "happy",
            "emotion": {"happy": 95.0, "sad": 5.0},
            "region": {"x": 50, "y": 50, "w": 100, "h": 100}
        }]
        result = mock_detector.predict_from_frame(dummy_bgr_frame)
        assert result.emotion == "Happy"

    @patch("modules.emotion_detector.DeepFace.analyze")
    def test_confidence_is_float(self, mock_analyze, mock_detector, dummy_bgr_frame):
        mock_analyze.return_value = [{
            "dominant_emotion": "happy",
            "emotion": {"happy": 95.0},
            "region": {"x": 50, "y": 50, "w": 100, "h": 100}
        }]
        result = mock_detector.predict_from_frame(dummy_bgr_frame)
        assert isinstance(result.confidence, float)

    @patch("modules.emotion_detector.DeepFace.analyze")
    def test_face_box_set_when_face_found(self, mock_analyze, mock_detector, dummy_bgr_frame):
        mock_analyze.return_value = [{
            "dominant_emotion": "happy",
            "emotion": {"happy": 95.0},
            "region": {"x": 50, "y": 50, "w": 100, "h": 100}
        }]
        result = mock_detector.predict_from_frame(dummy_bgr_frame)
        assert result.face_box is not None
        assert isinstance(result.face_box, FaceBox)


class TestNoFaceDetected:

    @patch("modules.emotion_detector.DeepFace.analyze")
    def test_no_face_returns_no_face_emotion(self, mock_analyze, dummy_bgr_frame):
        """When deepface raises ValueError, return No Face."""
        mock_analyze.side_effect = ValueError("Face not found")
        detector = EmotionDetector(confidence_threshold=0.40)
        
        result = detector.predict_from_frame(dummy_bgr_frame)
        assert result.emotion    == "No Face"
        assert result.confidence == 0.0
        assert result.face_box   is None
        assert result.face_found is False


class TestConfidenceThreshold:

    @patch("modules.emotion_detector.DeepFace.analyze")
    def test_low_confidence_falls_back_to_neutral(self, mock_analyze, dummy_bgr_frame):
        """Predictions below threshold must be overridden with 'Neutral'."""
        detector = EmotionDetector(confidence_threshold=0.80)

        # Make CNN return 'angry' with low confidence (50%)
        mock_analyze.return_value = [{
            "dominant_emotion": "angry",
            "emotion": {"angry": 50.0},
            "region": {"x": 50, "y": 50, "w": 100, "h": 100}
        }]
        
        result = detector.predict_from_frame(dummy_bgr_frame)
        assert result.emotion == "Neutral", \
            "Low-confidence prediction should fall back to Neutral"


# ─────────────────────────────────────────────────────────────
# 3. Frame annotation
# ─────────────────────────────────────────────────────────────
class TestAnnotateFrame:

    @patch("modules.emotion_detector.DeepFace.analyze")
    def test_annotate_returns_ndarray(self, mock_analyze, mock_detector, dummy_bgr_frame):
        mock_analyze.return_value = [{
            "dominant_emotion": "happy",
            "emotion": {"happy": 95.0},
            "region": {"x": 50, "y": 50, "w": 100, "h": 100}
        }]
        result    = mock_detector.predict_from_frame(dummy_bgr_frame)
        annotated = mock_detector.annotate_frame(dummy_bgr_frame, result)
        assert isinstance(annotated, np.ndarray)

    @patch("modules.emotion_detector.DeepFace.analyze")
    def test_annotate_does_not_mutate_original(self, mock_analyze, mock_detector, dummy_bgr_frame):
        mock_analyze.return_value = [{
            "dominant_emotion": "happy",
            "emotion": {"happy": 95.0},
            "region": {"x": 50, "y": 50, "w": 100, "h": 100}
        }]
        original  = dummy_bgr_frame.copy()
        result    = mock_detector.predict_from_frame(dummy_bgr_frame)
        _         = mock_detector.annotate_frame(dummy_bgr_frame, result)
        np.testing.assert_array_equal(dummy_bgr_frame, original)

    def test_annotate_no_face_frame(self, dummy_bgr_frame):
        """Annotating a 'No Face' result should not crash."""
        no_face = EmotionResult(emotion="No Face", confidence=0.0, face_box=None)
        detector = EmotionDetector()
        annotated = detector.annotate_frame(dummy_bgr_frame, no_face)
        assert annotated.shape == dummy_bgr_frame.shape


# ─────────────────────────────────────────────────────────────
# 4. Edge cases
# ─────────────────────────────────────────────────────────────
class TestEdgeCases:

    @patch("modules.emotion_detector.DeepFace.analyze")
    def test_very_small_frame(self, mock_analyze, mock_detector):
        """Tiny 64×64 frame must not crash the pipeline."""
        mock_analyze.return_value = [{
            "dominant_emotion": "neutral",
            "emotion": {"neutral": 99.0},
            "region": {"x": 0, "y": 0, "w": 64, "h": 64}
        }]
        tiny_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = mock_detector.predict_from_frame(tiny_frame)
        assert isinstance(result, EmotionResult)
