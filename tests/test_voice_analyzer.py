"""
tests/test_voice_analyzer.py  —  UNIT TESTS FOR MODULE 5
─────────────────────────────────────────────────────────────
Verifies:
• Audio data ingestion
• Mocked transcription handling
• RMS Energy calculation (Vocal Vitality)
• Sentiment mapping fallback
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from modules.voice_analyzer import VoiceAnalyzer, VoiceResult

@pytest.fixture
def analyzer():
    return VoiceAnalyzer()

def test_vitality_calculation_with_dummy_data(analyzer):
    """Verify that RMS energy extraction works on synthetic byte data."""
    # Create a simple sine wave buffer (1 second, 44100Hz)
    duration = 1.0
    fs = 44100
    f = 440.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # Loud signal
    audio_data = (np.sin(2 * np.pi * f * t) * 32767).astype(np.int16)
    bytes_data = audio_data.tobytes()

    score = analyzer._extract_vitality(bytes_data)
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # Pure sine waves at max amplitude are high energy

def test_vitality_with_silence(analyzer):
    """Verify that silence results in low vitality."""
    silence = np.zeros(44100, dtype=np.int16).tobytes()
    score = analyzer._extract_vitality(silence)
    assert score < 0.1

def test_analyze_audio_no_data(analyzer):
    """Handles None input gracefully."""
    result = analyzer.analyze_audio(None)
    assert result is None

@patch("speech_recognition.Recognizer.recognize_google")
@patch("speech_recognition.AudioFile")
def test_full_analysis_mocked(mock_audio_file, mock_recognize, analyzer):
    """Test full pipeline with mocked STT."""
    mock_recognize.return_value = "I am feeling very energetic today"
    
    # Mocking AudioFile context manager
    mock_audio_file.return_value.__enter__.return_value = "audio_source"
    
    # We pass some dummy bytes
    dummy_bytes = np.random.randint(-100, 100, 1000, dtype=np.int16).tobytes()
    
    result = analyzer.analyze_audio(dummy_bytes)
    
    assert isinstance(result, VoiceResult)
    assert result.transcript == "I am feeling very energetic today"
    assert result.wellbeing_score > 0.4  # Vitality + Sentiment blend
    assert result.sentiment.label in ["Positive", "Neutral", "Negative"]

def test_sentiment_fallback_on_stt_failure(analyzer):
    """Verify result structure when transcription fails."""
    # We force an error in transcription
    with patch("speech_recognition.Recognizer.record", side_effect=Exception("API Error")):
        dummy_bytes = np.zeros(1000, dtype=np.int16).tobytes()
        result = analyzer.analyze_audio(dummy_bytes)
        
        assert result.transcript == "[Transcription failed]"
        assert result.sentiment is None
        # Score should be purely derived from vitality (which is 0 in this case)
        assert result.wellbeing_score == 0.0
