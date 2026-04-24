"""
modules/voice_analyzer.py  —  MODULE 5
─────────────────────────────────────────────────────────────
Voice Analysis Engine (Speech-to-Text + Acoustics).

Responsibilities
────────────────
• Accept raw audio bytes from Streamlit camera/mic input
• Temporarily store and convert audio to WAV format
• Transcribe speech to text using Google Web Speech (Free Tier)
• Integrate with SentimentAnalyzer (Module 2) for text sentiment
• Extract 'Vitality' heuristic (RMS Energy) to detect stress/fatigue
• Return a VoiceResult with transcript, sentiment, and vitality score

Public API
──────────
    analyzer = VoiceAnalyzer()
    result   = analyzer.analyze_audio(audio_bytes, sample_rate=44100)
        → VoiceResult(transcript, compound, vitality, wellbeing_score)
"""

import os
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
from loguru import logger

# ── Project imports ───────────────────────────────────────────
from config import LOG_DIR, TRANSCRIPTION_LANG
from modules.sentiment_analyzer import SentimentAnalyzer, SentimentResult

logger.add(LOG_DIR / "voice_analyzer.log", rotation="5 MB", level="DEBUG")


@dataclass
class VoiceResult:
    """
    Combined output of voice transcription and sentiment analysis.
    """
    transcript:      str
    sentiment:       Optional[SentimentResult]
    vitality_score:  float    # 0.0 (tired/quiet) to 1.0 (energetic/loud)
    wellbeing_score: float    # combined 0.0 to 1.0

    def to_dict(self) -> dict:
        return {
            "transcript":      self.transcript,
            "sentiment":       self.sentiment.to_dict() if self.sentiment else None,
            "vitality_score":  self.vitality_score,
            "wellbeing_score": self.wellbeing_score,
        }


class VoiceAnalyzer:
    """
    Multimodal voice analysis engine.
    """

    def __init__(self, language: str = TRANSCRIPTION_LANG):
        self.language = language
        self.recognizer = sr.Recognizer()
        self.sentiment_engine = SentimentAnalyzer(backend="vader")
        logger.info(f"VoiceAnalyzer initialised (Language: {self.language})")

    def _extract_vitality(self, audio_segment: AudioSegment) -> float:
        """
        Heuristic for 'Vitality' based on RMS Energy (volume).
        Normalizes dBFS to a [0, 1] range where -20dB is Peak (1.0) and -50dB is Silence/Mumble (0.0).
        """
        rms = audio_segment.rms
        if rms == 0:
            return 0.0
        
        # dBFS is usually 0 (max) to -90 (silence)
        db = audio_segment.dBFS
        # Map -50 (Low) to -15 (High) -> [0, 1]
        low, high = -50.0, -15.0
        vitality = (db - low) / (high - low)
        return float(np.clip(vitality, 0.0, 1.0))

    def analyze_audio(self, audio_bytes: bytes) -> VoiceResult:
        """
        Full pipeline: Audio Bytes → WAV → Transcription → Sentiment + Vitality.
        """
        if not audio_bytes:
            logger.warning("Empty audio bytes received.")
            return VoiceResult("", None, 0.5, 0.5)

        try:
            # 1. Load Bytes into Pydub
            audio_io = io.BytesIO(audio_bytes)
            # We assume it's coming from Streamlit's mic recorder which is usually WebM or WAV
            # but we can try to let pydub guess the format.
            audio_seg = AudioSegment.from_file(audio_io)
            
            # 2. Extract Vitality (Acoustic energy)
            vitality = self._extract_vitality(audio_seg)
            
            # 3. Transcribe using SpeechRecognition
            # SR requires a WAV/AIFF/FLAC file or AudioData
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                audio_seg.export(tmp_path, format="wav")
            
            transcript = ""
            try:
                with sr.AudioFile(tmp_path) as source:
                    audio_data = self.recognizer.record(source)
                    transcript = self.recognizer.recognize_google(audio_data, language=self.language)
                logger.debug(f"Transcription successful: {transcript}")
            except sr.UnknownValueError:
                logger.warning("Speech Recognition could not understand audio.")
                transcript = "[Unrecognized Speech]"
            except sr.RequestError as e:
                logger.error(f"Could not request results from Google Speech; {e}")
                transcript = "[API Error]"
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            # 4. Sentiment Analysis on transcript
            sent_res = None
            if transcript and transcript not in ("[Unrecognized Speech]", "[API Error]"):
                sent_res = self.sentiment_engine.analyze(transcript)
                sent_score = sent_res.wellbeing_score
            else:
                sent_score = 0.5  # Neutral fallback

            # 5. Combine scores (Vitality + Sentiment)
            # Vitality (0.1 weight) + Sentiment (0.9 weight)
            combined_wb = round(0.9 * sent_score + 0.1 * vitality, 4)

            return VoiceResult(
                transcript=transcript,
                sentiment=sent_res,
                vitality_score=round(vitality, 4),
                wellbeing_score=combined_wb
            )

        except Exception as e:
            logger.exception(f"Error during voice analysis: {e}")
            # Fallback for when pydub fails due to missing ffmpeg
            if "ffmpeg" in str(e).lower() or "avconv" in str(e).lower():
                logger.warning("pydub energy extraction failed (ffmpeg missing). Using raw byte heuristic.")
                try:
                    # Very crude energy heuristic from raw bytes
                    raw_energy = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    rms = np.sqrt(np.mean(raw_energy**2))
                    vitality = np.clip(rms / 1000.0, 0.0, 1.0)
                    return VoiceResult("[Transcription Unavailable - ffmpeg missing]", None, float(vitality), 0.5)
                except:
                    pass
            return VoiceResult("[Error]", None, 0.5, 0.5)


if __name__ == "__main__":
    # Test stub (requires an actual audio file)
    print("Voice Analyzer Module loaded.")
    analyzer = VoiceAnalyzer()
