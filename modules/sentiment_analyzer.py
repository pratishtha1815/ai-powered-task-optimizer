"""
modules/sentiment_analyzer.py  —  MODULE 2
─────────────────────────────────────────────────────────────
Textual Sentiment Analyzer.

Responsibilities
────────────────
• Accept raw daily check-in text from an employee
• Run VADER for fast, inference-free sentiment scoring (default)
• Optional upgrade path to a HuggingFace transformer model
• Apply NLTK preprocessing (tokenise, clean, remove noise)
• Return a SentimentResult with score, label, and well-being contribution

Public API
──────────
    analyzer = SentimentAnalyzer()              # VADER backend (default)
    analyzer = SentimentAnalyzer(backend="hf")  # HuggingFace backend

    result = analyzer.analyze("I feel overwhelmed and exhausted today.")
        → SentimentResult(
              text      = "I feel overwhelmed and exhausted today.",
              compound  = -0.6124,
              label     = "Negative",
              pos       = 0.0,
              neu       = 0.328,
              neg       = 0.672,
              wellbeing_score = 0.194
          )
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from loguru import logger

# ── Project imports ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SENT_POS_THRESH, SENT_NEG_THRESH, LOG_DIR
)

logger.add(LOG_DIR / "sentiment_analyzer.log", rotation="5 MB", level="DEBUG")

# Valid backend identifiers
BackendType = Literal["vader", "hf"]


# ─────────────────────────────────────────────────────────────
# NLTK resource bootstrap (runs once, silent if already present)
# ─────────────────────────────────────────────────────────────
def _ensure_nltk_resources() -> None:
    """Download required NLTK corpora if not already present."""
    import nltk
    resources = [
        ("tokenizers/punkt",       "punkt"),
        ("tokenizers/punkt_tab",   "punkt_tab"),
        ("corpora/stopwords",      "stopwords"),
        ("corpora/wordnet",        "wordnet"),
    ]
    for check_path, resource_name in resources:
        try:
            nltk.data.find(check_path)
        except Exception:
            logger.info(f"Downloading NLTK resource: {resource_name}")
            nltk.download(resource_name, quiet=True)


# ─────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────
@dataclass
class SentimentResult:
    """
    Full output of one sentiment analysis pass.

    Attributes
    ----------
    text        : str   – the original (raw) input text
    compound    : float – overall sentiment score in [-1.0, +1.0]
                          (VADER compound / mapped HF score)
    label       : str   – 'Positive' | 'Neutral' | 'Negative'
    pos         : float – proportion of positive sentiment [0, 1]
    neu         : float – proportion of neutral sentiment  [0, 1]
    neg         : float – proportion of negative sentiment [0, 1]
    backend     : str   – which backend produced this result
    wellbeing_score : float – normalised score in [0, 1] for Task Engine
    keywords    : list[str] – top emotional keywords extracted from text
    """
    text:            str
    compound:        float
    label:           str
    pos:             float   = 0.0
    neu:             float   = 0.0
    neg:             float   = 0.0
    backend:         str     = "vader"
    keywords:        list    = field(default_factory=list)

    @property
    def wellbeing_score(self) -> float:
        """
        Map compound score [-1, +1] → well-being [0, 1].
        Formula:  (compound + 1) / 2
        """
        return round((self.compound + 1.0) / 2.0, 4)

    def to_dict(self) -> dict:
        """Serialisable representation for logging / DB storage."""
        return {
            "text":            self.text,
            "compound":        self.compound,
            "label":           self.label,
            "pos":             self.pos,
            "neu":             self.neu,
            "neg":             self.neg,
            "backend":         self.backend,
            "wellbeing_score": self.wellbeing_score,
            "keywords":        self.keywords,
        }


# ─────────────────────────────────────────────────────────────
# Text Preprocessor
# ─────────────────────────────────────────────────────────────
class TextPreprocessor:
    """
    Lightweight NLP preprocessor for check-in text.

    Applies:
    • Lowercase normalisation
    • URL / email / special-character removal
    • Contraction expansion (won't → will not, etc.)
    • NLTK word tokenisation
    • Stopword filtering (keeps negations like 'not', 'no')
    • Returns cleaned string + extracted emotional keywords
    """

    # Common English contractions
    _CONTRACTIONS: dict[str, str] = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "i'm":   "i am",     "i've":  "i have", "i'd": "i would",
        "i'll":  "i will",   "it's":  "it is",  "he's": "he is",
        "she's": "she is",   "we're": "we are", "they're": "they are",
        "you're":"you are",  "wasn't":"was not", "isn't": "is not",
        "aren't":"are not",  "didn't":"did not", "doesn't":"does not",
        "hadn't":"had not",  "hasn't":"has not", "haven't":"have not",
        "couldn't":"could not","shouldn't":"should not","wouldn't":"would not",
    }

    # Emotional keyword vocabulary per valence
    _EMOTION_KEYWORDS: dict[str, list[str]] = {
        "stress":   ["stressed", "stress", "overwhelmed", "pressure",
                     "anxious", "panic", "tense"],
        "fatigue":  ["tired", "exhausted", "fatigue", "drained", "sleepy",
                     "burnt", "burnout", "worn"],
        "positive": ["happy", "excited", "motivated", "great", "excellent",
                     "productive", "energized", "confident", "good"],
        "negative": ["sad", "angry", "frustrated", "upset", "depressed",
                     "hopeless", "terrible", "awful", "bad"],
    }
    _ALL_EMOTION_WORDS: set[str] = {
        w for words in _EMOTION_KEYWORDS.values() for w in words
    }

    def __init__(self):
        _ensure_nltk_resources()
        from nltk.corpus import stopwords
        # Keep negation words so VADER scores them correctly
        _base_stopwords = set(stopwords.words("english"))
        _negations       = {"no", "not", "nor", "never", "neither"}
        self._stopwords  = _base_stopwords - _negations

    def clean(self, text: str) -> str:
        """Return a cleaned version of text suitable for VADER."""
        # Lowercase
        text = text.lower()
        # Expand contractions
        for contraction, expansion in self._CONTRACTIONS.items():
            text = text.replace(contraction, expansion)
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # Remove emails
        text = re.sub(r"\S+@\S+", " ", text)
        # Remove non-alphanumeric (keep spaces and hyphens)
        text = re.sub(r"[^a-z0-9\s\-]", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_keywords(self, text: str) -> list[str]:
        """
        Return emotional keywords found in the (already-cleaned) text.
        Preserves order of appearance.
        """
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
        return [t for t in tokens if t in self._ALL_EMOTION_WORDS]


# ─────────────────────────────────────────────────────────────
# VADER Backend
# ─────────────────────────────────────────────────────────────
class _VaderBackend:
    """
    Thin wrapper around vaderSentiment's SentimentIntensityAnalyzer.
    Preloads the lexicon once; thread-safe for repeated calls.
    """

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._sia = SentimentIntensityAnalyzer()
        logger.info("VADER SentimentIntensityAnalyzer loaded.")

    def score(self, clean_text: str) -> dict:
        """
        Returns {'compound': float, 'pos': float, 'neu': float, 'neg': float}.
        """
        return self._sia.polarity_scores(clean_text)


# ─────────────────────────────────────────────────────────────
# HuggingFace Backend (optional, GPU-friendly)
# ─────────────────────────────────────────────────────────────
class _HuggingFaceBackend:
    """
    Uses 'distilbert-base-uncased-finetuned-sst-2-english' by default.
    Maps binary POSITIVE/NEGATIVE output to a [-1, +1] compound score.

    Requires: pip install transformers torch
    """

    HF_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self, model_name: str = HF_MODEL):
        try:
            from transformers import pipeline as hf_pipeline
            logger.info(f"Loading HuggingFace model: {model_name} (may take a moment…)")
            self._pipe = hf_pipeline(
                "sentiment-analysis",
                model=model_name,
                truncation=True,
                max_length=512,
            )
            logger.info("HuggingFace pipeline ready.")
        except ImportError as e:
            raise ImportError(
                "HuggingFace backend requires 'transformers' and 'torch'.\n"
                "  pip install transformers torch\n"
                f"  Original error: {e}"
            )

    def score(self, clean_text: str) -> dict:
        """
        Returns VADER-compatible dict:
        {'compound': float[-1,1], 'pos': float, 'neu': 0.0, 'neg': float}
        """
        output = self._pipe(clean_text)[0]
        label  = output["label"]    # 'POSITIVE' or 'NEGATIVE'
        score  = output["score"]    # confidence [0.5, 1.0]

        if label == "POSITIVE":
            compound = score            # positive range ~ [0.5, 1.0]
            pos, neg = score, 1 - score
        else:
            compound = -score           # negative range ~ [-1.0, -0.5]
            pos, neg = 1 - score, score

        return {
            "compound": round(compound, 4),
            "pos":      round(pos,      4),
            "neu":      0.0,
            "neg":      round(neg,      4),
        }


# ─────────────────────────────────────────────────────────────
# SentimentAnalyzer — main class
# ─────────────────────────────────────────────────────────────
class SentimentAnalyzer:
    """
    Employee check-in text → SentimentResult.

    Parameters
    ----------
    backend : 'vader' | 'hf'
        'vader' (default) — fast, no GPU required, excellent for short texts.
        'hf'              — HuggingFace DistilBERT; better for nuanced prose.
    hf_model : str
        Custom HuggingFace model name/path (ignored when backend='vader').
    pos_threshold : float
        Compound score above which text is labelled 'Positive'. Default 0.05.
    neg_threshold : float
        Compound score below which text is labelled 'Negative'. Default -0.05.
    """

    def __init__(
        self,
        backend:       BackendType = "vader",
        hf_model:      str         = _HuggingFaceBackend.HF_MODEL,
        pos_threshold: float       = SENT_POS_THRESH,
        neg_threshold: float       = SENT_NEG_THRESH,
    ):
        self.backend_name  = backend
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

        self._preprocessor = TextPreprocessor()

        if backend == "vader":
            self._backend = _VaderBackend()
        elif backend == "hf":
            self._backend = _HuggingFaceBackend(model_name=hf_model)
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose 'vader' or 'hf'."
            )

    # ── Label mapping ─────────────────────────────────────────
    def _compound_to_label(self, compound: float) -> str:
        if compound >= self.pos_threshold:
            return "Positive"
        elif compound <= self.neg_threshold:
            return "Negative"
        else:
            return "Neutral"

    # ── Core analysis ─────────────────────────────────────────
    def analyze(self, text: str) -> SentimentResult:
        """
        Full pipeline: raw text → SentimentResult.

        Parameters
        ----------
        text : str
            Raw employee check-in text. Any length, any case.

        Returns
        -------
        SentimentResult
            If text is empty, returns a Neutral baseline result.
        """
        text = text.strip()
        if not text:
            logger.warning("Empty text received; returning Neutral baseline.")
            return SentimentResult(
                text="",
                compound=0.0,
                label="Neutral",
                pos=0.0,
                neu=1.0,
                neg=0.0,
                backend=self.backend_name,
                keywords=[],
            )

        clean    = self._preprocessor.clean(text)
        keywords = self._preprocessor.extract_keywords(clean)
        scores   = self._backend.score(clean)

        label    = self._compound_to_label(scores["compound"])

        result = SentimentResult(
            text     = text,
            compound = round(scores["compound"], 4),
            label    = label,
            pos      = round(scores.get("pos", 0.0), 4),
            neu      = round(scores.get("neu", 0.0), 4),
            neg      = round(scores.get("neg", 0.0), 4),
            backend  = self.backend_name,
            keywords = keywords,
        )

        logger.debug(
            f"Sentiment [{self.backend_name}] | "
            f"label={result.label} | compound={result.compound:.4f} | "
            f"wellbeing={result.wellbeing_score:.4f} | "
            f"keywords={result.keywords}"
        )
        return result

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyse a list of texts and return a list of SentimentResults."""
        return [self.analyze(t) for t in texts]

    def summary_score(self, texts: list[str]) -> float:
        """
        Convenience: analyse multiple texts and return the mean well-being score.
        Useful for weekly trend aggregation.
        """
        if not texts:
            return 0.50   # neutral baseline
        results = self.analyze_batch(texts)
        scores  = [r.wellbeing_score for r in results]
        return round(sum(scores) / len(scores), 4)


# ─────────────────────────────────────────────────────────────
# Quick standalone test  (python modules/sentiment_analyzer.py)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_texts = [
        "I feel absolutely great today! Really motivated and productive.",
        "Everything is fine, nothing special happened.",
        "I am completely overwhelmed and stressed. I can't keep up with all the deadlines.",
        "Feeling a bit tired but manageable.",
        "I love working on this project, it excites me every day!",
        "",  # edge case: empty input
    ]

    print("\n" + "─" * 60)
    print("  Sentiment Analyzer — VADER Demo")
    print("─" * 60)

    analyzer = SentimentAnalyzer(backend="vader")
    for text in demo_texts:
        r = analyzer.analyze(text)
        display = r.text[:55] + "…" if len(r.text) > 55 else r.text or "<empty>"
        print(
            f"  [{r.label:8s}]  compound={r.compound:+.4f}  "
            f"wellbeing={r.wellbeing_score:.3f}  | {display}"
        )
        if r.keywords:
            print(f"             ↳ keywords: {r.keywords}")
    print("─" * 60 + "\n")
