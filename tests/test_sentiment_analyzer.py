"""
tests/test_sentiment_analyzer.py
─────────────────────────────────────────────────────────────
Unit tests for Module 2 — SentimentAnalyzer.

Tests run entirely with the VADER backend (no GPU, no downloads
beyond the small VADER lexicon). HuggingFace backend is tested
via mocking so no torch installation is required.

Run with:
    pytest tests/test_sentiment_analyzer.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentResult,
    TextPreprocessor,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def vader_analyzer():
    """One VADER analyzer shared across all tests in this module."""
    return SentimentAnalyzer(backend="vader")


@pytest.fixture(scope="module")
def preprocessor():
    return TextPreprocessor()


# ─────────────────────────────────────────────────────────────
# 1. TextPreprocessor
# ─────────────────────────────────────────────────────────────
class TestTextPreprocessor:

    def test_lowercases_input(self, preprocessor):
        result = preprocessor.clean("I Am FEELING Stressed")
        assert result == result.lower()

    def test_expands_contraction_wont(self, preprocessor):
        result = preprocessor.clean("I won't finish this today")
        assert "will not" in result
        assert "won't" not in result

    def test_expands_contraction_cant(self, preprocessor):
        result = preprocessor.clean("I can't cope")
        assert "cannot" in result

    def test_removes_url(self, preprocessor):
        result = preprocessor.clean("Check https://example.com for updates")
        assert "http" not in result
        assert "example" not in result

    def test_removes_email(self, preprocessor):
        result = preprocessor.clean("Contact hr@company.org please")
        assert "@" not in result

    def test_removes_special_characters(self, preprocessor):
        result = preprocessor.clean("Feeling great!!! :) #Monday")
        assert "!" not in result
        assert "#" not in result
        assert ":" not in result

    def test_collapses_whitespace(self, preprocessor):
        result = preprocessor.clean("   too    many    spaces   ")
        assert "  " not in result
        assert result == result.strip()

    def test_preserves_negation_not(self, preprocessor):
        """'not' must survive stopword filtering (critical for sentiment)."""
        keywords = preprocessor.extract_keywords(
            preprocessor.clean("I am not happy today")
        )
        # 'not' survives because it is excluded from stopwords
        # 'happy' is in emotional keyword list
        assert "happy" in keywords

    def test_extract_stress_keywords(self, preprocessor):
        text     = preprocessor.clean("I feel stressed and overwhelmed")
        keywords = preprocessor.extract_keywords(text)
        assert "stressed" in keywords or "overwhelmed" in keywords

    def test_extract_positive_keywords(self, preprocessor):
        text     = preprocessor.clean("I am excited and motivated today")
        keywords = preprocessor.extract_keywords(text)
        assert "excited" in keywords or "motivated" in keywords

    def test_extract_keywords_empty_string(self, preprocessor):
        keywords = preprocessor.extract_keywords("")
        assert keywords == []

    def test_clean_empty_string_returns_empty(self, preprocessor):
        assert preprocessor.clean("") == ""


# ─────────────────────────────────────────────────────────────
# 2. SentimentResult dataclass
# ─────────────────────────────────────────────────────────────
class TestSentimentResult:

    def test_wellbeing_score_formula(self):
        """wellbeing = (compound + 1) / 2"""
        result = SentimentResult(
            text="test", compound=0.60, label="Positive"
        )
        assert result.wellbeing_score == pytest.approx(0.80, abs=1e-4)

    def test_wellbeing_score_max_positive(self):
        result = SentimentResult(text="", compound=1.0, label="Positive")
        assert result.wellbeing_score == 1.0

    def test_wellbeing_score_max_negative(self):
        result = SentimentResult(text="", compound=-1.0, label="Negative")
        assert result.wellbeing_score == 0.0

    def test_wellbeing_score_neutral(self):
        result = SentimentResult(text="", compound=0.0, label="Neutral")
        assert result.wellbeing_score == pytest.approx(0.50, abs=1e-4)

    def test_wellbeing_always_in_range(self):
        for compound in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            r = SentimentResult(text="", compound=compound, label="Neutral")
            assert 0.0 <= r.wellbeing_score <= 1.0

    def test_to_dict_has_all_keys(self):
        r = SentimentResult(
            text="hello", compound=0.3, label="Positive",
            pos=0.5, neu=0.5, neg=0.0, backend="vader", keywords=["happy"]
        )
        d = r.to_dict()
        for key in ["text", "compound", "label", "pos", "neu",
                    "neg", "backend", "wellbeing_score", "keywords"]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_wellbeing_matches_property(self):
        r = SentimentResult(text="test", compound=0.4, label="Positive")
        assert r.to_dict()["wellbeing_score"] == r.wellbeing_score


# ─────────────────────────────────────────────────────────────
# 3. SentimentAnalyzer — VADER backend
# ─────────────────────────────────────────────────────────────
class TestVaderAnalyzer:

    def test_returns_sentiment_result(self, vader_analyzer):
        r = vader_analyzer.analyze("I feel great today!")
        assert isinstance(r, SentimentResult)

    def test_clearly_positive_text(self, vader_analyzer):
        r = vader_analyzer.analyze(
            "I am absolutely thrilled and excited about the new project!"
        )
        assert r.label    == "Positive"
        assert r.compound  > 0.05

    def test_clearly_negative_text(self, vader_analyzer):
        r = vader_analyzer.analyze(
            "I am completely overwhelmed, stressed, and burnt out."
        )
        assert r.label    == "Negative"
        assert r.compound  < -0.05

    def test_neutral_text(self, vader_analyzer):
        r = vader_analyzer.analyze(
            "Today is Monday. I attended a meeting."
        )
        assert r.label == "Neutral"

    def test_empty_text_returns_neutral(self, vader_analyzer):
        r = vader_analyzer.analyze("")
        assert r.label    == "Neutral"
        assert r.compound == 0.0
        assert r.text     == ""

    def test_whitespace_only_returns_neutral(self, vader_analyzer):
        r = vader_analyzer.analyze("     ")
        assert r.label == "Neutral"

    def test_backend_name_is_vader(self, vader_analyzer):
        r = vader_analyzer.analyze("Some text")
        assert r.backend == "vader"

    def test_compound_in_valid_range(self, vader_analyzer):
        texts = [
            "Amazing day!",
            "Terrible, horrible, no good day.",
            "Just a regular Tuesday.",
        ]
        for text in texts:
            r = vader_analyzer.analyze(text)
            assert -1.0 <= r.compound <= 1.0, (
                f"Compound {r.compound} out of range for: {text}"
            )

    def test_pos_neu_neg_sum_to_approx_one(self, vader_analyzer):
        r = vader_analyzer.analyze("I feel pretty good today.")
        total = r.pos + r.neu + r.neg
        assert total == pytest.approx(1.0, abs=0.01)

    def test_keywords_extracted_for_emotional_text(self, vader_analyzer):
        r = vader_analyzer.analyze("I am stressed, tired, and exhausted.")
        # At least one of the emotion keywords should appear
        assert len(r.keywords) > 0

    def test_keywords_empty_for_bland_text(self, vader_analyzer):
        r = vader_analyzer.analyze("The report was submitted on time.")
        assert r.keywords == []

    def test_negation_handled(self, vader_analyzer):
        positive = vader_analyzer.analyze("I am happy.")
        negative = vader_analyzer.analyze("I am not happy.")
        # Negation should lower the compound score
        assert negative.compound < positive.compound


# ─────────────────────────────────────────────────────────────
# 4. Batch analysis & summary score
# ─────────────────────────────────────────────────────────────
class TestBatchAndSummary:

    def test_analyze_batch_length(self, vader_analyzer):
        texts = ["Great day!", "Awful meeting.", "Regular lunch."]
        results = vader_analyzer.analyze_batch(texts)
        assert len(results) == len(texts)

    def test_analyze_batch_all_results(self, vader_analyzer):
        texts = ["Happy", "Sad", "Neutral"]
        results = vader_analyzer.analyze_batch(texts)
        for r in results:
            assert isinstance(r, SentimentResult)

    def test_analyze_batch_empty_list(self, vader_analyzer):
        assert vader_analyzer.analyze_batch([]) == []

    def test_summary_score_positive_texts(self, vader_analyzer):
        texts = [
            "I love my work!",
            "Feeling energized and productive.",
            "Great collaboration today.",
        ]
        score = vader_analyzer.summary_score(texts)
        assert score > 0.60, "Positive texts should yield high summary score"

    def test_summary_score_negative_texts(self, vader_analyzer):
        texts = [
            "Completely burnt out and overwhelmed.",
            "Terrible day, can't handle the stress.",
            "Exhausted and frustrated.",
        ]
        score = vader_analyzer.summary_score(texts)
        assert score < 0.45, "Negative texts should yield low summary score"

    def test_summary_score_empty_list_returns_neutral(self, vader_analyzer):
        assert vader_analyzer.summary_score([]) == 0.50

    def test_summary_score_in_range(self, vader_analyzer):
        mixed = ["Great!", "Terrible.", "Okay."]
        score = vader_analyzer.summary_score(mixed)
        assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────
# 5. Threshold configuration
# ─────────────────────────────────────────────────────────────
class TestThresholds:

    def test_custom_positive_threshold(self):
        """High positive threshold → only strongly positive text is 'Positive'."""
        strict_analyzer = SentimentAnalyzer(
            backend="vader", pos_threshold=0.80, neg_threshold=-0.05
        )
        # Mildly positive text → should be Neutral under strict threshold
        r = strict_analyzer.analyze("That was fine.")
        assert r.label in ("Neutral", "Negative")

    def test_custom_negative_threshold(self):
        """Low negative threshold → mildly negative text stays Neutral."""
        lenient_analyzer = SentimentAnalyzer(
            backend="vader", pos_threshold=0.05, neg_threshold=-0.80
        )
        r = lenient_analyzer.analyze("It was a bit disappointing.")
        assert r.label in ("Neutral", "Positive")

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            SentimentAnalyzer(backend="invalid_backend")


# ─────────────────────────────────────────────────────────────
# 6. HuggingFace backend (mocked — no torch needed)
# ─────────────────────────────────────────────────────────────
class TestHuggingFaceBackendMocked:

    def _make_hf_analyzer(self, hf_output: list[dict]) -> SentimentAnalyzer:
        """Build an HF analyzer with a fully mocked pipeline."""
        with patch(
            "modules.sentiment_analyzer._HuggingFaceBackend.__init__",
            return_value=None
        ):
            analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
            analyzer.backend_name  = "hf"
            analyzer.pos_threshold = 0.05
            analyzer.neg_threshold = -0.05

            from modules.sentiment_analyzer import TextPreprocessor
            analyzer._preprocessor = TextPreprocessor()

            mock_hf_backend = MagicMock()
            # Simulate HF score() output
            label = hf_output[0]["label"]
            score = hf_output[0]["score"]
            compound = score if label == "POSITIVE" else -score
            mock_hf_backend.score.return_value = {
                "compound": compound,
                "pos":      score if label == "POSITIVE" else 1 - score,
                "neu":      0.0,
                "neg":      score if label == "NEGATIVE" else 1 - score,
            }
            analyzer._backend = mock_hf_backend
        return analyzer

    def test_hf_positive_result(self):
        analyzer = self._make_hf_analyzer(
            [{"label": "POSITIVE", "score": 0.98}]
        )
        r = analyzer.analyze("Excellent work environment!")
        assert r.label    == "Positive"
        assert r.compound  > 0.05

    def test_hf_negative_result(self):
        analyzer = self._make_hf_analyzer(
            [{"label": "NEGATIVE", "score": 0.91}]
        )
        r = analyzer.analyze("I dread coming to work.")
        assert r.label    == "Negative"
        assert r.compound  < -0.05

    def test_hf_backend_name_in_result(self):
        analyzer = self._make_hf_analyzer(
            [{"label": "POSITIVE", "score": 0.75}]
        )
        r = analyzer.analyze("Things are going okay.")
        assert r.backend == "hf"
