"""
tests/test_task_matcher.py
─────────────────────────────────────────────────────────────
Unit tests for Module 3 — TaskMatcher.

Tests use an in-memory task catalogue fixture so no file I/O
or seeding is required. All module logic is tested in isolation.

Run with:
    pytest tests/test_task_matcher.py -v
"""

import sys
import io
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.task_matcher import (
    TaskMatcher,
    TaskRecommendation,
    Task,
    resolve_tier,
    _DECISION_MATRIX,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_tasks_df() -> pd.DataFrame:
    """Minimal in-memory task catalogue covering all tiers."""
    rows = []
    # Low tier: cognitive_load 1–3
    for i in range(1, 7):
        cl = (i % 3) + 1   # loads: 2,3,1,2,3,1
        rows.append({
            "id": i, "title": f"Low Task {i}",
            "description": "A light admin task.",
            "category": "Administrative",
            "cognitive_load": cl,
            "estimated_hours": 0.5,
            "tags": "admin,low-stress",
            "tier": "Low",
        })
    # Medium tier: cognitive_load 4–6
    for i in range(7, 13):
        cl = ((i - 7) % 3) + 4
        rows.append({
            "id": i, "title": f"Medium Task {i}",
            "description": "A moderate engineering task.",
            "category": "Engineering",
            "cognitive_load": cl,
            "estimated_hours": 2.0,
            "tags": "engineering,medium",
            "tier": "Medium",
        })
    # High tier: cognitive_load 7–10
    for i in range(13, 19):
        cl = ((i - 13) % 4) + 7
        rows.append({
            "id": i, "title": f"High Task {i}",
            "description": "A complex architecture task.",
            "category": "Architecture",
            "cognitive_load": cl,
            "estimated_hours": 6.0,
            "tags": "architecture,complex",
            "tier": "High",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def matcher(sample_tasks_df, tmp_path) -> TaskMatcher:
    """TaskMatcher pre-loaded with the in-memory fixture."""
    m = TaskMatcher(shuffle_ties=False)
    m._df = sample_tasks_df      # inject directly, bypass CSV
    return m


# ─────────────────────────────────────────────────────────────
# 1. Task dataclass
# ─────────────────────────────────────────────────────────────
class TestTask:

    def test_from_row_parses_all_fields(self, sample_tasks_df):
        row  = sample_tasks_df.iloc[0]
        task = Task.from_row(row)
        assert task.id               == int(row["id"])
        assert task.title            == row["title"]
        assert task.category         == row["category"]
        assert task.cognitive_load   == int(row["cognitive_load"])
        assert task.estimated_hours  == float(row["estimated_hours"])
        assert isinstance(task.tags, list)

    def test_from_row_tags_as_list(self, sample_tasks_df):
        row  = sample_tasks_df.iloc[0]
        task = Task.from_row(row)
        assert isinstance(task.tags, list)
        assert len(task.tags) > 0

    def test_to_dict_has_all_keys(self, sample_tasks_df):
        task = Task.from_row(sample_tasks_df.iloc[0])
        d    = task.to_dict()
        for key in ["id", "title", "description", "category",
                    "cognitive_load", "estimated_hours", "tags", "tier"]:
            assert key in d


# ─────────────────────────────────────────────────────────────
# 2. Decision Matrix (resolve_tier)
# ─────────────────────────────────────────────────────────────
class TestResolveTier:

    @pytest.mark.parametrize("score, expected_tier, min_cl, max_cl", [
        (0.00,  "Low",         1, 3),
        (0.17,  "Low",         1, 3),
        (0.34,  "Low",         1, 3),
        (0.35,  "Low-Medium",  1, 5),
        (0.45,  "Low-Medium",  1, 5),
        (0.54,  "Low-Medium",  1, 5),
        (0.55,  "Medium",      4, 6),
        (0.62,  "Medium",      4, 6),
        (0.69,  "Medium",      4, 6),
        (0.70,  "Medium-High", 5, 8),
        (0.77,  "Medium-High", 5, 8),
        (0.84,  "Medium-High", 5, 8),
        (0.85,  "High",        7, 10),
        (0.95,  "High",        7, 10),
        (1.00,  "High",        7, 10),
    ])
    def test_tier_mapping(self, score, expected_tier, min_cl, max_cl):
        tier, lo, hi, _ = resolve_tier(score)
        assert tier == expected_tier, f"Score {score}: expected {expected_tier}, got {tier}"
        assert lo   == min_cl
        assert hi   == max_cl

    def test_explanation_contains_score(self):
        _, _, _, template = resolve_tier(0.20)
        explanation = template.format(score=0.20)
        assert "0.20" in explanation

    def test_all_matrix_bands_covered(self):
        """No gap in the decision matrix from 0.0 to 1.0."""
        step = 0.01
        score = 0.00
        while score <= 1.00:
            tier, lo, hi, _ = resolve_tier(round(score, 2))
            assert tier is not None, f"No tier for score {score}"
            score += step


# ─────────────────────────────────────────────────────────────
# 3. Well-being Score Computation
# ─────────────────────────────────────────────────────────────
class TestComputeWellbeingScore:

    def test_weighted_blend_default_weights(self, matcher):
        # 0.6 * 1.0 + 0.4 * 1.0 = 1.0
        assert matcher.compute_wellbeing_score(1.0, 1.0) == pytest.approx(1.0)

    def test_weighted_blend_zero(self, matcher):
        assert matcher.compute_wellbeing_score(0.0, 0.0) == pytest.approx(0.0)

    def test_weighted_blend_mixed(self, matcher):
        # 0.6 * 0.8 + 0.4 * 0.6 = 0.48 + 0.24 = 0.72
        result = matcher.compute_wellbeing_score(0.8, 0.6)
        assert result == pytest.approx(0.72, abs=1e-4)

    def test_score_clamped_to_zero(self):
        """Weights summing > 1 should not produce score > 1."""
        m = TaskMatcher(weight_emotion=0.8, weight_sentiment=0.8, shuffle_ties=False)
        m._df = pd.DataFrame([])  # won't be used here
        result = m.compute_wellbeing_score(1.0, 1.0)
        assert result <= 1.0

    def test_custom_weights(self):
        """Test with equal 50/50 weighting."""
        m = TaskMatcher(weight_emotion=0.5, weight_sentiment=0.5, shuffle_ties=False)
        result = m.compute_wellbeing_score(0.6, 0.8)
        assert result == pytest.approx(0.70, abs=1e-4)

    def test_output_rounded_to_4dp(self, matcher):
        result = matcher.compute_wellbeing_score(0.333333, 0.666666)
        # Result should have at most 4 decimal places
        assert result == round(result, 4)


# ─────────────────────────────────────────────────────────────
# 4. Recommend (full pipeline)
# ─────────────────────────────────────────────────────────────
class TestRecommend:

    def test_returns_task_recommendation(self, matcher):
        rec = matcher.recommend_from_scores(0.5, 0.5)
        assert isinstance(rec, TaskRecommendation)

    def test_top_n_respected(self, matcher):
        for n in [1, 3, 5]:
            rec = matcher.recommend_from_scores(0.5, 0.5, top_n=n)
            assert len(rec.tasks) <= n

    def test_all_tasks_are_task_objects(self, matcher):
        rec = matcher.recommend_from_scores(0.6, 0.6)
        for t in rec.tasks:
            assert isinstance(t, Task)

    def test_low_wellbeing_returns_low_cl_tasks(self, matcher):
        rec = matcher.recommend_from_scores(0.10, 0.10, top_n=6)
        for t in rec.tasks:
            assert t.cognitive_load <= 3, \
                f"Task '{t.title}' (CL={t.cognitive_load}) too heavy for low WB"

    def test_high_wellbeing_returns_high_cl_tasks(self, matcher):
        rec = matcher.recommend_from_scores(0.92, 0.92, top_n=6)
        for t in rec.tasks:
            assert t.cognitive_load >= 7, \
                f"Task '{t.title}' (CL={t.cognitive_load}) too light for high WB"

    def test_neutral_wellbeing_returns_medium_cl_tasks(self, matcher):
        rec = matcher.recommend_from_scores(0.62, 0.62, top_n=6)
        for t in rec.tasks:
            assert 4 <= t.cognitive_load <= 6, \
                f"Task '{t.title}' (CL={t.cognitive_load}) outside Medium range"

    def test_wellbeing_score_stored_in_result(self, matcher):
        rec = matcher.recommend_from_scores(0.80, 0.60)
        expected = matcher.compute_wellbeing_score(0.80, 0.60)
        assert rec.wellbeing_score == expected

    def test_emotion_and_sentiment_scores_stored(self, matcher):
        rec = matcher.recommend_from_scores(0.75, 0.55)
        assert rec.emotion_score   == pytest.approx(0.75, abs=1e-4)
        assert rec.sentiment_score == pytest.approx(0.55, abs=1e-4)

    def test_explanation_is_non_empty_string(self, matcher):
        rec = matcher.recommend_from_scores(0.5, 0.5)
        assert isinstance(rec.explanation, str)
        assert len(rec.explanation) > 10

    def test_tier_range_is_tuple_of_ints(self, matcher):
        rec = matcher.recommend_from_scores(0.5, 0.5)
        assert isinstance(rec.tier_range, tuple)
        assert len(rec.tier_range) == 2
        assert isinstance(rec.tier_range[0], int)

    def test_to_dict_serialisable(self, matcher):
        rec  = matcher.recommend_from_scores(0.5, 0.5)
        data = rec.to_dict()
        import json
        # Must not raise
        json.dumps(data)

    def test_recommend_accepts_raw_scores_directly(self, matcher):
        rec = matcher.recommend(emotion_score=0.7, sentiment_score=0.8)
        assert isinstance(rec, TaskRecommendation)


# ─────────────────────────────────────────────────────────────
# 5. Boundary / Edge Cases
# ─────────────────────────────────────────────────────────────
class TestEdgeCases:

    def test_score_zero(self, matcher):
        """Score of 0.0 should not crash and should return Low tasks."""
        rec = matcher.recommend_from_scores(0.0, 0.0)
        assert rec.tier == "Low"

    def test_score_one(self, matcher):
        """Score of 1.0 should not crash and should return High tasks."""
        rec = matcher.recommend_from_scores(1.0, 1.0)
        assert rec.tier == "High"

    def test_top_n_zero_returns_empty_list(self, matcher):
        rec = matcher.recommend_from_scores(0.5, 0.5, top_n=0)
        assert rec.tasks == []

    def test_top_n_larger_than_catalogue(self, matcher):
        """top_n > available tasks should not crash; returns what exists."""
        rec = matcher.recommend_from_scores(0.5, 0.5, top_n=999)
        assert isinstance(rec.tasks, list)

    def test_no_emotion_result_defaults_to_neutral(self, matcher):
        """Passing None for emotion_result should default e_score to 0.5."""
        rec = matcher.recommend(emotion_result=None, sentiment_score=0.5)
        expected_wb = matcher.compute_wellbeing_score(0.5, 0.5)
        assert rec.wellbeing_score == pytest.approx(expected_wb, abs=1e-4)


# ─────────────────────────────────────────────────────────────
# 6. Analytics helpers
# ─────────────────────────────────────────────────────────────
class TestAnalyticsHelpers:

    def test_available_categories_returns_list(self, matcher):
        cats = matcher.available_categories()
        assert isinstance(cats, list)
        assert len(cats) > 0

    def test_available_categories_sorted(self, matcher):
        cats = matcher.available_categories()
        assert cats == sorted(cats)

    def test_tier_summary_has_all_tiers(self, matcher):
        summary = matcher.tier_summary()
        for tier in ["Low", "Medium", "High"]:
            assert tier in summary

    def test_tier_summary_counts_correct(self, matcher, sample_tasks_df):
        summary = matcher.tier_summary()
        expected = sample_tasks_df["tier"].value_counts().to_dict()
        for tier, count in expected.items():
            assert summary[tier] == count
