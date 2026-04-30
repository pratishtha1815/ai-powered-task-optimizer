"""
modules/task_matcher.py  —  MODULE 3
─────────────────────────────────────────────────────────────
Task Matching Engine.

Responsibilities
────────────────
• Accept EmotionResult (Module 1) + SentimentResult (Module 2)
• Compute a combined Well-being Score (configurable weighted blend)
• Map the Well-being Score to a Cognitive Load Tier
• Load the task catalogue from data/tasks.csv
• Filter and rank tasks appropriate to the employee's current state
• Return a TaskRecommendation with the top-N matched tasks

Decision Matrix
───────────────
  Well-being Score  │  Cognitive Load Tier  │  Rationale
  ──────────────────┼───────────────────────┼──────────────────────
  0.00 – 0.35       │  LOW (1–3)            │  Severe stress/fatigue
  0.35 – 0.55       │  LOW + MEDIUM (1–5)   │  Mildly low energy
  0.55 – 0.70       │  MEDIUM (4–6)         │  Neutral / baseline
  0.70 – 0.85       │  MEDIUM + HIGH (5–8)  │  Good energy
  0.85 – 1.00       │  HIGH (7–10)          │  Peak performance

Public API
──────────
    matcher = TaskMatcher()
    rec     = matcher.recommend(emotion_result, sentiment_result, top_n=5)
        → TaskRecommendation(
              wellbeing_score = 0.612,
              tier            = "Medium",
              tasks           = [Task(...), Task(...), ...],
              explanation     = "..."
          )
"""

from __future__ import annotations

import sys
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

# ── Project imports ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_DIR, LOG_DIR,
    WEIGHT_EMOTION, WEIGHT_SENTIMENT, WEIGHT_VOICE
)

logger.add(LOG_DIR / "task_matcher.log", rotation="5 MB", level="DEBUG")

TASKS_CSV = DATA_DIR / "tasks.csv"
DEFAULT_TOP_N = 5


# ─────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────
@dataclass
class Task:
    """A single task from the catalogue."""
    id:               int
    title:            str
    description:      str
    category:         str
    cognitive_load:   int          # 1–10
    estimated_hours:  float
    tags:             list[str]
    tier:             str          # Low | Medium | High

    @classmethod
    def from_row(cls, row: pd.Series) -> "Task":
        tags = [t.strip() for t in str(row.get("tags", "")).split(",") if t.strip()]
        return cls(
            id              = int(row["id"]),
            title           = str(row["title"]),
            description     = str(row["description"]),
            category        = str(row["category"]),
            cognitive_load  = int(row["cognitive_load"]),
            estimated_hours = float(row["estimated_hours"]),
            tags            = tags,
            tier            = str(row.get("tier", "Medium")),
        )

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "title":            self.title,
            "description":      self.description,
            "category":         self.category,
            "cognitive_load":   self.cognitive_load,
            "estimated_hours":  self.estimated_hours,
            "tags":             self.tags,
            "tier":             self.tier,
        }


@dataclass
class TaskRecommendation:
    """
    Full output of one task-matching pass.

    Attributes
    ----------
    wellbeing_score : float  – combined [0, 1] score
    tier            : str    – resolved cognitive-load tier label
    tier_range      : tuple  – (min_cl, max_cl) filter applied
    tasks           : list[Task] – ordered recommendations (best first)
    explanation     : str    – human-readable rationale for HR/dashboard
    emotion_score   : float  – emotion contribution
    sentiment_score : float  – sentiment contribution
    """
    wellbeing_score: float
    tier:            str
    tier_range:      tuple[int, int]
    tasks:           list[Task]
    explanation:     str
    emotion_score:   float    = 0.0
    sentiment_score: float    = 0.0
    voice_score:     float    = 0.0

    def to_dict(self) -> dict:
        return {
            "wellbeing_score": self.wellbeing_score,
            "tier":            self.tier,
            "tier_range":      self.tier_range,
            "emotion_score":   self.emotion_score,
            "sentiment_score": self.sentiment_score,
            "voice_score":     self.voice_score,
            "explanation":     self.explanation,
            "tasks":           [t.to_dict() for t in self.tasks],
        }


# ─────────────────────────────────────────────────────────────
# Decision Matrix
# ─────────────────────────────────────────────────────────────
# Maps (label, min_wellbeing, max_wellbeing) → (tier_name, min_cl, max_cl, description)
_DECISION_MATRIX = [
    # (lower_bound, upper_bound, tier_label, min_cl, max_cl, explanation_template)
    (0.00, 0.35,
     "Low",
     1, 3,
     "Your well-being score indicates significant stress or fatigue "
     "(score: {score:.2f}). We've selected light, low-cognitive-load "
     "tasks to help you recover and regain focus."),

    (0.35, 0.55,
     "Low-Medium",
     1, 5,
     "Your well-being score suggests you're feeling a bit low on energy "
     "(score: {score:.2f}). A mix of easy and moderate tasks has been "
     "selected to keep momentum without overloading you."),

    (0.55, 0.70,
     "Medium",
     4, 6,
     "Your well-being score is in the normal range "
     "(score: {score:.2f}). Standard-complexity tasks have been matched "
     "to keep you productive and engaged."),

    (0.70, 0.85,
     "Medium-High",
     5, 8,
     "You're in a positive, energised state "
     "(score: {score:.2f}). We've included some higher-complexity tasks "
     "alongside your usual workload to make the most of your focus."),

    (0.85, 1.01,
     "High",
     7, 10,
     "Excellent well-being score detected "
     "(score: {score:.2f})! You're at peak performance — challenging, "
     "high-impact tasks have been recommended."),
]


def resolve_tier(wellbeing_score: float) -> tuple[str, int, int, str]:
    """
    Map a well-being score to a decision-matrix row.

    Returns
    -------
    (tier_label, min_cognitive_load, max_cognitive_load, explanation_template)
    """
    for lo, hi, label, min_cl, max_cl, template in _DECISION_MATRIX:
        if lo <= wellbeing_score < hi:
            return label, min_cl, max_cl, template
    # Fallback (score exactly 1.0)
    *_, last = _DECISION_MATRIX
    return last[2], last[3], last[4], last[5]


# ─────────────────────────────────────────────────────────────
# TaskMatcher — main class
# ─────────────────────────────────────────────────────────────
class TaskMatcher:
    """
    Combines emotion and sentiment signals into ranked task recommendations.

    Parameters
    ----------
    tasks_csv : Path | str
        Path to the task catalogue CSV. Defaults to config.DATA_DIR/tasks.csv.
    weight_emotion : float
        Weight for the emotion-derived well-being score. Default 0.60.
    weight_sentiment : float
        Weight for the sentiment well-being score. Default 0.40.
    shuffle_ties : bool
        If True, tasks with identical cognitive_load are randomly shuffled
        (adds variety across sessions). Default True.
    """

    def __init__(
        self,
        tasks_csv:        Path | str = TASKS_CSV,
        weight_emotion:   float      = WEIGHT_EMOTION,
        weight_sentiment: float      = WEIGHT_SENTIMENT,
        weight_voice:     float      = WEIGHT_VOICE,
        shuffle_ties:     bool       = True,
    ):
        self.tasks_csv        = Path(tasks_csv)
        self.weight_emotion   = weight_emotion
        self.weight_sentiment = weight_sentiment
        self.weight_voice     = weight_voice
        self.shuffle_ties     = shuffle_ties
        self._df: Optional[pd.DataFrame] = None     # lazy-loaded

    # ── Task catalogue loading ────────────────────────────────
    def _load_tasks(self) -> pd.DataFrame:
        """Load (and cache) the task CSV. Auto-seeds if missing."""
        if self._df is not None:
            return self._df

        if not self.tasks_csv.exists():
            logger.warning(
                f"tasks.csv not found at {self.tasks_csv}. "
                "Running seed_tasks automatically…"
            )
            from scripts.seed_tasks import seed_tasks
            seed_tasks(self.tasks_csv)

        self._df = pd.read_csv(self.tasks_csv)
        logger.info(f"Loaded {len(self._df)} tasks from {self.tasks_csv}")
        return self._df

    def reload_tasks(self) -> None:
        """Force reload of the CSV (useful after seeding or manual edits)."""
        self._df = None
        self._load_tasks()

    # ── Combined score calculation ─────────────────────────────
    def compute_wellbeing_score(
        self,
        emotion_score:   float,
        sentiment_score: float,
        voice_score:     float = 0.5,
    ) -> float:
        """
        Weighted blend of emotion, sentiment, and voice well-being scores.

        Parameters
        ----------
        emotion_score   : float [0, 1] — from EmotionResult.wellbeing_score()
        sentiment_score : float [0, 1] — from SentimentResult.wellbeing_score
        voice_score     : float [0, 1] — from VoiceResult.wellbeing_score

        Returns
        -------
        float [0, 1] combined well-being score (rounded to 4dp)
        """
        score = (
            self.weight_emotion   * emotion_score
            + self.weight_sentiment * sentiment_score
            + self.weight_voice     * voice_score
        )
        return round(min(max(score, 0.0), 1.0), 4)

    # ── Task filtering and ranking ─────────────────────────────
    def _filter_tasks(
        self,
        df: pd.DataFrame,
        min_cl: int,
        max_cl: int,
    ) -> list[Task]:
        """
        Filter tasks by cognitive load range and convert to Task objects.
        Tasks are sorted by cognitive_load ascending so lightest tasks
        appear first within a tier.
        """
        mask     = (df["cognitive_load"] >= min_cl) & (df["cognitive_load"] <= max_cl)
        filtered = df[mask].copy()

        if filtered.empty:
            logger.warning(
                f"No tasks found for cognitive_load in [{min_cl}, {max_cl}]. "
                "Returning all Low tasks as fallback."
            )
            filtered = df[df["cognitive_load"] <= 3].copy()

        # Sort: ascending cognitive_load, then shuffle ties if enabled
        if self.shuffle_ties:
            # Stable sort on load; shuffle within each load group
            groups = []
            for _, grp in filtered.groupby("cognitive_load"):
                grp = grp.sample(frac=1)    # shuffle within group
                groups.append(grp)
            filtered = pd.concat(groups).reset_index(drop=True)
        else:
            filtered = filtered.sort_values("cognitive_load").reset_index(drop=True)

        return [Task.from_row(row) for _, row in filtered.iterrows()]

    # ── Public API ────────────────────────────────────────────
    def recommend(
        self,
        emotion_result=None,
        sentiment_result=None,
        voice_result=None,
        emotion_score:   Optional[float] = None,
        sentiment_score: Optional[float] = None,
        voice_score:     Optional[float] = None,
        top_n: int = DEFAULT_TOP_N,
    ) -> TaskRecommendation:
        """
        Full pipeline: emotion + sentiment + voice → TaskRecommendation.

        Accepts either the result objects directly (from Modules 1 & 2)
        or raw float scores (for testing / manual overrides).

        Parameters
        ----------
        emotion_result   : EmotionResult from Module 1 (or None)
        sentiment_result : SentimentResult from Module 2 (or None)
        emotion_score    : float override [0,1] (used if emotion_result is None)
        sentiment_score  : float override [0,1] (used if sentiment_result is None)
        top_n            : number of tasks to return

        Returns
        -------
        TaskRecommendation
        """
        # ── Resolve scores ─────────────────────────────────────
        e_score = (
            emotion_score if emotion_score is not None
            else (emotion_result.wellbeing_score() if emotion_result else 0.5)
        )
        s_score = (
            sentiment_score if sentiment_score is not None
            else (sentiment_result.wellbeing_score if sentiment_result else 0.5)
        )
        v_score = (
            voice_score if voice_score is not None
            else (voice_result.wellbeing_score if voice_result else 0.5)
        )

        combined = self.compute_wellbeing_score(e_score, s_score, v_score)
        tier_label, min_cl, max_cl, template = resolve_tier(combined)
        explanation = template.format(score=combined)

        logger.info(
            f"Well-being: {combined:.4f} | Tier: {tier_label} "
            f"| CL range: [{min_cl}, {max_cl}]"
        )

        # ── Filter tasks ───────────────────────────────────────
        df    = self._load_tasks()
        tasks = self._filter_tasks(df, min_cl, max_cl)
        top   = tasks[:top_n]

        logger.debug(
            f"Returning {len(top)}/{len(tasks)} tasks for tier '{tier_label}'"
        )

        return TaskRecommendation(
            wellbeing_score = combined,
            tier            = tier_label,
            tier_range      = (min_cl, max_cl),
            tasks           = top,
            explanation     = explanation,
            emotion_score   = round(e_score, 4),
            sentiment_score = round(s_score, 4),
            voice_score     = round(v_score, 4),
        )

    def recommend_from_scores(
        self,
        emotion_score:   float,
        sentiment_score: float,
        voice_score:     float = 0.5,
        top_n:           int = DEFAULT_TOP_N,
    ) -> TaskRecommendation:
        """
        Convenience shorthand — pass raw floats directly.
        Useful for the Streamlit dashboard and testing.
        """
        return self.recommend(
            emotion_score=emotion_score,
            sentiment_score=sentiment_score,
            voice_score=voice_score,
            top_n=top_n,
        )

    # ── Analytics helpers (used by the HR Dashboard) ──────────
    def available_categories(self) -> list[str]:
        """Return the sorted list of unique task categories."""
        df = self._load_tasks()
        return sorted(df["category"].unique().tolist())

    def tier_summary(self) -> dict[str, int]:
        """Return count of tasks per cognitive-load tier."""
        df = self._load_tasks()
        return df["tier"].value_counts().to_dict()


# ─────────────────────────────────────────────────────────────
# Quick standalone demo  (python modules/task_matcher.py)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from scripts.seed_tasks import seed_tasks
    if not TASKS_CSV.exists():
        seed_tasks()

    matcher = TaskMatcher()

    scenarios = [
        ("Severe stress",      0.15, 0.10),
        ("Mildly low energy",  0.40, 0.40),
        ("Neutral / baseline", 0.60, 0.65),
        ("Good energy",        0.78, 0.80),
        ("Peak performance",   0.95, 0.90),
    ]

    print("\n" + "═" * 70)
    print("  Task Matching Engine — Demo")
    print("═" * 70)

    for label, e_score, s_score in scenarios:
        rec = matcher.recommend_from_scores(e_score, s_score, top_n=3)
        print(f"\n  ● {label:<22}  WB={rec.wellbeing_score:.3f}  Tier={rec.tier}")
        for t in rec.tasks:
            print(f"    [{t.cognitive_load:2d}] {t.title} ({t.estimated_hours}h)")

    print("\n" + "═" * 70 + "\n")
