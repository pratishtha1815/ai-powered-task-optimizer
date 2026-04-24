"""
modules/logger_alerts.py  —  MODULE 4
─────────────────────────────────────────────────────────────
Trend Logging & HR Alert System.

Responsibilities
────────────────
• Define SQLAlchemy ORM models for two tables:
    - employee_logs   → one row per employee check-in
    - hr_alerts       → one row per fired HR alert
• Persist daily emotion/sentiment snapshots to SQLite
• Calculate 7-day rolling well-being averages per employee
• Detect consecutive days below the HR alert threshold
• Fire (create) an HR Alert record automatically
• Expose query helpers used by the HR Analytics Dashboard

Database Tables
───────────────
  employee_logs
    id               INTEGER PK
    employee_id      TEXT       (unique employee identifier)
    employee_name    TEXT
    date             DATE       (YYYY-MM-DD, one row per employee per day)
    emotion          TEXT       (dominant emotion label)
    emotion_conf     REAL       (CNN confidence)
    sentiment_label  TEXT       (Positive/Neutral/Negative)
    sentiment_score  REAL       (VADER compound)
    wellbeing_score  REAL       ([0, 1])
    tier             TEXT       (task tier assigned)
    hr_alert_fired   BOOL
    notes            TEXT       (optional free-text)
    created_at       DATETIME

  hr_alerts
    id               INTEGER PK
    employee_id      TEXT
    employee_name    TEXT
    alert_date       DATE
    consecutive_days INTEGER
    avg_wellbeing    REAL
    status           TEXT       ('active' | 'dismissed' | 'resolved')
    notes            TEXT
    created_at       DATETIME

Public API
──────────
    db = WellbeingDB()                                # connects & creates tables
    db.log_entry(employee_id, employee_name,
                 emotion_result, sentiment_result,
                 recommendation)                      # save a check-in
    df = db.get_employee_trend(employee_id, days=30)  # trend DataFrame
    alerts = db.get_active_alerts()                   # list[dict]
    db.dismiss_alert(alert_id)                        # HR dismisses alert
    db.get_all_employees()                            # list[dict]
    db.employee_summary()                             # all employees' last score
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float,
    Integer, String, Text, create_engine, func, and_,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ── Project imports ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATABASE_URL, LOG_DIR,
    HR_CONSECUTIVE_DAYS, HR_THRESHOLD,
)

logger.add(LOG_DIR / "logger_alerts.log", rotation="5 MB", level="DEBUG")


# ─────────────────────────────────────────────────────────────
# ORM Models
# ─────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


class EmployeeLog(Base):
    """One row per employee check-in session."""
    __tablename__ = "employee_logs"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    employee_id      = Column(String(64),  nullable=False, index=True)
    employee_name    = Column(String(128), nullable=False)
    date             = Column(Date,        nullable=False, index=True)
    emotion          = Column(String(32),  nullable=False, default="Neutral")
    emotion_conf     = Column(Float,       nullable=False, default=0.0)
    sentiment_label  = Column(String(16),  nullable=False, default="Neutral")
    sentiment_score  = Column(Float,       nullable=False, default=0.0)
    wellbeing_score  = Column(Float,       nullable=False, default=0.5)
    tier             = Column(String(16),  nullable=False, default="Medium")
    voice_score      = Column(Float,       nullable=True)
    voice_transcript = Column(Text,        nullable=True)
    hr_alert_fired   = Column(Boolean,     nullable=False, default=False)
    notes            = Column(Text,        nullable=True)
    created_at       = Column(DateTime,    nullable=False,
                              default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "employee_id":     self.employee_id,
            "employee_name":   self.employee_name,
            "date":            str(self.date),
            "emotion":         self.emotion,
            "emotion_conf":    self.emotion_conf,
            "sentiment_label": self.sentiment_label,
            "sentiment_score": self.sentiment_score,
            "wellbeing_score": self.wellbeing_score,
            "tier":            self.tier,
            "voice_score":     self.voice_score,
            "voice_transcript":self.voice_transcript,
            "hr_alert_fired":  self.hr_alert_fired,
            "notes":           self.notes,
            "created_at":      str(self.created_at),
        }


class HRAlert(Base):
    """One row per raised HR alert."""
    __tablename__ = "hr_alerts"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    employee_id      = Column(String(64),  nullable=False, index=True)
    employee_name    = Column(String(128), nullable=False)
    alert_date       = Column(Date,        nullable=False)
    consecutive_days = Column(Integer,     nullable=False)
    avg_wellbeing    = Column(Float,       nullable=False)
    status           = Column(String(16),  nullable=False,
                              default="active")       # active|dismissed|resolved
    notes            = Column(Text,        nullable=True)
    created_at       = Column(DateTime,    nullable=False,
                              default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "employee_id":      self.employee_id,
            "employee_name":    self.employee_name,
            "alert_date":       str(self.alert_date),
            "consecutive_days": self.consecutive_days,
            "avg_wellbeing":    self.avg_wellbeing,
            "status":           self.status,
            "notes":            self.notes,
            "created_at":       str(self.created_at),
        }


class UserTask(Base):
    """Reflects a task accepted or completed by an employee."""
    __tablename__ = "user_tasks"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    employee_id      = Column(String(64),  nullable=False, index=True)
    task_id          = Column(Integer,     nullable=False)
    task_title       = Column(String(256), nullable=False)
    status           = Column(String(16),  nullable=False, default="accepted") # accepted|completed
    score_at_action  = Column(Float,       nullable=True)
    created_at       = Column(DateTime,    nullable=False, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "employee_id":     self.employee_id,
            "task_id":         self.task_id,
            "task_title":      self.task_title,
            "status":          self.status,
            "score_at_action": self.score_at_action,
            "created_at":      str(self.created_at),
        }


# ─────────────────────────────────────────────────────────────
# WellbeingDB — main class
# ─────────────────────────────────────────────────────────────
class WellbeingDB:
    """
    Thread-safe SQLite data access layer for the Task Optimizer.

    Parameters
    ----------
    db_url : str
        SQLAlchemy connection string. Defaults to config.DATABASE_URL.
        Pass 'sqlite:///:memory:' for in-memory testing.
    consecutive_days_threshold : int
        Number of consecutive bad days to trigger an HR alert.
    wellbeing_threshold : float
        Daily score below this value counts as a "bad" day.
    """

    def __init__(
        self,
        db_url: str                   = DATABASE_URL,
        consecutive_days_threshold: int = HR_CONSECUTIVE_DAYS,
        wellbeing_threshold: float      = HR_THRESHOLD,
    ):
        self.db_url               = db_url
        self.consec_days          = consecutive_days_threshold
        self.wb_threshold         = wellbeing_threshold

        self._engine  = create_engine(
            db_url,
            connect_args={"check_same_thread": False},  # safe for Streamlit
            echo=False,
        )
        self._Session = sessionmaker(bind=self._engine)
        Base.metadata.create_all(self._engine)
        logger.info(f"WellbeingDB connected → {db_url}")

    # ── Context-manager helper ────────────────────────────────
    def _session(self) -> Session:
        return self._Session()

    # ─────────────────────────────────────────────────────────
    # WRITE: Log a check-in entry
    # ─────────────────────────────────────────────────────────
    def log_entry(
        self,
        employee_id:      str,
        employee_name:    str,
        wellbeing_score:  float,
        emotion:          str          = "Neutral",
        emotion_conf:     float        = 0.0,
        sentiment_label:  str          = "Neutral",
        sentiment_score:  float        = 0.0,
        tier:             str          = "Medium",
        notes:            Optional[str] = None,
        log_date:         Optional[date] = None,
        # Also accept result objects directly (from Modules 1–3, 5)
        emotion_result    = None,
        sentiment_result  = None,
        voice_result      = None,
        recommendation    = None,
    ) -> EmployeeLog:
        """
        Save a daily employee check-in snapshot.

        You can either pass raw scalars (emotion, sentiment_label, etc.)
        OR pass the result objects from Modules 1/2/3 directly —
        the method will extract values from them automatically.

        Returns
        -------
        EmployeeLog — the persisted ORM object
        """
        # ── Extract from result objects if supplied ────────────
        if emotion_result is not None:
            emotion      = emotion_result.emotion
            emotion_conf = emotion_result.confidence

        if sentiment_result is not None:
            sentiment_label = sentiment_result.label
            sentiment_score = sentiment_result.compound

        voice_score_val = None
        voice_text      = None
        if voice_result is not None:
            voice_score_val = voice_result.wellbeing_score
            voice_text      = voice_result.transcript

        if recommendation is not None:
            tier            = recommendation.tier
            wellbeing_score = recommendation.wellbeing_score

        today = log_date or date.today()

        with self._session() as session:
            # Upsert: one log per employee per day
            existing = (
                session.query(EmployeeLog)
                .filter(
                    EmployeeLog.employee_id == employee_id,
                    EmployeeLog.date        == today,
                )
                .first()
            )

            if existing:
                # Update existing record for today
                existing.emotion         = emotion
                existing.emotion_conf    = emotion_conf
                existing.sentiment_label = sentiment_label
                existing.sentiment_score = round(sentiment_score, 4)
                existing.wellbeing_score = round(wellbeing_score, 4)
                existing.tier            = tier
                existing.voice_score      = voice_score_val
                existing.voice_transcript = voice_text
                existing.notes           = notes
                entry = existing
                logger.debug(f"Updated log for {employee_id} on {today}")
            else:
                entry = EmployeeLog(
                    employee_id      = employee_id,
                    employee_name    = employee_name,
                    date             = today,
                    emotion          = emotion,
                    emotion_conf     = round(emotion_conf, 4),
                    sentiment_label  = sentiment_label,
                    sentiment_score  = round(sentiment_score, 4),
                    wellbeing_score  = round(wellbeing_score, 4),
                    tier             = tier,
                    voice_score      = voice_score_val,
                    voice_transcript = voice_text,
                    hr_alert_fired   = False,
                    notes            = notes,
                )
                session.add(entry)
                logger.info(
                    f"Logged check-in: {employee_name} ({employee_id}) "
                    f"| date={today} | WB={wellbeing_score:.4f} "
                    f"| tier={tier}"
                )

            session.commit()
            session.refresh(entry)

            # ── Check if HR alert should fire ──────────────────
            alert_fired = self._evaluate_hr_alert(
                session, employee_id, employee_name, today
            )
            if alert_fired:
                entry.hr_alert_fired = True
                session.commit()

            return entry

    # ─────────────────────────────────────────────────────────
    # HR ALERT EVALUATION
    # ─────────────────────────────────────────────────────────
    def _evaluate_hr_alert(
        self,
        session:       Session,
        employee_id:   str,
        employee_name: str,
        check_date:    date,
    ) -> bool:
        """
        Check the last N days for consecutive below-threshold scores.
        If detected, create an HRAlert record (if not already active).

        Returns True if an alert was fired.
        """
        cutoff = check_date - timedelta(days=self.consec_days - 1)

        recent_logs = (
            session.query(EmployeeLog)
            .filter(
                EmployeeLog.employee_id == employee_id,
                EmployeeLog.date        >= cutoff,
                EmployeeLog.date        <= check_date,
            )
            .order_by(EmployeeLog.date.asc())
            .all()
        )

        if len(recent_logs) < self.consec_days:
            return False

        # All consecutive days must be below threshold
        bad_days = [l for l in recent_logs if l.wellbeing_score < self.wb_threshold]
        if len(bad_days) < self.consec_days:
            return False

        # Check no active alert already exists for this employee
        existing_alert = (
            session.query(HRAlert)
            .filter(
                HRAlert.employee_id == employee_id,
                HRAlert.status      == "active",
            )
            .first()
        )
        if existing_alert:
            logger.debug(
                f"Active alert already exists for {employee_id}. Skipping."
            )
            return False

        avg_wb = round(sum(l.wellbeing_score for l in recent_logs) / len(recent_logs), 4)
        alert  = HRAlert(
            employee_id      = employee_id,
            employee_name    = employee_name,
            alert_date       = check_date,
            consecutive_days = self.consec_days,
            avg_wellbeing    = avg_wb,
            status           = "active",
            notes            = (
                f"Auto-generated: {self.consec_days} consecutive days "
                f"with well-being score below {self.wb_threshold}. "
                f"Avg score: {avg_wb:.4f}."
            ),
        )
        session.add(alert)
        session.commit()

        logger.warning(
            f"🚨 HR ALERT FIRED for {employee_name} ({employee_id}) | "
            f"Consecutive days: {self.consec_days} | Avg WB: {avg_wb:.4f}"
        )
        return True

    # ─────────────────────────────────────────────────────────
    # READ: Employee trend data
    # ─────────────────────────────────────────────────────────
    def get_employee_trend(
        self,
        employee_id: str,
        days:        int = 30,
    ) -> pd.DataFrame:
        """
        Return up to `days` most-recent logs for an employee as a DataFrame.

        Columns: date, emotion, emotion_conf, sentiment_label,
                 sentiment_score, wellbeing_score, tier, hr_alert_fired
        """
        cutoff = date.today() - timedelta(days=days)
        with self._session() as session:
            rows = (
                session.query(EmployeeLog)
                .filter(
                    EmployeeLog.employee_id == employee_id,
                    EmployeeLog.date        >= cutoff,
                )
                .order_by(EmployeeLog.date.asc())
                .all()
            )
        if not rows:
            return pd.DataFrame()

        return pd.DataFrame([r.to_dict() for r in rows])

    def get_rolling_average(
        self,
        employee_id: str,
        window:      int = 7,
    ) -> Optional[float]:
        """
        Return the mean well-being score over the last `window` days.
        Returns None if fewer than 1 day of data exists.
        """
        df = self.get_employee_trend(employee_id, days=window)
        if df.empty:
            return None
        return round(df["wellbeing_score"].mean(), 4)

    # ─────────────────────────────────────────────────────────
    # READ: HR Alert queries
    # ─────────────────────────────────────────────────────────
    def get_active_alerts(self) -> list[dict]:
        """Return all active HR alerts (newest first)."""
        with self._session() as session:
            alerts = (
                session.query(HRAlert)
                .filter(HRAlert.status == "active")
                .order_by(HRAlert.created_at.desc())
                .all()
            )
        return [a.to_dict() for a in alerts]

    def get_all_alerts(self, limit: int = 100) -> list[dict]:
        """Return all alerts (active + dismissed + resolved), newest first."""
        with self._session() as session:
            alerts = (
                session.query(HRAlert)
                .order_by(HRAlert.created_at.desc())
                .limit(limit)
                .all()
            )
        return [a.to_dict() for a in alerts]

    def dismiss_alert(
        self,
        alert_id: int,
        notes:    Optional[str] = None,
    ) -> bool:
        """
        Mark an alert as dismissed. Returns True on success, False if not found.
        """
        with self._session() as session:
            alert = session.query(HRAlert).filter(HRAlert.id == alert_id).first()
            if not alert:
                logger.warning(f"Alert {alert_id} not found.")
                return False
            alert.status = "dismissed"
            if notes:
                alert.notes = (alert.notes or "") + f"\n[Dismissed] {notes}"
            session.commit()
            logger.info(f"Alert {alert_id} dismissed.")
        return True

    def resolve_alert(
        self,
        alert_id: int,
        notes:    Optional[str] = None,
    ) -> bool:
        """Mark an alert as resolved (employee recovered)."""
        with self._session() as session:
            alert = session.query(HRAlert).filter(HRAlert.id == alert_id).first()
            if not alert:
                return False
            alert.status = "resolved"
            if notes:
                alert.notes = (alert.notes or "") + f"\n[Resolved] {notes}"
            session.commit()
            logger.info(f"Alert {alert_id} resolved.")
        return True

    # ─────────────────────────────────────────────────────────
    # READ: Employee list & summary (for HR dashboard)
    # ─────────────────────────────────────────────────────────
    def get_all_employees(self) -> list[dict]:
        """
        Return a list of all unique employees ever logged, with their metadata.
        """
        with self._session() as session:
            rows = (
                session.query(
                    EmployeeLog.employee_id,
                    EmployeeLog.employee_name,
                    func.count(EmployeeLog.id).label("total_checkins"),
                    func.max(EmployeeLog.date).label("last_checkin"),
                    func.avg(EmployeeLog.wellbeing_score).label("avg_wellbeing"),
                )
                .group_by(EmployeeLog.employee_id)
                .order_by(EmployeeLog.employee_name.asc())
                .all()
            )
        return [
            {
                "employee_id":    r.employee_id,
                "employee_name":  r.employee_name,
                "total_checkins": r.total_checkins,
                "last_checkin":   str(r.last_checkin),
                "avg_wellbeing":  round(r.avg_wellbeing, 4),
            }
            for r in rows
        ]

    def employee_summary(self, days: int = 7) -> pd.DataFrame:
        """
        Return a DataFrame with one row per employee showing:
        employee_id, employee_name, last_score, avg_score (last N days),
        has_active_alert.
        """
        employees = self.get_all_employees()
        if not employees:
            return pd.DataFrame()

        active_ids = {a["employee_id"] for a in self.get_active_alerts()}

        rows = []
        for emp in employees:
            eid    = emp["employee_id"]
            trend  = self.get_employee_trend(eid, days=days)
            last_  = float(trend["wellbeing_score"].iloc[-1]) if not trend.empty else None
            avg_   = round(float(trend["wellbeing_score"].mean()), 4) if not trend.empty else None
            rows.append({
                "employee_id":       eid,
                "employee_name":     emp["employee_name"],
                "last_wellbeing":    last_,
                "avg_wellbeing":     avg_,
                "has_active_alert":  eid in active_ids,
                "total_checkins":    emp["total_checkins"],
            })

        return pd.DataFrame(rows)

    # ─────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────
    def clear_employee_data(self, employee_id: str) -> int:
        """Delete all logs for an employee. Returns number of rows deleted."""
        with self._session() as session:
            n = (
                session.query(EmployeeLog)
                .filter(EmployeeLog.employee_id == employee_id)
                .delete()
            )
            session.commit()
        logger.info(f"Deleted {n} logs for {employee_id}")
        return n

    def get_db_stats(self) -> dict:
        """Returns basic stats for monitoring / debugging."""
        with self._session() as session:
            n_logs    = session.query(func.count(EmployeeLog.id)).scalar()
            n_alerts  = session.query(func.count(HRAlert.id)).scalar()
            n_tasks   = session.query(func.count(UserTask.id)).scalar()
            n_active  = (
                session.query(func.count(HRAlert.id))
                .filter(HRAlert.status == "active")
                .scalar()
            )
        return {
            "total_logs":     n_logs,
            "total_alerts":   n_alerts,
            "total_tasks":    n_tasks,
            "active_alerts":  n_active,
            "db_url":         self.db_url,
        }

    # ─────────────────────────────────────────────────────────
    # WRITE: Task Acceptance
    # ─────────────────────────────────────────────────────────
    def accept_task(
        self,
        employee_id: str,
        task_id:     int,
        task_title:  str,
        score:       Optional[float] = None
    ) -> bool:
        """Log that an employee has accepted a recommended task."""
        try:
            with self._session() as session:
                entry = UserTask(
                    employee_id     = employee_id,
                    task_id         = task_id,
                    task_title      = task_title,
                    status          = "accepted",
                    score_at_action = score
                )
                session.add(entry)
                session.commit()
                logger.info(f"Task accepted: {employee_id} -> {task_title}")
                return True
        except Exception as e:
            logger.error(f"Failed to log task acceptance: {e}")
            return False

    def get_engagement_metrics(self, days: int = 30) -> dict:
        """Return total and daily counts of accepted tasks."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self._session() as session:
            total = session.query(UserTask).filter(UserTask.created_at >= cutoff).count()
            
            # Daily trends
            daily_query = (
                session.query(
                    func.date(UserTask.created_at).label("day"),
                    func.count(UserTask.id).label("count")
                )
                .filter(UserTask.created_at >= cutoff)
                .group_by("day")
                .all()
            )
            trends = {str(r.day): r.count for r in daily_query}
            
        return {
            "total_accepted": total,
            "daily_trends":   trends
        }


# ─────────────────────────────────────────────────────────────
# Quick standalone demo  (python modules/logger_alerts.py)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from datetime import date, timedelta

    print("\n" + "═" * 60)
    print("  Logger & HR Alert System — Demo (in-memory DB)")
    print("═" * 60)

    db = WellbeingDB(db_url="sqlite:///:memory:")

    # Simulate 4 days of data for two employees
    employees = [
        ("EMP001", "Alice Johnson"),
        ("EMP002", "Bob Smith"),
    ]

    for emp_id, emp_name in employees:
        scores = (
            [0.20, 0.18, 0.22, 0.19]    # Alice: 4 bad days
            if emp_id == "EMP001"
            else [0.75, 0.80, 0.72, 0.78]  # Bob: all good
        )
        for i, wb in enumerate(scores):
            log_date = date.today() - timedelta(days=3 - i)
            db.log_entry(
                employee_id=emp_id, employee_name=emp_name,
                wellbeing_score=wb, log_date=log_date,
                emotion="Stressed" if wb < 0.4 else "Happy",
                tier="Low" if wb < 0.4 else "High",
            )
            print(f"  Logged {emp_name} | {log_date} | WB={wb:.2f}")

    print("\n  ── Active HR Alerts ─────────────────────────────")
    for alert in db.get_active_alerts():
        print(
            f"  🚨 {alert['employee_name']} | "
            f"Consec={alert['consecutive_days']} | "
            f"AvgWB={alert['avg_wellbeing']:.3f}"
        )

    print("\n  ── Employee Summary ──────────────────────────────")
    print(db.employee_summary().to_string(index=False))
    print("\n  ── DB Stats ──────────────────────────────────────")
    print(db.get_db_stats())
    print("═" * 60 + "\n")
