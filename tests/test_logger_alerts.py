"""
tests/test_logger_alerts.py
─────────────────────────────────────────────────────────────
Unit tests for Module 4 — WellbeingDB (Logger & HR Alerts).

All tests use an in-memory SQLite database so no files are
created or left behind after the test run.

Run with:
    pytest tests/test_logger_alerts.py -v
"""

import sys
from datetime import date, timedelta
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.logger_alerts import WellbeingDB, EmployeeLog, HRAlert


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def db() -> WellbeingDB:
    """Fresh in-memory DB for every test."""
    return WellbeingDB(
        db_url="sqlite:///:memory:",
        consecutive_days_threshold=3,
        wellbeing_threshold=0.35,
    )


@pytest.fixture
def db_with_alice(db) -> WellbeingDB:
    """DB pre-populated with 5 days of good data for Alice."""
    for i in range(5):
        d = date.today() - timedelta(days=4 - i)
        db.log_entry(
            employee_id="EMP001", employee_name="Alice Johnson",
            wellbeing_score=0.75, log_date=d,
            emotion="Happy", tier="High",
        )
    return db


@pytest.fixture
def db_with_stressed_bob(db) -> WellbeingDB:
    """DB with 3 consecutive bad days for Bob (should trigger alert)."""
    for i in range(3):
        d = date.today() - timedelta(days=2 - i)
        db.log_entry(
            employee_id="EMP002", employee_name="Bob Smith",
            wellbeing_score=0.20, log_date=d,
            emotion="Stressed", tier="Low",
        )
    return db


# ─────────────────────────────────────────────────────────────
# 1. DB Initialization
# ─────────────────────────────────────────────────────────────
class TestInit:

    def test_db_creates_successfully(self, db):
        assert db is not None

    def test_db_stats_empty_on_init(self, db):
        stats = db.get_db_stats()
        assert stats["total_logs"]    == 0
        assert stats["total_alerts"]  == 0
        assert stats["active_alerts"] == 0

    def test_get_all_employees_empty_on_init(self, db):
        assert db.get_all_employees() == []

    def test_get_active_alerts_empty_on_init(self, db):
        assert db.get_active_alerts() == []


# ─────────────────────────────────────────────────────────────
# 2. log_entry — basic persistence
# ─────────────────────────────────────────────────────────────
class TestLogEntry:

    def test_log_entry_returns_employee_log(self, db):
        entry = db.log_entry(
            employee_id="EMP001", employee_name="Alice",
            wellbeing_score=0.75,
        )
        assert isinstance(entry, EmployeeLog)

    def test_log_entry_persisted(self, db):
        db.log_entry(
            employee_id="EMP001", employee_name="Alice",
            wellbeing_score=0.75,
        )
        assert db.get_db_stats()["total_logs"] == 1

    def test_log_entry_fields_correct(self, db):
        d = date.today() - timedelta(days=10)
        db.log_entry(
            employee_id="EMP001", employee_name="Alice Johnson",
            wellbeing_score=0.80, log_date=d,
            emotion="Happy", emotion_conf=0.92,
            sentiment_label="Positive", sentiment_score=0.65,
            tier="High",
        )
        df = db.get_employee_trend("EMP001", days=200)
        assert not df.empty
        row = df.iloc[0]
        assert row["employee_id"]     == "EMP001"
        assert row["employee_name"]   == "Alice Johnson"
        assert row["emotion"]         == "Happy"
        assert row["sentiment_label"] == "Positive"
        assert float(row["wellbeing_score"]) == pytest.approx(0.80, abs=1e-4)
        assert row["tier"]            == "High"

    def test_upsert_same_employee_same_day(self, db):
        """Logging twice on the same day should update, not add a new row."""
        d = date.today()
        db.log_entry(
            employee_id="EMP001", employee_name="Alice",
            wellbeing_score=0.60, log_date=d,
        )
        db.log_entry(
            employee_id="EMP001", employee_name="Alice",
            wellbeing_score=0.80, log_date=d,   # update
        )
        assert db.get_db_stats()["total_logs"] == 1
        df = db.get_employee_trend("EMP001", days=2)
        assert float(df.iloc[-1]["wellbeing_score"]) == pytest.approx(0.80)

    def test_multiple_employees_separate_rows(self, db):
        db.log_entry("EMP001", "Alice", wellbeing_score=0.75)
        db.log_entry("EMP002", "Bob",   wellbeing_score=0.40)
        assert db.get_db_stats()["total_logs"] == 2

    def test_log_entry_accepts_result_objects(self, db):
        """Passing mock result objects should not raise."""
        from unittest.mock import MagicMock
        e_result = MagicMock()
        e_result.emotion    = "Happy"
        e_result.confidence = 0.90

        s_result = MagicMock()
        s_result.label    = "Positive"
        s_result.compound = 0.55

        rec = MagicMock()
        rec.tier            = "High"
        rec.wellbeing_score = 0.78

        entry = db.log_entry(
            employee_id="EMP001", employee_name="Alice",
            wellbeing_score=0.0,   # will be overridden by rec
            emotion_result=e_result,
            sentiment_result=s_result,
            recommendation=rec,
        )
        assert entry.emotion   == "Happy"
        assert entry.tier      == "High"
        assert float(entry.wellbeing_score) == pytest.approx(0.78, abs=1e-4)


# ─────────────────────────────────────────────────────────────
# 3. get_employee_trend
# ─────────────────────────────────────────────────────────────
class TestEmployeeTrend:

    def test_trend_returns_dataframe(self, db_with_alice):
        df = db_with_alice.get_employee_trend("EMP001", days=30)
        assert isinstance(df, pd.DataFrame)

    def test_trend_row_count(self, db_with_alice):
        df = db_with_alice.get_employee_trend("EMP001", days=30)
        assert len(df) == 5

    def test_trend_sorted_ascending_date(self, db_with_alice):
        df = db_with_alice.get_employee_trend("EMP001", days=30)
        dates = pd.to_datetime(df["date"])
        assert list(dates) == sorted(dates)

    def test_trend_unknown_employee_returns_empty(self, db):
        df = db.get_employee_trend("UNKNOWN", days=30)
        assert df.empty

    def test_trend_respects_days_window(self, db):
        """Logs older than `days` should not appear."""
        old_date = date.today() - timedelta(days=40)
        db.log_entry("EMP001", "Alice", wellbeing_score=0.5, log_date=old_date)
        df = db.get_employee_trend("EMP001", days=30)
        assert df.empty


# ─────────────────────────────────────────────────────────────
# 4. Rolling average
# ─────────────────────────────────────────────────────────────
class TestRollingAverage:

    def test_rolling_avg_correct(self, db):
        scores = [0.60, 0.70, 0.80]
        for i, s in enumerate(scores):
            d = date.today() - timedelta(days=2 - i)
            db.log_entry("EMP001", "Alice", wellbeing_score=s, log_date=d)
        avg = db.get_rolling_average("EMP001", window=7)
        assert avg == pytest.approx(sum(scores) / len(scores), abs=1e-3)

    def test_rolling_avg_none_for_unknown_employee(self, db):
        assert db.get_rolling_average("UNKNOWN") is None

    def test_rolling_avg_single_entry(self, db):
        db.log_entry("EMP001", "Alice", wellbeing_score=0.65)
        avg = db.get_rolling_average("EMP001", window=7)
        assert avg == pytest.approx(0.65, abs=1e-4)


# ─────────────────────────────────────────────────────────────
# 5. HR Alert firing
# ─────────────────────────────────────────────────────────────
class TestHRAlerts:

    def test_alert_fires_after_consecutive_bad_days(self, db_with_stressed_bob):
        alerts = db_with_stressed_bob.get_active_alerts()
        assert len(alerts) == 1
        assert alerts[0]["employee_id"] == "EMP002"

    def test_alert_not_fired_for_good_employee(self, db_with_alice):
        assert db_with_alice.get_active_alerts() == []

    def test_alert_not_fired_below_required_consecutive_days(self, db):
        """Only 2 bad days when threshold is 3 — should NOT fire."""
        for i in range(2):
            d = date.today() - timedelta(days=1 - i)
            db.log_entry("EMP003", "Carol", wellbeing_score=0.20, log_date=d)
        assert db.get_active_alerts() == []

    def test_alert_not_fired_twice_for_same_employee(self, db_with_stressed_bob):
        """Second bad day batch should not create a duplicate active alert."""
        # Add another bad day for Bob
        db_with_stressed_bob.log_entry(
            "EMP002", "Bob Smith", wellbeing_score=0.18,
            log_date=date.today() + timedelta(days=1),
        )
        alerts = db_with_stressed_bob.get_active_alerts()
        assert len(alerts) == 1   # still only one active alert

    def test_alert_employee_name_correct(self, db_with_stressed_bob):
        alert = db_with_stressed_bob.get_active_alerts()[0]
        assert alert["employee_name"] == "Bob Smith"

    def test_alert_consecutive_days_field(self, db_with_stressed_bob):
        alert = db_with_stressed_bob.get_active_alerts()[0]
        assert alert["consecutive_days"] == 3

    def test_alert_avg_wellbeing_below_threshold(self, db_with_stressed_bob):
        alert = db_with_stressed_bob.get_active_alerts()[0]
        assert alert["avg_wellbeing"] < 0.35

    def test_alert_status_is_active(self, db_with_stressed_bob):
        alert = db_with_stressed_bob.get_active_alerts()[0]
        assert alert["status"] == "active"

    def test_good_day_after_bad_days_does_not_re_fire(self, db):
        """2 bad + 1 good: under threshold count, no alert."""
        for i in range(2):
            d = date.today() - timedelta(days=2 - i)
            db.log_entry("EMP004", "Dave", wellbeing_score=0.20, log_date=d)
        # Good day
        db.log_entry("EMP004", "Dave", wellbeing_score=0.90)
        assert db.get_active_alerts() == []


# ─────────────────────────────────────────────────────────────
# 6. Dismiss / Resolve alert
# ─────────────────────────────────────────────────────────────
class TestAlertManagement:

    def test_dismiss_alert_changes_status(self, db_with_stressed_bob):
        alert_id = db_with_stressed_bob.get_active_alerts()[0]["id"]
        db_with_stressed_bob.dismiss_alert(alert_id, notes="HR reviewed.")
        all_alerts = db_with_stressed_bob.get_all_alerts()
        assert all_alerts[0]["status"] == "dismissed"

    def test_dismiss_alert_removes_from_active(self, db_with_stressed_bob):
        alert_id = db_with_stressed_bob.get_active_alerts()[0]["id"]
        db_with_stressed_bob.dismiss_alert(alert_id)
        assert db_with_stressed_bob.get_active_alerts() == []

    def test_resolve_alert_changes_status(self, db_with_stressed_bob):
        alert_id = db_with_stressed_bob.get_active_alerts()[0]["id"]
        db_with_stressed_bob.resolve_alert(alert_id, notes="Employee recovered.")
        all_alerts = db_with_stressed_bob.get_all_alerts()
        assert all_alerts[0]["status"] == "resolved"

    def test_dismiss_nonexistent_alert_returns_false(self, db):
        assert db.dismiss_alert(9999) is False

    def test_resolve_nonexistent_alert_returns_false(self, db):
        assert db.resolve_alert(9999) is False


# ─────────────────────────────────────────────────────────────
# 7. Employee list & summary helpers
# ─────────────────────────────────────────────────────────────
class TestEmployeeHelpers:

    def test_get_all_employees_count(self, db):
        db.log_entry("EMP001", "Alice", wellbeing_score=0.75)
        db.log_entry("EMP002", "Bob",   wellbeing_score=0.60)
        emps = db.get_all_employees()
        assert len(emps) == 2

    def test_get_all_employees_fields(self, db):
        db.log_entry("EMP001", "Alice", wellbeing_score=0.75)
        emp = db.get_all_employees()[0]
        for key in ["employee_id", "employee_name", "total_checkins",
                    "last_checkin", "avg_wellbeing"]:
            assert key in emp

    def test_employee_summary_returns_dataframe(self, db_with_alice):
        df = db_with_alice.employee_summary()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_employee_summary_columns(self, db_with_alice):
        df = db_with_alice.employee_summary()
        for col in ["employee_id", "employee_name", "last_wellbeing",
                    "avg_wellbeing", "has_active_alert"]:
            assert col in df.columns

    def test_employee_summary_no_alert_flag(self, db_with_alice):
        df = db_with_alice.employee_summary()
        assert not df.iloc[0]["has_active_alert"]

    def test_employee_summary_with_alert_flag(self, db_with_stressed_bob):
        df = db_with_stressed_bob.employee_summary()
        row = df[df["employee_id"] == "EMP002"].iloc[0]
        assert row["has_active_alert"]

    def test_employee_summary_empty_db(self, db):
        df = db.employee_summary()
        assert df.empty

    def test_clear_employee_data(self, db_with_alice):
        n = db_with_alice.clear_employee_data("EMP001")
        assert n == 5
        assert db_with_alice.get_employee_trend("EMP001").empty


# ─────────────────────────────────────────────────────────────
# 8. DB stats
# ─────────────────────────────────────────────────────────────
class TestDBStats:

    def test_stats_after_logging(self, db_with_stressed_bob):
        stats = db_with_stressed_bob.get_db_stats()
        assert stats["total_logs"]   == 3
        assert stats["total_alerts"] == 1
        assert stats["active_alerts"] == 1

    def test_stats_after_dismiss(self, db_with_stressed_bob):
        alert_id = db_with_stressed_bob.get_active_alerts()[0]["id"]
        db_with_stressed_bob.dismiss_alert(alert_id)
        stats = db_with_stressed_bob.get_db_stats()
        assert stats["active_alerts"] == 0
        assert stats["total_alerts"]  == 1  # still exists, just not active
