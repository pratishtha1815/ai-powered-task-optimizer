"""
pages/hr_dashboard.py  —  DASHBOARD PAGE 2
─────────────────────────────────────────────────────────────
HR / Admin Analytics Dashboard.

Sections
────────
1. KPI row            — total employees, avg well-being, active alerts
2. Active alerts panel — animated alert cards with dismiss button
3. Employee selector  — dropdown + date range picker
4. Well-being trend   — Plotly line chart with rolling average
5. Emotion breakdown  — pie chart of emotion distribution
6. Employee summary   — sortable data table + CSV download
"""

from __future__ import annotations

import sys
import io
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.logger_alerts import WellbeingDB
from scripts.seed_tasks import seed_tasks, OUTPUT_PATH as TASKS_PATH


# ─────────────────────────────────────────────────────────────
# Shared DB singleton
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_db() -> WellbeingDB:
    return WellbeingDB()


# ─────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────
_CHART_LAYOUT = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(family="Outfit", color="#9ca3af", size=12),
    margin        = dict(l=10, r=10, t=40, b=10),
    legend        = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    xaxis         = dict(showgrid=False, zeroline=False),
    yaxis         = dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
)


def _build_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Well-being line chart with 7-day rolling average band."""
    fig = go.Figure()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["rolling"] = df["wellbeing_score"].rolling(window=7, min_periods=1).mean()

    # Threshold reference line
    fig.add_hline(
        y=0.35, line_dash="dot", line_color="#f85149",
        annotation_text="HR Alert Threshold",
        annotation_font_color="#f85149",
        annotation_position="bottom right",
    )

    # Neutral baseline
    fig.add_hline(
        y=0.50, line_dash="dot", line_color="#8b949e", line_width=0.8,
    )

    # Shaded rolling average band
    if len(df) > 1:
        fig.add_trace(go.Scatter(
            x    = pd.concat([df["date"], df["date"].iloc[::-1]]),
            y    = pd.concat([
                df["rolling"] + 0.08,
                (df["rolling"] - 0.08).iloc[::-1],
            ]),
            fill      = "toself",
            fillcolor = "rgba(88,166,255,0.07)",
            line      = dict(color="rgba(0,0,0,0)"),
            name      = "Rolling band",
            showlegend = False,
            hoverinfo  = "skip",
        ))

    # Daily scores
    colours = [
        "#3fb950" if s >= 0.70 else "#d29922" if s >= 0.45 else "#f85149"
        for s in df["wellbeing_score"]
    ]
    fig.add_trace(go.Scatter(
        x          = df["date"],
        y          = df["wellbeing_score"],
        mode       = "lines+markers",
        name       = "Daily Score",
        line       = dict(color="#58a6ff", width=2),
        marker     = dict(color=colours, size=8, line=dict(color="#0d1117", width=1.5)),
        hovertemplate = "%{x|%d %b}<br>Score: <b>%{y:.3f}</b><extra></extra>",
    ))

    # 7-day rolling average
    fig.add_trace(go.Scatter(
        x          = df["date"],
        y          = df["rolling"],
        mode       = "lines",
        name       = "7-day Avg",
        line       = dict(color="#bc8cff", width=2, dash="dash"),
        hovertemplate = "%{x|%d %b}<br>7d Avg: <b>%{y:.3f}</b><extra></extra>",
    ))

    fig.update_xaxes(
        gridcolor="#21262d", tickformat="%d %b", tickangle=-30,
        showline=False,
    )
    fig.update_yaxes(
        gridcolor="#21262d", range=[0, 1],
        tickformat=".2f",
    )
    fig.update_layout(
        title   = "Well-being Score Trend",
        height  = 320,
        **_CHART_LAYOUT,
    )
    return fig


def _build_emotion_pie(df: pd.DataFrame) -> go.Figure:
    """Donut chart of emotion distribution."""
    if "emotion" not in df.columns or df.empty:
        return go.Figure()

    counts = df["emotion"].value_counts().reset_index()
    counts.columns = ["emotion", "count"]

    palette = {
        "Happy":   "#3fb950", "Neutral": "#8b949e",
        "Surprise":"#d29922", "Sad":     "#58a6ff",
        "Fear":    "#bc8cff", "Angry":   "#f85149",
        "Disgust": "#e3b341", "No Face": "#484f58",
    }
    colours = [palette.get(e, "#8b949e") for e in counts["emotion"]]

    fig = go.Figure(go.Pie(
        labels      = counts["emotion"],
        values      = counts["count"],
        hole        = 0.60,
        marker      = dict(colors=colours,
                           line=dict(color="#0d1117", width=2)),
        textfont    = dict(color="#e6edf3", size=12),
        hovertemplate = "%{label}: <b>%{value}</b> days (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title  = "Emotion Distribution",
        height = 280,
        **_CHART_LAYOUT,
    )
    return fig


def _build_sentiment_bar(df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart: sentiment over time (Text vs Voice)."""
    if "sentiment_label" not in df.columns or df.empty:
        return go.Figure()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    palette = {"Positive": "#3fb950", "Neutral": "#8b949e", "Negative": "#f85149"}
    fig = go.Figure()

    # Text Sentiment
    for label, colour in palette.items():
        sub = df[df["sentiment_label"] == label]
        fig.add_trace(go.Bar(
            x     = sub["date"],
            y     = [0.5] * len(sub),
            name  = f"Text: {label}",
            marker_color = colour,
            offsetgroup = 0,
            hovertemplate = f"Text {label}: %{{x|%d %b}}<extra></extra>",
        ))

    # Voice sentiment fallback/mock (if voice_score exists)
    if "voice_score" in df.columns:
        for idx, row in df.iterrows():
            if pd.notnull(row["voice_score"]):
                v_label = "Positive" if row["voice_score"] >= 0.6 else "Negative" if row["voice_score"] <= 0.4 else "Neutral"
                fig.add_trace(go.Bar(
                    x = [row["date"]],
                    y = [0.5],
                    name = f"Voice: {v_label}",
                    marker_color = palette[v_label],
                    offsetgroup = 1,
                    showlegend = False,
                    hovertemplate = f"Voice {v_label}: %{{x|%d %b}}<br>Vitality: {row.get('voice_score', 0):.2f}<extra></extra>",
                ))

    fig.update_layout(
        title        = "Daily Sentiment Trends (Top: Text, Bottom: Voice)",
        barmode      = "stack",
        height       = 240,
        bargap       = 0.3,
        showlegend    = False,
        **_CHART_LAYOUT,
    )
    fig.update_xaxes(tickformat="%d %b", showgrid=False)
    fig.update_yaxes(visible=False)
    return fig


# ─────────────────────────────────────────────────────────────
# Alert card renderer
# ─────────────────────────────────────────────────────────────
def _render_alert_card(alert: dict, db: WellbeingDB) -> None:
    severity_pct = int((1 - alert["avg_wellbeing"]) * 100)
    col_info, col_btn = st.columns([5, 1])

    with col_info:
        st.markdown(f"""
        <div class="alert-card">
            <div style="display:flex; justify-content:space-between;
                        align-items:flex-start; flex-wrap:wrap; gap:8px;">
                <div>
                    <span style="font-size:1.2rem;">🚨</span>
                    <b style="color:#f85149; font-size:1rem; margin-left:6px;">
                        {alert["employee_name"]}
                    </b>
                    <span style="color:#8b949e; font-size:0.8rem; margin-left:8px;">
                        ID: {alert["employee_id"]}
                    </span>
                </div>
                <span style="background:rgba(248,81,73,0.2);
                             border:1px solid rgba(248,81,73,0.4);
                             border-radius:4px; padding:2px 10px;
                             font-size:0.75rem; color:#f85149; font-weight:600;">
                    ACTIVE ALERT
                </span>
            </div>
            <div style="margin-top:12px; display:flex; gap:24px;
                        font-size:0.85rem; flex-wrap:wrap;">
                <div>
                    <span style="color:#8b949e;">Consecutive bad days:</span>
                    <b style="color:#f85149; margin-left:6px;">
                        {alert["consecutive_days"]}
                    </b>
                </div>
                <div>
                    <span style="color:#8b949e;">Avg well-being:</span>
                    <b style="color:#f85149; margin-left:6px;">
                        {alert["avg_wellbeing"]:.3f}
                    </b>
                </div>
                <div>
                    <span style="color:#8b949e;">Severity:</span>
                    <b style="color:#f85149; margin-left:6px;">{severity_pct}%</b>
                </div>
                <div>
                    <span style="color:#8b949e;">Flagged on:</span>
                    <span style="color:#e6edf3; margin-left:6px;">{alert["alert_date"]}</span>
                </div>
            </div>
            <div style="margin-top:10px; font-size:0.8rem;
                        color:#8b949e; font-style:italic;">
                {(alert.get("notes") or "")[:120]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_btn:
        if st.button("✓ Dismiss", key=f"dismiss_{alert['id']}",
                     use_container_width=True):
            db.dismiss_alert(alert["id"], notes="Dismissed via HR Dashboard.")
            st.success("Alert dismissed.")
            st.rerun()


# ─────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────
def render_hr_dashboard() -> None:

    db = get_db()

    # ── Page header ──────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h1 style="font-size:1.8rem; font-weight:700;
                   margin:0; color:#e6edf3;">
            📊 HR Analytics Dashboard
        </h1>
        <p style="color:#8b949e; margin:4px 0 0; font-size:0.9rem;">
            Employee well-being trends, alerts, and insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Seed demo data button (sidebar) ──────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown('<div style="font-size:0.8rem; color:#8b949e; '
                    'margin-bottom:6px;">DEMO CONTROLS</div>',
                    unsafe_allow_html=True)
        if st.button("🌱 Seed Demo Data", use_container_width=True,
                     key="seed_demo_btn"):
            _seed_demo_data(db)
            st.success("Demo data loaded!")
            st.rerun()

        if st.button("🔄 Refresh Dashboard", use_container_width=True,
                     key="refresh_btn"):
            st.rerun()

    # ── KPI Row ───────────────────────────────────────────────
    employees     = db.get_all_employees()
    active_alerts = db.get_active_alerts()
    summary_df    = db.employee_summary(days=7)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        n_emp = len(employees)
        _render_kpi_card("Active Employees", f"{n_emp:02d}", icon="👥")
    with k2:
        avg_wb = summary_df["avg_wellbeing"].mean() * 100 if not summary_df.empty else 0
        _render_kpi_card("System Wellness", f"{avg_wb:.0f}%", icon="🧠")
    with k3:
        n_alerts = len(active_alerts)
        _render_kpi_card("Policy Alerts", f"{n_alerts:02d}", icon="🚨")
    with k4:
        engagement = db.get_engagement_metrics(30)["total_accepted"]
        _render_kpi_card("Tasks Accepted", f"{engagement:02d}", icon="✅")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Active Alerts Panel ───────────────────────────────────
    if active_alerts:
        st.markdown(
            f'<div class="section-header" style="color:#f85149;">'
            f'🚨 Active HR Alerts '
            f'<span style="background:#f8514922; border:1px solid #f8514944; '
            f'border-radius:999px; padding:2px 10px; font-size:0.75rem;">'
            f'{len(active_alerts)}</span></div>',
            unsafe_allow_html=True,
        )
        for alert in active_alerts:
            _render_alert_card(alert, db)
        st.markdown("---")
    else:
        st.markdown("""
        <div style="background:rgba(63,185,80,0.08); border:1px solid rgba(63,185,80,0.2);
                    border-radius:10px; padding:14px 20px; margin-bottom:16px;
                    font-size:0.9rem; color:#3fb950;">
            ✅ No active HR alerts — all employees are within healthy well-being ranges.
        </div>
        """, unsafe_allow_html=True)

    # ── Employee selector ─────────────────────────────────────
    if not employees:
        st.info(
            "📭 No employee data yet. Ask employees to complete their daily check-in, "
            "or click **Seed Demo Data** in the sidebar to load sample records."
        )
        return

    st.markdown('<div class="section-header">📈 Individual Trend Analysis</div>',
                unsafe_allow_html=True)

    sel_col, days_col = st.columns([3, 1])
    with sel_col:
        emp_options = {
            f"{e['employee_name']} ({e['employee_id']})": e["employee_id"]
            for e in employees
        }
        selected_label = st.selectbox(
            "Select Employee", list(emp_options.keys()), key="emp_selector"
        )
        selected_id = emp_options[selected_label]

    with days_col:
        window = st.selectbox(
            "Period", ["7 days", "14 days", "30 days", "90 days"],
            index=2, key="period_selector"
        )
        days = int(window.split()[0])

    trend_df = db.get_employee_trend(selected_id, days=days)

    if trend_df.empty:
        st.warning(f"No data for **{selected_label}** in the last {days} days.")
    else:
        # ── Trend chart ───────────────────────────────────────
        st.plotly_chart(
            _build_trend_chart(trend_df),
            use_container_width=True,
        )

        # ── Sub-charts row ────────────────────────────────────
        pie_col, bar_col = st.columns([2, 3])
        with pie_col:
            st.plotly_chart(
                _build_emotion_pie(trend_df),
                use_container_width=True,
            )
        with bar_col:
            st.plotly_chart(
                _build_sentiment_bar(trend_df),
                use_container_width=True,
            )

        # ── Recent log table ──────────────────────────────────
        st.markdown(
            '<div class="section-header">📋 Recent Check-ins & Multimodal Insights</div>',
            unsafe_allow_html=True,
        )
        display_cols = [
            "date", "emotion", "sentiment_label", "voice_score",
            "voice_transcript", "wellbeing_score", "tier",
        ]
        display_df = trend_df[[c for c in display_cols if c in trend_df.columns]].copy()
        display_df = display_df.sort_values("date", ascending=False).head(14)
        display_df.columns = [
            c.replace("_", " ").title() for c in display_df.columns
        ]
        st.dataframe(
            display_df,
            use_container_width=True,
            height=300,
            hide_index=True,
        )

        # ── Download button ───────────────────────────────────
        csv_buffer = io.StringIO()
        trend_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label     = "⬇️ Download Full CSV",
            data      = csv_buffer.getvalue(),
            file_name = f"{selected_id}_wellbeing_{date.today()}.csv",
            mime      = "text/csv",
            key       = "download_csv",
        )

    st.markdown("---")

    # ── Task Engagement Metrics ───────────────────────────────
    st.markdown('<div class="section-header">✅ Task Engagement Metrics</div>', unsafe_allow_html=True)
    st.plotly_chart(_build_engagement_chart(), use_container_width=True)
    st.markdown("---")

    # ── All Employees Summary Table ───────────────────────────
    st.markdown('<div class="section-header">👥 All Employees Overview</div>',
                unsafe_allow_html=True)

    if not summary_df.empty:
        # Color-code well-being
        def _highlight(val):
            if isinstance(val, float):
                if val >= 0.70: return "color: #3fb950"
                if val >= 0.45: return "color: #d29922"
                return "color: #f85149"
            return ""

        st.dataframe(
            summary_df.style.applymap(
                _highlight,
                subset=["last_wellbeing", "avg_wellbeing"]
            ),
            use_container_width=True,
            height=280,
            hide_index=True,
        )

        # Download all employees
        csv_all = io.StringIO()
        summary_df.to_csv(csv_all, index=False)
        st.download_button(
            label     = "⬇️ Download All Employees CSV",
            data      = csv_all.getvalue(),
            file_name = f"all_employees_summary_{date.today()}.csv",
            mime      = "text/csv",
            key       = "download_all_csv",
        )

    # ── DB Statistics footer ──────────────────────────────────
    st.markdown("---")
    stats = db.get_db_stats()
    st.markdown(f"""
    <div style="display:flex; gap:32px; font-size:0.8rem; color:#8b949e;
                padding:8px 0; flex-wrap:wrap;">
        <span>🗄️ DB: <b style="color:#e6edf3;">{stats['db_url']}</b></span>
        <span>📝 Total logs: <b style="color:#e6edf3;">{stats['total_logs']}</b></span>
        <span>🚨 Total alerts: <b style="color:#e6edf3;">{stats['total_alerts']}</b></span>
        <span>📅 Report date: <b style="color:#e6edf3;">{date.today()}</b></span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _render_kpi_card(label: str, value: str, delta: str = "", icon: str = "📈") -> None:
    st.markdown(f"""
    <div class="metric-card" style="padding: 24px !important;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:0.85rem; color:var(--text-muted); font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">{label}</div>
                <div style="font-size:2.2rem; font-weight:800; color:var(--text-primary); margin-top:4px;">{value}</div>
                {f'<div style="font-size:0.8rem; color:var(--accent-green); margin-top:4px;">{delta}</div>' if delta else ''}
            </div>
            <div style="font-size:2rem; opacity:0.8;">{icon}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _build_engagement_chart() -> go.Figure:
    """Build a gauge or bar chart for task acceptance engagement."""
    db = get_db()
    stats = db.get_engagement_metrics(7)
    trends = stats["daily_trends"]
    
    if not trends:
        return go.Figure().update_layout(title="No Engagement Data Yet")
        
    dates = list(trends.keys())
    counts = list(trends.values())
    
    fig = go.Figure(go.Scatter(
        x=dates, y=counts,
        fill='tozeroy',
        mode='lines+markers',
        line=dict(color= "#10b981", width=3),
        marker=dict(size=8, color="#10b981", line=dict(color="#0a0a0c", width=2)),
        fillcolor="rgba(16, 185, 129, 0.1)"
    ))
    fig.update_layout(
        title="Daily Task Acceptance Engagement",
        height=280,
        **_CHART_LAYOUT
    )
    return fig


def _seed_demo_data(db: WellbeingDB) -> None:
    """Insert synthetic 30-day data for demo purposes."""
    import random, math
    random.seed(42)

    demo_employees = [
        ("EMP001", "Alice Johnson",  "burnout"),    # severe stress
        ("EMP002", "Bob Smith",      "recovering"), # improving trend
        ("EMP003", "Carol Williams", "healthy"),    # consistently good
        ("EMP004", "David Chen",     "mixed"),      # variable
    ]

    emotions_by_profile = {
        "burnout":    ["Angry", "Sad", "Fear", "Stressed", "Sad"],
        "recovering": ["Sad", "Neutral", "Neutral", "Happy", "Happy"],
        "healthy":    ["Happy", "Happy", "Neutral", "Happy", "Surprise"],
        "mixed":      ["Happy", "Sad", "Neutral", "Angry", "Happy"],
    }
    sentiment_map = {
        0.0: ("Negative", -0.60),
        1.0: ("Positive",  0.60),
        0.5: ("Neutral",   0.00),
    }

    for emp_id, name, profile in demo_employees:
        for day_offset in range(30, -1, -1):
            log_date = date.today() - timedelta(days=day_offset)
            t        = day_offset / 30.0

            # Base score trajectory per profile
            if profile == "burnout":
                base = 0.20 + 0.05 * math.sin(day_offset)
            elif profile == "recovering":
                base = 0.25 + (1 - t) * 0.55
            elif profile == "healthy":
                base = 0.72 + 0.08 * math.sin(day_offset * 0.5)
            else:
                base = 0.50 + 0.30 * math.sin(day_offset * 0.8)

            wb        = max(0.05, min(0.99, base + random.gauss(0, 0.05)))
            emotion   = random.choice(emotions_by_profile[profile])
            sent_label = "Positive" if wb > 0.6 else ("Negative" if wb < 0.4 else "Neutral")
            sent_score = 0.5 if sent_label == "Positive" else (-0.5 if sent_label == "Negative" else 0.0)
            tier       = "Low" if wb < 0.35 else ("Medium" if wb < 0.70 else "High")

            db.log_entry(
                employee_id     = emp_id,
                employee_name   = name,
                wellbeing_score = round(wb, 4),
                emotion         = emotion,
                emotion_conf    = round(random.uniform(0.55, 0.95), 2),
                sentiment_label = sent_label,
                sentiment_score = round(sent_score + random.gauss(0, 0.1), 3),
                tier            = tier,
                log_date        = log_date,
            )


def _render_alert_card(alert: dict, db: WellbeingDB) -> None:
    """Render an action card for a system alert."""
    a_id = alert["id"]
    st.markdown(f"""
    <div style="background:rgba(248,81,73,0.08); border:1px solid rgba(248,81,73,0.25);
                border-radius:12px; padding:18px 22px; margin-bottom:16px;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <div style="font-size:0.75rem; color:#f85149; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Critical Policy Violation
                </div>
                <div style="font-size:1.1rem; font-weight:700; color:var(--text-primary);">
                    {alert['employee_name']} ({alert['employee_id']})
                </div>
                <div style="font-size:0.9rem; color:var(--text-muted); margin-top:4px;">
                    Condition: <b>{alert['condition']}</b> (Score: {alert['wellbeing_score']:.2f})
                </div>
            </div>
            <div style="font-size:0.8rem; color:#8b949e; text-align:right;">
                {alert['timestamp'][:16]}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Action Button
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("Dismiss", key=f"dismiss_{a_id}", use_container_width=True):
            db.dismiss_alert(a_id)
            st.rerun()


def _wellbeing_colour(score: float) -> str:
    """Map score to semantic CSS variable."""
    if score >= 0.70: return "var(--accent-green)"
    if score >= 0.40: return "var(--accent-blue)"
    return "var(--accent-red)"


def _highlight(val):
    """Style helper for the employee summary table."""
    try:
        val = float(val)
        if val < 0.35: return "color: #f85149; font-weight: bold;"
        if val > 0.75: return "color: #3fb950; font-weight: bold;"
    except:
        pass
    return ""
