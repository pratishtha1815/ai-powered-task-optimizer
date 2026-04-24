"""
pages/employee_view.py  —  DASHBOARD PAGE 1
─────────────────────────────────────────────────────────────
Employee check-in view.

Layout (left → right)
──────────────────────
Left column  (40%) : Webcam capture + Emotion result card
Right column (60%) : Daily check-in form + Wellbeing gauge +
                     Task recommendations
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EMOTION_LABELS
from modules.sentiment_analyzer import SentimentAnalyzer
from modules.voice_analyzer import VoiceAnalyzer, VoiceResult
from modules.task_matcher import TaskMatcher
from modules.logger_alerts import WellbeingDB
from scripts.seed_tasks import seed_tasks, OUTPUT_PATH as TASKS_PATH
from streamlit_mic_recorder import mic_recorder


# ─────────────────────────────────────────────────────────────
# Lazy singleton helpers (cached across reruns)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_sentiment_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer(backend="vader")


@st.cache_resource(show_spinner=False)
def get_voice_analyzer() -> VoiceAnalyzer:
    return VoiceAnalyzer()


@st.cache_resource(show_spinner=False)
def get_task_matcher() -> TaskMatcher:
    if not TASKS_PATH.exists():
        seed_tasks()
    return TaskMatcher()


@st.cache_resource(show_spinner=False)
def get_db() -> WellbeingDB:
    return WellbeingDB()


@st.cache_resource
def get_emotion_detector():
    """
    Try to load the real EmotionDetector; fall back gracefully if the
    CNN model has not been trained yet (common during development).
    """
    try:
        from modules.emotion_detector import EmotionDetector
        return EmotionDetector()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# UI Helpers
# ─────────────────────────────────────────────────────────────
_EMOTION_COLOURS = {
    "Happy":    ("#3fb950", "😄"),
    "Neutral":  ("#8b949e", "😐"),
    "Surprise": ("#d29922", "😲"),
    "Sad":      ("#58a6ff", "😢"),
    "Fear":     ("#bc8cff", "😨"),
    "Angry":    ("#f85149", "😠"),
    "Disgust":  ("#e3b341", "🤢"),
    "No Face":  ("#484f58", "👁️"),
    "Demo":     ("#58a6ff", "🎭"),
}

_TIER_COLOURS = {
    "Low":         ("#3fb950", "🟢"),
    "Low-Medium":  ("#58a6ff", "🔵"),
    "Medium":      ("#d29922", "🟡"),
    "Medium-High": ("#f0883e", "🟠"),
    "High":        ("#f85149", "🔴"),
}

_SENTIMENT_COLOURS = {
    "Positive": ("#3fb950", "✅"),
    "Neutral":  ("#8b949e", "➖"),
    "Negative": ("#f85149", "⚠️"),
}


def _wellbeing_colour(score: float) -> str:
    if score >= 0.70: return "#3fb950"
    if score >= 0.45: return "#d29922"
    return "#f85149"


def _render_emotion_card(emotion: str, confidence: float) -> None:
    colour, icon = _EMOTION_COLOURS.get(emotion, ("#8b949e", "❓"))
    st.markdown(f"""
    <div class="metric-card" style="text-align:center; padding: 30px 20px !important;">
        <div style="font-size:3.5rem; margin-bottom:12px; filter: drop-shadow(0 0 10px {colour}44);">{icon}</div>
        <div style="font-size:1.8rem; font-weight:800; color:{colour}; letter-spacing:-0.01em;">
            {emotion}
        </div>
        <div style="font-size:0.9rem; color:var(--text-muted); margin-top:8px;">
            Confidence: <b style="color:{colour}; font-weight:700;">{confidence:.0%}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_wellbeing_gauge(score: float, emotion_wb: float, sentiment_wb: float, voice_wb: float) -> None:
    colour    = _wellbeing_colour(score)
    pct       = int(score * 100)
    bar_width = int(score * 100)

    from config import WEIGHT_EMOTION, WEIGHT_SENTIMENT, WEIGHT_VOICE
    w_e = int(WEIGHT_EMOTION * 100)
    w_s = int(WEIGHT_SENTIMENT * 100)
    w_v = int(WEIGHT_VOICE * 100)

    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:0.85rem; color:var(--text-muted); text-transform:uppercase; 
                    font-weight:600; letter-spacing:0.1em; margin-bottom:12px; text-align:center;">
            Optimization Score
        </div>
        <div style="text-align:center;">
            <span style="font-size:4.5rem; font-weight:800; color:{colour}; line-height:1;
                         text-shadow: 0 0 20px {colour}22;">{pct}</span>
            <span style="font-size:1.2rem; color:var(--text-muted); font-weight:500;">/100</span>
        </div>
        
        <div style="margin: 20px 0 10px;">
            <div style="background:rgba(255,255,255,0.05); border-radius:999px; height:12px; overflow:hidden; border:1px solid rgba(255,255,255,0.05);">
                <div style="background: linear-gradient(90deg, {colour}88, {colour}); 
                            width:{bar_width}%; height:100%; border-radius:999px; 
                            box-shadow: 0 0 15px {colour}44;"></div>
            </div>
        </div>
        
        <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:var(--text-muted); margin-bottom:24px; padding:0 4px;">
            <span style="color:{'#f85149' if score < 0.35 else 'var(--text-muted)'};">Overload</span>
            <span style="color:{'#d29922' if 0.35 <= score < 0.65 else 'var(--text-muted)'};">Balanced</span>
            <span style="color:{'#3fb950' if score >= 0.65 else 'var(--text-muted)'};">Peak Flow</span>
        </div>

        <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:12px; padding-top:15px; border-top: 1px solid var(--glass-border);">
            <div style="text-align:center;">
                <div style="color:var(--text-muted); font-size:0.75rem; font-weight:600; margin-bottom:4px;">FACE</div>
                <div style="color:{colour}; font-size:1.1rem; font-weight:700;">{emotion_wb:.2f}</div>
                <div style="font-size:0.6rem; color:var(--text-muted);">weight {w_e}%</div>
            </div>
            <div style="text-align:center;">
                <div style="color:var(--text-muted); font-size:0.75rem; font-weight:600; margin-bottom:4px;">TEXT</div>
                <div style="color:{colour}; font-size:1.1rem; font-weight:700;">{sentiment_wb:.2f}</div>
                <div style="font-size:0.6rem; color:var(--text-muted);">weight {w_s}%</div>
            </div>
            <div style="text-align:center;">
                <div style="color:var(--text-muted); font-size:0.75rem; font-weight:600; margin-bottom:4px;">VOICE</div>
                <div style="color:{colour}; font-size:1.1rem; font-weight:700;">{voice_wb:.2f}</div>
                <div style="font-size:0.6rem; color:var(--text-muted);">weight {w_v}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_task_card(task, rank: int, employee_id: str, wb_score: float) -> None:
    tier_colour, tier_icon = _TIER_COLOURS.get(
        "Low" if task.cognitive_load <= 3
        else "Medium" if task.cognitive_load <= 6
        else "High",
        ("#8b949e", "⚪")
    )
    hours_label = f"{task.estimated_hours:.1f}h"
    tags_html   = " ".join(
        f'<span style="background:rgba(255,255,255,0.05); border:1px solid var(--glass-border); '
        f'border-radius:4px; padding:3px 8px; font-size:0.7rem; '
        f'color:var(--text-muted); font-weight:500;">{t}</span>'
        for t in task.tags[:3]
    )
    
    st.markdown(f"""
    <div class="task-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
            <div style="display:flex; gap:12px; align-items:center;">
                <div style="background:var(--accent-blue)22; color:var(--accent-blue); 
                            width:24px; height:24px; border-radius:50%; 
                            display:flex; align-items:center; justify-content:center;
                            font-size:0.75rem; font-weight:800; border:1px solid var(--accent-blue)44;">
                    {rank}
                </div>
                <div style="font-weight:700; font-size:1.1rem; color:var(--text-primary);">{task.title}</div>
            </div>
            <div style="display:flex; gap:10px; align-items:center;">
                <span style="font-size:0.8rem; color:var(--text-muted); font-weight:500;">⏱ {hours_label}</span>
                <span style="background:{tier_colour}22; color:{tier_colour}; 
                             border:1px solid {tier_colour}44; 
                             border-radius:6px; padding:4px 10px; 
                             font-size:0.8rem; font-weight:700; letter-spacing:0.02em;">
                    CL {task.cognitive_load}
                </span>
            </div>
        </div>
        <div style="font-size:0.92rem; color:var(--text-muted); margin-bottom:16px; line-height:1.5; padding-left:36px;">
            {task.description}
        </div>
        <div style="margin-left:36px; display:flex; flex-wrap:wrap; gap:8px; align-items:center;">
            <span style="font-size:0.8rem; color:var(--text-muted); font-weight:600; margin-right:4px;">
                📁 {task.category}
            </span>
            {tags_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Interactive Accept Button ──
    btn_key = f"accept_{employee_id}_{task.id}_{rank}"
    if btn_key not in st.session_state:
        st.session_state[btn_key] = False

    col_btn, col_spacer = st.columns([1, 2])
    with col_btn:
        if not st.session_state[btn_key]:
            # Use custom CSS class for the button
            st.markdown('<div class="accept-button">', unsafe_allow_html=True)
            if st.button(f"✅ Accept Task", key=f"btn_{btn_key}"):
                db = get_db()
                if db.accept_task(employee_id, task.id, task.title, wb_score):
                    st.session_state[btn_key] = True
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.success("Accepted!")


# ─────────────────────────────────────────────────────────────
# Demo mode — simulate emotion from slider when model not ready
# ─────────────────────────────────────────────────────────────
def _demo_emotion_result(selected_emotion: str, confidence: float):
    """Return a mock EmotionResult-like object for demo mode."""
    class _MockResult:
        def __init__(self, emotion, conf):
            self.emotion    = emotion
            self.confidence = conf
        def wellbeing_score(self):
            weights = {
                "Happy":0.95,"Neutral":0.65,"Surprise":0.55,
                "Sad":0.35,"Fear":0.25,"Angry":0.15,"Disgust":0.10,
            }
            base = weights.get(self.emotion, 0.5)
            return round(base * self.confidence + 0.5*(1-self.confidence), 4)
    return _MockResult(selected_emotion, confidence)


# ─────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────
def render_employee_view() -> None:

    st.markdown("""
    <div style="margin-bottom:24px;">
        <h1 style="font-size:1.8rem; font-weight:700;
                   margin:0; color:#e6edf3;">
            👤 Employee Check-in
        </h1>
        <p style="color:#8b949e; margin:4px 0 0; font-size:0.9rem;">
            Daily well-being assessment and task recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Theme colors for Plotly
    _accent_blue   = "#58a6ff"
    _accent_cyan   = "#3fb950"
    _accent_purple = "#bc8cff"

    # ── Employee identity (sidebar-injected state) ────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown('<div style="font-size:0.8rem; color:#8b949e; '
                    'margin-bottom:6px;">EMPLOYEE PROFILE</div>',
                    unsafe_allow_html=True)
        emp_name = st.text_input("Your Name", value="", placeholder="e.g. Alice Johnson",
                                 key="emp_name")
        emp_id   = st.text_input("Employee ID", value="", placeholder="e.g. EMP001",
                                 key="emp_id")

    if not emp_name or not emp_id:
        st.info("👋 Please enter your **Name** and **Employee ID** in the sidebar to begin.")
        return

    # ── Layout: two columns ───────────────────────────────────
    col_cam, col_main = st.columns([4, 6], gap="large")

    # ══════════════════════════════════════════════════════════
    # LEFT COLUMN — Webcam / Emotion Detection
    # ══════════════════════════════════════════════════════════
    with col_cam:
        st.markdown('<div class="section-header">📷 Emotion Detection</div>',
                    unsafe_allow_html=True)

        detector    = get_emotion_detector()
        model_ready = detector is not None

        if model_ready:
            st.caption("📸 Capture your photo for real-time emotion analysis.")
            cam_on = st.toggle("📸 Enable Camera", key="cam_toggle")
            
            if cam_on:
                img_file = st.camera_input(
                    "Take a snapshot", key="cam_snap", label_visibility="collapsed"
                )
            else:
                img_file = None
            if img_file is not None:
                try:
                    pil_img   = Image.open(img_file).convert("RGB")
                    frame_bgr = np.array(pil_img)[:, :, ::-1]   # RGB→BGR
                    result    = detector.predict_from_frame(frame_bgr)
                    e_score   = result.wellbeing_score()
                    st.session_state["emotion_result"] = result
                    st.session_state["emotion_score"]  = e_score
                except Exception as e:
                    st.error(f"⚠️ Emotion analysis failed: {e}")
                    result  = None
                    e_score = 0.50
            else:
                result  = st.session_state.get("emotion_result", None)
                e_score = st.session_state.get("emotion_score", 0.50)

            if result:
                _render_emotion_card(result.emotion, result.confidence)

                if result.face_found and result.all_scores:
                    with st.expander("📊 All emotion probabilities", expanded=False):
                        for label, prob in sorted(
                            result.all_scores.items(), key=lambda x: -x[1]
                        ):
                            colour, _ = _EMOTION_COLOURS.get(label, ("#8b949e", ""))
                            st.markdown(
                                f'<div style="display:flex; justify-content:space-between; '
                                f'font-size:0.82rem; margin-bottom:6px;">'
                                f'<span style="color:#e6edf3;">{label}</span>'
                                f'<span style="color:{colour}; font-weight:600;">'
                                f'{prob:.1%}</span></div>',
                                unsafe_allow_html=True,
                            )
                            st.progress(prob)
            else:
                st.info("📸 Take a snapshot above to detect your emotion.")
                e_score = 0.50

        else:
            # ── Demo mode ─────────────────────────────────────
            st.markdown("""
            <div style="background:rgba(210,153,34,0.1); border:1px solid
                        rgba(210,153,34,0.3); border-radius:10px;
                        padding:14px 16px; margin-bottom:16px;
                        font-size:0.82rem; color:#d29922;">
                ⚠️ <b>Demo Mode</b> — Emotion Engine not ready.<br>
                Manually select your current emotion below.<br>
                <span style="color:#8b949e;">
                    (Ensure <code>deepface</code> and <code>tf-keras</code> are installed in your environment.)
                </span>
            </div>
            """, unsafe_allow_html=True)

            demo_emotion = st.selectbox(
                "How are you feeling?", EMOTION_LABELS,
                index=EMOTION_LABELS.index("Neutral"),
                key="demo_emotion",
            )
            demo_conf = st.slider(
                "Confidence level", 0.40, 1.00, 0.80, 0.05,
                key="demo_conf",
            )
            result  = _demo_emotion_result(demo_emotion, demo_conf)
            e_score = result.wellbeing_score()
            _render_emotion_card(result.emotion, result.confidence)
            st.session_state["emotion_result"] = result
            st.session_state["emotion_score"]  = e_score

        # ── Today's date ──────────────────────────────────────
        st.markdown(f"""
        <div class="metric-card" style="text-align:center; padding:14px;">
            <div style="font-size:0.75rem; color:#8b949e;">Check-in Date</div>
            <div style="font-size:1.05rem; font-weight:600;
                        color:#58a6ff; margin-top:4px;">
                📅 {date.today().strftime("%A, %d %B %Y")}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # RIGHT COLUMN — Text Check-in + Results + Tasks
    # ══════════════════════════════════════════════════════════
    with col_main:

        # ── Daily check-in text ───────────────────────────────
        st.markdown('<div class="section-header">✍️ Daily Check-in</div>',
                    unsafe_allow_html=True)
        checkin_text = st.text_area(
            "How are you feeling today? Describe your mood, energy level, "
            "or any challenges you're facing.",
            height=120,
            placeholder="e.g. I feel a bit overwhelmed with today's deadlines. "
                        "The project review meeting went well but I'm quite tired...",
            key="checkin_text",
        )

        st.markdown('<div class="section-header" style="margin-top:20px;">🌈 Manual Mood Calibration</div>', unsafe_allow_html=True)
        mood_map = {
            "Very Fatigued": 0.15,
            "Low Energy":    0.35,
            "Neutral":       0.60,
            "Good":          0.85,
            "Vibrant":       1.00
        }
        manual_mood = st.select_slider(
            "If the AI misreads you, override your mood here:",
            options=list(mood_map.keys()),
            value="Neutral",
            key="manual_mood"
        )
        manual_score = mood_map[manual_mood]

        st.markdown('<div style="margin:10px 0 16px;"><b>OR Speak your check-in:</b></div>', unsafe_allow_html=True)
        audio = mic_recorder(
            start_prompt="🎙️ Start Recording",
            stop_prompt="🛑 Stop & Process",
            key="voice_recorder",
        )

        if audio:
            st.audio(audio['bytes'])
            st.session_state["voice_bytes"] = audio['bytes']

        analyze_col, _ = st.columns([2, 3])
        with analyze_col:
            analyze_btn = st.button(
                "🔍 Analyse & Get Tasks", use_container_width=True,
                key="analyze_btn"
            )

        if analyze_btn or "last_recommendation" in st.session_state:
            if analyze_btn:
                # ── Fresh analysis ────────────────────────────
                with st.spinner("🧠 Analysing your multi-modal check-in…"):
                    analyzer  = get_sentiment_analyzer()
                    v_analyzer = get_voice_analyzer()
                    matcher   = get_task_matcher()
                    db        = get_db()

                    e_result = st.session_state.get("emotion_result", None)
                    e_score  = st.session_state.get("emotion_score", 0.60)
                    
                    # Blend with manual score (20% weight if changed from Neutral)
                    if st.session_state.get("manual_mood") != "Neutral":
                        e_score = (e_score * 0.8) + (manual_score * 0.2)

                    s_result = analyzer.analyze(checkin_text) if checkin_text.strip() else None
                    s_score  = s_result.wellbeing_score if s_result else 0.50

                    # NEW: Voice Analysis
                    v_bytes = st.session_state.get("voice_bytes", None)
                    v_result = v_analyzer.analyze_audio(v_bytes) if v_bytes else None
                    v_score  = v_result.wellbeing_score if v_result else 0.50

                    rec = matcher.recommend(
                        emotion_score=e_score,
                        sentiment_score=s_score,
                        voice_score=v_score,
                        top_n=5,
                    )
                    st.session_state["last_recommendation"] = rec
                    st.session_state["last_sentiment"]      = s_result
                    st.session_state["last_voice"]          = v_result
                    st.session_state["last_e_score"]        = e_score
                    st.session_state["last_s_score"]        = s_score
                    st.session_state["last_v_score"]        = v_score

                    # ── Persist to DB ──────────────────────────
                    db.log_entry(
                        employee_id      = emp_id,
                        employee_name    = emp_name,
                        wellbeing_score  = rec.wellbeing_score,
                        emotion          = e_result.emotion if e_result else "Neutral",
                        emotion_conf     = e_result.confidence if e_result else 0.0,
                        sentiment_label  = s_result.label if s_result else "Neutral",
                        sentiment_score  = s_result.compound if s_result else 0.0,
                        tier             = rec.tier,
                        voice_result     = v_result,
                        notes            = f"[Text: {checkin_text[:200]}]",
                    )

            rec      = st.session_state["last_recommendation"]
            s_result = st.session_state["last_sentiment"]
            v_result = st.session_state["last_voice"]
            e_score  = st.session_state["last_e_score"]
            s_score  = st.session_state["last_s_score"]
            v_score  = st.session_state["last_v_score"]

            # ── Sentiment result ──────────────────────────────
            st.markdown('<div class="section-header">💬 Sentiment Analysis</div>',
                        unsafe_allow_html=True)

            if s_result:
                sc, sl, sk = st.columns(3)
                sent_colour, sent_icon = _SENTIMENT_COLOURS.get(
                    s_result.label, ("#8b949e", "➖")
                )
                with sc:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center; padding:20px !important;">
                        <div style="font-size:0.8rem; color:var(--text-muted); font-weight:600; text-transform:uppercase;">Tone</div>
                        <div style="color:{sent_colour}; font-size:1.3rem; font-weight:800; margin-top:8px;">
                            {sent_icon} {s_result.label}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with sl:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center; padding:20px !important;">
                        <div style="font-size:0.8rem; color:var(--text-muted); font-weight:600; text-transform:uppercase;">Score</div>
                        <div style="color:var(--accent-blue); font-size:1.3rem; font-weight:800; margin-top:8px;">
                            {s_result.compound:+.3f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with sk:
                    kw_text = ", ".join(s_result.keywords[:3]) if s_result.keywords else "none"
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center; padding:20px !important;">
                        <div style="font-size:0.8rem; color:var(--text-muted); font-weight:600; text-transform:uppercase;">Context</div>
                        <div style="color:var(--text-muted); font-size:0.9rem; margin-top:8px; font-weight:500;">
                            {kw_text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("_No text entered — sentiment defaulted to Neutral (0.50)._")

            # ── Voice result ──────────────────────────────────
            if v_result:
                st.markdown('<div class="section-header">🎙️ Voice Analysis</div>',
                            unsafe_allow_html=True)
                vc, vt, vv = st.columns(3)
                with vc:
                    v_sent = v_result.sentiment.label if v_result.sentiment else "Neutral"
                    v_col, v_icon = _SENTIMENT_COLOURS.get(v_sent, ("#8b949e", "➖"))
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center; padding:20px !important;">
                        <div style="font-size:0.8rem; color:var(--text-muted); font-weight:600; text-transform:uppercase;">Vocal Tone</div>
                        <div style="color:{v_col}; font-size:1.2rem; font-weight:800; margin-top:8px;">
                            {v_icon} {v_sent}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with vt:
                    v_vit    = v_result.vitality_score
                    v_status = "Vibrant" if v_vit > 0.7 else "Balanced" if v_vit > 0.4 else "Fatigued"
                    v_s_col  = "#3fb950" if v_vit > 0.7 else "#d29922" if v_vit > 0.4 else "#f85149"
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center; padding:20px !important;">
                        <div style="font-size:0.8rem; color:var(--text-muted); font-weight:600; text-transform:uppercase;">Energy</div>
                        <div style="color:var(--accent-purple); font-size:1.3rem; font-weight:800; margin:8px 0 4px;">
                            {v_vit:.2f}
                        </div>
                        <div style="font-size:0.7rem; color:{v_s_col}; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">
                            {v_status}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with vv:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center; padding:20px !important;">
                        <div style="font-size:0.8rem; color:var(--text-muted); font-weight:600; text-transform:uppercase;">Speech Info</div>
                        <div style="color:var(--text-muted); font-size:0.78rem; margin-top:8px; font-style:italic;">
                            "{v_result.transcript[:35]}..."
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Well-being gauge ──────────────────────────────
            st.markdown('<div class="section-header">⚡ Overall Well-being Score</div>',
                        unsafe_allow_html=True)
            _render_wellbeing_gauge(rec.wellbeing_score, e_score, s_score, v_score)

            # ── Tier explanation ──────────────────────────────
            tier_colour, tier_icon = _TIER_COLOURS.get(rec.tier, ("#8b949e", "⚪"))
            st.markdown(f"""
            <div style="background:rgba(88,166,255,0.06);
                        border:1px solid rgba(88,166,255,0.15);
                        border-radius:10px; padding:14px 18px;
                        font-size:0.88rem; color:#8b949e; margin:12px 0;">
                <span style="font-size:1rem;">{tier_icon}</span>
                <b style="color:{tier_colour};"> {rec.tier} Load Tasks</b>
                &nbsp;—&nbsp;{rec.explanation}
            </div>
            """, unsafe_allow_html=True)

            # ── Task recommendations ──────────────────────────
            st.markdown(
                f'<div class="section-header">📋 Recommended Tasks '
                f'<span style="font-size:0.8rem; color:#8b949e; font-weight:400;">'
                f'(Top {len(rec.tasks)})</span></div>',
                unsafe_allow_html=True,
            )

            if rec.tasks:
                for i, task in enumerate(rec.tasks, start=1):
                    _render_task_card(task, i, emp_id, rec.wellbeing_score)

                # ── Log success ───────────────────────────────
                st.markdown("""
                <div style="display:flex; align-items:center; gap:8px;
                            font-size:0.8rem; color:#3fb950; margin-top:14px;">
                    ✅ Check-in saved to your wellness log.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No tasks found for your current tier. "
                           "Please run `python scripts/seed_tasks.py` first.")

        else:
            # ── Personal Dashboard (Before Check-in) ──────────
            db = get_db()
            history_df = db.get_employee_trend(emp_id, days=7)
            
            st.markdown(f"""
            <div style="background:var(--bg-card); border:1px solid #30363d;
                        border-radius:12px; padding:30px; margin-top:8px; 
                        border-left: 4px solid var(--accent-blue);">
                <div style="font-size:1.4rem; font-weight:700; color:var(--accent-blue);">
                    Welcome back, {emp_name}! 👋
                </div>
                <div style="font-size:0.9rem; color:var(--text-muted); margin-top:4px;">
                    Enter your details above to start today's multi-modal check-in.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if not history_df.empty:
                st.markdown('<div class="section-header" style="margin-top:25px;">📉 Your Weekly Trend</div>', unsafe_allow_html=True)
                
                # Simple Plotly Chart
                import plotly.express as px
                fig = px.line(
                    history_df, x="date", y="wellbeing_score",
                    markers=True,
                    template="plotly_dark",
                    color_discrete_sequence=[_accent_blue]
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=200,
                    xaxis_title=None,
                    yaxis_title="Well-being",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                c1, c2 = st.columns(2)
                last_entry = history_df.iloc[-1]
                avg_score = history_df["wellbeing_score"].mean()
                
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.75rem; color:var(--text-muted);">Last Score</div>
                        <div style="font-size:1.2rem; font-weight:700; color:var(--accent-cyan);">{last_entry['wellbeing_score']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.75rem; color:var(--text-muted);">7-Day Average</div>
                        <div style="font-size:1.2rem; font-weight:700; color:var(--accent-purple);">{avg_score:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02); border:1px dashed #30363d;
                            border-radius:12px; padding:40px; text-align:center;
                            color:#8b949e; margin-top:20px;">
                    <div style="font-size:2.5rem; margin-bottom:12px;">📊</div>
                    <div style="font-size:1rem; font-weight:600; color:#e6edf3;">
                        First Check-in?
                    </div>
                    <div style="font-size:0.85rem; margin-top:8px;">
                        Use the camera, text, and voice tools on the left to start building your well-being history.
                    </div>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    render_employee_view()
