"""
app.py — Application Entry Point
─────────────────────────────────────────────────────────────
Main Streamlit application. Renders the top-level navigation
and routes between the Employee View and HR/Admin Dashboard.

Run with:
    streamlit run app.py
"""

import streamlit as st

# ── Page config (must be FIRST Streamlit call) ────────────────
st.set_page_config(
    page_title="AI-Powered Task Optimizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* ── Root Variables (Premium Obsidian Theme) ── */
:root {
    --bg-primary:    #0a0a0c;
    --bg-secondary:  #121214;
    --glass-bg:      rgba(25, 25, 30, 0.7);
    --glass-border:  rgba(255, 255, 255, 0.08);
    --accent-blue:   #3b82f6;
    --accent-glow:   rgba(59, 130, 246, 0.4);
    --accent-green:  #10b981;
    --accent-red:    #ef4444;
    --accent-purple: #8b5cf6;
    --text-primary:  #f3f4f6;
    --text-muted:    #9ca3af;
    --radius:        16px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Glassmorphic Cards ── */
.metric-card, .task-card, [data-testid="stMetric"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius) !important;
    padding: 20px !important;
    margin-bottom: 16px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.metric-card:hover, .task-card:hover {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 20px var(--accent-glow) !important;
    transform: translateY(-2px) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--glass-border) !important;
}

/* ── Custom Buttons (Premium) ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
}

/* ── Secondary/Accept Button ── */
.accept-button > button {
    background: rgba(16, 185, 129, 0.1) !important;
    color: var(--accent-green) !important;
    border: 1px solid var(--accent-green) !important;
    margin-top: 8px !important;
}
.accept-button > button:hover {
    background: var(--accent-green) !important;
    color: #000 !important;
}

/* ── Section Headers ── */
.section-header {
    font-size: 1.25rem;
    font-weight: 600;
    margin: 32px 0 16px;
    padding-left: 12px;
    border-left: 4px solid var(--accent-blue);
    display: flex;
    align-items: center;
    gap: 12px;
}

/* ── Modern Sliders & Inputs ── */
.stSlider [data-baseweb="slider"] { margin-bottom: 25px; }
.stProgress > div > div { background-color: var(--accent-blue) !important; }

/* ── Button Glow & Pulse ── */
@keyframes pulse-glow {
    0% { box-shadow: 0 0 5px var(--accent-blue)44; }
    50% { box-shadow: 0 0 20px var(--accent-blue)88; }
    100% { box-shadow: 0 0 5px var(--accent-blue)44; }
}
.stButton > button {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button:active {
    transform: scale(0.98);
}
/* Focus on the main analyze button */
div[data-testid="stForm"] .stButton > button, 
.stButton > button[kind="primary"],
.stButton > button:contains("Analyse") {
    animation: pulse-glow 3s infinite;
    border: 1px solid var(--accent-blue)88 !important;
}

/* ── Utility ── */
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Sidebar — Brand + Navigation
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 30px;">
        <div style="font-size:2.4rem; margin-bottom:8px;">🧠</div>
        <div style="font-size:1.15rem; font-weight:700;
                    background: linear-gradient(135deg,#58a6ff,#bc8cff);
                    -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent;">
            Task Optimizer
        </div>
        <div style="font-size:0.7rem; color:#8b949e;
                    letter-spacing:0.1em; text-transform:uppercase; margin-top:4px;">
            AI-Powered · Well-being
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    view = st.radio(
        "Navigate to",
        options=["👤  Employee View", "📊  HR Dashboard"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#8b949e; line-height:1.8;">
        <b style="color:#e6edf3;">System Status</b><br>
        📡 Emotion Engine: Ready<br>
        💬 NLP Pipeline: Ready<br>
        🎙️ Vocal Vitality: Ready<br>
        🗄️ Database: Connected<br>
        ⚡ Task Engine: Ready
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Wellness Quote ──
    import random
    quotes = [
        "Focus on being productive instead of busy.",
        "Your well-being is your best productivity tool.",
        "Small steps lead to big results. Keep going!",
        "Balance is not something you find, it's something you create.",
        "Rest is a part of the process, not a reward for it."
    ]
    quote = random.choice(quotes)
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03); border-radius:8px; padding:12px; 
                border-left: 3px solid var(--accent-purple); margin-top:20px;">
        <div style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; margin-bottom:4px;">Daily Insight</div>
        <div style="font-size:0.8rem; color:var(--text-secondary); font-style:italic; line-height:1.4;">
            "{quote}"
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.7rem; color:#484f58; text-align:center;">
        Capstone Project · 8th Semester<br>
        AI-Powered Task Optimizer v1.0
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Route to page
# ─────────────────────────────────────────────────────────────
if "Employee" in view:
    from pages.employee_view import render_employee_view
    render_employee_view()
else:
    from pages.hr_dashboard import render_hr_dashboard
    render_hr_dashboard()
