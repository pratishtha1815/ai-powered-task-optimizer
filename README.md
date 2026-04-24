# AI-Powered Task Optimizer 🧠⚡

An intelligent, multi-modal well-being and productivity dashboard designed to monitor and optimize employee task load based on their current psychological state.

Built as an **8th-Semester Engineering Capstone Project**.

---

## 📖 Project Overview

The AI-Powered Task Optimizer uses a modern machine learning pipeline to determine an employee's "Well-being Score" and dynamically recommends tasks from a catalog mapped to cognitive load.

### Core Architecture (The 4 Modules)

1. **Facial Emotion Recognition (DeepFace Engine)**
   - Powered by the industry-standard **DeepFace** library.
   - Provides out-of-the-box support for 7 primary emotions without requiring manual training.
   - Leverages pre-trained weights Managed by DeepFace, ensuring high accuracy and zero-setup.
2. **Textual Sentiment Analysis (NLP)**
   - Analyzes daily check-in text via `vaderSentiment` (Rule-based lexical analysis, fast and efficient).
   - Designed with an optional HuggingFace `distilbert` upgrade path for deep contextual understanding.
3. **Voice Analysis (Vocal Vitality)**
   - **STT**: Transcribes check-in audio via Google Web Speech API.
   - **Vitality Heuristic**: Extracts RMS Energy (volume/consistency) using `pydub` to detect vocal fatigue.
4. **Task Matching Engine (Multi-modal)**
   - Blends Emotion (40%), Text (30%), and Voice (30%) into a unified Well-being Score.
   - Maps scores against a 5-tier Decision Matrix to rank tasks by cognitive load.
5. **Trend Logging & HR Alert System**
   - Persists snapshot data to a SQLite database (`SQLAlchemy` ORM) with dedicated voice tracking.
   - Calculates 7-day rolling averages and auto-generates severity alerts if well-being drops below safe thresholds.
6. **Task Engagement Tracking (NEW)**
   - Tracks individual task acceptance rates per employee.
   - Enables HR to measure the effectiveness of AI recommendations via high-fidelity engagement metrics.

---

## 🎨 System Roles & Dashboards (Streamlit)

The web dashboard provides two primary interfaces, built natively on Streamlit with custom CSS.

### 👤 Employee View
**Purpose:** Daily Check-in & Personal Workflow
- **Premium Obsidian Theme**: High-performance glassmorphic interface with reactive animations.
- Live webcam capture (Powered by DeepFace) with model warmup logic.
- Text area for daily unstructured journaling.
- **Voice Check-in**: Live audio recording for vitality and sentiment analysis (Robust fallback enabled).
- Real-time gauge of the active Well-being Score (3-way split).
- **Interactive Tasks**: Accept recommended tasks directly to log engagement.

### 📊 HR / Admin Dashboard
**Purpose:** Global Health Monitoring
- **Engagement Analytics**: Track acceptance trends and recommendation accuracy.
- KPI indicators (Total Employees, System Wellness, Policy Alerts, Tasks Accepted).
- Dedicated panel handling **Active HR Alerts** with one-click dismiss features.
- Drill-down `Plotly` graphs: Rolling trend charts, emotion donut charts, stacked sentiment labels.
- CSV export capabilities.

---

## 🚀 Installation & Quick Start

### 1. Requirements
- Python 3.10+
- A virtual environment is highly recommended.

### 2. Setup

```powershell
# Clone the repository and cd into it
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate   # On Windows
# source venv/bin/activate # On macOS/Linux

# Install requirements
pip install -r requirements.txt

# Seed the task database with sample records
python scripts/seed_tasks.py
```

### 3. Running the Dashboard

```powershell
streamlit run app.py
```

*The application will open in your default browser at `http://localhost:8501`.*

---

## 🧪 Testing

The codebase includes an exhaustive test suite (**150 tests**) across all modules built with `pytest`. The tests utilize mocked classes and in-memory databases, meaning they run fast and require **zero setup, datasets, or camera permissions to execute.**

```powershell
pytest tests/ -v
```

---

## 🛠️ Configuration & Customization

All internal thresholds, weights, and labels are centralized in `config.py`. You can adjust:
- **Weights**: Multi-modal balance between Emotion (40%), Sentiment (30%), and Voice (30%).
- **HR Thresholds**: Configure when an employee should be flagged for review (default: 3 days of low scores).
- **Sentiment Tiers**: Adjust the positive/negative compound sensitivity.
