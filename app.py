import streamlit as st
import os, random, csv, datetime, time
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import timedelta
from pathlib import Path
import altair as alt
import re

# ---------- Configuration ----------
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
CSV_FILE = DATA_DIR / "11_Plus_Exam_Prep.csv"
LOG_FILE = BASE_DIR / "results.csv"

QUESTION_LIMIT = 25
TARGET_EXAM_DATE = datetime.date(2026, 9, 15)
TARGET_ACCURACY = 0.85  # 85%

# ---------- File checking helper ----------
def setup_directories():
    """Create necessary directories if they don't exist"""
    try:
        # Create data directory if it doesn't exist
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {DATA_DIR}")
    except Exception as e:
        print(f"Note: Could not create directory: {str(e)}")
        print(f"Using current directory: {Path.cwd()}")

setup_directories()

# ---------- Enhanced NVR Visual Enhancement Function ----------
def enhance_nvr_display(text):
    """
    Convert text descriptions to Unicode symbols for NVR questions.
    This function handles: circle -> ●, square -> ■, triangle -> ▲, etc.
    """
    # Comprehensive mapping of text descriptions to Unicode symbols
    shape_map = {
        # Basic shapes
        'circle': '●',
        'square': '■',
        'triangle': '▲',
        'hexagon': '⬢',
        'pentagon': '⬟',
        'octagon': '⬡',
        'rectangle': '▭',
        'diamond': '◆',
        'star': '★',
        'heart': '♥',
        
        # Filled/empty variations
        'filled circle': '●',
        'empty circle': '○',
        'hollow circle': '○',
        'full circle': '●',
        
        'filled square': '■',
        'empty square': '□',
        'hollow square': '□',
        'full square': '■',
        
        'filled triangle': '▲',
        'empty triangle': '△',
        'hollow triangle': '△',
        'full triangle': '▲',
        
        # Arrows and directions
        'up': '↑',
        'down': '↓',
        'left': '←',
        'right': '→',
        'up arrow': '↑',
        'down arrow': '↓',
        'left arrow': '←',
        'right arrow': '→',
        'arrow up': '↑',
        'arrow down': '↓',
        'arrow left': '←',
        'arrow right': '→',
        
        # Shading patterns
        'shaded': '▓',
        'unshaded': '▒',
        'half shaded': '▒',
        'striped': '▒',
        'checkered': '▒',
        'pattern': '▒',
        
        # Lines and dots
        'thick line': '━━',
        'thin line': '──',
        'dotted line': '┄┄',
        'dashed line': '╌╌',
        'dot': '•',
        'large dot': '●',
        'small dot': '•',
        'medium dot': '•',
        
        # Size descriptions
        'small': '⊙',
        'medium': '◉',
        'large': '●',
        'extra large': '⦿',
        
        # Colors (for reference, though we use symbols)
        'black': '⬛',
        'white': '⬜',
        'gray': '◼',
        'grey': '◼',
        
        # Rotation terms
        'rotated': '↻',
        'mirror': '⇄',
        'reflected': '⇄',
        'flipped': '⇅',
        
        # Sequence indicators
        'next': '→',
        'previous': '←',
        'first': '①',
        'second': '②',
        'third': '③',
        'fourth': '④',
    }
    
    enhanced_text = str(text)
    
    # Replace text descriptions with symbols (case-insensitive, whole word matching)
    for word, symbol in shape_map.items():
        # Use regex to match whole words only
        pattern = r'\b' + re.escape(word) + r'\b'
        enhanced_text = re.sub(pattern, symbol, enhanced_text, flags=re.IGNORECASE)
    
    # Additional processing for common patterns
    # Convert sequences like "circle, square, triangle" to "●, ■, ▲"
    enhanced_text = re.sub(r'circle', '●', enhanced_text, flags=re.IGNORECASE)
    enhanced_text = re.sub(r'square', '■', enhanced_text, flags=re.IGNORECASE)
    enhanced_text = re.sub(r'triangle', '▲', enhanced_text, flags=re.IGNORECASE)
    enhanced_text = re.sub(r'hexagon', '⬢', enhanced_text, flags=re.IGNORECASE)
    enhanced_text = re.sub(r'pentagon', '⬟', enhanced_text, flags=re.IGNORECASE)
    enhanced_text = re.sub(r'octagon', '⬡', enhanced_text, flags=re.IGNORECASE)
    
    return enhanced_text

# ---------- Logging ----------
def write_log_header_if_needed():
    """Create results file with header if it doesn't exist"""
    try:
        # Ensure the directory exists
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        if not LOG_FILE.exists():
            with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["date", "module", "topic", "score", "total", "seconds"])
            print(f"✅ Created results file: {LOG_FILE}")
    except Exception as e:
        print(f"❌ Error creating results file: {str(e)}")

def log_result(module, topic, score, total, seconds):
    """Log quiz results to CSV file"""
    try:
        # Debug info
        print(f"📝 Attempting to log result to: {LOG_FILE}")
        print(f"📁 Directory exists: {LOG_FILE.parent.exists()}")
        
        # Ensure the directory exists
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        write_log_header_if_needed()
        
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.date.today().isoformat(),
                module, topic, score, total, seconds
            ])
        print(f"✅ Result logged successfully: {module}, {topic}, {score}/{total}, {seconds}s")
    except Exception as e:
        print(f"❌ Error logging result: {str(e)}")
        # Try an alternative location as backup
        try:
            alt_file = Path("quiz_results_backup.csv")
            with open(alt_file, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    datetime.date.today().isoformat(),
                    module, topic, score, total, seconds
                ])
            print(f"📋 Result saved to backup file: {alt_file}")
        except Exception as alt_e:
            print(f"❌ Also failed to write to backup: {str(alt_e)}")

# ---------- Data helpers ----------
def check_csv_file() -> bool:
    if not CSV_FILE.exists():
        return False
    try:
        df = pd.read_csv(CSV_FILE)
        required_columns = ["Type", "Question", "Option1", "Option2", "Option3", "Option4", "Answer"]
        return all(col in df.columns for col in required_columns)
    except Exception as e:
        return False

def load_questions(question_type=None, limit=None) -> List[Dict[str, Any]]:
    if not CSV_FILE.exists():
        return []
    try:
        df = pd.read_csv(CSV_FILE)

        if question_type and question_type != "Mixed (All available)":
            type_mapping = {
                "Maths": "Maths",
                "Vocabulary": "Vocabulary",
                "Verbal Reasoning": "Verbal Reasoning",
                "NVR": "Non-Verbal-Reasoning",
            }
            csv_type = type_mapping.get(question_type)
            if csv_type:
                df = df[df["Type"] == csv_type]
            if df.empty:
                return []

        questions = []
        for _, row in df.iterrows():
            question = {
                "q": row["Question"],
                "options": [row["Option1"], row["Option2"], row["Option3"], row["Option4"]],
                "answer": row["Answer"],
                "type": row["Type"]
            }
            questions.append(question)

        if limit and len(questions) > limit:
            return random.sample(questions, limit)
        return questions
    except Exception as e:
        return []

# ---------- Results load and sanitation ----------
def sanitize_results_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        raw = pd.read_csv(path, header=0)
        expected_cols = ["date", "module", "topic", "score", "total", "seconds"]
        if raw.shape[1] > 6:
            raw = raw.iloc[:, :6]
        if raw.shape[1] == 6:
            raw.columns = expected_cols

        raw = raw.dropna(how="all")
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        for c in ["score", "total", "seconds"]:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

        raw = raw.dropna(subset=["date", "module", "topic", "score", "total", "seconds"])
        raw["module"] = raw["module"].astype(str)
        raw["topic"] = raw["topic"].astype(str)

        raw.to_csv(path, index=False, encoding="utf-8")
    except Exception as e:
        pass

def load_results(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(columns=["date", "module", "topic", "score", "total", "seconds", "accuracy"])

    sanitize_results_file(log_path)
    try:
        df = pd.read_csv(log_path)
        base_cols = ["date", "module", "topic", "score", "total", "seconds"]
        df = df[base_cols]

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df["total"] = pd.to_numeric(df["total"], errors="coerce")
        df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")

        df = df.dropna(subset=["date", "module", "topic", "score", "total", "seconds"])
        df["accuracy"] = (df["score"] / df["total"]).clip(lower=0, upper=1)
        return df
    except:
        return pd.DataFrame()

def summarize_mastery(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summary = df.groupby("topic").agg({
        "score": "sum",
        "total": "sum",
        "seconds": "mean",
        "accuracy": "mean",
    }).reset_index()
    summary["mastery"] = pd.cut(summary["accuracy"], bins=[0, 0.5, 0.8, 1.0], labels=["Low", "Medium", "High"])
    return summary

def prepare_mastery_over_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.groupby(["date", "topic"]).agg({"accuracy": "mean"}).reset_index()

# ---------- Performance Analytics ----------
def calculate_learning_trajectory(df: pd.DataFrame):
    if len(df) < 3:
        return None
    try:
        df_sorted = df.sort_values("date").copy()
        start_date = df_sorted["date"].min()
        df_sorted["days"] = (df_sorted["date"] - start_date).dt.days

        X = df_sorted["days"].values
        y = df_sorted["accuracy"].values

        n = len(X)
        if n < 2:
            return None

        sum_x = np.sum(X)
        sum_y = np.sum(y)
        sum_xy = np.sum(X * y)
        sum_xx = np.sum(X * X)

        denom = n * sum_xx - sum_x * sum_x
        if denom == 0:
            return None

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        y_pred = slope * X + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        today = pd.Timestamp(datetime.date.today())
        exam_ts = pd.Timestamp(TARGET_EXAM_DATE)
        days_until_exam = int((exam_ts - today).days)

        current_day = df_sorted["days"].max()
        current_accuracy = slope * current_day + intercept
        predicted_accuracy = slope * (current_day + max(days_until_exam, 0)) + intercept

        current_accuracy = float(np.clip(current_accuracy, 0, 1))
        predicted_accuracy = float(np.clip(predicted_accuracy, 0, 1))

        current_to_target_gap = TARGET_ACCURACY - current_accuracy
        required_daily_improvement = (current_to_target_gap / days_until_exam) if days_until_exam > 0 else 0

        return {
            "slope": slope,
            "intercept": intercept,
            "current_accuracy": current_accuracy,
            "predicted_accuracy": predicted_accuracy,
            "required_daily_improvement": required_daily_improvement,
            "days_until_exam": days_until_exam,
            "r_squared": r_squared,
            "trend_data": df_sorted,
            "start_date": start_date,
        }
    except:
        return None

def calculate_performance_metrics(df: pd.DataFrame):
    if df.empty:
        return {}
    try:
        today = datetime.date.today()
        last_7_days = pd.Timestamp(today - timedelta(days=7))
        last_30_days = pd.Timestamp(today - timedelta(days=30))

        df_recent = df[df["date"] >= last_7_days]
        df_month = df[df["date"] >= last_30_days]

        metrics = {
            "overall_accuracy": df["accuracy"].mean(),
            "recent_accuracy": df_recent["accuracy"].mean() if not df_recent.empty else 0,
            "monthly_accuracy": df_month["accuracy"].mean() if not df_month.empty else 0,
            "total_questions": int(df["total"].sum()),
            "total_quizzes": int(len(df)),
            "best_score": df["accuracy"].max(),
            "improvement_7d": 0,
            "consistency": (1 - df["accuracy"].std()) if len(df) > 1 else 1,
        }

        if len(df) >= 2 and not df_recent.empty:
            recent_avg = metrics["recent_accuracy"]
            older_data = df[df["date"] < last_7_days]
            if not older_data.empty:
                older_avg = older_data["accuracy"].mean()
                if older_avg > 0:
                    metrics["improvement_7d"] = (recent_avg - older_avg) / older_avg
        return metrics
    except:
        return {}

# ---------- Session State ----------
def reset_session():
    st.session_state.update({
        "questions": [],
        "q_index": 0,
        "score": 0,
        "start_time": None,
        "choice": None,
        "finished": False,
        "question_start": None,
        "module_label": None,
        "quiz_completed": False,
        "user_answers": [],
        "answered_current": False,
        "shuffled_options": {},
    })

def clear_all_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# ---------- Main App ----------
st.set_page_config(
    page_title="11+ Practice App",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://streamlit.io',
        'Report a bug': None,
        'About': "Elisa's 11+ Learning App - Mobile Optimized"
    }
)

# ---------- Mobile-Optimized CSS ----------
st.markdown("""
    <style>
    .stApp {
        background-color: #000000 !important;
        color: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .stButton > button {
        background-color: #2E8B57 !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 20px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        margin: 8px 0 !important;
        min-height: 44px !important;
        min-width: 44px !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background-color: #3DA56C !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }

    .stRadio > div {
        background: #1a1a1a !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin: 12px 0 !important;
        border: 2px solid #333 !important;
    }

    .stRadio > div > label {
        font-size: 18px !important;
        padding: 14px 0 !important;
        margin: 4px 0 !important;
        color: white !important;
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
        min-height: 44px !important;
    }

    @media (max-width: 768px) {
        [data-testid="column"] {
            width: 100% !important;
            margin-bottom: 16px !important;
        }
        
        h1 { font-size: 28px !important; }
        h2 { font-size: 24px !important; }
        h3 { font-size: 20px !important; }
        
        .stDataFrame {
            font-size: 14px !important;
        }
    }

    .stProgress > div > div {
        background-color: #2E8B57 !important;
        height: 20px !important;
        border-radius: 10px !important;
    }

    .dataframe {
        background-color: #1a1a1a !important;
        color: white !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    .stAlert {
        border-radius: 12px !important;
        padding: 16px !important;
        margin: 12px 0 !important;
    }

    .stSuccess {
        background-color: #1a3a1a !important;
        border-color: #2E8B57 !important;
        color: #90EE90 !important;
    }

    .stError {
        background-color: #3a1a1a !important;
        border-color: #8B2E2E !important;
        color: #FF6B6B !important;
    }

    .stInfo {
        background-color: #1a2a3a !important;
        border-color: #2E578B !important;
        color: #87CEEB !important;
    }

    .nvr-shape {
        font-size: 32px !important;
        text-align: center !important;
        margin: 24px 0 !important;
        line-height: 1.5 !important;
        letter-spacing: 8px !important;
    }
    
    .nvr-question {
        font-size: 24px !important;
        line-height: 1.6 !important;
        margin: 20px 0 !important;
        padding: 15px !important;
        background: #1a1a1a !important;
        border-radius: 12px !important;
        border: 2px solid #333 !important;
    }
    
    .nvr-option {
        font-size: 22px !important;
        margin: 10px 0 !important;
        padding: 12px !important;
    }

    @media (max-width: 768px) {
        .nvr-shape {
            font-size: 24px !important;
            letter-spacing: 4px !important;
        }
        .nvr-question {
            font-size: 20px !important;
        }
        .nvr-option {
            font-size: 18px !important;
        }
    }

    [data-testid="metric-container"] {
        background-color: #1a1a1a !important;
        border: 2px solid #333 !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }

    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            min-width: 280px !important;
        }
    }

    .stNumberInput input, .stTextInput input, .stSelectbox select {
        font-size: 18px !important;
        padding: 12px !important;
        min-height: 48px !important;
        border-radius: 10px !important;
    }

    .main .block-container {
        max-width: 100% !important;
        overflow-x: hidden !important;
    }
    </style>
""", unsafe_allow_html=True)

# App title (separate markdown call)
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="color: #2E8B57; margin-bottom: 10px;">🧠 11+ Practice App</h1>
    <p style="color: #ccc; font-size: 18px;">Mobile-friendly learning for Elisaveta</p>
</div>
""", unsafe_allow_html=True)

# Check if CSV file exists
if not CSV_FILE.exists():
    st.error(f"❌ CSV file not found. Please ensure '11_Plus_Exam_Prep.csv' is in the 'data' folder.")
    st.info(f"Looking for file at: {CSV_FILE}")
    st.info("Current directory structure should be:")
    st.code("""
    Elisa-smart-learning/
    ├── app.py
    ├── requirements.txt
    ├── .streamlit/config.toml
    ├── data/11_Plus_Exam_Prep.csv   ← This file is missing
    ├── .gitignore
    └── README.md
    """)
else:
    if not check_csv_file():
        st.error("❌ CSV file format is incorrect. Please check the required columns.")
    else:
        st.success("✅ App initialized successfully!")

# ---------- Mode Selector ----------
mode_type = st.sidebar.radio("Choose Mode", ["Kid Mode", "Parent Mode"], horizontal=True)
if mode_type == "Kid Mode":
    page = st.sidebar.radio("Choose view", ["Quiz", "My Progress"], horizontal=True)
else:
    page = st.sidebar.radio("Choose view", ["Quiz", "Dashboard", "Data Info", "Predictive Analytics", "Debug"], horizontal=True)

# ---------- Refresh Button ----------
with st.sidebar:
    st.markdown("---")
    if st.button("🔄 Refresh App", use_container_width=True, type="secondary"):
        clear_all_session()
        st.rerun()

# ---------- Data Info ----------
if page == "Data Info" and mode_type == "Parent Mode":
    st.title("📊 Data Information")
    try:
        df = pd.read_csv(CSV_FILE)
        st.success("✅ CSV file loaded successfully")

        st.markdown("### Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Questions", len(df))
            st.metric("Maths", len(df[df["Type"] == "Maths"]))
        with col2:
            st.metric("Vocabulary", len(df[df["Type"] == "Vocabulary"]))
            st.metric("Verbal Reasoning", len(df[df["Type"] == "Verbal Reasoning"]))
        with col3:
            st.metric("NVR", len(df[df["Type"] == "Non-Verbal-Reasoning"]))

        st.markdown("### Sample Questions (First 10)")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("### Question Type Distribution")
        type_counts = df["Type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        
        # Using Altair bar chart
        chart = alt.Chart(type_counts).mark_bar().encode(
            x='Type:N',
            y='Count:Q',
            color='Type:N',
            tooltip=['Type', 'Count']
        ).properties(
            title="Question Types Distribution",
            height=400
        )
        st.altair_chart(chart, use_container_width=True)
        
        # Also show as a table
        st.dataframe(type_counts, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")

# ---------- Debug Page ----------
elif page == "Debug" and mode_type == "Parent Mode":
    st.title("🔧 Debug Information")
    
    st.markdown("### File System Info")
    st.write(f"Current directory: {Path.cwd()}")
    st.write(f"BASE_DIR: {BASE_DIR}")
    st.write(f"LOG_FILE path: {LOG_FILE}")
    st.write(f"LOG_FILE exists: {LOG_FILE.exists()}")
    
    if LOG_FILE.exists():
        st.markdown("### Current Results File Content")
        try:
            df = pd.read_csv(LOG_FILE)
            st.dataframe(df)
            st.write(f"Total records: {len(df)}")
            st.write(f"File size: {LOG_FILE.stat().st_size} bytes")
            st.write(f"Last modified: {datetime.datetime.fromtimestamp(LOG_FILE.stat().st_mtime)}")
        except Exception as e:
            st.error(f"Error reading results file: {e}")
    else:
        st.warning("Results file doesn't exist yet. Take a quiz first!")
    
    # Check if we can write to file
    st.markdown("### Test Write Permission")
    if st.button("Test Write to File"):
        try:
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.date.today().isoformat(), "test", "test", 1, 1, 10])
            st.success("✅ Successfully wrote test data to file")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to write: {e}")
    
    # Show session state info
    st.markdown("### Session State")
    session_keys = list(st.session_state.keys())
    st.write(f"Session keys: {session_keys}")
    
    # Check for backup file
    backup_file = Path("quiz_results_backup.csv")
    if backup_file.exists():
        st.markdown("### Backup File Content")
        try:
            backup_df = pd.read_csv(backup_file)
            st.dataframe(backup_df)
            st.write(f"Backup records: {len(backup_df)}")
        except Exception as e:
            st.error(f"Error reading backup file: {e}")

# ---------- Quiz ----------
elif page == "Quiz":
    if "questions" not in st.session_state:
        reset_session()

    modules = ["Maths", "Vocabulary", "Verbal Reasoning", "NVR", "Mixed (All available)"]

    with st.sidebar:
        st.markdown("### 🎯 Quiz Settings")
        mode = st.selectbox("Choose quiz type", modules)
        num_questions = st.slider("Number of questions", 5, QUESTION_LIMIT, 10)
        per_question_seconds = st.number_input("Time limit per question (seconds)", min_value=0, value=0, step=5)
        start_button = st.button("🚀 Start / Restart Quiz", use_container_width=True, type="primary")

    if start_button:
        questions = load_questions(mode, limit=num_questions)
        if not questions:
            st.error("❌ Could not load questions. Please check the CSV file in the data folder.")
        else:
            module_label = "mixed" if mode == "Mixed (All available)" else mode.lower().replace(" ", "_")
            st.session_state.shuffled_options = {}

            st.session_state.update({
                "questions": questions,
                "q_index": 0,
                "score": 0,
                "start_time": time.time(),
                "choice": None,
                "finished": False,
                "question_start": time.time(),
                "module_label": module_label,
                "quiz_completed": False,
                "user_answers": [None] * len(questions),
                "answered_current": False,
            })
            st.rerun()

    if st.session_state.get("questions"):
        idx = st.session_state.q_index
        total = len(st.session_state.questions)
        elapsed = int(time.time() - st.session_state.start_time) if st.session_state.start_time else 0

        # Progress display
        if mode_type == "Kid Mode":
            st.markdown("### 🌟 Your Adventure")
            st.progress(idx / total)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Question", f"{idx + 1} of {total}")
            with col2:
                st.metric("Score", st.session_state.score)
        else:
            st.markdown("### 📊 Quiz Progress")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Question", f"{idx + 1} / {total}")
            with col2:
                st.metric("Score", st.session_state.score)
            with col3:
                st.metric("Time", f"{elapsed}s")

        current = st.session_state.questions[idx]
        
        # Question display
        st.markdown(f"### **Question {idx + 1}**")
        
        current_question = current.get("q", "")
        question_type = current.get("type", "")
        
        # Detect NVR questions more reliably
        is_nvr_question = (
            "Non-Verbal-Reasoning" in str(question_type) or 
            mode == "NVR" or
            any(term in str(current_question).lower() for term in [
                'circle', 'square', 'triangle', 'arrow', 'shaded', 'unshaded', 
                'filled', 'empty', 'rotation', 'mirror', 'reflect', 'shape',
                'hexagon', 'pentagon', 'octagon', 'pattern', 'sequence'
            ])
        )
        
        if is_nvr_question:
            # Enhanced NVR display
            visual_question = enhance_nvr_display(current_question)
            st.markdown(f'<div class="nvr-question">**{visual_question}**</div>', unsafe_allow_html=True)
            
            # Add decorative shapes for visual appeal
            st.markdown('<div class="nvr-shape">● ■ ▲ ⬢ ⬟ ⬡ ★ ♥</div>', unsafe_allow_html=True)
            
            # Process options
            raw_options = [str(o) for o in current.get("options", [])]
            options = [enhance_nvr_display(opt) for opt in raw_options]
            
            # Store original answers for comparison
            original_answer = str(current.get("answer", "")).strip()
            enhanced_answer = enhance_nvr_display(original_answer)
        else:
            # Regular question display
            st.markdown(f"**{current_question}**")
            options = [str(o) for o in current.get("options", [])]
            enhanced_answer = str(current.get("answer", "")).strip()

        # Answer selection
        question_key = f"q{idx}"
        if question_key not in st.session_state.get("shuffled_options", {}):
            shuffled = options.copy()
            random.shuffle(shuffled)
            st.session_state.shuffled_options[question_key] = shuffled
        else:
            shuffled = st.session_state.shuffled_options[question_key]

        # Display options
        if is_nvr_question:
            selected_answer = st.radio(
                "Select your answer:", 
                shuffled, 
                key=f"choice_{idx}"
            )
        else:
            selected_answer = st.radio("Select your answer:", shuffled, key=f"choice_{idx}")
        
        st.session_state.user_answers[idx] = selected_answer

        # Timer (if enabled)
        timed_out = False
        if per_question_seconds > 0:
            elapsed_q = int(time.time() - st.session_state.get("question_start", time.time()))
            remaining = max(0, per_question_seconds - elapsed_q)
            
            if remaining <= 0:
                timed_out = True
                st.error("⏱️ Time's up!")
            else:
                st.progress(remaining / per_question_seconds)
                st.info(f"⏳ Time remaining: {remaining} seconds")

        # Navigation buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            next_btn = st.button("➡️ Next", use_container_width=True)
        with col2:
            skip_btn = st.button("⏭️ Skip", use_container_width=True)
        with col3:
            finish_btn = st.button("🏁 Finish", use_container_width=True)

        # Answer checking logic
        if next_btn:
            if st.session_state.user_answers[idx] is not None:
                selected = str(st.session_state.user_answers[idx]).strip()
                
                # For NVR questions, compare enhanced versions
                if is_nvr_question:
                    # Enhanced comparison
                    selected_enhanced = enhance_nvr_display(selected).strip()
                    answer_enhanced = enhanced_answer.strip()
                    
                    if timed_out:
                        st.error("Time's up - marked as incorrect")
                    elif selected_enhanced == answer_enhanced:
                        st.session_state.score += 1
                        st.success("✅ Correct! Great job!")
                    else:
                        st.error(f"❌ Incorrect. The correct answer was: **{enhanced_answer}**")
                else:
                    # Regular comparison
                    correct = str(current.get("answer")).strip()
                    
                    if timed_out:
                        st.error("Time's up - marked as incorrect")
                    elif selected == correct:
                        st.session_state.score += 1
                        st.success("✅ Correct!")
                    else:
                        st.error(f"❌ Incorrect. The correct answer was: **{correct}**")
                
                time.sleep(1)
                if idx < total - 1:
                    st.session_state.q_index += 1
                    st.session_state.question_start = time.time()
                    st.rerun()
                else:
                    st.session_state.finished = True
            else:
                st.warning("Please select an answer before proceeding")
        
        elif skip_btn:
            if idx < total - 1:
                st.session_state.q_index += 1
                st.session_state.question_start = time.time()
                st.rerun()
            else:
                st.session_state.finished = True
        
        elif finish_btn:
            st.session_state.finished = True

        # Completion
        if st.session_state.finished and not st.session_state.get("quiz_completed", False):
            duration = int(time.time() - st.session_state.start_time)
            score = st.session_state.score
            # Log the result
            log_result(st.session_state.get("module_label", "unknown"), mode, score, total, duration)
            st.session_state.quiz_completed = True

            st.balloons()
            st.success("🎉 Quiz Complete!")
            
            if mode_type == "Kid Mode":
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #2E8B57, #3DA56C); padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0;">
                    <h2 style="color: white;">Amazing job, Elisaveta! 🌟</h2>
                    <p style="font-size: 24px; color: white; margin: 10px 0;">You scored <strong>{score} out of {total}</strong> 🏆</p>
                    <p style="font-size: 20px; color: white;">Time: <strong>{duration} seconds</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                accuracy = score / total if total > 0 else 0
                days_until_exam = (TARGET_EXAM_DATE - datetime.date.today()).days
                
                if accuracy >= TARGET_ACCURACY:
                    st.success(f"🌈 You're already at your target! {accuracy:.1%} accuracy! 🎯")
                else:
                    st.info(f"🎯 Goal: {TARGET_ACCURACY:.0%} by September 2026 ({days_until_exam} days to go!)")
            else:
                st.markdown(f"""
                ### Results Summary
                - **Score:** {score} / {total} ({score/total:.1%})
                - **Time:** {duration} seconds
                - **Average time per question:** {duration/total:.1f} seconds
                """)
            
            if st.button("🔄 Start New Quiz", use_container_width=True):
                reset_session()
                st.rerun()
    else:
        # Welcome screen
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2E8B57, #3DA56C); padding: 40px; border-radius: 20px; text-align: center; margin: 20px 0;">
            <h2 style="color: white; margin-bottom: 20px;">
                Welcome Elisaveta! 🙂<br>Добре дошли Елисавета! 🙂
            </h2>
            <p style="color: white; font-size: 20px; margin-bottom: 30px;">
                Choose a quiz type in the sidebar and click<br>
                <strong>"Start / Restart Quiz"</strong> to begin your learning journey!
            </p>
            <div style="font-size: 48px; margin-top: 20px;">
                🧠 📚 ✏️ 🎯
            </div>
        </div>
        """, unsafe_allow_html=True)
# ---------- My Progress (Kid Mode) ----------
elif page == "My Progress" and mode_type == "Kid Mode":
    st.title("🌟 My Learning Journey")
    df = load_results(LOG_FILE)
    
    if df.empty:
        st.info("📝 No quiz data yet. Complete a quiz to see your progress!")
    else:
        days_until_exam = (TARGET_EXAM_DATE - datetime.date.today()).days
        accuracy = df["accuracy"].mean()

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Quizzes", len(df))
        with col2:
            st.metric("Overall Accuracy", f"{accuracy:.1%}")
        with col3:
            st.metric("Days Until Exam", days_until_exam)

        # Progress chart using Altair
        st.subheader("📈 Progress Over Time")
        weekly_avg = df.groupby(pd.Grouper(key="date", freq="W"))["accuracy"].mean().reset_index()
        if not weekly_avg.empty:
            # Create Altair line chart
            line = alt.Chart(weekly_avg).mark_line(
                point=True,
                color='#2E8B57',
                strokeWidth=3
            ).encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('accuracy:Q', title='Accuracy', axis=alt.Axis(format='%')),
                tooltip=['date:T', alt.Tooltip('accuracy:Q', format='.1%')]
            ).properties(
                height=400,
                title='Weekly Average Accuracy'
            )
            
            # Add target line
            target_rule = alt.Chart(pd.DataFrame({'y': [TARGET_ACCURACY]})).mark_rule(
                color='red',
                strokeDash=[5, 5]
            ).encode(y='y:Q')
            
            target_text = alt.Chart(pd.DataFrame({'x': [weekly_avg['date'].min()], 'y': [TARGET_ACCURACY]})).mark_text(
                text=f'Target: {TARGET_ACCURACY:.0%}',
                align='left',
                dx=5,
                dy=-5,
                color='red'
            ).encode(x='x:T', y='y:Q')
            
            chart = (line + target_rule + target_text)
            st.altair_chart(chart, use_container_width=True)

# ---------- Dashboard ----------
elif page == "Dashboard":
    st.title("📊 Topic Mastery Dashboard")
    df = load_results(LOG_FILE)
    
    if df.empty:
        st.info("No quiz data available yet.")
    else:
        st.markdown("#### 📅 Filter by Date Range")
        min_date, max_date = df["date"].min(), df["date"].max()
        start_date, end_date = st.date_input("Select range", [min_date, max_date])
        filtered_df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

        if not filtered_df.empty:
            st.markdown("#### 📊 Topic Summary")
            summary = summarize_mastery(filtered_df)
            st.dataframe(
                summary.style.format({"accuracy": "{:.1%}", "seconds": "{:.1f}"}),
                use_container_width=True
            )

# ---------- Predictive Analytics ----------
elif page == "Predictive Analytics" and mode_type == "Parent Mode":
    st.title("🔮 Predictive Analytics")
    df = load_results(LOG_FILE)
    
    if df.empty:
        st.info("Complete some quizzes to see analytics.")
    else:
        metrics = calculate_performance_metrics(df)
        
        st.subheader("📊 Performance Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Accuracy", f"{metrics.get('overall_accuracy', 0):.1%}")
        with col2:
            st.metric("Recent Accuracy (7d)", f"{metrics.get('recent_accuracy', 0):.1%}")
        with col3:
            st.metric("Total Questions", metrics.get("total_questions", 0))

# ---------- Footer ----------
# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 14px; padding: 20px 0;">
    Made with ❤️ for Elisaveta | Streamlit Cloud | Mobile Optimized
</div>
""", unsafe_allow_html=True)
