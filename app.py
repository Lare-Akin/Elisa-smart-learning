import streamlit as st
import os, random, csv, datetime, time
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import timedelta
from pathlib import Path
import altair as alt
import re
import json

# ---------- Configuration ----------
try:
    BASE_DIR = Path(__file__).parent.resolve()
except:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
CSV_FILE = DATA_DIR / "11_Plus_Exam_Prep.csv"
JSON_FILE = DATA_DIR / "quiz_results.json"

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
        
        # Create JSON file if it doesn't exist
        if not JSON_FILE.exists():
            with open(JSON_FILE, 'w') as f:
                json.dump([], f)
            print(f"Created JSON file: {JSON_FILE}")
            
    except Exception as e:
        print(f"Note: Could not create directory: {str(e)}")
        print(f"Using current directory: {Path.cwd()}")

setup_directories()

# ---------- JSON Functions ----------
def log_result(module, topic, score, total, seconds):
    """Log quiz results to JSON file"""
    try:
        # Load existing results
        if JSON_FILE.exists():
            with open(JSON_FILE, 'r') as f:
                results = json.load(f)
        else:
            results = []
        
        # Create new result entry
        new_result = {
            "date": datetime.date.today().isoformat(),
            "module": module,
            "topic": topic,
            "score": score,
            "total": total,
            "seconds": seconds,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # Append and save
        results.append(new_result)
        with open(JSON_FILE, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"✅ Result logged to JSON: {module}, {topic}, {score}/{total}, {seconds}s")
        return True
        
    except Exception as e:
        print(f"❌ Error logging to JSON: {e}")
        return False

def load_results() -> pd.DataFrame:
    """Load results from JSON file"""
    try:
        if not JSON_FILE.exists():
            return pd.DataFrame(columns=["date", "module", "topic", "score", "total", "seconds", "accuracy"])
        
        with open(JSON_FILE, 'r') as f:
            results = json.load(f)
        
        if results:
            df = pd.DataFrame(results)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["accuracy"] = (df["score"] / df["total"]).clip(lower=0, upper=1)
                print(f"✅ Loaded {len(df)} records from JSON file")
                return df
                
    except Exception as e:
        print(f"❌ Error loading from JSON: {e}")
    
    return pd.DataFrame(columns=["date", "module", "topic", "score", "total", "seconds", "accuracy"])

def clear_results():
    """Clear all results from JSON file"""
    try:
        with open(JSON_FILE, 'w') as f:
            json.dump([], f)
        print("✅ Cleared all results from JSON file")
        return True
    except Exception as e:
        print(f"❌ Error clearing results: {e}")
        return False

# ---------- Enhanced NVR Visual Enhancement Function ----------
def enhance_nvr_display(text):
    """
    Convert text descriptions to Unicode symbols for NVR questions.
    This function handles: circle -> ●, square -> ■, triangle -> ▲, etc.
    """
    shape_map = {
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
        'shaded': '▓',
        'unshaded': '▒',
        'half shaded': '▒',
        'striped': '▒',
        'checkered': '▒',
        'pattern': '▒',
        'thick line': '━━',
        'thin line': '──',
        'dotted line': '┄┄',
        'dashed line': '╌╌',
        'dot': '•',
        'large dot': '●',
        'small dot': '•',
        'medium dot': '•',
        'small': '⊙',
        'medium': '◉',
        'large': '●',
        'extra large': '⦿',
        'black': '⬛',
        'white': '⬜',
        'gray': '◼',
        'grey': '◼',
        'rotated': '↻',
        'mirror': '⇄',
        'reflected': '⇄',
        'flipped': '⇅',
        'next': '→',
        'previous': '←',
        'first': '①',
        'second': '②',
        'third': '③',
        'fourth': '④',
    }
    
    enhanced_text = str(text)
    for word, symbol in shape_map.items():
        pattern = r'\b' + re.escape(word) + r'\b'
        enhanced_text = re.sub(pattern, symbol, enhanced_text, flags=re.IGNORECASE)

    enhanced_text = re.sub(r'circle', '●', enhanced_text, flags=re.IGNORECASE)
    enhanced_text = re.sub(r'square', '■', enhanced_text, flags=re.IGNORECASE)
    enhanced_text = re.sub(r'triangle', '▲', enhanced_text, flags=re.IGNORECASE)
    enhanced_text = re.sub(r'hexagon', '⬢', enhanced_text, flags=re.IGNORECASE)
    enhanced_text = re.sub(r'pentagon', '⬟', enhanced_text, flags=re.IGNORECASE)
    enhanced_text = re.sub(r'octagon', '⬡', enhanced_text, flags=re.IGNORECASE)
    
    return enhanced_text

# ---------- Data helpers ----------
def check_csv_file() -> bool:
    if not CSV_FILE.exists():
        return False
    try:
        df = pd.read_csv(CSV_FILE)
        required_columns = ["Type", "Question", "Option1", "Option2", "Option3", "Option4", "Answer"]
        return all(col in df.columns for col in required_columns)
    except Exception:
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
    except Exception:
        return []

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

        df_sorted["trend_accuracy"] = slope * df_sorted["days"] + intercept

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
        'About': "Elisa's 11+ Learning App - Mobile Optimized with JSON Storage"
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
    /* Hide admin panel initially */
    .admin-panel {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="color: #2E8B57; margin-bottom: 10px;">🧠 11+ Practice App</h1>
    <p style="color: #ccc; font-size: 18px;">Elisaveta's Mobile Friendly Learning Platofrm</p>
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
        if JSON_FILE.exists():
            df = load_results()
            if not df.empty:
                st.info(f"✅ Found {len(df)} quiz results")

# ---------- Mode Selector ----------
mode_type = st.sidebar.radio("Choose Mode", ["Kid Mode", "Parent Mode"], horizontal=True)
if mode_type == "Kid Mode":
    page = st.sidebar.radio("Choose view", ["Quiz", "My Progress"], horizontal=True)
else:
    # Removed "Data Info" and "Debug" from Parent Mode
    page = st.sidebar.radio("Choose view", ["Quiz", "Dashboard", "Predictive Analytics"], horizontal=True)

# ---------- Secret Admin Panel Activation ----------
with st.sidebar:
    st.markdown("---")
    
    # Simple refresh button (always visible)
    if st.button("🔄 Refresh App", use_container_width=True, type="secondary"):
        clear_all_session()
        st.rerun()
    
    # Hidden admin panel - only appears after clicking "Help" 3 times
    if 'help_clicks' not in st.session_state:
        st.session_state.help_clicks = 0
    
    # Add a small, inconspicuous "Help" link
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("❓", help="Click for help", use_container_width=False):
            st.session_state.help_clicks += 1
    
    # Show admin panel after 3 clicks on the help button
    if st.session_state.help_clicks >= 3:
        st.markdown("---")
        st.markdown("### 🔒 Admin Panel (Hidden)")
        st.warning("⚠️ **Parent Access Only** - Be careful with these options!")
        
        # Show current data stats
        if JSON_FILE.exists():
            try:
                with open(JSON_FILE, 'r') as f:
                    results = json.load(f)
                st.info(f"**Currently stored:** {len(results)} quiz results")
                
                # Download button
                if results:
                    json_str = json.dumps(results, indent=2)
                    st.download_button(
                        label="📥 Backup Quiz Data",
                        data=json_str,
                        file_name=f"elisa_quiz_backup_{datetime.date.today()}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                # Clear data with confirmation
                if st.button("🗑️ Clear All Quiz Data", use_container_width=True, type="secondary"):
                    # Double confirmation
                    st.error("⚠️ **WARNING: This will delete ALL quiz data permanently!**")
                    confirm = st.checkbox("I understand this cannot be undone")
                    password = st.text_input("Enter 'DELETE' to confirm", type="password")
                    
                    if st.button("🚨 PERMANENTLY DELETE ALL DATA", type="primary", use_container_width=True):
                        if confirm and password == "DELETE":
                            if clear_results():
                                st.success("✅ All data cleared!")
                                st.session_state.help_clicks = 0  # Reset click counter
                                st.rerun()
                        else:
                            st.error("❌ Confirmation failed. Data NOT deleted.")
                
                # Reset click counter
                if st.button("Hide Admin Panel", use_container_width=True):
                    st.session_state.help_clicks = 0
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error accessing data: {e}")

# ---------- Quiz ----------
if page == "Quiz":
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
        st.markdown(f"### **Question {idx + 1}**")
        
        current_question = current.get("q", "")
        question_type = current.get("type", "")
        
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
            visual_question = enhance_nvr_display(current_question)
            st.markdown(f'<div class="nvr-question">**{visual_question}**</div>', unsafe_allow_html=True)
            st.markdown('<div class="nvr-shape">● ■ ▲ ⬢ ⬟ ⬡ ★ ♥</div>', unsafe_allow_html=True)
            raw_options = [str(o) for o in current.get("options", [])]
            options = [enhance_nvr_display(opt) for opt in raw_options]
            original_answer = str(current.get("answer", "")).strip()
            enhanced_answer = enhance_nvr_display(original_answer)
        else:
            st.markdown(f"**{current_question}**")
            options = [str(o) for o in current.get("options", [])]
            enhanced_answer = str(current.get("answer", "")).strip()

        question_key = f"q{idx}"
        if question_key not in st.session_state.get("shuffled_options", {}):
            shuffled = options.copy()
            random.shuffle(shuffled)
            st.session_state.shuffled_options[question_key] = shuffled
        else:
            shuffled = st.session_state.shuffled_options[question_key]

        if is_nvr_question:
            selected_answer = st.radio(
                "Select your answer:", 
                shuffled, 
                key=f"choice_{idx}"
            )
        else:
            selected_answer = st.radio("Select your answer:", shuffled, key=f"choice_{idx}")
        
        st.session_state.user_answers[idx] = selected_answer

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

        col1, col2, col3 = st.columns(3)
        with col1:
            next_btn = st.button("➡️ Next", use_container_width=True)
        with col2:
            skip_btn = st.button("⏭️ Skip", use_container_width=True)
        with col3:
            finish_btn = st.button("🏁 Finish", use_container_width=True)

        if next_btn:
            if st.session_state.user_answers[idx] is not None:
                selected = str(st.session_state.user_answers[idx]).strip()
                
                if is_nvr_question:
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

        if st.session_state.finished and not st.session_state.get("quiz_completed", False):
            duration = int(time.time() - st.session_state.start_time)
            score = st.session_state.score
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
    df = load_results()
    
    if df.empty:
        st.info("📝 No quiz data yet. Complete a quiz to see your progress!")
    else:
        days_until_exam = (TARGET_EXAM_DATE - datetime.date.today()).days
        accuracy = df["accuracy"].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Quizzes", len(df))
        with col2:
            st.metric("Overall Accuracy", f"{accuracy:.1%}")
        with col3:
            st.metric("Days Until Exam", days_until_exam)

        st.subheader("📈 Progress Over Time")
        weekly_avg = df.groupby(pd.Grouper(key="date", freq="W"))["accuracy"].mean().reset_index()
        if not weekly_avg.empty:
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
    df = load_results()
    
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
            
            # Add download button for data
            st.markdown("---")
            st.markdown("### 💾 Export Data")
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Results as CSV",
                data=csv,
                file_name=f"elisa_quiz_results_{datetime.date.today()}.csv",
                mime="text/csv",
            )

# ---------- Predictive Analytics ----------
elif page == "Predictive Analytics" and mode_type == "Parent Mode":
    st.title("🔮 Predictive Analytics: 11+ Exam Readiness")
    df = load_results()
    
    if df.empty:
        st.info("Complete some quizzes to see analytics.")
    else:
        metrics = calculate_performance_metrics(df)
        trajectory = calculate_learning_trajectory(df)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Accuracy", f"{metrics.get('overall_accuracy', 0):.1%}")
        with col2:
            st.metric("Recent Accuracy (7d)", f"{metrics.get('recent_accuracy', 0):.1%}")
        with col3:
            st.metric("Total Questions", metrics.get("total_questions", 0))

        st.markdown("---")

        if trajectory is None:
            st.info("Not enough data for trend modelling. Try at least 3 quiz sessions.")
        else:
            st.subheader("📈 Accuracy Trend and Forecast")
            trend_df = trajectory["trend_data"]
            slope = trajectory["slope"]
            intercept = trajectory["intercept"]
            days_until_exam = trajectory["days_until_exam"]

            # Build forecast
            last_day = trend_df["days"].max()
            future_days = max(days_until_exam, 0)
            forecast_days = np.arange(last_day, last_day + future_days + 1)

            forecast_df = pd.DataFrame({
                "days": forecast_days,
                "accuracy": slope * forecast_days + intercept,
                "segment": "Forecast"
            })
            forecast_df["accuracy"] = forecast_df["accuracy"].clip(0, 1)

            hist_df = trend_df[["days", "accuracy"]].copy()
            hist_df["segment"] = "History"

            combined = pd.concat([hist_df, forecast_df], ignore_index=True)

            chart = alt.Chart(combined).mark_line(point=True).encode(
                x=alt.X("days:Q", title="Days Since First Quiz"),
                y=alt.Y("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("segment:N", title=""),
                tooltip=["days", alt.Tooltip("accuracy:Q", format=".1%"), "segment"]
            ).properties(
                height=400,
                title="Accuracy Trend and Forecast to Exam Day"
            )

            target_line = alt.Chart(pd.DataFrame({"y": [TARGET_ACCURACY]})).mark_rule(
                color="red",
                strokeDash=[5, 5]
            ).encode(y="y:Q")

            st.altair_chart(chart + target_line, use_container_width=True)

            current_acc = trajectory["current_accuracy"]
            predicted_acc = trajectory["predicted_accuracy"]

            # Simple logistic mapping from predicted accuracy to "probability of passing"
            prob_pass = 1 / (1 + np.exp(-12 * (predicted_acc - 0.75)))

            st.subheader("📌 Prediction Summary")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Current Accuracy", f"{current_acc*100:.1f}%")
            with c2:
                st.metric("Predicted Exam Accuracy", f"{predicted_acc*100:.1f}%")
            with c3:
                st.metric("Days Until Exam", days_until_exam)
            with c4:
                st.metric("Probability of Passing", f"{prob_pass*100:.1f}%")

            # Daily improvement needed
            gap_to_target = TARGET_ACCURACY - current_acc
            req_daily = trajectory["required_daily_improvement"]
            if days_until_exam > 0:
                if predicted_acc >= TARGET_ACCURACY:
                    st.success("🎉 Based on current trend, Elisa is on track to meet or exceed the target by exam day.")
                else:
                    st.warning(
                        f"At the current pace, Elisa is predicted to reach about {predicted_acc*100:.1f}% accuracy.\n"
                        f"She needs roughly {req_daily*100:.2f}% extra improvement per day to hit the {TARGET_ACCURACY*100:.0f}% target."
                    )
            else:
                st.info("Exam date has passed or is today — predictions beyond this point are less meaningful.")

            st.markdown("---")

            # Topic-level mastery visualisation
            st.subheader("🧠 Topic Mastery (by Quiz Topic)")
            mastery = summarize_mastery(df)
            if not mastery.empty:
                mastery_chart = alt.Chart(mastery).mark_bar().encode(
                    x=alt.X("topic:N", title="Topic"),
                    y=alt.Y("accuracy:Q", title="Average Accuracy", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("mastery:N", title="Mastery Level"),
                    tooltip=["topic", alt.Tooltip("accuracy:Q", format=".1%"), "mastery"]
                ).properties(
                    height=350,
                    title="Average Accuracy by Topic"
                )
                st.altair_chart(mastery_chart, use_container_width=True)
                st.dataframe(
                    mastery[["topic", "accuracy", "seconds", "mastery"]].sort_values("accuracy"),
                    use_container_width=True
                )
            else:
                st.info("Not enough topic-level data to compute mastery.")

            st.markdown("---")

            # Natural language summary for parents
            st.subheader("📝 Parent Insight Summary")
            overall = metrics.get("overall_accuracy", 0)
            recent = metrics.get("recent_accuracy", 0)
            improvement_7d = metrics.get("improvement_7d", 0)
            consistency = metrics.get("consistency", 0)

            summary_lines = []

            summary_lines.append(
                f"- Elisa's **overall accuracy** so far is **{overall*100:.1f}%**, "
                f"with **{metrics.get('total_quizzes', 0)}** quizzes completed."
            )
            summary_lines.append(
                f"- In the **last 7 days**, her average accuracy is **{recent*100:.1f}%**."
            )
            if improvement_7d > 0:
                summary_lines.append(
                    f"- Compared to earlier weeks, she has improved by roughly **{improvement_7d*100:.1f}%** over the last week."
                )
            elif improvement_7d < 0:
                summary_lines.append(
                    f"- There is a slight dip of about **{abs(improvement_7d)*100:.1f}%** compared to earlier weeks — "
                    "a great opportunity to revisit a few trickier topics."
                )
            else:
                summary_lines.append(
                    "- Her performance has been **stable** over the last week."
                )

            summary_lines.append(
                f"- Her performance **consistency** (how variable scores are) is around **{consistency*100:.1f}%** "
                "(higher is more consistent)."
            )

            if predicted_acc >= TARGET_ACCURACY:
                summary_lines.append(
                    f"- If she continues at this pace, she is **on track** to meet or exceed the **{TARGET_ACCURACY*100:.0f}%** target by September."
                )
            else:
                summary_lines.append(
                    f"- At the current pace, she may reach around **{predicted_acc*100:.1f}%** by the exam. "
                    "A few extra focused sessions on weaker topics could significantly boost this."
                )

            st.markdown("\n".join(summary_lines))

# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 14px; padding: 20px 0;">
    Made with ❤️ for Elisaveta | Streamlit Cloud | JSON Storage | Mobile Optimized
</div>
""", unsafe_allow_html=True)
