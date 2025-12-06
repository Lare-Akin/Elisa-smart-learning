import streamlit as st
import os, random, csv, datetime, time
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
from pathlib import Path

# ---------- Mobile Detection ----------
def is_mobile():
    """Detect mobile device based on screen width"""
    try:
        # Try to get screen width from query params (for testing)
        # In production, we'll use CSS media queries and JavaScript
        return st.session_state.get('is_mobile', False)
    except:
        return False

# ---------- Mobile CSS Injector ----------
def inject_mobile_css():
    """Inject mobile-optimized CSS with dynamic viewport"""
    mobile_css = """
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
    /* Base mobile-first styles */
    * {
        box-sizing: border-box;
    }
    
    body, .stApp {
        background-color: #000 !important;
        color: #fff !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        -webkit-tap-highlight-color: transparent;
        -webkit-text-size-adjust: 100%;
    }
    
    /* Minimum touch target size (Apple HIG: 44x44px) */
    .stButton > button, 
    .stDownloadButton > button,
    .stRadio > div > label,
    .stSelectbox > div > div,
    .stSlider > div {
        min-height: 44px !important;
        min-width: 44px !important;
        touch-action: manipulation;
    }
    
    /* Mobile-optimized buttons */
    .stButton > button {
        background-color: #2E8B57 !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 24px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        margin: 8px 0 !important;
        width: 100% !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #3DA56C !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Mobile-optimized radio buttons */
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
        border-radius: 8px !important;
        transition: background 0.3s ease !important;
    }
    
    .stRadio > div > label:hover {
        background: #2a2a2a !important;
    }
    
    .stRadio > div > label:active {
        background: #3a3a3a !important;
    }
    
    /* Mobile sidebar optimization */
    @media (max-width: 768px) {
        .sidebar .sidebar-content {
            padding: 20px 15px !important;
        }
        
        [data-testid="stSidebar"] {
            min-width: 280px !important;
            max-width: 300px !important;
        }
        
        .sidebar .stButton > button {
            margin: 12px 0 !important;
            font-size: 16px !important;
            padding: 12px 20px !important;
        }
        
        .sidebar .stRadio > div,
        .sidebar .stSelectbox > div,
        .sidebar .stNumberInput > div,
        .sidebar .stSlider > div {
            margin: 16px 0 !important;
        }
    }
    
    /* Mobile-responsive columns */
    @media (max-width: 768px) {
        .stHorizontalBlock > div,
        [data-testid="column"] {
            flex: 0 0 100% !important;
            width: 100% !important;
            margin-bottom: 16px !important;
        }
        
        /* Stack metrics vertically on mobile */
        [data-testid="metric-container"] {
            margin: 8px 0 !important;
            padding: 16px !important;
            border-radius: 12px !important;
            border: 2px solid #333 !important;
            background: #1a1a1a !important;
        }
    }
    
    /* Mobile-optimized progress bars */
    .stProgress > div > div {
        background: #2E8B57 !important;
        height: 20px !important;
        border-radius: 10px !important;
    }
    
    /* Mobile-optimized tables */
    .dataframe {
        font-size: 15px !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 2px solid #333 !important;
    }
    
    .dataframe thead th {
        background: #2E8B57 !important;
        color: white !important;
        padding: 16px !important;
        font-size: 16px !important;
    }
    
    .dataframe tbody td {
        padding: 14px !important;
        border-bottom: 1px solid #333 !important;
    }
    
    /* Mobile-optimized headers */
    h1 {
        font-size: 28px !important;
        line-height: 1.3 !important;
        margin-bottom: 24px !important;
        text-align: center !important;
    }
    
    h2 {
        font-size: 24px !important;
        line-height: 1.4 !important;
        margin: 20px 0 16px 0 !important;
    }
    
    h3 {
        font-size: 20px !important;
        line-height: 1.4 !important;
        margin: 18px 0 14px 0 !important;
    }
    
    @media (max-width: 768px) {
        h1 {
            font-size: 24px !important;
            margin-bottom: 20px !important;
        }
        
        h2 {
            font-size: 20px !important;
            margin: 16px 0 12px 0 !important;
        }
        
        h3 {
            font-size: 18px !important;
            margin: 14px 0 10px 0 !important;
        }
    }
    
    /* NVR shapes for mobile */
    .nvr-shape {
        font-size: 32px !important;
        text-align: center !important;
        margin: 24px 0 !important;
        line-height: 1.8 !important;
        letter-spacing: 8px !important;
    }
    
    @media (max-width: 768px) {
        .nvr-shape {
            font-size: 28px !important;
            letter-spacing: 6px !important;
            margin: 20px 0 !important;
        }
    }
    
    /* Mobile-optimized quiz container */
    .quiz-container {
        background: #1a1a1a !important;
        border-radius: 16px !important;
        padding: 20px !important;
        margin: 16px 0 !important;
        border: 2px solid #2E8B57 !important;
    }
    
    @media (max-width: 768px) {
        .quiz-container {
            padding: 16px !important;
            margin: 12px 0 !important;
        }
    }
    
    /* Mobile-optimized input fields */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div {
        font-size: 18px !important;
        padding: 14px 16px !important;
        border-radius: 10px !important;
        border: 2px solid #333 !important;
        background: #2a2a2a !important;
        color: white !important;
        min-height: 48px !important;
    }
    
    /* Mobile alerts and messages */
    .stAlert {
        border-radius: 12px !important;
        padding: 16px 20px !important;
        margin: 12px 0 !important;
        border: 2px solid !important;
        font-size: 16px !important;
    }
    
    .stSuccess {
        background: #1a3a1a !important;
        border-color: #2E8B57 !important;
        color: #90EE90 !important;
    }
    
    .stError {
        background: #3a1a1a !important;
        border-color: #8B2E2E !important;
        color: #FF6B6B !important;
    }
    
    .stInfo {
        background: #1a2a3a !important;
        border-color: #2E578B !important;
        color: #87CEEB !important;
    }
    
    /* Mobile-optimized plotly charts */
    .js-plotly-plot {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 2px solid #333 !important;
    }
    
    /* Hide sidebar on mobile by default, show toggle */
    @media (max-width: 768px) {
        [data-testid="collapsedControl"] {
            display: block !important;
        }
        
        /* Make sidebar overlay full screen on mobile */
        [data-testid="stSidebar"] {
            top: 0 !important;
            height: 100vh !important;
        }
    }
    
    /* Mobile-safe scrolling */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
    }
    
    /* Mobile keyboard optimization */
    input, textarea, select {
        font-size: 16px !important; /* Prevents iOS zoom */
    }
    
    /* Smooth transitions */
    * {
        -webkit-transition: all 0.3s ease;
        transition: all 0.3s ease;
    }
    </style>
    
    <script>
    // Detect mobile device and store in session
    function detectMobile() {
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        const viewportWidth = window.innerWidth;
        
        // Consider mobile if width < 768 or device is mobile
        const isMobileDevice = isMobile || viewportWidth < 768;
        
        // Update CSS variables for mobile
        if (isMobileDevice) {
            document.body.classList.add('mobile-device');
            // Store in localStorage for Streamlit to potentially access
            localStorage.setItem('is_mobile', 'true');
        }
        
        // Set viewport height for mobile browsers
        document.documentElement.style.setProperty('--vh', window.innerHeight * 0.01 + 'px');
    }
    
    // Run on load and resize
    window.addEventListener('load', detectMobile);
    window.addEventListener('resize', detectMobile);
    window.addEventListener('orientationchange', detectMobile);
    
    // Prevent double-tap zoom
    document.addEventListener('touchstart', function(event) {
        if (event.touches.length > 1) {
            event.preventDefault();
        }
    }, { passive: false });
    
    // Prevent pull-to-refresh on mobile
    document.body.style.overscrollBehavior = 'none';
    </script>
    """
    st.markdown(mobile_css, unsafe_allow_html=True)

# ---------- Configuration ----------
# Use relative paths for deployment
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CSV_FILE = DATA_DIR / "11_Plus_Exam_Prep.csv"
LOG_FILE = BASE_DIR / "results.csv"

QUESTION_LIMIT = 25
TARGET_EXAM_DATE = datetime.date(2026, 9, 15)
TARGET_ACCURACY = 0.85  # 85%

# ---------- File checking helper ----------
def setup_directories():
    """Create necessary directories if they don't exist"""
    DATA_DIR.mkdir(exist_ok=True)
    
# Call this at startup
setup_directories()

# ---------- NVR Visual Enhancement Function ----------
def enhance_nvr_display(text):
    """Replace shape names with Unicode symbols"""
    shape_map = {
        'circle': '●',
        'square': '■',
        'triangle': '▲',
        'hexagon': '⬢',
        'pentagon': '⬟',
        'octagon': '⬡',
        'filled circle': '●',
        'empty circle': '○',
        'filled square': '■', 
        'empty square': '□',
        'filled triangle': '▲',
        'empty triangle': '△',
        'up': '↑',
        'down': '↓', 
        'left': '←',
        'right': '→',
        'shaded': '▓',
        'unshaded': '▒',
        'thick line': '━━',
        'thin line': '──',
        'dot': '•',
        'large dot': '●',
        'small dot': '•',
        'star': '★',
        'heart': '♥'
    }
    
    enhanced_text = str(text)
    for word, symbol in shape_map.items():
        enhanced_text = enhanced_text.replace(word, symbol)
    
    return enhanced_text

# ---------- Logging ----------
def write_log_header_if_needed():
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["date", "module", "topic", "score", "total", "seconds"])

def log_result(module, topic, score, total, seconds):
    write_log_header_if_needed()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.date.today().isoformat(),
            module, topic, score, total, seconds
        ])

# ---------- Data helpers ----------
def check_csv_file() -> bool:
    if not CSV_FILE.exists():
        return False
    try:
        df = pd.read_csv(CSV_FILE)
        required_columns = ["Type", "Question", "Option1", "Option2", "Option3", "Option4", "Answer"]
        return all(col in df.columns for col in required_columns)
    except Exception as e:
        st.error(f"Error checking CSV: {str(e)}")
        return False

def load_questions(question_type=None, limit=None) -> List[Dict[str, Any]]:
    if not CSV_FILE.exists():
        st.error(f"❌ CSV file not found: {CSV_FILE}")
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
                st.error(f"❌ No questions found for type: {question_type}")
                return []

        questions = []
        for _, row in df.iterrows():
            question = {
                "q": row["Question"],
                "options": [row["Option1"], row["Option2"], row["Option3"], row["Option4"]],
                "answer": row["Answer"],
            }
            questions.append(question)

        if limit and len(questions) > limit:
            return random.sample(questions, limit)
        return questions
    except Exception as e:
        st.error(f"❌ Error loading CSV file: {str(e)}")
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
        st.warning(f"Could not sanitize results file: {e}")

def load_results(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(columns=["date", "module", "topic", "score", "total", "seconds", "accuracy"])

    sanitize_results_file(log_path)
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

def calculate_performance_metrics(df: pd.DataFrame):
    if df.empty:
        return {}
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

# Inject mobile CSS first
inject_mobile_css()

# App title with mobile-friendly design
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="color: #2E8B57; font-size: 2.5rem; margin-bottom: 10px;">🧠 11+ Practice App</h1>
    <p style="color: #ccc; font-size: 1.2rem;">Mobile-friendly learning for Elisaveta</p>
</div>
""", unsafe_allow_html=True)

# CSV presence check
if not check_csv_file():
    st.error(f"❌ CSV file not found or invalid: {CSV_FILE}")
    st.info("Please ensure the CSV file exists in the data folder with the correct format.")
    st.stop()

# ---------- Mode Selector ----------
mode_type = st.sidebar.radio("Choose Mode", ["Kid Mode", "Parent Mode"], horizontal=True)
if mode_type == "Kid Mode":
    page = st.sidebar.radio("Choose view", ["Quiz", "My Progress"], horizontal=True)
else:
    page = st.sidebar.radio("Choose view", ["Quiz", "Dashboard", "Data Info", "Predictive Analytics"], horizontal=True)

# ---------- Mobile-optimized sidebar ----------
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
        st.success(f"✅ CSV file loaded successfully")

        st.markdown("### Dataset Overview")
        cols = st.columns(2)  # 2 columns on mobile instead of 4
        with cols[0]:
            st.metric("Total Questions", len(df))
            st.metric("Maths Questions", len(df[df["Type"] == "Maths"]))
        with cols[1]:
            st.metric("Vocabulary Questions", len(df[df["Type"] == "Vocabulary"]))
            st.metric("Verbal Reasoning", len(df[df["Type"] == "Verbal Reasoning"]))

        st.markdown("### Sample Questions")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("### Question Type Distribution")
        type_counts = df["Type"].value_counts()
        st.bar_chart(type_counts)
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")

# ---------- Quiz ----------
elif page == "Quiz":
    if "questions" not in st.session_state:
        reset_session()

    modules = ["Maths", "Vocabulary", "Verbal Reasoning", "NVR", "Mixed (All available)"]

    if mode_type == "Parent Mode":
        try:
            df = pd.read_csv(CSV_FILE)
            st.sidebar.markdown("### 📊 Data Overview")
            total_questions = len(df)
            st.sidebar.info(f"Total questions: {total_questions}")
        except:
            st.sidebar.warning("Could not load data stats")

    # Mobile-optimized quiz controls
    with st.sidebar:
        st.markdown("### Quiz Settings")
        mode = st.selectbox("Choose quiz", modules)
        num_questions = st.slider("Number of questions", 5, QUESTION_LIMIT, 10)
        per_question_seconds = st.number_input("Time limit per question (sec)", min_value=0, value=0, step=5)
        start_button = st.button("🎮 Start / Restart Quiz", use_container_width=True)

    if start_button:
        questions = load_questions(mode, limit=num_questions)
        if not questions:
            st.sidebar.error("❌ Could not load questions. Check if CSV file exists and has data.")
            st.stop()

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

    if st.session_state.questions:
        idx = st.session_state.q_index
        total = len(st.session_state.questions)
        elapsed = int(time.time() - st.session_state.start_time) if st.session_state.start_time else 0

        # Mobile-optimized progress display
        if mode_type == "Kid Mode":
            st.markdown('<div class="quiz-container">', unsafe_allow_html=True)
            st.markdown("### 🌟 Your Adventure Progress")
            st.progress(idx / total)
            
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"**Question {idx+1} of {total}** 🎯")
            with cols[1]:
                st.markdown(f"⭐ **Score:** {st.session_state.score}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="quiz-container">', unsafe_allow_html=True)
            st.markdown("### 📊 Progress Dashboard")
            cols = st.columns(2)  # 2 columns on mobile
            with cols[0]:
                st.metric("Question", f"{idx + 1} / {total}")
                st.metric("Score", f"{st.session_state.score}")
            with cols[1]:
                st.metric("Time Elapsed", f"{elapsed} sec")
                st.metric("Module", mode)
            st.markdown('</div>', unsafe_allow_html=True)

        current = st.session_state.questions[idx]
        
        # Question display
        st.markdown('<div class="quiz-container">', unsafe_allow_html=True)
        st.markdown(f"### **Question {idx + 1}**")
        
        current_question = current.get("q", "")
        is_nvr_question = (
            "non-verbal" in str(current_question).lower() or 
            mode == "NVR" or
            any(shape in str(current_question).lower() for shape in [
                'circle', 'square', 'triangle', 'arrow', 'shaded', 'unshaded', 
                'filled', 'empty', 'rotation', 'mirror', 'reflect', 'shape'
            ])
        )
        
        if is_nvr_question:
            visual_question = enhance_nvr_display(current_question)
            st.markdown(f"**{visual_question}**")
            st.markdown('<div class="nvr-shape">🔺 🟦 ⬛ ⚫ 🔶</div>', unsafe_allow_html=True)
            raw_options = [str(o) for o in current.get("options", [])]
            options = [enhance_nvr_display(opt) for opt in raw_options]
        else:
            st.write(current_question)
            options = [str(o) for o in current.get("options", [])]

        # Mobile-optimized answer selection
        question_key = f"q{idx}_{hash(current.get('q', '')) % 10000}"
        if question_key not in st.session_state.get("shuffled_options", {}):
            if "shuffled_options" not in st.session_state:
                st.session_state.shuffled_options = {}
            shuffled = options.copy()
            random.shuffle(shuffled)
            st.session_state.shuffled_options[question_key] = shuffled
        else:
            shuffled = st.session_state.shuffled_options[question_key]

        choice_key = f"choice_{idx}"
        current_answer = st.session_state.user_answers[idx]
        default_index = shuffled.index(current_answer) if (current_answer in shuffled) else 0

        st.markdown("### Select your answer:")
        selected_answer = st.radio(
            "",
            shuffled,
            index=default_index,
            key=choice_key,
            label_visibility="collapsed"
        )
        st.session_state.user_answers[idx] = selected_answer
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Countdown timer (mobile-optimized)
        timed_out = False
        if per_question_seconds > 0:
            if st.session_state.get("question_start") is None:
                st.session_state["question_start"] = time.time()
            elapsed_q = int(time.time() - st.session_state["question_start"])
            remaining = per_question_seconds - elapsed_q
            
            if remaining <= 0:
                timed_out = True
                st.error("⏱️ Time's up!" if mode_type != "Kid Mode" else "⏱️ Time's up! Let's zoom to the next one 🚀")
            else:
                progress = remaining / per_question_seconds
                st.progress(progress)
                st.info(f"**Time remaining:** {remaining} seconds")
            
            if hasattr(st, "experimental_autorefresh"):
                st.experimental_autorefresh(interval=1000, key=f"countdown_{idx}")

        # Mobile-optimized action buttons (vertical stack on mobile)
        action_cols = st.columns(3)
        with action_cols[0]:
            next_btn = st.button("➡️ Next", use_container_width=True)
        with action_cols[1]:
            skip = st.button("⏭️ Skip", use_container_width=True, type="secondary")
        with action_cols[2]:
            finish = st.button("🏁 Finish", use_container_width=True, type="secondary")

        action_taken = None
        if next_btn:
            action_taken = "next"
        elif skip:
            action_taken = "skip"
        elif finish:
            action_taken = "finish"

        if action_taken == "next":
            if st.session_state.user_answers[idx] is not None:
                correct = str(current.get("answer")).strip().lower()
                selected = str(st.session_state.user_answers[idx]).strip().lower()
                if timed_out:
                    st.error("Too slow — marked incorrect." if mode_type != "Kid Mode" else "⏱️ Ran out of time — that's okay! Ready for the next one?")
                elif selected == correct:
                    st.session_state.score += 1
                    st.success("✅ Correct!" if mode_type != "Kid Mode" else "🎉 Brilliant! You got it right! 🌟")
                else:
                    if mode_type == "Kid Mode":
                        correct_answer = current.get('answer')
                        if is_nvr_question:
                            correct_answer = enhance_nvr_display(correct_answer)
                        st.error(f"❌ Not quite — the right answer was **{correct_answer}**. Keep going, you're learning! 💪")
                    else:
                        st.error(f"❌ Incorrect. Correct answer: {current.get('answer')}")
            else:
                st.warning("Please select an answer" if mode_type != "Kid Mode" else "Please select an answer before clicking Next")
                action_taken = None

        if action_taken == "next":
            st.session_state.question_start = time.time()
            if idx < total - 1:
                st.session_state.q_index += 1
                st.rerun()
            else:
                st.session_state.finished = True
        elif action_taken == "skip":
            if idx < total - 1:
                st.session_state.q_index += 1
                st.session_state.question_start = time.time()
                st.rerun()
            else:
                st.session_state.finished = True
        elif action_taken == "finish":
            st.session_state.finished = True

        # Completion
        if st.session_state.finished and not st.session_state.get("quiz_completed", False):
            duration = int(time.time() - st.session_state.start_time)
            score = st.session_state.score
            log_result(st.session_state.get("module_label", "unknown"), mode, score, total, duration)
            st.session_state.quiz_completed = True

            st.markdown("---")
            if mode_type == "Kid Mode":
                st.markdown("""
                <div style="text-align: center; background: linear-gradient(135deg, #2E8B57, #3DA56C); 
                         padding: 30px; border-radius: 20px; margin: 20px 0;">
                    <h2 style="color: white; margin-bottom: 20px;">🎉 Amazing job, Elisaveta! 🌟</h2>
                    <p style="font-size: 24px; color: white;">You scored <strong>{score} out of {total}</strong> 🏆</p>
                    <p style="font-size: 20px; color: white;">⏱️ Time: <strong>{duration} seconds</strong></p>
                </div>
                """.format(score=score, total=total, duration=duration), unsafe_allow_html=True)

                accuracy = score / total if total > 0 else 0
                days_until_exam = (TARGET_EXAM_DATE - datetime.date.today()).days
                
                if accuracy >= TARGET_ACCURACY:
                    st.success(f"🌈 You're already at your target! {accuracy:.1%} accuracy! 🎯")
                else:
                    st.info(f"🎯 Goal: {TARGET_ACCURACY:.0%} by September 2026 ({days_until_exam} days to go!)")
                
                if score == total or (total >= 5 and score / total >= 0.8):
                    st.balloons()
                    st.success("🌈 Fantastic work! You're a superstar! ⭐")
            else:
                st.subheader("🎉 Quiz Complete")
                st.write(f"**Score:** {score} / {total}")
                st.write(f"**Duration:** {duration} seconds")
                st.button("Restart Quiz", use_container_width=True)
    else:
        # Welcome screen - mobile optimized
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #2E8B57, #3DA56C);
            padding: 40px 20px;
            border-radius: 20px;
            text-align: center;
            border: 3px solid #FFFFFF;
            margin: 20px 0;
        ">
            <h2 style="color: white; font-size: 2rem; margin-bottom: 20px;">
                Welcome Elisaveta! 🙂<br>Добре дошли Елисавета! 🙂
            </h2>
            <p style="color: white; font-size: 1.4rem; margin-bottom: 30px;">
                Choose a topic in the sidebar and click<br>
                <strong>"Start / Restart Quiz"</strong> to begin!
            </p>
            <div style="font-size: 3rem; margin-top: 20px;">
                🧠 📚 ✏️ 🎯
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------- My Progress (Kid Mode) ----------
elif page == "My Progress" and mode_type == "Kid Mode":
    st.title("🌟 My Learning Journey")
    df = load_results(LOG_FILE)
    
    if df.empty:
        st.info("No quiz data yet. Complete some quizzes to see your progress! 🚀")
    else:
        days_until_exam = (TARGET_EXAM_DATE - datetime.date.today()).days
        accuracy = df["accuracy"].mean()

        # Mobile-optimized metrics
        cols = st.columns(2)  # 2 columns on mobile
        with cols[0]:
            st.metric("Total Quizzes", len(df))
            st.metric("Overall Accuracy", f"{accuracy:.1%}")
        with cols[1]:
            st.metric("Days Until Exam", days_until_exam)
            target_status = "🎯 On Track" if accuracy >= TARGET_ACCURACY else "📈 Keep Going"
            st.metric("Goal Status", target_status)

        st.subheader("📈 My Progress Over Time")
        weekly_avg = df.groupby(pd.Grouper(key="date", freq="W"))["accuracy"].mean().reset_index()
        if not weekly_avg.empty:
            fig = px.line(weekly_avg, x="date", y="accuracy", title="Weekly Average Accuracy",
                          labels={"accuracy": "Accuracy", "date": "Date"})
            fig.add_hline(y=TARGET_ACCURACY, line_dash="dash", line_color="red",
                          annotation_text=f"Target: {TARGET_ACCURACY:.0%}")
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(
                font=dict(size=16),
                height=400  # Fixed height for mobile
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Performance by Topic")
        topic_performance = df.groupby("topic")["accuracy"].mean().sort_values(ascending=False)
        
        for topic, acc in topic_performance.items():
            with st.container():
                cols = st.columns([3, 1])
                with cols[0]:
                    st.write(f"**{topic}**")
                with cols[1]:
                    st.write(f"{acc:.1%}")
                st.progress(min(1.0, acc))

        st.subheader("💫 Keep Going!")
        if accuracy >= TARGET_ACCURACY:
            st.success(f"""
            🎉 **Amazing!** You're already meeting your target of {TARGET_ACCURACY:.0%}!
            
            Keep up the great work! 🌟
            """)
        else:
            improvement_needed = TARGET_ACCURACY - accuracy
            st.info(f"""
            🎯 **Goal Status:** You're {improvement_needed:.1%} away from your target.
            
            **Tip:** Practice your weakest topics daily to improve faster! 💪
            """)

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

        st.markdown("#### 📊 Topic Summary")
        summary = summarize_mastery(filtered_df)
        st.dataframe(
            summary.style.format({"accuracy": "{:.1%}", "seconds": "{:.1f}"}),
            use_container_width=True
        )

        st.markdown("### 📈 Mastery Over Time")
        mastery_df = prepare_mastery_over_time(filtered_df)
        if not mastery_df.empty:
            pivoted = mastery_df.pivot(index="date", columns="topic", values="accuracy")
            st.line_chart(pivoted)
        else:
            st.info("Not enough data to show mastery evolution.")

        st.markdown("### 📤 Export Insights")
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results as CSV",
            data=csv_data,
            file_name="quiz_results_filtered.csv",
            mime="text/csv",
            use_container_width=True
        )

# ---------- Predictive Analytics ----------
elif page == "Predictive Analytics" and mode_type == "Parent Mode":
    st.title("🔮 Predictive Analytics & Progress Tracking")
    try:
        df = load_results(LOG_FILE)
        
        if df.empty:
            st.info("No quiz data available yet. Complete some quizzes to see analytics.")
        else:
            metrics = calculate_performance_metrics(df)
            trajectory = calculate_learning_trajectory(df)

            st.subheader("📊 Current Performance Overview")
            cols = st.columns(2)  # 2 columns on mobile
            with cols[0]:
                st.metric("Overall Accuracy", f"{metrics['overall_accuracy']:.1%}" if metrics['overall_accuracy'] > 0 else "0%")
                st.metric("Recent Accuracy (7d)", f"{metrics['recent_accuracy']:.1%}" if metrics['recent_accuracy'] > 0 else "0%")
            with cols[1]:
                st.metric("Best Score", f"{metrics['best_score']:.1%}" if metrics['best_score'] > 0 else "0%")
                st.metric("Total Questions", metrics["total_questions"])

            cols = st.columns(2)
            with cols[0]:
                st.metric("Consistency", f"{metrics['consistency']:.2f}")
            with cols[1]:
                st.metric("7-Day Improvement", f"{metrics['improvement_7d']:+.1%}")

            if trajectory:
                cols = st.columns(2)
                with cols[0]:
                    st.metric("Days Until Exam", trajectory["days_until_exam"])
                with cols[1]:
                    current_vs_target = "🎯 On Track" if trajectory["predicted_accuracy"] >= TARGET_ACCURACY else "⚠️ Needs Improvement"
                    st.metric("Target Status", current_vs_target)

                st.subheader("📈 Learning Trajectory")
                
                # Progress bars
                current_accuracy = trajectory["current_accuracy"]
                predicted_accuracy = trajectory["predicted_accuracy"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Current Accuracy**")
                    st.progress(current_accuracy)
                    st.metric("Current", f"{current_accuracy:.1%}")
                with col2:
                    st.markdown("**Predicted Exam Accuracy**")
                    st.progress(predicted_accuracy)
                    st.metric("Predicted", f"{predicted_accuracy:.1%}")

                # Forecast summary
                st.subheader("🎯 Forecast Summary")
                forecast_data = pd.DataFrame({
                    "Metric": [
                        "Days Until Exam",
                        "Current Accuracy", 
                        "Predicted Exam Accuracy",
                        "Required Daily Improvement",
                        "Model Confidence (R²)"
                    ],
                    "Value": [
                        f"{trajectory['days_until_exam']} days",
                        f"{current_accuracy:.1%}",
                        f"{predicted_accuracy:.1%}",
                        f"{trajectory['required_daily_improvement']:+.4%}",
                        f"{trajectory['r_squared']:.3f}"
                    ]
                })
                st.dataframe(forecast_data, use_container_width=True, hide_index=True)

                # Weekly performance
                st.subheader("📅 Weekly Performance")
                weekly_avg = df.groupby(pd.Grouper(key="date", freq="W"))["accuracy"].mean().reset_index()
                weekly_avg["Week"] = weekly_avg["date"].dt.strftime("%Y-%m-%d")
                weekly_avg = weekly_avg.rename(columns={"accuracy": "Accuracy"})
                weekly_avg["Accuracy"] = weekly_avg["Accuracy"].apply(lambda x: f"{x:.1%}")
                st.dataframe(weekly_avg[["Week", "Accuracy"]], use_container_width=True, hide_index=True)

                # Recommendations
                st.subheader("💡 Recommended Actions")
                if trajectory["predicted_accuracy"] < TARGET_ACCURACY:
                    st.warning("""
                    **Areas for Improvement:**
                    - Increase practice frequency to 20-30 minutes daily
                    - Focus on weaker topics identified in dashboard
                    - Review incorrect answers thoroughly
                    - Consider timed practice sessions
                    """)
                else:
                    st.success("""
                    **Maintenance Strategy:**
                    - Maintain current practice schedule (15-20 minutes daily)
                    - Continue mixed topic practice
                    - Focus on speed and accuracy
                    - Regular review of challenging concepts
                    """)
            else:
                st.warning("""
                **Need more data for predictions:**
                - Complete at least 3 quizzes on different days
                - Try different topics to build comprehensive data
                """)

    except Exception as e:
        st.error(f"Error loading analytics data: {str(e)}")
        st.info("Try completing a new quiz to generate fresh data.")

# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 14px; padding: 20px 0;">
    Made with ❤️ for Elisaveta | Mobile Optimized | Streamlit Cloud
</div>
""", unsafe_allow_html=True)