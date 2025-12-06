🧠 Elisa Smart Learning App

Welcome to the Elisa Smart Learning App — a mobile-friendly, interactive quiz platform designed to help children prepare for the UK 11+ exams in a joyful and data-informed way. Built with Streamlit, this app blends engaging quiz mechanics with predictive analytics to support both learners and parents.

🎯 Purpose

This app was created to support Elisaveta's learning journey by offering:

Fun, accessible quizzes across key 11+ subjects

Visual feedback and progress tracking

Predictive insights to guide long-term mastery

It’s optimized for mobile use and designed with emotional comfort, clarity, and accessibility in mind.

✨ Features

👧 Kid Mode

Quiz Adventure: Choose from Maths, Vocabulary, Verbal Reasoning, NVR, or Mixed mode

Progress Tracker: See weekly accuracy trends and countdown to exam day

Visual Enhancements: NVR questions use Unicode shapes for intuitive display

👨‍👩‍👧 Parent Mode

Dashboard: Topic-level mastery summaries with accuracy and timing

Data Info: View question distribution and sample questions

Predictive Analytics: Forecast future accuracy and daily improvement needs

📊 Analytics

Accuracy trends over time

Weekly and monthly performance metrics

Linear regression-based learning trajectory

🚀 Setup Instructions

1. Clone the repository

git clone https://github.com/your-username/Elisa-smart-learning.git
cd Elisa-smart-learning

2. Install dependencies

Ensure you have Python 3.9+ (Streamlit Cloud uses Python 3.13).

pip install -r requirements.txt

3. Add your data

Place your question CSV file in the data/ folder:

data/11_Plus_Exam_Prep.csv

Required columns:

Type, Question, Option1, Option2, Option3, Option4, Answer

4. Run the app

streamlit run app.py

📁 Repo Structure

Elisa-smart-learning/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── data/
│   └── 11_Plus_Exam_Prep.csv
├── .gitignore
└── README.md

🛡️ License

This project is licensed under the CC0-1.0 license — feel free to use, remix, and share.

💡 Credits

Created with love by Lare-Akin, blending data science, UX design, and joyful learning.

🙋 Support

For help or suggestions, open an issue or visit Streamlit.io.

Enjoy learning — and good luck, Elisaveta! 🌟
