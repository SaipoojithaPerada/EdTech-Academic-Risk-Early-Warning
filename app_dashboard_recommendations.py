import os
import glob
import joblib
import pandas as pd
import streamlit as st

# Optional plotting library
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Academic Risk Early Warning System",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Academic Risk Early Warning System")
st.caption(
    "Machine-learning based academic performance prediction, "
    "risk monitoring and personalized student recommendations."
)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_CANDIDATES = [
    os.path.join(BASE_DIR, "dataset", "student_performance.csv"),
    os.path.join(BASE_DIR, "dataset", "StudentsPerformance.csv"),
    os.path.join(BASE_DIR, "student_performance.csv"),
    os.path.join(BASE_DIR, "StudentsPerformance.csv"),
]

MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "model", "model.pkl"),
    os.path.join(BASE_DIR, "model", "decision_tree_model.pkl"),
    os.path.join(BASE_DIR, "model", "decision_tree.pkl"),
    os.path.join(BASE_DIR, "model", "student_risk_model.pkl"),
    os.path.join(BASE_DIR, "model.pkl"),
    os.path.join(BASE_DIR, "decision_tree_model.pkl"),
]

def find_existing_file(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path

    # Search the project recursively for common model files
    for pattern in ["*.pkl", "*.joblib"]:
        matches = glob.glob(os.path.join(BASE_DIR, "**", pattern), recursive=True)
        if matches:
            return matches[0]

    return None

DATASET_PATH = next(
    (p for p in DATASET_CANDIDATES if os.path.exists(p)),
    None
)
MODEL_PATH = find_existing_file(MODEL_CANDIDATES)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

df = None
model = None

if DATASET_PATH:
    try:
        df = load_dataset(DATASET_PATH)
    except Exception as e:
        st.warning(f"Dataset could not be loaded: {e}")

if MODEL_PATH:
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Model could not be loaded: {e}")

# ---------------------------------------------------------
# EXPECTED MODEL FEATURES
# Your current trained model uses 14 features:
# Gender and FinalGrade were removed.
# ---------------------------------------------------------
FEATURES = [
    "StudyHours",
    "Attendance",
    "Resources",
    "Extracurricular",
    "Motivation",
    "Internet",
    "Age",
    "LearningStyle",
    "OnlineCourses",
    "Discussions",
    "AssignmentCompletion",
    "ExamScore",
    "EduTech",
    "StressLevel"
]

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def risk_from_grade(grade):
    """
    The dataset shows lower FinalGrade values associated with
    better academic performance. Therefore:
      0 -> Low Risk
      1 -> Medium Risk
      2/3 -> High Risk
    """
    grade = int(grade)

    if grade == 0:
        return "LOW RISK", "🟢"
    elif grade == 1:
        return "MEDIUM RISK", "🟡"
    else:
        return "HIGH RISK", "🔴"


def build_recommendations(values, predicted_grade):
    """
    Personalized rule-based recommendations.
    These complement the ML prediction; they do not replace it.
    """
    recommendations = []

    attendance = values["Attendance"]
    assignments = values["AssignmentCompletion"]
    exam = values["ExamScore"]
    study = values["StudyHours"]
    motivation = values["Motivation"]
    stress = values["StressLevel"]
    online = values["OnlineCourses"]
    discussions = values["Discussions"]
    resources = values["Resources"]
    internet = values["Internet"]
    edutech = values["EduTech"]

    # Academic risk related
    if attendance < 75:
        recommendations.append(
            f"📅 Improve attendance from {attendance}% toward at least 75%."
        )

    if assignments < 70:
        recommendations.append(
            f"📝 Increase assignment completion from {assignments}% "
            "by completing pending work on time."
        )

    if exam < 50:
        recommendations.append(
            f"📚 Exam performance is low ({exam}). Schedule regular practice "
            "tests and revision sessions."
        )
    elif exam < 65:
        recommendations.append(
            f"📚 Exam score is {exam}. Increase revision and practice before the next assessment."
        )

    if study < 10:
        recommendations.append(
            "⏰ Increase focused study time gradually and follow a consistent weekly schedule."
        )

    if motivation == 0:
        recommendations.append(
            "🎯 Set small weekly academic goals and track progress to improve motivation."
        )

    if stress >= 2:
        recommendations.append(
            "🧘 Stress level is elevated. Add short breaks, sleep consistently, "
            "and seek academic support when needed."
        )

    if discussions == 0:
        recommendations.append(
            "💬 Participate more in class discussions or peer-learning activities."
        )

    if online < 5:
        recommendations.append(
            "💻 Use additional online learning resources for difficult topics."
        )

    if resources == 0:
        recommendations.append(
            "📖 Make better use of available learning resources and study materials."
        )

    if internet == 0:
        recommendations.append(
            "🌐 Ensure reliable internet access for online learning activities."
        )

    if edutech == 0:
        recommendations.append(
            "🤖 Consider using educational technology tools such as practice quizzes "
            "and interactive learning platforms."
        )

    # If no weaknesses are detected
    if not recommendations:
        if predicted_grade == 0:
            recommendations.append(
                "🌟 Student is performing well. Continue the current learning habits."
            )
        else:
            recommendations.append(
                "👍 Maintain current study habits and monitor academic progress regularly."
            )

    return recommendations


def risk_message(risk):
    if risk == "LOW RISK":
        return "Student is currently showing strong academic indicators."
    elif risk == "MEDIUM RISK":
        return "Student may benefit from targeted academic support and regular monitoring."
    return "Student needs early intervention and closer academic monitoring."


# ---------------------------------------------------------
# BASIC VALIDATION
# ---------------------------------------------------------
if df is not None:
    missing = [f for f in FEATURES + ["FinalGrade"] if f not in df.columns]
else:
    missing = []

if missing:
    st.warning(
        "The loaded dataset does not contain all expected columns: "
        + ", ".join(missing)
    )

if model is None:
    st.error(
        "No trained model file was found. Make sure your saved .pkl or .joblib "
        "model is inside the project folder/model folder."
    )

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["🔮 Student Prediction", "📊 Dashboard", "💡 Personalized Recommendations"]
)

# =========================================================
# TAB 1 - PREDICTION
# =========================================================
with tab1:
    st.header("🔮 Predict Academic Risk")
    st.write("Enter the student's current academic information.")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.number_input(
            "Study Hours",
            min_value=0,
            max_value=100,
            value=20,
            step=1
        )

        attendance = st.number_input(
            "Attendance (%)",
            min_value=0,
            max_value=100,
            value=75,
            step=1
        )

        resources = st.number_input(
            "Resources",
            min_value=0,
            max_value=100,
            value=50,
            step=1
        )

        extracurricular = st.number_input(
            "Extracurricular",
            min_value=0,
            max_value=1,
            value=0,
            step=1,
            help="Use 1 for yes and 0 for no."
        )

        motivation = st.number_input(
            "Motivation",
            min_value=0,
            max_value=1,
            value=1,
            step=1,
            help="Use 1 for motivated and 0 for low motivation."
        )

        internet = st.number_input(
            "Internet",
            min_value=0,
            max_value=1,
            value=1,
            step=1,
            help="Use 1 for available and 0 for unavailable."
        )

        age = st.number_input(
            "Age",
            min_value=10,
            max_value=80,
            value=20,
            step=1
        )

    with col2:
        learning_style = st.number_input(
            "Learning Style",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )

        online_courses = st.number_input(
            "Online Courses",
            min_value=0,
            max_value=100,
            value=10,
            step=1
        )

        discussions = st.number_input(
            "Discussions",
            min_value=0,
            max_value=1,
            value=1,
            step=1,
            help="Use 1 for active participation and 0 for low participation."
        )

        assignment_completion = st.number_input(
            "Assignment Completion (%)",
            min_value=0,
            max_value=100,
            value=75,
            step=1
        )

        exam_score = st.number_input(
            "Exam Score",
            min_value=0,
            max_value=100,
            value=70,
            step=1
        )

        edutech = st.number_input(
            "EduTech Usage",
            min_value=0,
            max_value=1,
            value=1,
            step=1,
            help="Use 1 for yes and 0 for no."
        )

        stress_level = st.number_input(
            "Stress Level",
            min_value=0,
            max_value=5,
            value=1,
            step=1
        )

    values = {
        "StudyHours": study_hours,
        "Attendance": attendance,
        "Resources": resources,
        "Extracurricular": extracurricular,
        "Motivation": motivation,
        "Internet": internet,
        "Age": age,
        "LearningStyle": learning_style,
        "OnlineCourses": online_courses,
        "Discussions": discussions,
        "AssignmentCompletion": assignment_completion,
        "ExamScore": exam_score,
        "EduTech": edutech,
        "StressLevel": stress_level,
    }

    st.divider()

    predict_button = st.button(
        "🔍 Predict Academic Risk",
        type="primary",
        use_container_width=True
    )

    if predict_button:
        if model is None:
            st.error("Model is not available.")
        else:
            input_df = pd.DataFrame([values])[FEATURES]

            try:
                prediction = int(model.predict(input_df)[0])
                risk, icon = risk_from_grade(prediction)

                st.subheader("📊 Prediction Result")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric("Predicted Final Grade", prediction)

                with c2:
                    st.metric("Risk Level", f"{icon} {risk}")

                with c3:
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(input_df)[0]
                        confidence = float(max(probabilities)) * 100
                        st.metric("Prediction Confidence", f"{confidence:.1f}%")
                    else:
                        confidence = None
                        st.metric("Prediction Confidence", "N/A")

                if risk == "LOW RISK":
                    st.success(
                        f"🟢 {risk}: {risk_message(risk)}"
                    )
                elif risk == "MEDIUM RISK":
                    st.warning(
                        f"🟡 {risk}: {risk_message(risk)}"
                    )
                else:
                    st.error(
                        f"🔴 {risk}: {risk_message(risk)}"
                    )

                st.subheader("💡 Immediate Recommendations")

                recs = build_recommendations(values, prediction)

                for rec in recs:
                    st.info(rec)

                # Save latest prediction for recommendation tab
                st.session_state["latest_values"] = values
                st.session_state["latest_prediction"] = prediction
                st.session_state["latest_risk"] = risk

            except Exception as e:
                st.error(
                    "Prediction failed. This usually means the saved model expects "
                    "a different feature order or number of features."
                )
                st.code(str(e))

# =========================================================
# TAB 2 - DASHBOARD
# =========================================================
with tab2:
    st.header("📊 Academic Performance Dashboard")

    if df is None:
        st.warning("Dataset could not be found.")
    else:
        # Dataset overview
        total_students = len(df)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("👨‍🎓 Total Students", f"{total_students:,}")

        with c2:
            st.metric(
                "📅 Avg Attendance",
                f"{df['Attendance'].mean():.1f}%"
            )

        with c3:
            st.metric(
                "📝 Avg Assignment Completion",
                f"{df['AssignmentCompletion'].mean():.1f}%"
            )

        with c4:
            st.metric(
                "📚 Avg Exam Score",
                f"{df['ExamScore'].mean():.1f}"
            )

        st.divider()

        # Grade distribution
        st.subheader("🎓 Final Grade Distribution")

        grade_counts = (
            df["FinalGrade"]
            .value_counts()
            .sort_index()
            .rename_axis("FinalGrade")
            .reset_index(name="Students")
        )

        if PLOTLY_AVAILABLE:
            fig_grade = px.bar(
                grade_counts,
                x="FinalGrade",
                y="Students",
                text="Students",
                title="Number of Students by Final Grade"
            )
            fig_grade.update_layout(xaxis_title="Final Grade", yaxis_title="Students")
            st.plotly_chart(fig_grade, use_container_width=True)
        else:
            st.bar_chart(grade_counts.set_index("FinalGrade"))

        # Risk distribution
        st.subheader("🚦 Academic Risk Distribution")

        risk_counts = df["FinalGrade"].map(
            lambda x: risk_from_grade(x)[0]
        ).value_counts()

        risk_order = ["LOW RISK", "MEDIUM RISK", "HIGH RISK"]
        risk_counts = risk_counts.reindex(risk_order, fill_value=0)

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric("🟢 Low Risk", int(risk_counts["LOW RISK"]))

        with r2:
            st.metric("🟡 Medium Risk", int(risk_counts["MEDIUM RISK"]))

        with r3:
            st.metric("🔴 High Risk", int(risk_counts["HIGH RISK"]))

        if PLOTLY_AVAILABLE:
            risk_df = risk_counts.reset_index()
            risk_df.columns = ["Risk Level", "Students"]

            fig_risk = px.pie(
                risk_df,
                names="Risk Level",
                values="Students",
                title="Academic Risk Levels"
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.bar_chart(risk_counts)

        # Academic indicators by grade
        st.subheader("📈 Academic Indicators by Final Grade")

        avg_by_grade = (
            df.groupby("FinalGrade")[
                ["Attendance", "AssignmentCompletion", "ExamScore", "StudyHours"]
            ]
            .mean()
            .round(2)
        )

        st.dataframe(avg_by_grade, use_container_width=True)

        if PLOTLY_AVAILABLE:
            chart_df = avg_by_grade.reset_index().melt(
                id_vars="FinalGrade",
                var_name="Metric",
                value_name="Average"
            )

            fig_metrics = px.bar(
                chart_df,
                x="FinalGrade",
                y="Average",
                color="Metric",
                barmode="group",
                title="Average Academic Indicators by Grade"
            )
            st.plotly_chart(fig_metrics, use_container_width=True)

        # Scatter relationship
        st.subheader("📊 Attendance vs Exam Score")

        if PLOTLY_AVAILABLE:
            sample_df = df.sample(
                min(2000, len(df)),
                random_state=42
            )

            fig_scatter = px.scatter(
                sample_df,
                x="Attendance",
                y="ExamScore",
                color="FinalGrade",
                hover_data=["AssignmentCompletion", "StudyHours"],
                title="Attendance vs Exam Score"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Install Plotly to display interactive charts: pip install plotly")

# =========================================================
# TAB 3 - PERSONALIZED RECOMMENDATIONS
# =========================================================
with tab3:
    st.header("💡 Personalized Student Recommendations")
    st.write(
        "Recommendations are generated from the student's academic indicators "
        "and the predicted risk level."
    )

    if "latest_values" not in st.session_state:
        st.info(
            "First go to **Student Prediction**, enter student details, "
            "and click **Predict Academic Risk**."
        )
    else:
        latest_values = st.session_state["latest_values"]
        latest_prediction = st.session_state["latest_prediction"]
        latest_risk = st.session_state["latest_risk"]

        risk, icon = risk_from_grade(latest_prediction)

        st.subheader(f"{icon} Current Assessment: {risk}")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Attendance", f"{latest_values['Attendance']}%")

        with c2:
            st.metric(
                "Assignment Completion",
                f"{latest_values['AssignmentCompletion']}%"
            )

        with c3:
            st.metric("Exam Score", latest_values["ExamScore"])

        with c4:
            st.metric("Study Hours", latest_values["StudyHours"])

        st.divider()

        recs = build_recommendations(
            latest_values,
            latest_prediction
        )

        if risk == "LOW RISK":
            st.success(
                "🌟 The student's current indicators are generally positive."
            )
        elif risk == "MEDIUM RISK":
            st.warning(
                "⚠️ The student should be monitored and given targeted support."
            )
        else:
            st.error(
                "🚨 The student should receive early academic intervention."
            )

        for i, rec in enumerate(recs, 1):
            st.write(f"**{i}.** {rec}")

        st.subheader("📌 Priority Actions")

        priority_actions = []

        if latest_values["Attendance"] < 75:
            priority_actions.append("Increase attendance.")

        if latest_values["AssignmentCompletion"] < 70:
            priority_actions.append("Complete pending assignments.")

        if latest_values["ExamScore"] < 60:
            priority_actions.append("Increase exam preparation and practice.")

        if latest_values["StudyHours"] < 10:
            priority_actions.append("Create a consistent study schedule.")

        if latest_values["StressLevel"] >= 2:
            priority_actions.append("Manage stress and seek appropriate support.")

        if not priority_actions:
            priority_actions.append(
                "Maintain current learning habits and continue monitoring progress."
            )

        for action in priority_actions:
            st.checkbox(action, value=False)

        st.caption(
            "Note: The ML model predicts the grade category. "
            "The personalized recommendations are rule-based and use the "
            "student's entered indicators to suggest practical interventions."
        )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.caption(
    "Academic Risk Early Warning System • Decision Tree ML • Streamlit"
)
