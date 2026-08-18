import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EdTech Academic Risk Early Warning System",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = os.path.join("model", "academic_risk_model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found!")
    st.write("Expected location:")
    st.code("model/academic_risk_model.pkl")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("❌ Could not load the model.")
    st.write(str(e))
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🎓 EdTech Academic Risk Early Warning System")

st.markdown(
    """
    ### 📊 AI-Powered Student Performance Prediction
    Enter a student's academic and learning information to predict
    their performance level and receive personalized recommendations.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎯 Student Information")

st.sidebar.markdown(
    """
    Enter realistic values for the student.

    **Study Hours** means study hours **per day**.
    """
)


# ============================================================
# INPUTS
# ============================================================

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# COLUMN 1
# ------------------------------------------------------------

with col1:

    st.subheader("📚 Academic Information")

    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=1.0,
        max_value=10.0,
        value=4.5,
        step=0.1
    )

    attendance = st.slider(
        "Attendance (%)",
        min_value=0,
        max_value=100,
        value=80
    )

    assignment_completion = st.slider(
        "Assignment Completion (%)",
        min_value=0,
        max_value=100,
        value=75
    )

    online_courses = st.number_input(
        "Online Courses",
        min_value=0,
        max_value=30,
        value=10
    )

    discussions = st.selectbox(
        "Participates in Discussions?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


# ------------------------------------------------------------
# COLUMN 2
# ------------------------------------------------------------

with col2:

    st.subheader("🧠 Learning & Personal Factors")

    resources = st.selectbox(
        "Learning Resources Available?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    motivation = st.selectbox(
        "Motivation Level",
        options=[0, 1],
        format_func=lambda x: "High" if x == 1 else "Low"
    )

    internet = st.selectbox(
        "Internet Access?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    gender = st.selectbox(
        "Gender",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    age = st.number_input(
        "Age",
        min_value=15,
        max_value=30,
        value=20
    )

    learning_style = st.number_input(
        "Learning Style",
        min_value=0,
        max_value=5,
        value=2
    )

    edutech = st.selectbox(
        "Uses EdTech Tools?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    stress_level = st.selectbox(
        "Stress Level",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Low",
            1: "Moderate",
            2: "High",
            3: "Very High"
        }[x]
    )


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔮 Predict Academic Risk",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # CREATE INPUT DATA
    # IMPORTANT:
    # EXACTLY 14 FEATURES USED DURING MODEL TRAINING
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "StudyHours": [study_hours],
        "Attendance": [attendance],
        "Resources": [resources],
        "Extracurricular": [extracurricular],
        "Motivation": [motivation],
        "Internet": [internet],
        "Gender": [gender],
        "Age": [age],
        "LearningStyle": [learning_style],
        "OnlineCourses": [online_courses],
        "Discussions": [discussions],
        "AssignmentCompletion": [assignment_completion],
        "EduTech": [edutech],
        "StressLevel": [stress_level]
    })

    # --------------------------------------------------------
    # ENSURE EXACT FEATURE ORDER
    # --------------------------------------------------------

    feature_order = [
        "StudyHours",
        "Attendance",
        "Resources",
        "Extracurricular",
        "Motivation",
        "Internet",
        "Gender",
        "Age",
        "LearningStyle",
        "OnlineCourses",
        "Discussions",
        "AssignmentCompletion",
        "EduTech",
        "StressLevel"
    ]

    input_data = input_data[feature_order]

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(input_data)[0]

        # ----------------------------------------------------
        # OPTIONAL PROBABILITY
        # ----------------------------------------------------

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            probability = max(probabilities) * 100

        # ----------------------------------------------------
        # MAP FINAL GRADE
        # ----------------------------------------------------

        grade_names = {
            0: "Excellent",
            1: "Good",
            2: "Average",
            3: "At Risk"
        }

        risk_names = {
            0: "🟢 Low Risk",
            1: "🟢 Low Risk",
            2: "🟠 Medium Risk",
            3: "🔴 High Risk"
        }

        grade_name = grade_names.get(
            int(prediction),
            "Unknown"
        )

        risk_name = risk_names.get(
            int(prediction),
            "Unknown Risk"
        )

        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        st.success("✅ Prediction completed successfully!")

        st.subheader("📊 Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Predicted Grade",
                str(int(prediction))
            )

        with result_col2:

            st.metric(
                "Performance Level",
                grade_name
            )

        with result_col3:

            st.metric(
                "Risk Level",
                risk_name
            )

        if probability is not None:

            st.info(
                f"🤖 Model confidence: {probability:.2f}%"
            )

        # ----------------------------------------------------
        # STUDENT PROFILE
        # ----------------------------------------------------

        st.divider()

        st.subheader("👨‍🎓 Student Profile")

        profile_col1, profile_col2, profile_col3, profile_col4 = st.columns(4)

        with profile_col1:
            st.metric(
                "Study Hours/Day",
                f"{study_hours:.1f}"
            )

        with profile_col2:
            st.metric(
                "Attendance",
                f"{attendance}%"
            )

        with profile_col3:
            st.metric(
                "Assignments",
                f"{assignment_completion}%"
            )

        with profile_col4:
            st.metric(
                "Stress Level",
                str(stress_level)
            )

        # ----------------------------------------------------
        # PERSONALIZED RECOMMENDATIONS
        # ----------------------------------------------------

        st.divider()

        st.subheader("💡 Personalized Recommendations")

        recommendations = []

        # Study hours
        if study_hours < 3:
            recommendations.append(
                "📚 Increase study time gradually to around 3–5 hours per day."
            )
        elif study_hours > 8:
            recommendations.append(
                "⏰ Your study time is very high. Include breaks and maintain a healthy schedule."
            )
        else:
            recommendations.append(
                "✅ Your daily study time is within a reasonable range."
            )

        # Attendance
        if attendance < 75:
            recommendations.append(
                "⚠️ Improve attendance. Try to maintain at least 75–80% attendance."
            )
        else:
            recommendations.append(
                "✅ Attendance is good. Continue attending classes regularly."
            )

        # Assignments
        if assignment_completion < 70:
            recommendations.append(
                "📝 Complete assignments regularly. Set weekly assignment targets."
            )
        elif assignment_completion < 85:
            recommendations.append(
                "📈 Try to improve assignment completion above 85%."
            )
        else:
            recommendations.append(
                "✅ Assignment completion is strong."
            )

        # Motivation
        if motivation == 0:
            recommendations.append(
                "🧠 Work on motivation using small daily goals and progress tracking."
            )
        else:
            recommendations.append(
                "🔥 Good motivation. Continue setting clear academic goals."
            )

        # Internet
        if internet == 0:
            recommendations.append(
                "🌐 Improve access to reliable internet resources for online learning."
            )

        # Resources
        if resources == 0:
            recommendations.append(
                "📖 Use additional learning resources such as textbooks, notes, and educational platforms."
            )

        # Discussions
        if discussions == 0:
            recommendations.append(
                "💬 Participate more in class discussions to improve understanding."
            )
        else:
            recommendations.append(
                "💬 Good participation in discussions. Keep interacting with classmates and instructors."
            )

        # Online courses
        if online_courses < 5:
            recommendations.append(
                "💻 Consider taking additional online courses related to your subjects."
            )

        # EdTech
        if edutech == 0:
            recommendations.append(
                "🤖 Consider using educational technology tools for practice, revision, and personalized learning."
            )

        # Stress
        if stress_level >= 2:
            recommendations.append(
                "🧘 High stress detected. Use breaks, proper sleep, exercise, and time management."
            )
        elif stress_level == 1:
            recommendations.append(
                "🙂 Maintain a balanced study schedule to prevent stress from increasing."
            )
        else:
            recommendations.append(
                "✅ Stress level appears manageable."
            )

        # Risk-based recommendation
        if prediction == 3:

            recommendations.append(
                "🚨 High-risk student: recommend academic intervention, mentor support, and regular progress monitoring."
            )

        elif prediction == 2:

            recommendations.append(
                "⚠️ Medium-risk student: monitor academic progress and provide targeted support."
            )

        else:

            recommendations.append(
                "🌟 Continue the current learning strategy while monitoring performance."
            )

        # ----------------------------------------------------
        # DISPLAY RECOMMENDATIONS
        # ----------------------------------------------------

        for recommendation in recommendations:
            st.write(recommendation)

        # ----------------------------------------------------
        # EARLY WARNING MESSAGE
        # ----------------------------------------------------

        st.divider()

        st.subheader("🚨 Early Warning System")

        if prediction == 3:

            st.error(
                """
                HIGH RISK ALERT

                This student may require immediate academic support.
                Recommended actions:
                • Faculty/mentor intervention
                • Additional study support
                • Assignment monitoring
                • Attendance monitoring
                • Regular follow-up
                """
            )

        elif prediction == 2:

            st.warning(
                """
                MEDIUM RISK ALERT

                The student should be monitored regularly.
                Focus on improving academic consistency and weak areas.
                """
            )

        else:

            st.success(
                """
                LOW RISK

                The student is currently performing relatively well.
                Continue monitoring and maintain good learning habits.
                """
            )

        # ----------------------------------------------------
        # INPUT DATA TABLE
        # ----------------------------------------------------

        st.divider()

        st.subheader("📋 Model Input Data")

        st.dataframe(
            input_data,
            use_container_width=True
        )

    except Exception as e:

        st.error("❌ Prediction failed.")

        st.code(str(e))

        st.info(
            """
            Make sure the model was trained using the same 14 features:

            StudyHours
            Attendance
            Resources
            Extracurricular
            Motivation
            Internet
            Gender
            Age
            LearningStyle
            OnlineCourses
            Discussions
            AssignmentCompletion
            EduTech
            StressLevel
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 EdTech Academic Risk Early Warning System | "
    "AI/ML Student Performance Prediction"
)