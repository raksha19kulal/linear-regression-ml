
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Student GPA Prediction",
    page_icon="📚"
)


@st.cache_resource
def load_model():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "best_gpa_model.pkl"
    )
    return joblib.load(path)


model = load_model()

st.title("Student GPA Prediction")
st.write("Enter student details to estimate GPA.")


# Dataset category labels and their original numeric codes

gender_options = {
    "Male - 0": 0,
    "Female - 1": 1
}

ethnicity_options = {
    "Caucasian - 0": 0,
    "African American - 1": 1,
    "Asian - 2": 2,
    "Other - 3": 3
}

education_options = {
    "None - 0": 0,
    "High School - 1": 1,
    "Some College - 2": 2,
    "Bachelor's - 3": 3,
    "Higher - 4": 4
}

yes_no_options = {
    "No - 0": 0,
    "Yes - 1": 1
}

support_options = {
    "None - 0": 0,
    "Low - 1": 1,
    "Moderate - 2": 2,
    "High - 3": 3,
    "Very High - 4": 4
}


with st.form("student_form"):

    st.subheader("Academic Details")

    age = st.selectbox(
        "Age (years)",
        [15, 16, 17, 18]
    )

    study_time = st.number_input(
        "Study Time Weekly (hours)",
        min_value=0.0,
        max_value=20.0,
        value=10.0,
        step=0.5
    )

    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=30,
        value=2,
        step=1
    )

    st.subheader("Student Background")

    gender_label = st.selectbox(
        "Gender",
        list(gender_options.keys())
    )

    ethnicity_label = st.selectbox(
        "Ethnicity",
        list(ethnicity_options.keys())
    )

    education_label = st.selectbox(
        "Parental Education",
        list(education_options.keys())
    )

    tutoring_label = st.selectbox(
        "Receives Tutoring?",
        list(yes_no_options.keys())
    )

    support_label = st.selectbox(
        "Parental Support",
        list(support_options.keys())
    )

    extracurricular_label = st.selectbox(
        "Participates in Extracurricular Activities?",
        list(yes_no_options.keys())
    )

    sports_label = st.selectbox(
        "Participates in Sports?",
        list(yes_no_options.keys())
    )

    music_label = st.selectbox(
        "Participates in Music Activities?",
        list(yes_no_options.keys())
    )

    volunteering_label = st.selectbox(
        "Participates in Volunteering?",
        list(yes_no_options.keys())
    )

    submitted = st.form_submit_button("Predict GPA")


if submitted:

    # Convert the selected labels into numeric dataset codes

    student = pd.DataFrame([{
        "Age": age,
        "Gender": gender_options[gender_label],
        "Ethnicity": ethnicity_options[ethnicity_label],
        "ParentalEducation": education_options[education_label],
        "StudyTimeWeekly": study_time,
        "Absences": absences,
        "Tutoring": yes_no_options[tutoring_label],
        "ParentalSupport": support_options[support_label],
        "Extracurricular": yes_no_options[extracurricular_label],
        "Sports": yes_no_options[sports_label],
        "Music": yes_no_options[music_label],
        "Volunteering": yes_no_options[volunteering_label]
    }])

    try:
        prediction = model.predict(student)[0]

        st.success(
            f"Estimated GPA: {prediction:.2f} / 4.00"
        )

        st.caption(
            "This is a model estimate, not a guaranteed academic result."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
