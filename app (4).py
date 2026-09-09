"""
Alzheimer's Disease Risk Prediction — Deployment App
Run with:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Alzheimer's Risk Predictor", page_icon="🧠", layout="centered")

# Resolve model files relative to THIS script's location, not the process's
# working directory — Streamlit Cloud's cwd doesn't always match app.py's folder.
APP_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model_path = APP_DIR / "best_model.pkl"
    scaler_path = APP_DIR / "scaler.pkl"
    info_path = APP_DIR / "model_info.json"

    missing = [p.name for p in (model_path, scaler_path, info_path) if not p.exists()]
    if missing:
        st.error(
            f"Missing required file(s) next to app.py: {', '.join(missing)}. "
            f"Looked in: {APP_DIR}. "
            "Make sure best_model.pkl, scaler.pkl, and model_info.json are committed "
            "to the repo in the same folder as app.py."
        )
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    info = json.load(open(info_path))
    return model, scaler, info

model, scaler, info = load_artifacts()
FEATURES = info["features"]
NEEDS_SCALING = info["needs_scaling"]

st.title("🧠 Alzheimer's Disease Risk Predictor")
st.caption(
    f"Model: **{info['best_model_name']}**  |  "
    f"Test ROC-AUC: **{info['test_metrics']['ROC_AUC']:.3f}**  |  "
    f"Test Accuracy: **{info['test_metrics']['Accuracy']:.3f}**"
)
st.info(
    "This tool is a decision-support aid for research/educational use only. "
    "It does not replace clinical diagnosis by a qualified professional.",
    icon="⚠️",
)

st.divider()
st.subheader("Patient Information")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Demographics**")
        Age = st.slider("Age", 60, 90, 75)
        Gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
        Ethnicity = st.selectbox("Ethnicity", [0, 1, 2, 3],
                                  format_func=lambda x: ["Caucasian", "African American", "Asian", "Other"][x])
        EducationLevel = st.selectbox("Education Level", [0, 1, 2, 3],
                                       format_func=lambda x: ["None", "High School", "Bachelor's", "Higher Education"][x])

        st.markdown("**Lifestyle**")
        BMI = st.slider("BMI", 15.0, 40.0, 25.0)
        Smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        AlcoholConsumption = st.slider("Alcohol Consumption (units/week)", 0.0, 20.0, 5.0)
        PhysicalActivity = st.slider("Physical Activity (hrs/week)", 0.0, 10.0, 3.0)
        DietQuality = st.slider("Diet Quality (0-10)", 0.0, 10.0, 5.0)
        SleepQuality = st.slider("Sleep Quality (4-10)", 4.0, 10.0, 7.0)

        st.markdown("**Medical History**")
        FamilyHistoryAlzheimers = st.selectbox("Family History of Alzheimer's", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        CardiovascularDisease = st.selectbox("Cardiovascular Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        Diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        Depression = st.selectbox("Depression", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        HeadInjury = st.selectbox("History of Head Injury", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        Hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    with col2:
        st.markdown("**Clinical Measurements**")
        SystolicBP = st.slider("Systolic BP (mmHg)", 90, 180, 130)
        DiastolicBP = st.slider("Diastolic BP (mmHg)", 60, 120, 80)
        CholesterolTotal = st.slider("Total Cholesterol (mg/dL)", 150.0, 300.0, 200.0)
        CholesterolLDL = st.slider("LDL Cholesterol (mg/dL)", 50.0, 200.0, 100.0)
        CholesterolHDL = st.slider("HDL Cholesterol (mg/dL)", 20.0, 100.0, 50.0)
        CholesterolTriglycerides = st.slider("Triglycerides (mg/dL)", 50.0, 400.0, 150.0)

        st.markdown("**Cognitive & Functional**")
        MMSE = st.slider("MMSE Score (0-30, lower = more impaired)", 0.0, 30.0, 24.0)
        FunctionalAssessment = st.slider("Functional Assessment (0-10, lower = worse)", 0.0, 10.0, 6.0)
        MemoryComplaints = st.selectbox("Memory Complaints", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        BehavioralProblems = st.selectbox("Behavioral Problems", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        ADL = st.slider("ADL Score (0-10, lower = worse)", 0.0, 10.0, 6.0)

        st.markdown("**Symptoms**")
        Confusion = st.selectbox("Confusion", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        Disorientation = st.selectbox("Disorientation", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        PersonalityChanges = st.selectbox("Personality Changes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        DifficultyCompletingTasks = st.selectbox("Difficulty Completing Tasks", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        Forgetfulness = st.selectbox("Forgetfulness", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    submitted = st.form_submit_button("Predict Risk", use_container_width=True, type="primary")

if submitted:
    row = {k: v for k, v in locals().items() if k in FEATURES}
    patient_df = pd.DataFrame([row])[FEATURES]

    X_in = scaler.transform(patient_df) if NEEDS_SCALING else patient_df.values
    pred = model.predict(X_in)[0]
    proba = model.predict_proba(X_in)[0, 1]

    st.divider()
    st.subheader("Result")

    if pred == 1:
        st.error(f"⚠️ **Higher risk of Alzheimer's Disease** — predicted probability: **{proba:.1%}**")
    else:
        st.success(f"✅ **Lower risk of Alzheimer's Disease** — predicted probability: **{proba:.1%}**")

    st.progress(float(proba))
    st.caption(
        "This is a statistical estimate from a machine-learning model trained on historical "
        "data. Always confirm with a qualified clinician using full diagnostic workup."
    )
