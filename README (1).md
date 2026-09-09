# Alzheimer's Disease Risk Predictor

A Streamlit web app that predicts Alzheimer's Disease risk from patient health,
lifestyle, and cognitive data, using a tuned XGBoost model
(test ROC-AUC 0.95, accuracy 94.4%) trained on 2,149 patient records.

⚠️ **For research/educational use only — not a substitute for clinical diagnosis.**

## Live demo

Deployed on Streamlit Community Cloud: _add your link here after deploying_

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app — patient input form + risk prediction |
| `best_model.pkl` | Trained XGBoost classifier |
| `scaler.pkl` | Fitted StandardScaler (kept for pipeline consistency) |
| `model_info.json` | Feature list, scaling flag, and test-set metrics |
| `requirements.txt` | Python dependencies |

## Model summary

- **Algorithm:** XGBoost (tuned via GridSearchCV, 5-fold CV)
- **Test accuracy:** 94.4% · **Test ROC-AUC:** 0.950
- **Top predictors:** MemoryComplaints, BehavioralProblems, MMSE,
  ADL, FunctionalAssessment
