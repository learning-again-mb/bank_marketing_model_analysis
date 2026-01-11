import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# =========================================================
# Term Deposit Subscription Predictor (Multi-Model)
# Models + preprocessing artifacts are created
# in the notebook and saved as .pkl files in the same folder.
# =========================================================

st.set_page_config(page_title="Term Deposit Subscription Predictor", layout="wide")

APP_DIR = Path(".")  # keep .pkl files next to this script

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model" / "all_models.pkl"
    return joblib.load(model_path)

model = load_model()


# Required artifacts saved from the notebook
FEATURE_COLS_FILE = "feature_columns.pkl"     # list[str] : exact X columns used in training
SCALER_FILE = "scaler.pkl"                    # fitted StandardScaler (used for KNN/NB in notebook)
PREPROC_META_FILE = "preprocess_meta.pkl"     # dict with {"min_balance": <float>} used in balance_log


# -----------------------------
# Load artifacts 
# -----------------------------
@st.cache_resource
def load_artifacts():
    models = {}
    for name, fname in model.items():
        models[name] = joblib.load(APP_DIR / fname)

    feature_cols = joblib.load(APP_DIR / FEATURE_COLS_FILE)
    scaler = joblib.load(APP_DIR / SCALER_FILE)
    meta = joblib.load(APP_DIR / PREPROC_META_FILE)

    return models, feature_cols, scaler, meta


def build_feature_vector(raw_row: dict, feature_cols: list[str], meta: dict) -> pd.DataFrame:
    """
      1) balance_log = log1p(balance - min_balance + 1)
      2) drop month, day, duration 
      3) one-hot encoding (get_dummies)
      4) align to training feature columns
    """
    df = pd.DataFrame([raw_row])

    # balance_log 
    min_balance = float(meta.get("min_balance", 0.0))
    df["balance_log"] = np.log1p(df["balance"] - min_balance + 1)

    # Drop columns 
    for col in ["month", "day", "duration"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # One-hot encoding
    df_enc = pd.get_dummies(df)

    # Align to the exact training feature set
    X = df_enc.reindex(columns=feature_cols, fill_value=0)

    return X


def predict_subscription(model, X: pd.DataFrame):
    """
    Returns:
      pred (0/1), proba (float|None)
    """
    pred = int(model.predict(X)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = float(model.predict_proba(X)[0, 1])
        except Exception:
            proba = None
    return pred, proba


# -----------------------------
# App UI
# -----------------------------
st.title("Term Deposit Subscription Predictor")
st.caption("Predict whether a client will subscribe to a term deposit (y). Choose among multiple trained models.")

# Load notebook artifacts
try:
    models, feature_cols, scaler, meta = load_artifacts()
except Exception as e:
    st.error(
        "Could not load one or more required .pkl files. "
        "Place all saved model/artifact files in the same folder as this app and retry."
    )
    st.exception(e)
    st.stop()

# Sidebar: model selection + inputs
st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox("Choose classifier", list(MODEL_FILES.keys()), index=0)

st.sidebar.header("Client Inputs")

age = st.sidebar.number_input("age", min_value=18, max_value=120, value=35)
balance = st.sidebar.number_input("balance", value=0)
campaign = st.sidebar.number_input("campaign (contacts in this campaign)", min_value=1, value=1)
pdays = st.sidebar.number_input("pdays (days since previous contact; -1 means never)", value=-1)
previous = st.sidebar.number_input("previous (contacts before this campaign)", min_value=0, value=0)

job = st.sidebar.selectbox("job", [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
    "self-employed", "services", "student", "technician", "unemployed", "unknown"
])

marital = st.sidebar.selectbox("marital", ["divorced", "married", "single"])
education = st.sidebar.selectbox("education", ["primary", "secondary", "tertiary", "unknown"])
default = st.sidebar.selectbox("default", ["no", "yes"])
housing = st.sidebar.selectbox("housing", ["no", "yes"])
loan = st.sidebar.selectbox("loan", ["no", "yes"])
contact = st.sidebar.selectbox("contact", ["cellular", "telephone", "unknown"])
poutcome = st.sidebar.selectbox("poutcome", ["failure", "other", "success", "unknown"])

threshold = st.sidebar.slider("Decision threshold (probability)", 0.05, 0.95, 0.50, 0.05)

predict_btn = st.sidebar.button("Predict Subscription")

# Raw input row (before dummy encoding)
raw_row = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome,
}

# Models that were trained on scaled data 
SCALED_MODELS = {"K-Nearest Neighbours", "Naive Bayes (Gaussian)"}

if predict_btn:
    model = models[selected_model_name]

    X_input = build_feature_vector(raw_row, feature_cols, meta)

    # Apply scaling only for the models that used scaled features 
    if selected_model_name in SCALED_MODELS:
        X_used = scaler.transform(X_input)
        pred = int(model.predict(X_used)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = float(model.predict_proba(X_used)[0, 1])
            except Exception:
                proba = None
    else:
        pred, proba = predict_subscription(model, X_input)

    final_pred = pred
    if proba is not None:
        final_pred = int(proba >= threshold)

    st.subheader("Prediction Result")
    st.write(f"**Selected model:** {selected_model_name}")

    if final_pred == 1:
        st.success("Client is **LIKELY TO SUBSCRIBE** to the term deposit (y = yes).")
    else:
        st.error("Client is **UNLIKELY TO SUBSCRIBE** to the term deposit (y = no).")

    if proba is not None:
        st.write(f"**Predicted probability of subscription (y=1):** {proba*100:.2f}%")
        st.write(f"**Threshold applied:** {threshold:.2f}")

    st.write("### Input Features Sent to Model")
    st.dataframe(X_input)
else:
    st.info("Fill the details on the left sidebar and click **Predict Subscription**.")
