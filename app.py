import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Import training functions
from model.logistic_regression import train_logistic_regression
from model.decision_tree import train_decision_tree
from model.knn import train_knn
from model.naive_bayes import train_naive_bayes
from model.random_forest import train_random_forest
from model.xgboost_model import train_xgboost


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Dry Bean Classification",
    layout="wide"
)

st.title("🌱 Dry Bean Multi-Class Classification")
st.markdown(
    """
    This Streamlit application compares **six machine learning classification models**
    on the **Dry Bean Dataset** using standard evaluation metrics.
    """
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Configuration")

model_name = st.sidebar.selectbox(
    "Select Classification Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Dry Bean CSV Dataset",
    type=["csv"]
)

# ---------------- METRIC FUNCTION ----------------
def evaluate_model(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1 Score": f1_score(y_true, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="weighted"
        )
    }

# ---------------- MAIN LOGIC ----------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "Class" not in df.columns:
        st.error("❌ Dataset must contain a 'Class' column")
        st.stop()

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ---------------- DATA PREPARATION ----------------
    X = df.drop("Class", axis=1)
    y = df["Class"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------- MODEL SELECTION ----------------
    if model_name == "Logistic Regression":
        model = train_logistic_regression(X_train_scaled, y_train)
        X_eval = X_test_scaled

    elif model_name == "Decision Tree":
        model = train_decision_tree(X_train, y_train)
        X_eval = X_test

    elif model_name == "kNN":
        model = train_knn(X_train_scaled, y_train)
        X_eval = X_test_scaled

    elif model_name == "Naive Bayes":
        model = train_naive_bayes(X_train_scaled, y_train)
        X_eval = X_test_scaled

    elif model_name == "Random Forest":
        model = train_random_forest(X_train, y_train)
        X_eval = X_test

    else:  # XGBoost
        model = train_xgboost(X_train, y_train)
        X_eval = X_test

    # ---------------- PREDICTION ----------------
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)

    metrics = evaluate_model(y_test, y_pred, y_prob)

    # ---------------- METRICS DISPLAY ----------------
    st.subheader(f"📈 Evaluation Metrics – {model_name}")

    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    c1.metric("Precision", f"{metrics['Precision']:.4f}")

    c2.metric("Recall", f"{metrics['Recall']:.4f}")
    c2.metric("F1 Score", f"{metrics['F1 Score']:.4f}")

    c3.metric("AUC", f"{metrics['AUC']:.4f}")
    c3.metric("MCC", f"{metrics['MCC']:.4f}")

    # ---------------- CONFUSION MATRIX ----------------
    st.subheader("🧮 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[str(c) for c in label_encoder.classes_],
        yticklabels=[str(c) for c in label_encoder.classes_],
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

else:
    st.info("⬅️ Upload the Dry Bean CSV file to start the analysis")
