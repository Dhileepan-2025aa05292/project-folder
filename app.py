import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

# 1. Page Configuration
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("📞 Telco Customer Churn Prediction App")
st.markdown("Download the sample test data, then upload it to evaluate performance.")

# --- DOWNLOAD BUTTON ---
st.subheader("1. Prepare Test Data")
sample_file_path = "final_test_upload.csv" 

if os.path.exists(sample_file_path):
    with open(sample_file_path, "rb") as file:
        st.download_button(
            label="📥 Download Sample Test CSV",
            data=file,
            file_name="final_test_upload.csv",
            mime="text/csv"
        )
else:
    st.warning("Ensure 'final_test_upload.csv' is in your GitHub root.")

st.divider()

# 2. Sidebar - Model Selection
st.sidebar.header("Settings")
model_option = st.sidebar.selectbox(
    "Choose a Machine Learning Model",
    ("Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost")
)

# 3. Load Assets
@st.cache_resource
def load_assets(name):
    model_name_clean = name.replace(" ", "_").lower()
    model = joblib.load(f'model/{model_name_clean}.pkl')
    scaler = joblib.load('model/scaler.pkl')
    return model, scaler

# 4. Main App Logic
st.subheader("2. Evaluate Model")
uploaded_file = st.file_uploader("Upload your test CSV file", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    target_col = 'Churn' # Updated to match your request
    
    if target_col in test_df.columns:
        y_true = test_df[target_col]
        X_test_raw = test_df.drop(columns=[target_col])
        
        try:
            model, scaler = load_assets(model_option)
            
            # Align features and scale
            trained_features = model.feature_names_in_
            X_test_aligned = X_test_raw[trained_features]
            X_test_scaled = scaler.transform(X_test_aligned)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            # 5. Display Metrics
            st.subheader(f"📊 Evaluation Metrics: {model_option}")
            cols = st.columns(6)
            metrics = {
                "Accuracy": accuracy_score(y_true, y_pred),
                "AUC": roc_auc_score(y_true, y_proba),
                "Precision": precision_score(y_true, y_pred),
                "Recall": recall_score(y_true, y_pred),
                "F1": f1_score(y_true, y_pred),
                "MCC": matthews_corrcoef(y_true, y_pred)
            }
            for i, (label, val) in enumerate(metrics.items()):
                cols[i].metric(label, f"{val:.4f}")
            
            # 6. Confusion Matrix
            st.subheader("📉 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error(f"Column '{target_col}' not found in uploaded file.")
