import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

# 1. Page Configuration
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("📞 Telco Customer Churn Prediction App")
st.markdown("Upload test data to evaluate performance across 6 different ML models.")

# 2. Sidebar - Model Selection
st.sidebar.header("Settings")
model_option = st.sidebar.selectbox(
    "Choose a Machine Learning Model",
    ("Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost")
)

# 3. Load Models and Scaler
@st.cache_resource
def load_assets(name):
    # Ensure filenames match your .pkl files in the /model folder
    model_name_clean = name.replace(" ", "_").lower()
    model_path = f'model/{model_name_clean}.pkl'
    model = joblib.load(model_path)
    scaler = joblib.load('model/scaler.pkl')
    return model, scaler

# 4. Main App Logic
uploaded_file = st.file_uploader("Upload your test CSV file", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    target_col = 'Churn'
    
    if target_col in test_df.columns:
        y_true = test_df[target_col]
        X_test_raw = test_df.drop(columns=[target_col])
        
        try:
            # Load assets based on selection
            model, scaler = load_assets(model_option)
            
            # --- THE FIX: Align features exactly with Scaler/Model expectations ---
            trained_features = model.feature_names_in_
            X_test_aligned = X_test_raw[trained_features]
            
            # Preprocessing: Scale features
            X_test_scaled = scaler.transform(X_test_aligned)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Probability calculation (handles models without predict_proba like some SVMs/NB)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_proba = y_pred
            
            # 5. Display Metrics
            st.subheader(f"📊 Evaluation Metrics: {model_option}")
            m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
            
            m_col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
            m_col2.metric("AUC", f"{roc_auc_score(y_true, y_proba):.4f}")
            m_col3.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
            m_col4.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
            m_col5.metric("F1", f"{f1_score(y_true, y_pred):.4f}")
            m_col6.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")
            
            # 6. Confusion Matrix
            st.subheader("📉 Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)
            
            # Display Results
            st.subheader("📋 Prediction Results (Sample)")
            result_df = X_test_raw.copy()
            result_df['Actual_Churn'] = y_true.values
            result_df['Predicted_Churn'] = y_pred
            st.write(result_df.head(10))
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.info("Check if your uploaded CSV has the correct encoded columns and the model files exist in /model/.")
            
    else:
        st.error(f"The uploaded CSV must contain a '{target_col}' column.")
else:
    st.info("Please upload a CSV file to begin.")
