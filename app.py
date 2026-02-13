%%writefile app.py
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
    model_path = f'model/{name.replace(" ", "_").lower()}.pkl'
    model = joblib.load(model_path)
    scaler = joblib.load('model/scaler.pkl')
    return model, scaler

# 4. Main App Logic
uploaded_file = st.file_uploader("Upload your test CSV file", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    # Consistent use of 'Churn' column
    target_col = 'Churn'
    
    if target_col in test_df.columns:
        X_test = test_df.drop(target_col, axis=1)
        y_true = test_df[target_col]
        
        # Load assets
        model, scaler = load_assets(model_option)
        
        try:
            # --- THE FIX: Align feature names and order ---
            # This ensures X_test matches exactly what the scaler/model expects
            X_test = X_test[model.feature_names_in_]
            
            # Preprocessing: Scale features
            X_test_scaled = scaler.transform(X_test)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            # 5. Display Metrics (Required Features)
            st.subheader(f"📊 Evaluation Metrics: {model_option}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
            col2.metric("AUC Score", f"{roc_auc_score(y_true, y_proba):.4f}")
            col3.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
            col4.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
            col5.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")
            col6.metric("MCC Score", f"{matthews_corrcoef(y_true, y_pred):.4f}")
            
            # 6. Confusion Matrix (Required Feature)
            st.subheader("📉 Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)
            
            # Display raw predictions
            st.subheader("📋 Prediction Results (Sample)")
            result_df = X_test.copy()
            result_df['Actual_Churn'] = y_true.values
            result_df['Predicted_Churn'] = y_pred
            st.write(result_df.head(10))
            
        except KeyError as e:
            st.error(f"Feature Mismatch: The uploaded file is missing columns or has incorrect headers. Error: {e}")
            st.info("Ensure your test CSV has the exact encoded features used during training.")
            
    else:
        st.error(f"The uploaded CSV must contain a '{target_col}' column for evaluation.")
else:
    st.info("Please upload a CSV file to begin.")
