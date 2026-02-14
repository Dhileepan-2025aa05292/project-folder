import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

# 1. Page Configuration & Custom Theme
st.set_page_config(page_title="Telco Churn Analytics", layout="wide", page_icon="📞")

# Custom CSS for a polished look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .stSubheader { color: #1e3a8a; font-weight: bold; }
    h1 { color: #1e3a8a; text-align: center; }
    </style>
    """, unsafe_allow_stdio=True)

st.title("📞 Telco Customer Churn Prediction")
st.markdown("<p style='text-align: center; color: gray;'>M.Tech AIML - Machine Learning Assignment 2</p>", unsafe_allow_stdio=True)
st.divider()

# 2. Sidebar - Model Selection & Info
with st.sidebar:
    st.header("⚙️ Configuration")
    model_option = st.selectbox(
        "Select Classification Model",
        ("Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost")
    )
    st.info("Ensure the uploaded CSV matches the preprocessed feature set (19+ features).")

# 3. Load Assets
@st.cache_resource
def load_assets(name):
    # Mapping for specific naming cases
    filename = name.replace(" ", "_").lower()
    model_path = f'model/{filename}.pkl'
    model = joblib.load(model_path)
    scaler = joblib.load('model/scaler.pkl')
    return model, scaler

# 4. Main App Logic
col_main, col_side = st.columns([3, 1])

with col_side:
    st.subheader("📤 Data Input")
    uploaded_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    target_col = 'Churn'
    
    if target_col in test_df.columns:
        with col_main:
            # Load assets
            model, scaler = load_assets(model_option)
            
            X_test = test_df.drop(target_col, axis=1)
            y_true = test_df[target_col]
            
            try:
                # Align features
                X_test_aligned = X_test[scaler.feature_names_in_]
                X_test_scaled = scaler.transform(X_test_aligned)
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
                
                # 5. Dashboard Metrics
                st.subheader(f"📊 Model Performance: {model_option}")
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col4, m_col5, m_col6 = st.columns(3)
                
                m_col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
                m_col2.metric("AUC Score", f"{roc_auc_score(y_true, y_proba):.4f}")
                m_col3.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
                m_col4.metric("Recall (Sensitivity)", f"{recall_score(y_true, y_pred):.4f}")
                m_col5.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")
                m_col6.metric("MCC Score", f"{matthews_corrcoef(y_true, y_pred):.4f}")
                
                st.divider()

                # 6. Visualization Layout
                v_col1, v_col2 = st.columns(2)
                
                with v_col1:
                    st.subheader("📉 Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
                    ax.set_xlabel('Predicted Label')
                    ax.set_ylabel('True Label')
                    st.pyplot(fig)
                
                with v_col2:
                    st.subheader("📋 Prediction Samples")
                    result_df = pd.DataFrame({
                        'Actual': y_true.values,
                        'Predicted': y_pred
                    }).head(10)
                    st.dataframe(result_df.style.background_gradient(cmap='Blues'), use_container_width=True)

            except KeyError as e:
                st.error(f"⚠️ Feature Mismatch: Your CSV is missing column {e}")
                st.info("Check if your test data was encoded with the same 'get_dummies' logic as training.")
    else:
        st.error(f"❌ Target column '{target_col}' not found in the uploaded file.")
else:
    with col_main:
        st.write("---")
        st.warning("Waiting for data upload... Please use the sidebar to upload your test_data.csv")
        # Visual placeholder
        st.image("https://img.freepik.com", width=400)
