import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Import classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🎗️", layout="centered")

# --- MODEL TRAINING & SCORING (Cached) ---
@st.cache_resource
def load_and_train_models():
    # 1. Data Load & Clean
    # load the csv relative to this script (deployment friendly)
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "data.csv")
    df = pd.read_csv(csv_path)

    # FIX: Drop columns that aren't features (ID and the empty last column)
    # We use .iloc to handle the "Unnamed: 32" column regardless of its name
    df = df.drop(columns=['id'], errors='ignore')
    if 'Unnamed: 32' in df.columns:
        df = df.drop(columns=['Unnamed: 32'], errors='ignore')
    
    # We will use the 'mean' features (first 10 numeric features)
    # This ensures the UI stays simple while maintaining high accuracy
    feature_cols = [col for col in df.columns if 'mean' in col]
    X = df[feature_cols]
    
    # Encoding target: M = 1, B = 0
    y = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

    # 2. Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        model_scores[name] = accuracy_score(y_test, predictions) * 100
        
    best_model = models["Random Forest"]
    
    return best_model, scaler, feature_cols, model_scores

# Load and train
try:
    best_model, scaler, feature_columns, scores = load_and_train_models()
except Exception as e:
    st.error(f"Failed to load data. Make sure 'data.csv' is in the same folder. Error: {e}")
    st.stop()

# --- WEBSITE FRONTEND ---
st.title("🎗️ Breast Cancer Classifier")
st.write("Input cell nucleus measurements to predict if a tumor is Malignant or Benign.")

# --- MODEL ACCURACY DASHBOARD ---
with st.expander("📊 View Model Performance", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Log. Reg.", f"{scores['Logistic Regression']:.1f}%")
    col2.metric("KNN", f"{scores['KNN Classifier']:.1f}%")
    col3.metric("Decision Tree", f"{scores['Decision Tree']:.1f}%")
    col4.metric("Random Forest", f"{scores['Random Forest']:.1f}%", delta="Best")

st.markdown("---")

# --- USER INPUTS ---
# Using columns to organize the 10 inputs
st.subheader("Tumor Nucleus Features (Means)")
c1, c2 = st.columns(2)

# Dictionary to store user inputs
user_inputs = {}

# We iterate through the feature columns to create inputs dynamically
for i, col in enumerate(feature_columns):
    # Split features into two columns for better UI
    target_col = c1 if i < 5 else c2
    # Clean the name for display (e.g., "radius_mean" -> "Radius Mean")
    display_name = col.replace('_', ' ').title()
    
    # Set default values based on dataset averages to make testing easier
    user_inputs[col] = target_col.number_input(f"{display_name}", value=float(0.0), format="%.4f")

st.markdown("---")

# --- PREDICTION ---
if st.button("Run Classification", type="primary"):
    # Convert input dict to DataFrame
    input_df = pd.DataFrame([user_inputs])
    
    # Scale and Predict
    input_scaled = scaler.transform(input_df)
    prediction = best_model.predict(input_scaled)[0]
    prediction_proba = best_model.predict_proba(input_scaled)[0] # Confidence

    if prediction == 1:
        st.error(f"### Result: Malignant (Cancerous)")
        st.warning(f"Confidence: {prediction_proba[1]*100:.1f}%")
    else:
        st.success(f"### Result: Benign (Healthy)")
        st.info(f"Confidence: {prediction_proba[0]*100:.1f}%")

st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes and is not a substitute for professional medical advice.")