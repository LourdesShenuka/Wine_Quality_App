import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib, json, os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# -------------------
# Load Data & Model
# -------------------
@st.cache_data
def load_data():
    return pd.read_csv("WineQT.csv")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

def load_aux():
    metrics = json.load(open("metrics.json")) if os.path.exists("metrics.json") else None
    features = json.load(open("features.json")) if os.path.exists("features.json") else None
    return metrics, features

df = load_data()

# Drop Id column if present
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# Recreate binary target column
df["good"] = (df["quality"] >= 6).astype(int)

model = load_model()
metrics_file, features = load_aux()

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Home","Explore Data","Visualizations","Model Performance","Predict"])

# -------------------
# Pages
# -------------------
if page == "Home":
    st.title("🍷 Wine Quality Prediction App")
    st.write("""
    This app demonstrates an **end-to-end ML workflow**:  
    - Data exploration  
    - Model training  
    - Deployment with Streamlit  
    - Real-time predictions  

    **Dataset**: Wine Quality (from Kaggle, `WineQT.csv`)  
    **Target**: Binary classification → Good wine (quality ≥ 6) vs Bad wine (< 6).
    """)

elif page == "Explore Data":
    st.title("📊 Dataset Overview")
    st.write("**Shape**:", df.shape)
    st.write("**Columns**:", df.columns.tolist())
    st.dataframe(df.head())

    st.subheader("🔍 Interactive Filter")
    col = st.selectbox("Choose a feature to filter", options=[c for c in df.columns if c not in ["Id"]])
    min_v, max_v = float(df[col].min()), float(df[col].max())
    low, high = st.slider("Range", min_v, max_v, (min_v, max_v))
    st.dataframe(df[(df[col] >= low) & (df[col] <= high)].head(50))

elif page == "Visualizations":
    st.title("📈 Visualizations")

    # Histogram
    st.subheader("Histogram by Feature")
    feat = st.selectbox("Select feature", options=[c for c in df.columns if c not in ["Id","quality","good"]])
    fig = px.histogram(df, x=feat, color="good", nbins=30, barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

    # Scatter plot
    st.subheader("Scatter Plot")
    col1, col2 = st.columns(2)
    with col1:
        x_feat = st.selectbox("X-axis", options=[c for c in df.columns if c not in ["Id","quality","good"]])
    with col2:
        y_feat = st.selectbox("Y-axis", options=[c for c in df.columns if c not in ["Id","quality","good"]])
    fig2 = px.scatter(df, x=x_feat, y=y_feat, color="good")
    st.plotly_chart(fig2, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig3, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.drop(columns=["Id","quality","good"]).corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig3)

elif page == "Model Performance":
    st.title("🧪 Model Performance")
    if metrics_file:
        st.write("**Saved Metrics from Training**")
        st.json(metrics_file)
    else:
        st.warning("No metrics.json found. Using quick evaluation instead.")

    # Quick confusion matrix
    X = df.drop(columns=["quality","good"])
    y = df["good"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred)
    st.write(f"Accuracy: **{acc:.3f}**, F1-score: **{f1:.3f}**")

    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

elif page == "Predict":
    st.title("🔮 Predict Wine Quality")
    st.write("Enter the wine’s chemical properties to predict if it’s **Good (≥6)** or **Bad (<6)**.")

    inputs = {}
    for feat in features:
        default_val = float(df[feat].mean())
        min_val, max_val = float(df[feat].min()), float(df[feat].max())
        inputs[feat] = st.number_input(
            feat, min_value=min_val, max_value=max_val, value=default_val
        )

    row = pd.DataFrame([inputs])

    if st.button("Predict"):
        with st.spinner("Running model..."):
            proba = model.predict_proba(row)[0,1]
            pred = int(proba >= 0.5)
        st.success(f"Prediction: **{'Good Wine 🍷✅' if pred==1 else 'Bad Wine 🍷❌'}**")

        st.write(f"Confidence: {proba:.2%}")
