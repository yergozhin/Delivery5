import json
from pathlib import Path

import streamlit as st
from predict import predict

# Iris species names
species_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
MLFLOW_UI_URL = "http://localhost:5001"
META_PATH = Path(__file__).parent / "model_meta.json"


@st.cache_data
def load_meta():
    """Load model metadata produced by train_model.py."""
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def build_footer(meta):
    version = meta.get("version", "n/a")
    best_model = meta.get("best_model", "n/a")
    metrics = meta.get("metrics", {})
    accuracy = metrics.get("accuracy", "n/a")
    run_id = meta.get("mlflow_run_id")
    run_label = run_id if run_id else "n/a"
    run_link = MLFLOW_UI_URL if not run_id else f"{MLFLOW_UI_URL}/#/models/IrisModel"
    return (
        f"Version: {version} • "
        f"Best model: {best_model} • "
        f"MLflow run: [{run_label}]({run_link}) • "
        f"Accuracy: {accuracy}"
    )


st.title("Iris Species Predictor")
st.write("Enter the measurements of an iris flower to predict its species.")

# Input fields for the 4 features
st.header("Flower Measurements")
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

# Predict button
if st.button("Predict"):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    prediction = predict(features)
    species = species_names[prediction]
    
    st.success(f"Predicted Species: **{species}**")
    st.write(f"Class: {prediction}")

# Footer with metadata from MLflow training
meta = load_meta()
st.markdown("---")
st.markdown("#### Model Info")
st.markdown(build_footer(meta))

