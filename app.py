import streamlit as st
import pandas as pd
from drug_discovery import get_model

# Load the trained ML model
@st.cache_resource
def load_model():
    return get_model()  # Get model from drug_discovery.py
drugs['Binding Affinity (Ki/IC50)'] = pd.to_numeric(drugs['Binding Affinity (Ki/IC50)'], errors='coerce')
drugs.drop(columns=['Binding Affinity (Ki/IC50)'],axis=1)

# Prediction function
def predict_drug_properties(h_bond_acceptors, binding_affinity, bioavailability, toxicity_numeric, qed_score):
    input_df = pd.DataFrame([{
        "H-Bond Acceptors": h_bond_acceptors,
      
        "Bioavailability": bioavailability,
        "Toxicity Numeric": toxicity_numeric,
        "QED score": qed_score
    }])

    model = load_model()
    prediction = model.predict(input_df)
    return prediction[0]

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Drug Discovery Predictor", layout="centered")

st.title("🧪 Drug Discovery ML Predictor")

st.markdown("Enter the molecular properties below to predict drug-likeness.")

# Input sliders / number inputs
h_bond_acceptors = st.slider("🔹 H-Bond Acceptors", 0, 15, 2)
binding_affinity = st.number_input("🔹 Binding Affinity (Ki/IC50)", value=0.5)
bioavailability = st.number_input("🔹 Bioavailability", value=0.5)
toxicity_numeric = st.number_input("🔹 Toxicity Score (0–1)", min_value=0.0, max_value=1.0, value=0.3)
qed_score = st.number_input("🔹 QED Score (0–1)", min_value=0.0, max_value=1.0, value=0.6)

# Predict button
if st.button("🚀 Predict Drug Property"):
    prediction = predict_drug_properties(
        h_bond_acceptors, binding_affinity, bioavailability, toxicity_numeric, qed_score
    )
    st.success(f"✅ Prediction: **{prediction}**")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for drug discovery research.")
