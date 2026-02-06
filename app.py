import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Multi-Domain Churn Prediction", layout="centered")

st.title("📊 Multi-Domain Customer Churn Prediction")
st.markdown("Transfer Learning Model (Telecom → Banking)")

# =============================
# LOAD MODEL & SCALER
# =============================
@st.cache_resource
def load_artifacts():
    model = load_model("transfer_model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# =============================
# INPUT SECTION
# =============================
st.header("Enter Customer Details")

account_length = st.number_input("Tenure / Account Length", min_value=0, value=5)
total_day_charge = st.number_input("Balance / Financial Usage", min_value=0.0, value=5000.0)
customer_service_calls = st.number_input("Complaints", min_value=0, value=0)
international_plan = st.selectbox("Has Credit Card / International Plan", [0,1])
voice_mail_plan = st.selectbox("Is Active Member / Voice Plan", [0,1])
total_day_minutes = st.number_input("Estimated Salary / Usage Proxy", min_value=0.0, value=3000.0)
total_intl_calls = st.number_input("Number of Products / Service Usage", min_value=0, value=1)

# =============================
# PREDICT BUTTON
# =============================
if st.button("Predict Churn Risk"):

    input_data = np.array([[
        account_length,
        total_day_charge,
        customer_service_calls,
        international_plan,
        voice_mail_plan,
        total_day_minutes,
        total_intl_calls
    ]])

    input_scaled = scaler.transform(input_data)
    probability = model.predict(input_scaled)[0][0]

    st.subheader("Prediction Result")

    st.metric("Churn Probability", f"{probability*100:.2f}%")

    if probability >= 0.75:
        st.error("⚠ High Risk Customer")
    elif probability >= 0.4:
        st.warning("⚠ Medium Risk Customer")
    else:
        st.success("✅ Low Risk Customer")

    st.progress(float(probability))

    st.markdown("---")
    st.markdown("### Business Recommendation")

    if probability >= 0.75:
        st.write("Immediate retention offer recommended.")
    elif probability >= 0.4:
        st.write("Targeted engagement strategy suggested.")
    else:
        st.write("Customer considered stable.")
