
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from huggingface_hub import hf_hub_download
import logging

logging.basicConfig(
    level=logging.INFO,   # 👈 enables INFO logs
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Download and load the model
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="bkrishnamukund/Wellness-Tourism-Prediction",
                             filename="best_wellness_tourism_prediction_model_v1.joblib")
    return joblib.load(model_path)

model = load_model()

training_columns = [
    'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch',
    'Occupation', 'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups',
    'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips',
    'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 
    'Designation', 'MonthlyIncome'
]

st.title("Wellness Tourism Purchase Prediction")
st.write("Predict whether a customer is likely to purchase a wellness tourism package.")

st.header("Single Customer Prediction")

with st.form("single_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100)
        typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
        citytier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        maritalstatus = st.selectbox("Marital Status", ["Single", "Unmarried", "Married", "Divorced"])

    with col2:
        num_persons = st.number_input("Number of Persons Visiting", min_value=1)
        preferred_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
        num_trips = st.number_input("Number of Trips (Yearly)", min_value=0)
        passport = st.selectbox("Passport", [0, 1])
        owncar = st.selectbox("Own Car", [0, 1])
        num_children = st.number_input("Number of Children (<5 yrs)", min_value=0)

    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5)
    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe"]
    )
    followups = st.number_input("Number of Followups", min_value=0)
    pitch_duration = st.number_input("Duration of Pitch (minutes)", min_value=1)
    designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    income = st.number_input("Monthly Income", min_value=0.0, step=1000.0)

    submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([{
            "Age": age,
            "TypeofContact": typeofcontact,
            "CityTier": citytier,
            "Occupation": occupation,
            "Gender": gender,
            "MaritalStatus": maritalstatus,
            "NumberOfPersonVisiting": num_persons,
            "PreferredPropertyStar": preferred_star,
            "NumberOfTrips": num_trips,
            "Passport": passport,
            "OwnCar": owncar,
            "NumberOfChildrenVisiting": num_children,
            "PitchSatisfactionScore": pitch_score,
            "ProductPitched": product_pitched,
            "NumberOfFollowups": followups,
            "DurationOfPitch": pitch_duration,
            "Designation": designation,
            "MonthlyIncome": income
        }])

        # Reorder input_df columns to match training
        input_df_ordered = input_df[training_columns]

        # Make prediction
        prediction = model.predict(input_df_ordered)[0]

        # Get probability of positive class (1)
        probability = model.predict_proba(input_df_ordered)[0][1]

        # Optional: log the ordered input_df
        logging.info("Input data (ordered): %s", input_df_ordered.head().to_dict(orient="records"))
        logging.info("Prediction: %s, Probability: %.4f", prediction, probability)
        
        if prediction == 1:
            st.success(f"✅ Customer is likely to purchase (Probability: {probability:.2f})")
        else:
            st.warning(f"❌ Customer is unlikely to purchase (Probability: {probability:.2f})")

st.header("Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write("Preview of uploaded data:")
  st.dataframe(df.head())

  if st.button("Predict Batch"):
      try:
          preds = model.predict(df)
          probs = model.predict_proba(df)[:, 1]

          df_out = df.copy()
          df_out["Predicted_ProdTaken"] = preds
          df_out["Purchase_Probability"] = np.round(probs, 2)

          st.success("Batch prediction completed!")
          st.dataframe(df_out)

          csv = df_out.to_csv(index=False)
          st.download_button(
              "Download Predictions CSV",
              csv,
              file_name="wellness_predictions.csv",
              mime="text/csv"
          )

      except Exception as e:
          st.error(str(e))
