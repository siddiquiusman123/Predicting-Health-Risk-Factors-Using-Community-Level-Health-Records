import pandas as pd
import numpy as np
import joblib
import streamlit as st

# load The Dataset

dataset = pd.read_csv("Health_data.csv")

st.title("Health Prediction App")

# load The model , scalar , encoder

model = joblib.load("health_prd_model.pkl")
onehot_encoder = joblib.load("health_prd_oh_encoder.pkl")
label_encoder = joblib.load("health_prd_label_encoder.pkl")
scalar = joblib.load("health_prd_scaler.pkl")

# user input

gender = st.radio("Select Gender:", ["Male", "Female"])
st.write("You selected:", gender)
age = st.number_input("Age",min_value=10 , max_value=100 , step=1)
st.write(f"Your Age : {age}")
systolic_bp = st.number_input("Systolic BP")
diastolic_bp = st.number_input("Diastolic BP")
cholestrol = st.number_input("Cholesterol")
height_cm = st.number_input("Height (cm)")
weight_kg = st.number_input("Weight (kg)")
bmi = st.number_input("BMI")
smoker = st.radio("Smoker",[True , False])
diabetes = st.radio("Diabetes",[True , False])

# converting into data frame

colums = [ 'Gender', 'Age', 'Systolic BP', 'Diastolic BP', 'Cholesterol',
       'Height (cm)', 'Weight (kg)', 'BMI', 'Smoker', 'Diabetes']

input_df = pd.DataFrame([[gender,age,systolic_bp,diastolic_bp,cholestrol,
                          height_cm,weight_kg,bmi,smoker,diabetes]],
                          columns=colums)

# Encoding Nominal Columns

nominal_col = input_df.select_dtypes(include=['object','bool']).columns
enc_input = onehot_encoder.transform(input_df[nominal_col])
enc_df = pd.DataFrame(enc_input,columns=onehot_encoder.get_feature_names_out(nominal_col))
final_input = pd.concat([input_df.drop(columns=nominal_col,axis=1),enc_df],axis=1)
numeric_col = dataset.select_dtypes(include=['int64','float64']).columns
final_input[numeric_col] = scalar.transform(final_input[numeric_col])

if st.button("Predict Health"):
    prediction_encoded = model.predict(final_input)
    orginal_prediction = label_encoder.inverse_transform(prediction_encoded)
    st.success(f"Predicted Health Status: {orginal_prediction[0]}")