import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import streamlit as st
from PIL import Image
import os

# Load the trained model
#with open(r"C:\Users\plangote\plangote.pkl", 'rb') as file:
    #model = pickle.load(file)

import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression  # Just an example dataset
# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Example: Create a synthetic dataset (replace this with your actual data)
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the regressor model
regressor_random = RandomForestRegressor()

# Fit the model to the training data
regressor_random.fit(X_train, y_train)


# Save the trained model using pickle
with open(r"C:\Users\plangote\regressor_random_model.pkl", 'wb') as file:
    pickle.dump(regressor_random, file)

print("Model saved successfully.")




    # Path to the car photos directory
CAR_PHOTOS_DIR = r"C:\Users\plangote\car_photo"

# Function to preprocess input data
def preprocess_data(data):
    features = [
        'modelYear',
        'Engine Displacement',
        'Power Steering',
        'transmission',
        'Engine Type',
        'Turbo Charger',
        'Mileage',
        'Max Power',
        'Torque',
        'No Of Airbags',
        'Leather Seats',
        'Air Conditioner',
        'Bluetooth',
        'Touch Screen',
        'Wheel Size',
        'Alloy Wheels',
        'Roof Rail',
        'Rear Camera',
        'Length',
        'Drive Type'
    ]
    
    df = pd.DataFrame([data], columns=features)
    
    categorical_features = [
        'transmission', 'Engine Type', 'Turbo Charger',
        'Leather Seats', 'Air Conditioner', 'Bluetooth', 
        'Touch Screen', 'Alloy Wheels', 'Roof Rail', 
        'Rear Camera', 'Drive Type'
    ]
    numeric_features = [
        'modelYear', 'Engine Displacement', 'Power Steering',
        'Mileage', 'Max Power', 'Torque', 'No Of Airbags', 
        'Wheel Size', 'Length'
    ]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    df_transformed = preprocessor.fit_transform(df)
    
    return df_transformed

# Streamlit app
st.title("Vehicle Price Prediction")

# Input fields for the user
model_year = st.number_input("Model Year", min_value=2000, max_value=2024, value=2020)
engine_displacement = st.number_input("Engine Displacement (L)", min_value=0, format="%d")  # Changed to integer
power_steering = st.selectbox("Power Steering", ["Yes", "No"])
transmission = st.selectbox("Transmission", ["Automatic", "Manual", "CVT", "DCT", "Direct Drive"])
engine_type = st.selectbox("Engine Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
turbo_charger = st.selectbox("Turbo Charger", ["Yes", "No"])
mileage = st.number_input("Mileage (km/l)", min_value=0, format="%d")  # Changed to integer
max_power = st.number_input("Max Power (hp)", min_value=0, format="%d")
torque = st.number_input("Torque (Nm)", min_value=0, format="%d")
no_of_airbags = st.number_input("No Of Airbags", min_value=0, max_value=10, format="%d")
leather_seats = st.selectbox("Leather Seats", ["Yes", "No"])
air_conditioner = st.selectbox("Air Conditioner", ["Yes", "No"])
bluetooth = st.selectbox("Bluetooth", ["Yes", "No"])
touch_screen = st.selectbox("Touch Screen", ["Yes", "No"])
wheel_size = st.number_input("Wheel Size (inches)", min_value=0, format="%d")
alloy_wheels = st.selectbox("Alloy Wheels", ["Yes", "No"])
roof_rail = st.selectbox("Roof Rail", ["Yes", "No"])
rear_camera = st.selectbox("Rear Camera", ["Yes", "No"])
length = st.number_input("Length (mm)", min_value=0, format="%d")
drive_type = st.selectbox("Drive Type", ["FWD", "RWD", "AWD"])

# Create a dictionary with the input data
input_data = {
    'modelYear': model_year,
    'Engine Displacement': engine_displacement,
    'Power Steering': 1 if power_steering == "Yes" else 0,
    'transmission': transmission,
    'Engine Type': engine_type,
    'Turbo Charger': 1 if turbo_charger == "Yes" else 0,
    'Mileage': mileage,
    'Max Power': max_power,
    'Torque': torque,
    'No Of Airbags': no_of_airbags,
    'Leather Seats': 1 if leather_seats == "Yes" else 0,
    'Air Conditioner': 1 if air_conditioner == "Yes" else 0,
    'Bluetooth': 1 if bluetooth == "Yes" else 0,
    'Touch Screen': 1 if touch_screen == "Yes" else 0,
    'Wheel Size': wheel_size,
    'Alloy Wheels': 1 if alloy_wheels == "Yes" else 0,
    'Roof Rail': 1 if roof_rail == "Yes" else 0,
    'Rear Camera': 1 if rear_camera == "Yes" else 0,
    'Length': length,
    'Drive Type': drive_type
}

# Preprocess the input data
processed_data = preprocess_data(input_data)

# Predict the price
if st.button("Predict Price"):
    prediction = model.predict(processed_data)
    st.write(f"Predicted Price: ${prediction[0]:,.2f}")

    # Display car photo
    car_model = st.selectbox("Select Car Model", ["ModelA", "ModelB", "ModelC"])  # Replace with actual car models
    photo_path = os.path.join(CAR_PHOTOS_DIR, f"{car_model}.jpg")  # Path to car photos
    if os.path.exists(photo_path):
        image = Image.open(photo_path)
        st.image(image, caption=f"{car_model} Photo", use_column_width=True)
    else:
        st.write("Photo not available for the selected model.")
