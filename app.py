import streamlit as st
import pandas as pd
import joblib
import json
import os
from sklearn.preprocessing import StandardScaler
import zipfile
import tempfile

# Function to load the pre-trained model and scaler
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# Function to load model metadata
def load_metadata(metadata_path):
    with open(metadata_path, 'r') as json_file:
        metadata = json.load(json_file)
    return metadata

# Function to make predictions
def make_prediction(model, scaler, input_data):
    # Ensure the input data is in the correct shape for prediction
    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)
    return prediction[0]  # Return the single prediction value

# Streamlit App UI
st.title("Stock Change Prediction Application")

# Allow user to upload the folder containing the pre-trained model
results_folder = st.file_uploader("Upload the Results Folder (ZIP)", type=["zip"])

if results_folder is not None:
    # Unzip the folder (assuming the user uploads a .zip file)
    with tempfile.TemporaryDirectory() as tempdir:
        with zipfile.ZipFile(results_folder, 'r') as zip_ref:
            zip_ref.extractall(tempdir)
            
            # List the extracted files for debugging
            extracted_files = os.listdir(tempdir)
            st.write(f"Files extracted: {extracted_files}")

            # Check if the model and metadata files exist
            model_filename = None
            metadata_filename = None

            # Search for the model and metadata files (pkl and json)
            for file in extracted_files:
                st.write(f"Checking file: {file}")
                if file.endswith(".pkl"):
                    model_filename = os.path.join(tempdir, file)
                    st.write(f"Found model: {model_filename}")
                if file.endswith(".json"):
                    metadata_filename = os.path.join(tempdir, file)
                    st.write(f"Found metadata: {metadata_filename}")

            if model_filename and metadata_filename:
                model = load_model(model_filename)
                metadata = load_metadata(metadata_filename)
                st.success("Model and metadata loaded successfully.")
            else:
                st.error("Model or metadata files not found in the uploaded folder.")
                st.write("Please ensure the ZIP folder contains the model (.pkl) and metadata (.json) files.")
                st.stop()

# Inputs for the user to provide the upcoming values
st.header("Input Upcoming Values")

gdp_value = st.number_input("Enter the upcoming GDP value:", min_value=-100.0, max_value=100.0, value=0.0)
inflation_value = st.number_input("Enter the upcoming Inflation value:", min_value=-10.0, max_value=10.0, value=0.0)
interest_rate_value = st.number_input("Enter the upcoming Interest Rate value:", min_value=-10.0, max_value=100.0, value=0.0)
vix_value = st.number_input("Enter the upcoming VIX value:", min_value=0.0, max_value=100.0, value=20.0)

# Prepare the input data for prediction
input_data = [gdp_value, inflation_value, interest_rate_value, vix_value]

# Load the scaler used during training
# Check if scaler parameters are available in the metadata
if 'scaler' in metadata and 'mean' in metadata['scaler'] and 'std' in metadata['scaler']:
    # If the metadata contains scaler info, initialize with those values
    scaler = StandardScaler()
    scaler.mean_ = np.array(metadata["scaler"]["mean"])
    scaler.scale_ = np.array(metadata["scaler"]["std"])
else:
    # If no scaler info in metadata, initialize a new scaler (default behavior)
    scaler = StandardScaler()

# Make prediction when the button is pressed
if st.button("Predict Stock Change"):
    # Make the prediction
    predicted_stock_change = make_prediction(model, scaler, input_data)

    # Display the predicted stock change
    st.subheader(f"Predicted Stock Change: {predicted_stock_change:.4f}")
    st.write("This is the predicted percentage change in stock based on the provided economic inputs.")
