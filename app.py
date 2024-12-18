import streamlit as st
import pickle
import os

# Title of the Streamlit app
st.title("Pickle File Viewer")

# Upload file input widget
uploaded_file = st.file_uploader("Choose a .pkl file", type="pkl")

# Check if a file is uploaded
if uploaded_file is not None:
    try:
        # Load the pickle file
        data = pickle.load(uploaded_file)
        
        # Display the data based on its type
        if isinstance(data, dict):
            st.write("The content of the pickle file is a dictionary:")
            st.json(data)
        elif isinstance(data, list):
            st.write("The content of the pickle file is a list:")
            st.write(data)
        else:
            st.write("The content of the pickle file:")
            st.write(data)
    except Exception as e:
        st.error(f"Error loading the file: {e}")
