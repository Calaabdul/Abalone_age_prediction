import streamlit as st 
import pickle
import pandas as pd

st.title("🐚 Abalone Age Prediction App")
st.write(
    """
    Enter the physical measurements of an abalone to predict its age (number of rings).
    """
)

# Load model
@st.cache_resource
def load_model():
    with open('model/abalone_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Input fields
sex = st.selectbox("Sex", ["M", "F", "I"])
length = st.number_input("Length (mm)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
height = st.number_input("Height (mm)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
whole_weight = st.number_input("Whole Weight (g)", min_value=0.0, max_value=3.0, value=0.3, step=0.01)
shell_weight = st.number_input("Shell Weight (g)", min_value=0.0, max_value=2.0, value=0.4, step=0.01)


input_df = pd.DataFrame({
        "sex": [sex],
        "length": [length],
        "height": [height],
        "whole_weight": [whole_weight],
        "shell_weight": [shell_weight]
    })

# Predict button
if st.button("Predict Age"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted number of rings (age): {prediction:.2f}")








