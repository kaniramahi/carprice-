import pandas as pd
import pickle
import streamlit as st
import sklearn
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline

# Load the trained model
with open('dt_modelcar2.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)

# Load the data
df = pd.read_pickle('datacar2.pkl',)

# Create the Streamlit app
st.title('🚗 Used Car Selling Price Prediction App')
st.header('Fill in the details to predict the used car selling price')

# Input fields
brand = st.selectbox('Brand', df['brand'].unique())
km_driven = st.number_input('Enter the kilometers reading of the vehicle', value=300000)
fuel = st.selectbox('Fuel', df['fuel'].unique())
seller_type = st.selectbox('Seller Type', df['seller_type'].unique())
transmission = st.selectbox('Transmission', df['transmission'].unique(), index=0)
owner = st.selectbox('Owner', df['owner'].unique())
age = st.number_input('Age of the vehicle in years (1-32)', value=10)

# Prepare the input data
model = pd.DataFrame({
    'brand': [brand],
    'km_driven': [km_driven],
    'fuel': [fuel],
    'seller_type': [seller_type],
    'transmission': [transmission],
    'owner': [owner],
    'age': [age]
})

st.markdown("---")
st.markdown("### Prediction Results")
st.markdown("Click the button below to see the predicted price.")

# Make the prediction
predict_price = loaded_pipeline.predict(model)[0]

# Display the predicted selling price
st.sidebar.markdown('Predicted Selling Price')
st.write('The predicted selling price of the used car is',predict_price, 'units.')