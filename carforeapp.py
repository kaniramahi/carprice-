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

st.markdown("---")
st.markdown("### Prediction Results")
st.markdown("Click the button below to see the predicted price.")


# Prepare the input data
if st.sidebar.button('Used Car Price Prediction :money_with_wings:'):
        test = np.array([brand, km_driven, fuel, seller_type, transmission, owner, age])
        test = test.reshape([1,7])
        predicted_price = model.predict(test)[0]
        predicted_price_rounded = round(predicted_price, 2)
        st.sidebar.success(f'Predicted Price: {predicted_price_rounded} :dollar:')
 

# Make the prediction
predict_price = loaded_pipeline.predict(model)[0]

# Display the predicted selling price
st.markdown('Predicted Selling Price')
st.write('The predicted selling price of the used car is',predicted_price_rounded, 'units.')