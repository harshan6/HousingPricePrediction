import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import base64  # Import base64 module for encoding images

# Load the saved Keras model
model = load_model('prediction_model.keras')

# Function to predict house prices
def predict_price(features):
    # Process features as needed (scaling, encoding, etc.)
    # Example: scaling features assuming 'preprocessor' is a fitted scaler
    # features = preprocessor.transform(features)

    # Make predictions using the loaded model
    predictions = model.predict(features)

    return predictions

# Function to display example inputs and explanations
def show_example_inputs():
    st.subheader('Example Inputs and Explanations')
    st.markdown("""
    Below are example inputs for features related to predicting house prices, along with brief explanations:

        1. longitude: A measure of how far west a house is; a higher value is farther west
        2. latitude: A measure of how far north a house is; a higher value is farther north
        3. housingMedianAge: Median age of a house within a block; a lower number is a newer building
        4. totalRooms: Total number of rooms within a block
        5. totalBedrooms: Total number of bedrooms within a block
        6. population: Total number of people residing within a block
        7. households: Total number of households, a group of people residing within a home unit, for a block
        8. medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
        9. medianHouseValue: Median house value for households within a block (measured in US Dollars)
        10. oceanProximity: Location of the house w.r.t ocean/sea
            """)

# Streamlit UI
st.title('House Price Prediction')


# Example input fields (replace with actual UI input elements)
longitude = st.number_input('Longitude')
latitude = st.number_input('Latitude')
housing_median_age = st.number_input('Housing Median Age')
total_rooms = st.number_input('Total Rooms')
total_bedrooms = st.number_input('Total Bedrooms')
population = st.number_input('Population')
households = st.number_input('Households')
median_income = st.number_input('Median Income')
ocean_proximity = st.selectbox('Ocean Proximity', ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'])

# Button to show example inputs and explanations
if st.button('Show Example Inputs and Explanations'):
    show_example_inputs()

# Example feature array (replace with actual feature handling)
features = np.array([[longitude, latitude, housing_median_age, total_rooms,
                      total_bedrooms, population, households, median_income]])

# Perform any necessary preprocessing on 'features' to match the expected input shape
# Example: One-hot encode 'ocean_proximity' if needed

# Button to predict
if st.button('Predict'):
    # Ensure 'features' has the correct shape (1, 13) if needed
    # Example: Add one-hot encoded 'ocean_proximity' feature
    if ocean_proximity == 'NEAR BAY':
        features = np.append(features, [[1, 0, 0, 0, 0]], axis=1)
    elif ocean_proximity == '<1H OCEAN':
        features = np.append(features, [[0, 1, 0, 0, 0]], axis=1)
    elif ocean_proximity == 'INLAND':
        features = np.append(features, [[0, 0, 1, 0, 0]], axis=1)
    elif ocean_proximity == 'NEAR OCEAN':
        features = np.append(features, [[0, 0, 0, 1, 0]], axis=1)
    elif ocean_proximity == 'ISLAND':
        features = np.append(features, [[0, 0, 0, 0, 1]], axis=1)

    # Make sure 'features' has shape (1, 13)
    assert features.shape == (1, 13), f'Expected shape (1, 13), got {features.shape}'

    # Perform prediction with the model
    prediction = predict_price(features)
    st.write(f'Predicted Price: ${prediction[0][0]:,.2f}')
