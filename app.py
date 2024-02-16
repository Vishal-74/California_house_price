import numpy as np
import streamlit as st
import pickle

loaded_model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Define a function to preprocess ocean_proximity using one-hot encoding
def preprocess_ocean_proximity(ocean_proximity):
    categories = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    encoded_proximity = [1 if category == ocean_proximity else 0 for category in categories]
    return encoded_proximity

def preprocess_input_features(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity):
    rooms_per_house = total_rooms / households
    bedrooms_ratio = total_bedrooms / total_rooms
    people_per_house = population / households

    processed_ocean_proximity = preprocess_ocean_proximity(ocean_proximity)
    return [longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income,
            rooms_per_house, bedrooms_ratio, people_per_house] + processed_ocean_proximity

def house_price_prediction(input_data):
    input_data = np.asarray(input_data)
    input_data_reshaped = input_data.reshape(1, -1)
    predicted_price = loaded_model.predict(input_data_reshaped)
    return predicted_price[0]

def main():
    # Add image and project title in the sidebar
    st.sidebar.image('california.jpg.webp', width=200, caption='California', use_column_width=True, output_format='JPEG')
    st.sidebar.markdown('**<span style="color:blue; font-size:24px;">California House Price Prediction</span>**', unsafe_allow_html=True)

    # Main content in the main area
    st.title('California House Price Prediction')

    # Create a layout with two columns
    col1, col2 = st.columns(2)

    # Input features in the first column
    with col1:
        longitude = st.number_input('Longitude')
        latitude = st.number_input('Latitude')
        housing_median_age = st.number_input('Housing Median Age')
        total_rooms = st.number_input('Total Rooms')
        total_bedrooms = st.number_input('Total Bedrooms')

    # Input features in the second column
    with col2:
        population = st.number_input('Population')
        households = st.number_input('Households')
        median_income = st.number_input('Median Income')
        ocean_proximity = st.selectbox('Ocean Proximity', ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

    # Preprocess input features
    input_data = preprocess_input_features(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity)

    # Predict button
    if st.button('Predict House Price'):
        predicted_price = house_price_prediction(input_data)
        st.success(f'Predicted House Price: ${predicted_price:,.2f}')

if __name__ == '__main__':
    main()
