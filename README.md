# California House Price Prediction

Welcome to the California House Price Prediction project! This project aims to predict house prices in California using machine learning techniques.

![image](https://github.com/Vishal-74/California_house_price/assets/115347234/0231a723-11be-4f24-a7b7-a7196b3a805e)


## Problem Statement

The California housing market is dynamic and diverse, making accurate house price prediction a challenging task. This project addresses the need for reliable predictions to assist stakeholders in making informed decisions regarding buying, selling, or investing in real estate properties.

## Dataset

The dataset used in this project contains various housing features such as location coordinates, housing median age, total rooms, total bedrooms, population, households, median income, and ocean proximity. We preprocess the data, including one-hot encoding for categorical variables like ocean proximity, to ensure compatibility with machine learning algorithms.

## Model Development

We leverage machine learning algorithms, specifically Random Forest, to analyze the dataset and generate accurate predictions of house prices. The model is trained on historical housing data, allowing it to learn patterns and relationships within the dataset. The trained model is then deployed to predict house prices based on user-provided input features.

## Usage
0. **You can directly run app.py**

1. **Install Dependencies**: Ensure you have Python installed along with necessary libraries such as NumPy, Pandas, Streamlit, and Scikit-learn. 

2. **Clone the Repository**: Clone this repository to your local machine using `git clone`.

3. **Download Model**: Download the pre-trained model file (`random_forest_model.pkl`) from the provided Google Drive link and place it in the project directory.

4. **Run the Application**: Run the Streamlit application by executing `streamlit run app.py` in your terminal. This will launch the web interface where you can input housing features and obtain predictions for house prices.


#

## Acknowledgements

We would like to express our gratitude to kaggle for providing the housing dataset used in this project. Additionally, we thank the open-source community for developing and maintaining the tools and libraries utilized in this project.
