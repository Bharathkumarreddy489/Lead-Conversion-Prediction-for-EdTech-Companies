import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the trained XGBoost model and preprocessor
model = joblib.load('models/xgb_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Columns for input features in the original dataset
input_columns = [
    'age', 'current_occupation', 'first_interaction', 'profile_completed', 'website_visits',
    'time_spent_on_website', 'page_views_per_visit', 'last_activity', 'print_media_type1',
    'print_media_type2', 'digital_media', 'educational_channels', 'referral'
]

# Preprocessing function for input data
def preprocess_input(input_data):
    # Creating DataFrame from input data
    input_df = pd.DataFrame([input_data])

    # Apply preprocessor (loaded during runtime)
    input_data_processed = preprocessor.transform(input_df)
    return input_data_processed

# Streamlit user interface
st.title('Lead Conversion Prediction for EdTech Companies')
st.markdown("""
This project aims to analyze and build machine learning models to predict lead conversion for EdTech Companies. 
By leveraging historical lead data, the goal is to identify key factors influencing lead conversion and prioritize leads most likely to convert into paid customers. 
A comprehensive lead profile will be developed, showcasing the characteristics of high-conversion leads, thus optimizing resource allocation and improving marketing efforts.
""")

# Collecting input data from the user
input_data = {}

# Collecting numerical inputs
input_data['age'] = st.number_input('Age', min_value=0, max_value=100, value=57)
input_data['website_visits'] = st.number_input('Website Visits', min_value=0, max_value=1000, value=7)
input_data['time_spent_on_website'] = st.number_input('Time Spent on Website (minutes)', min_value=0, max_value=2000, value=1639)
input_data['page_views_per_visit'] = st.number_input('Page Views per Visit', min_value=0.0, max_value=50.0, value=1.861)

# Collecting categorical inputs (string-based columns)
input_data['current_occupation'] = st.selectbox('Current Occupation', ['Professional', 'Student', 'Unemployed'])
input_data['first_interaction'] = st.selectbox('First Interaction', ['Mobile App', 'Website'])
input_data['profile_completed'] = st.selectbox('Profile Completed', ['High', 'Medium', 'Low'])
input_data['last_activity'] = st.selectbox('Last Activity', ['Email Activity', 'Phone Activity', 'Website Activity'])
input_data['print_media_type1'] = st.selectbox('Print Media Type 1', ['Yes', 'No'])
input_data['print_media_type2'] = st.selectbox('Print Media Type 2', ['Yes', 'No'])
input_data['digital_media'] = st.selectbox('Digital Media', ['Yes', 'No'])
input_data['educational_channels'] = st.selectbox('Educational Channels', ['Yes', 'No'])
input_data['referral'] = st.selectbox('Referral', ['Yes', 'No'])

# When the user clicks on the "Predict" button
if st.button('Predict'):
    try:
        # Preprocess the input data
        processed_input = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(processed_input)
        
        # Display the result
        st.write(f'Prediction: {"Positive Lead" if prediction[0] == 1 else "Negative Lead"}')
        
    except ValueError as e:
        st.error(f"Error: {e}")
