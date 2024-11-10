# Importing Libraries
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np


# Load pre-trained models and data
@st.cache_data
def load_data():
    return pd.read_csv('Data/train.csv')

# Load the trained model and scaler
@st.cache_resource
def load_models():
    with open("Lgbm_Tuned_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler, selected_features = pickle.load(scaler_file)
    return model, scaler, selected_features

# Load data and models
data = load_data()
model, scaler, selected_features = load_models()

# Title and Description
st.title("Employee Promotion Prediction App")
st.write("""
This application predicts the eligibility of employees for promotion within a multinational corporation.
Provide the employee's details below, and the model will assess their promotion eligibility.
""")

# CSS Styling for app layout
st.markdown(
    """
    <style>
    /* Background and layout */
    .main { background-color: #f0f2f6; padding: 20px; }
    h1, h2, h3 { color: #5B5EA6; text-align: center; }
    
    /* Input box styling */
    .stSelectbox, .stCheckbox, .stSlider { 
        font-size: 16px; color: #333333;
    }
    
    /* Button styling */
    .stButton button { 
        background-color: #5B5EA6; color: white; 
        border-radius: 8px; padding: 10px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Helper function to encode categorical and ordinal inputs
def encode_inputs(inputs):
    # Frequency encode nominal features
    department_encoding = data['department'].value_counts(normalize=True)
    recruitment_encoding = data['recruitment_channel'].value_counts(normalize=True)
    
    inputs['department_freq_encoded'] = department_encoding.get(inputs['department'], 0)
    inputs['recruitment_channel_freq_encoded'] = recruitment_encoding.get(inputs['recruitment_channel'], 0)

    # Ordinal encoding with specified order
    age_order = ['Under 20', '20-30', '30-40', '40-50', '50-60', 'Over 60']
    service_order = ['Less than 2 years', '2-5 years', '5-10 years', 'Over 10 years']
    score_order = ['Low', 'Medium', 'High', 'Very High']
    education_order = ['Below Secondary', "Bachelor's", "Master's & above"]
    
    inputs['age_group'] = age_order.index(inputs['age_group']) if inputs['age_group'] in age_order else -1
    inputs['service_group'] = service_order.index(inputs['service_group']) if inputs['service_group'] in service_order else -1
    inputs['score_group'] = score_order.index(inputs['score_group']) if inputs['score_group'] in score_order else -1
    inputs['education'] = education_order.index(inputs['education']) if inputs['education'] in education_order else -1

    return inputs

# Collect user inputs
department = st.selectbox("Department", options=data['department'].unique())
recruitment_channel = st.selectbox("Recruitment Channel", options=data['recruitment_channel'].unique())
age_group = st.selectbox("Age Group", options=['Under 20', '20-30', '30-40', '40-50', '50-60', 'Over 60'])
service_group = st.selectbox("Service Group", options=['Less than 2 years', '2-5 years', '5-10 years', 'Over 10 years'])
score_group = st.selectbox("Performance Score", options=['Low', 'Medium', 'High', 'Very High'])
education = st.selectbox("Education Level", options=['Below Secondary', "Bachelor's", "Master's & above"])
KPIs_met = st.checkbox("KPIs met >80%")
awards_won = st.checkbox("Awards Won")
career_velocity = st.slider("Career Velocity", min_value=0.0, max_value=1.0, step=0.01)
training_frequency = st.slider("Training Frequency", min_value=0.0, max_value=1.0, step=0.01)
training_effectiveness = st.slider("Training Effectiveness", min_value=0.0, max_value=100.0, step=0.1)
relative_performance = st.slider("Relative Performance", min_value=-2.0, max_value=2.0, step=0.01)
consistent_performer = st.checkbox("Consistent Performer")

# Create dictionary for inputs
user_inputs = {
    'department': department,
    'recruitment_channel': recruitment_channel,
    'age_group': age_group,
    'service_group': service_group,
    'score_group': score_group,
    'education': education,
    'KPIs_met >80%': int(KPIs_met),
    'awards_won?': int(awards_won),
    'career_velocity': career_velocity,
    'training_frequency': training_frequency,
    'training_effectiveness': training_effectiveness,
    'relative_performance': relative_performance,
    'consistent_performer': int(consistent_performer)
}

# Encode inputs
encoded_inputs = encode_inputs(user_inputs)

# Convert to DataFrame and reorder columns for model prediction
input_df = pd.DataFrame([encoded_inputs])
selected_features = ['training_effectiveness', 'department_freq_encoded', 'relative_performance', 
                     'career_velocity', 'training_frequency', 'recruitment_channel_freq_encoded', 
                     'age_group', 'score_group', 'KPIs_met >80%', 'awards_won?', 
                     'education', 'service_group', 'consistent_performer']

# Select only the required features for prediction
input_df = scaler.transform(input_data[selected_features])


# Make Prediction
if st.button("Predict Promotion Eligibility"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.write("### Prediction: Eligible for Promotion")
        st.image("https://www.bing.com/th/id/OGC.41f3175b2e13b48a21709f280604453b?pid=1.7&rurl=https%3a%2f%2fgifdb.com%2fimages%2fhigh%2fnickelodeon-spongebob-squarepants-ready-for-promotion-3dwi74fsa7214agw.gif&ehk=IRNPw%2fCrEob5bRMWrFZY503VWUvKJPdM8%2fp%2fbrtS1c0%3d", caption="Promotion Achieved!")
    else:
        st.write("### Prediction: Not Eligible for Promotion")
