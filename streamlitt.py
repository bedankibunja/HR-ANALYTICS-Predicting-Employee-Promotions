import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load pre-trained models and data
@st.cache_data
def load_data():
    return pd.read_csv('Data/train.csv')

# Load the trained model and scaler
@st.cache_resource
def load_models():
    with open("xgb_weighted_model.pkl", "rb") as model_file:
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
threshold = st.slider("Set Threshold", min_value=0.5, max_value=1.0, step=0.05, value=0.6)

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
input_df = input_df[selected_features]

# Scale the inputs using the pre-loaded scaler
input_df = pd.DataFrame(scaler.transform(input_df), columns=selected_features)

# Make Prediction with a Custom Threshold
if st.button("Predict Promotion Eligibility"):
    # Get model's probability of the positive class (1)
    prob = model.predict_proba(input_df)[:, 1]  # Get probability for class 1 (eligible for promotion)
    
    # Apply the custom threshold
    if prob >= threshold:
        prediction = 1  # Eligible for promotion
    else:
        prediction = 0  # Not eligible for promotion

    # Display the prediction
    if prediction == 1:
        st.write(f"### Prediction: Eligible for Promotion (Probability: {prob[0]:.2f})")
        st.image("https://www.bing.com/th/id/OGC.41f3175b2e13b48a21709f280604453b?pid=1.7&rurl=https%3a%2f%2fgifdb.com%2fimages%2fhigh%2fnickelodeon-spongebob-squarepants-ready-for-promotion-3dwi74fsa7214agw.gif&ehk=IRNPw%2fCrEob5bRMWrFZY503VWUvKJPdM8%2fp%2fbrtS1c0%3d", caption="Promotion Achieved!")
    else:
        st.write(f"### Prediction: Not Eligible for Promotion (Probability: {prob[0]:.2f})")



