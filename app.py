from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
with open("Lgbm_Tuned_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler, selected_features = pickle.load(scaler_file)

# Helper function for encoding inputs
def encode_inputs(data):
    department_encoding = {'Sales': 0.2, 'Operations': 0.15}  # Example encodings, replace with actual values
    recruitment_encoding = {'LinkedIn': 0.1, 'Indeed': 0.2}  # Example encodings, replace with actual values
    
    data['department_freq_encoded'] = department_encoding.get(data.get('department'), 0)
    data['recruitment_channel_freq_encoded'] = recruitment_encoding.get(data.get('recruitment_channel'), 0)

    age_order = ['Under 20', '20-30', '30-40', '40-50', '50-60', 'Over 60']
    service_order = ['Less than 2 years', '2-5 years', '5-10 years', 'Over 10 years']
    score_order = ['Low', 'Medium', 'High', 'Very High']
    education_order = ['Below Secondary', "Bachelor's", "Master's & above"]
    
    data['age_group'] = age_order.index(data.get('age_group', 'Under 20')) if data.get('age_group') in age_order else -1
    data['service_group'] = service_order.index(data.get('service_group', 'Less than 2 years')) if data.get('service_group') in service_order else -1
    data['score_group'] = score_order.index(data.get('score_group', 'Low')) if data.get('score_group') in score_order else -1
    data['education'] = education_order.index(data.get('education', 'Below Secondary')) if data.get('education') in education_order else -1

    return data

# Home route to display the form
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Extract input data
        data = {
            'department': request.form['department'],
            'recruitment_channel': request.form['recruitment_channel'],
            'age_group': request.form['age_group'],
            'service_group': request.form['service_group'],
            'score_group': request.form['score_group'],
            'education': request.form['education'],
            'KPIs_met >80%': int('KPIs_met' in request.form),
            'awards_won?': int('awards_won' in request.form),
            'career_velocity': float(request.form['career_velocity']),
            'training_frequency': float(request.form['training_frequency']),
            'training_effectiveness': float(request.form['training_effectiveness']),
            'relative_performance': float(request.form['relative_performance']),
            'consistent_performer': int('consistent_performer' in request.form),
        }
        
        # Encode inputs and make prediction
        encoded_data = encode_inputs(data)
        input_features = [encoded_data[feature] for feature in selected_features]
        scaled_features = scaler.transform([input_features])
        prediction = model.predict(scaled_features)[0]

        prediction = "Eligible for Promotion" if prediction == 1 else "Not Eligible for Promotion"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
