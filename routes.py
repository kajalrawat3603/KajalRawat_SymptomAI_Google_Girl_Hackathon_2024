from flask import  render_template, flash,request, redirect, url_for, send_file, session
from werkzeug.utils import secure_filename
import os
from application import app
import secrets
from flask import session
import os
import pickle
import sqlite3
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

secret_key = secrets.token_hex(16)

print("Generated Secret Key:", secret_key)
app.secret_key = secret_key

@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("index.html")


# Function to create or connect to the database
def create_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            email TEXT,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Create the database on app startup
create_db()

def insert_user(username, email, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', (username, email, password))
    conn.commit()
    conn.close()



# Create the users table when the application starts

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        # Check if passwords match
        if password != confirm_password:
            return 'Passwords do not match'
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        existing_user = cursor.fetchone()
        if existing_user:
            conn.close()
            return 'Username already exists'
        # Add user to the database
        insert_user(username, email,password)
        conn.close()
        flash('Signup successful', 'success')
        return render_template("index.html")
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            # If user exists, redirect to home page or any other page
            return redirect(url_for('index'))
        else:
            # If user does not exist, render the login form again with an error message
            error = 'Invalid username or password'
    return render_template('login.html', error=error)


diabetes_model = pickle.load(open("application/Models/diabetes_model.h5", "rb"))
heart_disease_model = pickle.load(open("application/Models/heart_disease_model.h5", "rb"))
parkinsons_model = pickle.load(open("application/Models/parkinsons_model.h5", "rb"))
svm_model = load('application/Models/disease_symptom.joblib')
# Load the dataset and other necessary files
df_precautions = pd.read_csv('application/Datasets/symptom/symptom_precaution.csv')
df_description = pd.read_csv('application/Datasets/symptom/symptom_Description.csv')
df_severity = pd.read_csv('application/Datasets/symptom/symptom_severity.csv')

from sklearn.preprocessing import LabelEncoder

# Initialize label encoder
label_encoder = LabelEncoder()

# Fit the LabelEncoder with all possible categories
label_encoder.fit(df_precautions['Disease'])
@app.route('/symptoms', methods=['POST', 'GET'])
def symptoms():
    if request.method == 'POST':
        symptoms = [0] * 17 
        
        # Get the symptoms from the form and update the input list accordingly
        symptom_keys = ['itching', 'skin_rash', 'nodal_skin_eruptions',         'continuous_sneezing', 'shivering', 'chills',
                        'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
                        'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety']
        
        symptom_weights = {
            'itching': 1,
            'skin_rash': 3,
            'nodal_skin_eruptions': 4,
            'continuous_sneezing': 4,
            'shivering': 5,
            'chills': 3,
            'joint_pain': 3,
            'stomach_pain': 5,
            'acidity': 3,
            'ulcers_on_tongue': 4,
            'muscle_wasting': 3,
            'vomiting': 5,
            'burning_micturition': 6,
            'spotting_urination': 6,
            'fatigue': 4,
            'weight_gain': 3,
            'anxiety': 4,
            'cold_hands_and_feets': 5,
            'mood_swings': 3,
            'weight_loss': 3,
            'restlessness': 5,
            'lethargy': 2,
            'patches_in_throat': 6,
            'irregular_sugar_level': 5,
            'cough': 4,
            'high_fever': 7,
            'sunken_eyes': 3,
            'breathlessness': 4,
            'sweating': 3,
            'dehydration': 4,
            'indigestion': 5,
            'headache': 3,
            'yellowish_skin': 3,
            'dark_urine': 4,
            'nausea': 5,
            'loss_of_appetite': 4,
            'pain_behind_the_eyes': 4,
            'back_pain': 3,
            'constipation': 4,
            'abdominal_pain': 4,
            'diarrhoea': 6,
            'mild_fever': 5,
            'yellow_urine': 4,
            'yellowing_of_eyes': 4,
            'acute_liver_failure': 6,
            'fluid_overload': 6,
            'swelling_of_stomach': 7,
            'swelled_lymph_nodes': 6,
            'malaise': 6,
            'blurred_and_distorted_vision': 5,
            'phlegm': 5,
            'throat_irritation': 4,
            'redness_of_eyes': 5,
            'sinus_pressure': 4,
            'runny_nose': 5,
            'congestion': 5,
            'chest_pain': 7,
            'weakness_in_limbs': 7,
            'fast_heart_rate': 5,
            'pain_during_bowel_movements': 5,
            'pain_in_anal_region': 6,
            'bloody_stool': 5,
            'irritation_in_anus': 6,
            'neck_pain': 5,
            'dizziness': 4,
            'cramps': 4,
            'bruising': 4,
            'obesity': 4,
            'swollen_legs': 5,
            'swollen_blood_vessels': 5,
            'puffy_face_and_eyes': 5,
            'enlarged_thyroid': 6,
            'brittle_nails': 5,
            'swollen_extremeties': 5,
            'excessive_hunger': 4,
            'extra_marital_contacts': 5,
            'drying_and_tingling_lips': 4,
            'slurred_speech': 4,
            'knee_pain': 3,
            'hip_joint_pain': 2,
            'muscle_weakness': 2,
            'stiff_neck': 4,
            'swelling_joints': 5,
            'movement_stiffness': 5,
            'spinning_movements': 6,
            'loss_of_balance': 4,
            'unsteadiness': 4,
            'weakness_of_one_body_side': 4,
            'loss_of_smell': 3,
            'bladder_discomfort': 4,
            'foul_smell_ofurine': 5,
            'continuous_feel_of_urine': 6,
            'passage_of_gases': 5,
            'internal_itching': 4,
            'toxic_look_(typhos)': 5,
            'depression': 3,
            'irritability': 2,
            'muscle_pain': 2,
            'altered_sensorium': 2,
            'red_spots_over_body': 3,
            'belly_pain': 4,
            'abnormal_menstruation': 6,
            'dischromic_patches': 6,
            'watering_from_eyes': 4,
            'increased_appetite': 5,
            'polyuria': 4,
            'family_history': 5,
            'mucoid_sputum': 4,
            'rusty_sputum': 4,
            'lack_of_concentration': 3,
            'visual_disturbances': 3,
            'receiving_blood_transfusion': 5,
            'receiving_unsterile_injections': 2,
            'coma': 7,
            'stomach_bleeding': 6,
            'distention_of_abdomen': 4,
            'history_of_alcohol_consumption': 5,
            'blood_in_sputum': 5,
            'prominent_veins_on_calf': 6,
            'palpitations': 4,
            'painful_walking': 2,
            'pus_filled_pimples': 2,
            'blackheads': 2,
            'scurring': 2,
            'skin_peeling': 3,
            'silver_like_dusting': 2,
            'small_dents_in_nails': 2,
            'inflammatory_nails': 2,
            'blister': 4,
            'red_sore_around_nose': 2,
            'yellow_crust_ooze': 3,
            'prognosis': 5
        }

        for i, key in enumerate(symptom_keys):
            if request.form.get(key):
                symptoms[i] = symptom_weights[key]
        
        # Call the model to predict the disease using the symptoms
        input_data = [symptoms]
        predictions = svm_model.predict(input_data)
        predicted_disease_names = label_encoder.inverse_transform(predictions)
        precautions_list = []
        descriptions_list = []

        for disease_name in predicted_disease_names:
            precautions_row = df_precautions[df_precautions['Disease'] == disease_name].iloc[:, 1:].values.tolist()[0]
            precautions_list.append(precautions_row)
            description = df_description[df_description['Disease'] == disease_name]['Description'].values[0]
            
            descriptions_list.append(description)

        return render_template('symptoms.html', predicted_diseases=predicted_disease_names, precautions=precautions_list, descriptions=descriptions_list)
    return render_template('symptoms.html')
    
    
    # Fetch additional information for the predicted disease
    



@app.route('/test', methods=['GET', 'POST'])
def test():
    return render_template('test.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        # Get user input
        user_input = [float(request.form[field]) for field in request.form]
        # Make prediction
        diab_prediction = diabetes_model.predict([user_input])
        if diab_prediction[0] == 1:
            result = 'The person is diabetic'
        else:
            result = 'The person is not diabetic'
        return render_template('diabetes.html', result=result)
    return render_template('diabetes.html')

@app.route('/heart_disease', methods=['GET', 'POST'])
def heart_disease():
    if request.method == 'POST':
        # Get user input
        user_input = [float(request.form[field]) for field in request.form]
        # Make prediction
        heart_prediction = heart_disease_model.predict([user_input])
        if heart_prediction[0] == 1:
            result = 'The person is having heart disease'
        else:
            result = 'The person does not have any heart disease'
        return render_template('heart_disease.html', result=result)
    return render_template('heart_disease.html')

@app.route('/parkinsons', methods=['GET', 'POST'])
def parkinsons():
    if request.method == 'POST':
        # Get user input
        user_input = [float(request.form[field]) for field in request.form]
        # Make prediction
        parkinsons_prediction = parkinsons_model.predict([user_input])
        if parkinsons_prediction[0] == 1:
            result = "The person has Parkinson's disease"
        else:
            result = "The person does not have Parkinson's disease"
        return render_template('parkinsons.html', result=result)
    return render_template('parkinsons.html')


if __name__ == '__main__':
    app.run(debug=True)