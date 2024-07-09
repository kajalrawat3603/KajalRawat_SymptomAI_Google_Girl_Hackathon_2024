import io
import re
from flask import render_template, flash, request, redirect, url_for, send_file, session,jsonify
from werkzeug.utils import secure_filename
import os
import secrets
import sqlite3
from datetime import datetime
from fpdf import FPDF
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder
import pickle
from functools import wraps
from io import BytesIO
from application import app
from application import utils
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

secret_key = secrets.token_hex(16)

print("Generated Secret Key:", secret_key)
app.secret_key = secret_key

@app.before_request
def before_request():
    session.modified = True

# Routes for signup, login, logout, and index
@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("index.html")

# Function to create or connect to the users database
def create_users_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        );
    ''')
    conn.commit()
    conn.close()

# Function to create or connect to the reports database
def create_reports_db():
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            report_type TEXT NOT NULL,
            report_data BLOB NOT NULL,
            time TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
    ''')
    conn.commit()
    conn.close()

create_users_db()
create_reports_db()

def insert_user(username, email, password):
    hashed_password = generate_password_hash(password)
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', (username, email, hashed_password))
    conn.commit()
    conn.close()

def insert_report(user_id, report_type, report_data):
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    current_datetime = datetime.now().isoformat()
    cursor.execute('INSERT INTO reports (user_id, report_type, report_data, time) VALUES (?, ?, ?, ?)',
                   (user_id, report_type, report_data, current_datetime))
    conn.commit()
    conn.close()

def generate_report(user_id, report_type, result, pos, neg, user_data):
    logo_path = "application/static/images/logo.png"
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    page_width, page_height = letter
    title_height = 60
    title_y = page_height - title_height
    
    c.setFillColorRGB(245, 245, 245)
    c.rect(0, title_y, page_width, title_height, fill=True)
    
    c.setFillColorRGB(0.02, 0.02, 0.42)
    c.setFont("Helvetica-Bold", 24)
    title_text = "symptomAI Report"
    title_text_width = c.stringWidth(title_text, "Helvetica-Bold", 24)
    title_text_x = (page_width - title_text_width) / 2
    title_text_y = title_y + (title_height - 24) / 2 
    c.drawString(title_text_x, title_text_y, title_text)
    logo_x = 20  
    logo_y = title_y + (title_height - 40) / 2
    c.drawImage(logo_path, logo_x, logo_y, width=40, height=40, mask='auto')
    
    c.setFont("Helvetica-Bold", 16)
    c.setFillColorRGB(0, 0, 0) 
    c.drawString(100, 700, f"Report Type: {report_type}")

    c.setFont("Helvetica", 12)
    y_position = 675
    for label, value in user_data.items():
        c.drawString(100, y_position, f"{label}: {value}")
        y_position -= 15
    
    c.setFont("Helvetica-Bold", 14)
    y_position -= 20
    c.drawString(100, y_position, "Result:")
    y_position -= 15
    c.setFont("Helvetica", 12)
    c.drawString(100, y_position, result)
    if report_type!='Symptom Analysis':
        c.setFont("Helvetica-Bold", 14)
        y_position -= 20
        c.drawString(100, y_position, "Positive Result:")
        y_position -= 15
        c.setFont("Helvetica", 12)
        c.drawString(100, y_position,f"{pos}%")
        c.setFont("Helvetica-Bold", 14)
        y_position -= 20
        c.drawString(100, y_position, "Negitive Result:")
        y_position -= 15
        c.setFont("Helvetica", 12)
        c.drawString(100, y_position,f"{neg}%")
    c.showPage()
    c.save()
    
    pdf_buffer.seek(0)
    pdf_data = pdf_buffer.read()
    
    insert_report(user_id, report_type, pdf_data)


# Function to require login for specific routes
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Load machine learning models and data
diabetes_model = load("application/Models/diabetes_model.joblib")
heart_disease_model = load(open("application/Models/heart_disease_model.joblib", "rb"))
parkinsons_model = pickle.load(open("application/Models/parkinsons_model.joblib", "rb"))
svm_model = load('application/Models/disease_symptom.joblib')
df_precautions = pd.read_csv('application/Datasets/symptom/symptom_precaution.csv')
df_description = pd.read_csv('application/Datasets/symptom/symptom_Description.csv')
df_severity = pd.read_csv('application/Datasets/symptom/symptom_severity.csv')
label_encoder = LabelEncoder()
label_encoder.fit(df_precautions['Disease'])



@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if len(username) < 3 or len(username) > 25:
            flash('Username must be between 3 and 25 characters long', 'danger')
            return redirect(url_for('signup'))

        email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        if not re.match(email_regex, email):
            flash('Invalid email address', 'danger')
            return redirect(url_for('signup'))

        if len(password) < 8:
            flash('Password must be at least 8 characters long', 'danger')
            return redirect(url_for('signup'))

        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('signup'))

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        existing_user = cursor.fetchone()
        if existing_user:
            conn.close()
            flash('Username already exists', 'danger')
            return redirect(url_for('signup'))

        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        existing_email = cursor.fetchone()
        if existing_email:
            conn.close()
            flash('Email already associated with an account', 'danger')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password)
        cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', (username, email, hashed_password))
        conn.commit()
        conn.close()

        flash('Signup successful', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, password FROM users WHERE username = ? AND email = ?', (username, email))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session.permanent = True
            return redirect(url_for('index'))
        else:
            flash('Invalid username/email or password', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.logged_in=False
    return redirect(url_for('login'))

@app.route('/report_trends', methods=['GET'])
@login_required
def report_trends():
    user_id = session.get('user_id')
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    cursor.execute('SELECT time, report_type FROM reports WHERE user_id = ?', (user_id,))
    reports = cursor.fetchall()
    conn.close()

    # Process the data to extract the trends
    trends = {}
    for report in reports:
        time, report_type = report
        date = datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%f')
        date_str = date.strftime('%Y-%m-%d')
        if report_type not in trends:
            trends[report_type] = {}
        if date_str not in trends[report_type]:
            trends[report_type][date_str] = 0
        trends[report_type][date_str] += 1

    return jsonify(trends)

@app.route('/test', methods=['GET', 'POST'])
def test():
    return render_template('test.html')

from sklearn.preprocessing import StandardScaler

@app.route('/diabetes', methods=['GET', 'POST'])
@login_required
def diabetes():
    if request.method == 'POST':
        user_id = session.get('user_id')
        
        labels = {
            'Pregnancies': float(request.form['Pregnancies']),
            'Glucose Level': float(request.form['Glucose']),
            'Blood Pressure': float(request.form['BloodPressure']),
            'Skin Thickness': float(request.form['SkinThickness']),
            'Insulin Level': float(request.form['Insulin']),
            'BMI': float(request.form['BMI']),
            'Diabetes Pedigree Function': float(request.form['DiabetesPedigreeFunction']),
            'Age': float(request.form['Age']),
        }
        
        # Standardize the input features
        scaler = StandardScaler()
        X = scaler.fit_transform([list(labels.values())])
        
        diab_prediction = diabetes_model.predict(X)
        diab_probabilities = diabetes_model.predict_proba(X)
        
        result = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
        prob_diabetic = round(diab_probabilities[0][1] * 100, 2)
        prob_not_diabetic = round(diab_probabilities[0][0] * 100, 2)
        
        generate_report(user_id, 'Diabetes', result,prob_diabetic,prob_not_diabetic, labels)
        
        return render_template('diabetes.html', result=result, prob_diabetic=prob_diabetic, prob_not_diabetic=prob_not_diabetic)
    
    return render_template('diabetes.html')


@app.route('/heart_disease', methods=['GET', 'POST'])
@login_required
def heart_disease():
    if request.method == 'POST':
        user_id = session.get('user_id')
        
        labels = {
            'Age': float(request.form['age']),
            'Sex (1:male and 0:female)': float(request.form['sex']),
            'Chest Pain Type': float(request.form['cp']),
            'Resting Blood Pressure (mm Hg)': float(request.form['trestbps']),
            'Serum Cholesterol (mg/dL)': float(request.form['chol']),
            'Fasting Blood Sugar (mg/dL)': float(request.form['fbs']),
            'Resting Electrocardiographic Results': float(request.form['restecg']),
            'Maximum Heart Rate achieved': float(request.form['thalach']),
            'Exercise Induced Angina': float(request.form['exang']),
            'ST depression induced by exercise': float(request.form['oldpeak']),
            'Slope of the peak exercise ST segment': float(request.form['slope']),
            'Major vessels colored by fluoroscopy': float(request.form['ca']),
            'Thal': float(request.form['thal']),
        }
        
        heart_prediction = heart_disease_model.predict([list(labels.values())])
        heart_probabilities = heart_disease_model.predict_proba([list(labels.values())])
        
        result = 'The person is having heart disease' if heart_prediction[0] == 1 else 'The person does not have any heart disease'
        prob_heart_disease = round(heart_probabilities[0][1] * 100, 2)
        prob_no_heart_disease = round(heart_probabilities[0][0] * 100, 2)
        
        generate_report(user_id, 'Heart Disease', result, prob_heart_disease, prob_no_heart_disease,labels)
        
        return render_template('heart_disease.html', result=result, prob_heart_disease=prob_heart_disease, prob_no_heart_disease=prob_no_heart_disease)
    
    return render_template('heart_disease.html')


@app.route('/parkinsons', methods=['GET', 'POST'])
@login_required
def parkinsons():
    if request.method == 'POST':
        user_id = session.get('user_id')
        
        labels = {
            'Average vocal fundamental frequency (MDVP:Fo(Hz))': float(request.form['MDVPFo']),
            'Maximum vocal fundamental frequency (MDVP:Fhi(Hz))': float(request.form['MDVPFhi']),
            'Minimum vocal fundamental frequency (MDVP:Flo(Hz))': float(request.form['MDVPFlo']),
            'MDVP:Jitter(%)': float(request.form['MDVPJitter']),
            'MDVP:Jitter(Abs)': float(request.form['MDVPJitterAbs']),
            'MDVP:RAP': float(request.form['MDVPRAP']),
            'MDVP:PPQ': float(request.form['MDVPPPQ']),
            'Jitter:DDP': float(request.form['JitterDDP']),
            'MDVP:Shimmer': float(request.form['MDVPShimmer']),
            'MDVP:Shimmer(dB)': float(request.form['MDVPShimmerdB']),
            'Shimmer:APQ3': float(request.form['ShimmerAPQ3']),
            'Shimmer:APQ5': float(request.form['ShimmerAPQ5']),
            'MDVP:APQ': float(request.form['MDVPAPQ']),
            'Shimmer:DDA': float(request.form['ShimmerDDA']),
            'Noise-to-harmonics ratio (NHR)': float(request.form['NHR']),
            'Harmonics-to-noise ratio (HNR)': float(request.form['HNR']),
            'Relative vocal fundamental frequency variability (RPDE)': float(request.form['RPDE']),
            'DFA (signal fractal scaling exponent)': float(request.form['DFA']),
            'Detrended fluctuation analysis of pitch period': float(request.form['spread1']),
            'Detrended fluctuation analysis of amplitude period': float(request.form['spread2']),
            'Correlation dimension': float(request.form['D2']),
            'Pitch period entropy': float(request.form['PPE']),
        }
        
        parkinsons_prediction = parkinsons_model.predict([list(labels.values())])
        parkinsons_probabilities = parkinsons_model.predict_proba([list(labels.values())])
        
        result = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
        prob_parkinsons = round(parkinsons_probabilities[0][1] * 100, 2)
        prob_no_parkinsons = round(parkinsons_probabilities[0][0] * 100, 2)
        
        generate_report(user_id, 'Parkinsons Disease', result,prob_parkinsons, prob_no_parkinsons, labels)
        
        return render_template('parkinsons.html', result=result, prob_parkinsons=prob_parkinsons, prob_no_parkinsons=prob_no_parkinsons)
    
    return render_template('parkinsons.html')


@app.route('/symptoms', methods=['GET', 'POST'])
@login_required
def symptoms():
    if request.method == 'POST':
        selected_symptoms = []
        for key, weight in utils.symptom_weights.items():
            if request.form.get(key):
                selected_symptoms.append((key, weight))
        
        top_17_symptoms = sorted(selected_symptoms, key=lambda x: x[1], reverse=True)[:17]

        if len(top_17_symptoms) < 17:
            top_17_symptoms += [('other', 0)] * (17 - len(top_17_symptoms))
        
        input_data = [[weight for _, weight in top_17_symptoms]]
        predictions = svm_model.predict(input_data)
        predicted_disease_names = label_encoder.inverse_transform(predictions)
        precautions_list = []
        descriptions_list = []
        for disease_name in predicted_disease_names:
            if not df_precautions[df_precautions['Disease'] == disease_name].empty:
                precautions_row = df_precautions[df_precautions['Disease'] == disease_name].iloc[:, 1:].values.tolist()[0]
            else:
                precautions_row = ["Precautions not available because the disease cannot be identified."]
            precautions_list.append(precautions_row)
            
            if not df_description[df_description['Disease'] == disease_name].empty:
                description = df_description[df_description['Disease'] == disease_name]['Description'].values[0]
            else:
                description = "Disease description not available because the disease cannot be identified."
            descriptions_list.append(description)

        generate_report(session['user_id'], 'Symptom Analysis', ', '.join(predicted_disease_names), 0, 0, {k: v for k,v in selected_symptoms})

        return render_template('symptoms.html', selected_symptoms=[symptom for symptom, _ in top_17_symptoms], predicted_diseases=predicted_disease_names, precautions=precautions_list, descriptions=descriptions_list)

    return render_template('symptoms.html')


@app.route('/reset', methods=['POST','GET'])
def reset_symptoms():
    predicted_diseases = []
    precautions = []
    descriptions = []
    return render_template('symptoms.html', predicted_diseases=predicted_diseases, precautions=precautions, descriptions=descriptions)

# Routes for user reports and report downloads
@app.route('/reports')
@login_required
def user_reports():
    user_id = session.get('user_id')
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, time, report_type FROM reports WHERE user_id = ?', (user_id,))
    reports = cursor.fetchall()
    conn.close()
    return render_template('reports.html', reports=reports)

@app.route('/download_report/<int:report_id>')
def download_report(report_id):
    conn = sqlite3.connect('reports.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM reports WHERE id = ?', (report_id,))
    report = cursor.fetchone()
    conn.close()
    if report:
        report_id, user_id, report_type, report_data, time = report
        pdf_bytes = BytesIO(report_data)
        return send_file(pdf_bytes,as_attachment=True,download_name=f'report_{report_id}.pdf')
    else:
        return 'Report not found', 404


if __name__ == '__main__':
    app.run(debug=True)
