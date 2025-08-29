from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
from flask_cors import CORS
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# Flask app
app = Flask(__name__)
CORS(app)

# Import email configuration
try:
    from email_config import EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASSWORD, RECIPIENT_EMAIL
except ImportError:
    # Fallback configuration (you need to set up email_config.py)
    EMAIL_HOST = 'smtp.gmail.com'
    EMAIL_PORT = 587
    EMAIL_USER = 'your-email@gmail.com'  # Replace with your Gmail
    EMAIL_PASSWORD = 'your-app-password'  # Replace with your Gmail app password
    RECIPIENT_EMAIL = 'shreyasssanil62@gmail.com'

# Load datasets
sym_des = pd.read_csv("src/datasets/symtoms_df.csv")
precautions = pd.read_csv("src/datasets/precautions_df.csv")
workout = pd.read_csv("src/datasets/workout_df.csv")
description = pd.read_csv("src/datasets/description.csv")
medications = pd.read_csv('src/datasets/medications.csv')
diets = pd.read_csv("src/datasets/diets.csv")

# Load model
svc = pickle.load(open('src/models/svc.pkl', 'rb'))

# Helper functions
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([str(w) for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [str(col) for row in pre.values for col in row]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [str(med) for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [str(die) for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = [str(wrk) for wrk in wrkout.values]

    return desc, pre, med, die, wrkout

def send_email(subject, body):
    """Send email using Gmail SMTP with multiple fallback methods"""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = subject

        # Add body to email
        msg.attach(MIMEText(body, 'plain'))

        # Method 1: Try with regular SMTP (port 587)
        try:
            print("Trying SMTP method 1 (port 587)...")
            server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            text = msg.as_string()
            server.sendmail(EMAIL_USER, RECIPIENT_EMAIL, text)
            server.quit()
            print(f"Email sent successfully to {RECIPIENT_EMAIL}")
            return True
        except Exception as e1:
            print(f"Method 1 failed: {e1}")
            
            # Method 2: Try with SSL (port 465)
            try:
                print("Trying SMTP method 2 (port 465)...")
                server = smtplib.SMTP_SSL(EMAIL_HOST, 465)
                server.login(EMAIL_USER, EMAIL_PASSWORD)
                text = msg.as_string()
                server.sendmail(EMAIL_USER, RECIPIENT_EMAIL, text)
                server.quit()
                print(f"Email sent successfully to {RECIPIENT_EMAIL}")
                return True
            except Exception as e2:
                print(f"Method 2 failed: {e2}")
                
                # Method 3: Try without TLS
                try:
                    print("Trying SMTP method 3 (no TLS)...")
                    server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
                    server.login(EMAIL_USER, EMAIL_PASSWORD)
                    text = msg.as_string()
                    server.sendmail(EMAIL_USER, RECIPIENT_EMAIL, text)
                    server.quit()
                    print(f"Email sent successfully to {RECIPIENT_EMAIL}")
                    return True
                except Exception as e3:
                    print(f"Method 3 failed: {e3}")
                    return False
        
    except Exception as e:
        print(f"Error in send_email: {e}")
        return False

# Symptom dictionary
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}


# Model prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[int(svc.predict([input_vector])[0])]

# Routes
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms', [])
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400
    try:
        disease = get_predicted_value(symptoms)
        desc, pre, med, die, wrkout = helper(disease)
        response_data = {
            'disease': disease,
            'description': desc,
            'precautions': pre,
            'medications': med,
            'diets': die,
            'workouts': wrkout
        }
        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def send_feedback():
    try:
        data = request.get_json()
        
        # Extract feedback data
        name = data.get('name', 'Anonymous')
        email = data.get('email', 'No email provided')
        subject = data.get('subject', 'No subject')
        message = data.get('message', 'No message')
        rating = data.get('rating', 5)
        category = data.get('category', 'general')
        
        # Create email content
        email_subject = f"Health AI Feedback: {subject}"
        email_body = f"""
New Feedback from Health AI Website

Name: {name}
Email: {email}
Category: {category}
Rating: {rating}/5
Subject: {subject}

Message:
{message}

---
This feedback was submitted through the Health AI website.
        """
        
        # Try to send email
        email_sent = send_email(email_subject, email_body)
        
        if email_sent:
            return jsonify({
                'success': True,
                'message': 'Feedback sent successfully!'
            }), 200
        else:
            # Fallback: just log to console if email fails
            print("=" * 50)
            print("FEEDBACK EMAIL RECEIVED (Console Log)")
            print("=" * 50)
            print(f"To: {RECIPIENT_EMAIL}")
            print(f"Subject: {email_subject}")
            print(f"Body: {email_body}")
            print("=" * 50)
            
            return jsonify({
                'success': True,
                'message': 'Feedback received! (Email service not configured)'
            }), 200
        
    except Exception as e:
        print(f"Error sending feedback: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to send feedback. Please try again.'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
