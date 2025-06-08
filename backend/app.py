from flask import Flask, request, jsonify, session
from flask_cors import CORS
import speech_recognition as sr
import joblib
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from pydub import AudioSegment
import io

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Use environment variable for secret key, fallback to generated key if not set
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(32).hex())
CORS(app, supports_credentials=True)

# Load the trained model
model = joblib.load('diabetes_model.pkl')

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Define questions and their descriptions
questions = [
    ("Gender", "What is your gender? (male/female)"),
    ("Pregnancies", "How many times have you been pregnant? (Valid range: 0-17)"),  # Will be skipped for males
    ("Glucose", "What is your glucose level in mg/dL? (Normal range: 70-140 mg/dL, accepted range: 40-400 mg/dL)"),
    ("BloodPressure", "What is your blood pressure in mmHg? (Normal range: 90-140 mmHg, accepted range: 60-250 mmHg)"),
    ("SkinThickness", "What is your triceps skinfold thickness in mm? (Valid range: 0-99 mm)"),
    ("Insulin", "What is your insulin level in μU/mL? (Normal range: 2.6-24.9 μU/mL, accepted range: 0-846 μU/mL)"),
    ("BMI", "What is your BMI? (Normal range: 18.5-24.9, accepted range: 10-70)"),
    ("DiabetesPedigreeFunction", "What is your diabetes pedigree function value? (Valid range: 0.078-2.42)"),
    ("Age", "What is your age in years? (Valid range: 21-81 years)")
]

# Define follow-up questions based on certain conditions
follow_up_questions = {
    "Glucose": {
        "high": [
            ("GlucoseFasting", "Was this measurement taken while fasting? (yes/no)"),
            ("GlucoseTime", "When was this measurement taken? (morning/afternoon/evening)"),
            ("GlucoseSymptoms", "Are you experiencing any symptoms like increased thirst or frequent urination? (yes/no)"),
            ("GlucoseHistory", "Have you had high glucose readings before? (yes/no)"),
            ("GlucoseMedication", "Are you currently taking any diabetes medication? (yes/no)"),
            ("GlucoseFamily", "Do you have any family members with diabetes? (yes/no)"),
            ("GlucoseDiet", "How would you describe your typical diet? (healthy/moderate/poor)"),
            ("GlucoseExercise", "How often do you exercise? (never/occasionally/regularly)")
        ],
        "low": [
            ("GlucoseSymptoms", "Are you experiencing any symptoms like dizziness or sweating? (yes/no)"),
            ("GlucoseLastMeal", "When did you last eat? (hours ago)"),
            ("GlucoseMedication", "Are you taking any medications that might affect blood sugar? (yes/no)"),
            ("GlucoseHistory", "Have you had low glucose readings before? (yes/no)")
        ]
    },
    "BloodPressure": {
        "high": [
            ("BPMedication", "Are you currently taking any blood pressure medication? (yes/no)"),
            ("BPStress", "Are you currently experiencing stress? (yes/no)"),
            ("BPActivity", "Were you physically active before this measurement? (yes/no)"),
            ("BPHistory", "Have you had high blood pressure before? (yes/no)"),
            ("BPSalt", "How would you describe your salt intake? (low/moderate/high)"),
            ("BPFamily", "Do you have any family members with high blood pressure? (yes/no)"),
            ("BPSleep", "How many hours of sleep do you typically get? (hours)"),
            ("BPCaffeine", "How many caffeinated drinks do you have daily? (number)")
        ],
        "low": [
            ("BPSymptoms", "Are you experiencing any symptoms like dizziness or fatigue? (yes/no)"),
            ("BPMedication", "Are you currently taking any blood pressure medication? (yes/no)"),
            ("BPHistory", "Have you had low blood pressure before? (yes/no)"),
            ("BPHydration", "How much water do you drink daily? (glasses)"),
            ("BPStanding", "Do you feel dizzy when standing up quickly? (yes/no)")
        ]
    },
    "BMI": {
        "high": [
            ("BMIActivity", "How often do you exercise? (never/occasionally/regularly)"),
            ("BMIDiet", "How would you describe your diet? (healthy/moderate/poor)"),
            ("BMIWeightHistory", "Has your weight changed significantly in the last year? (yes/no)"),
            ("BMIFamily", "Do you have any family members with weight-related health issues? (yes/no)"),
            ("BMISleep", "How many hours of sleep do you typically get? (hours)"),
            ("BMISedentary", "How many hours do you spend sitting daily? (hours)"),
            ("BMIMealPattern", "How many meals do you eat per day? (number)"),
            ("BMISnacking", "How often do you snack between meals? (never/occasionally/frequently)")
        ],
        "low": [
            ("BMIAppetite", "Have you experienced any loss of appetite? (yes/no)"),
            ("BMIWeightHistory", "Has your weight changed significantly in the last year? (yes/no)"),
            ("BMIMedical", "Are you currently being treated for any medical conditions? (yes/no)"),
            ("BMIDiet", "How would you describe your diet? (healthy/moderate/poor)"),
            ("BMISymptoms", "Are you experiencing any other symptoms? (yes/no)"),
            ("BMIFamily", "Do you have any family members with similar weight patterns? (yes/no)")
        ]
    },
    "Age": {
        "elderly": [
            ("AgeActivity", "How would you describe your physical activity level? (sedentary/moderate/active)"),
            ("AgeMobility", "Do you have any mobility issues? (yes/no)"),
            ("AgeMedication", "How many medications do you take daily? (number)"),
            ("AgeSupport", "Do you have family or caregiver support? (yes/no)"),
            ("AgeCheckups", "How often do you get medical checkups? (monthly/quarterly/yearly/rarely)")
        ]
    }
}

# Define preventive measures information
preventive_measures = {
    "general": {
        "title": "General Preventive Measures",
        "measures": [
            "1. Regular Health Check-ups:",
            "   • Annual physical examination",
            "   • Regular blood sugar monitoring",
            "   • Blood pressure checks",
            "   • Cholesterol screening",
            "   • Eye examination (for diabetic retinopathy)",
            "   • Foot examination (for diabetic neuropathy)",
            "",
            "2. Healthy Diet:",
            "   • Follow a balanced diet rich in fruits, vegetables, and whole grains",
            "   • Limit processed foods and sugary drinks",
            "   • Control portion sizes",
            "   • Choose lean proteins",
            "   • Stay hydrated with water",
            "   • Limit alcohol consumption",
            "",
            "3. Physical Activity:",
            "   • Aim for 150 minutes of moderate exercise weekly",
            "   • Include both cardio and strength training",
            "   • Take regular breaks from sitting",
            "   • Find activities you enjoy",
            "   • Start slowly and gradually increase intensity",
            "",
            "4. Weight Management:",
            "   • Maintain a healthy BMI (18.5-24.9)",
            "   • Set realistic weight loss goals",
            "   • Track your progress",
            "   • Get support from healthcare providers",
            "   • Focus on sustainable lifestyle changes",
            "",
            "5. Stress Management:",
            "   • Practice relaxation techniques",
            "   • Get adequate sleep (7-8 hours)",
            "   • Maintain work-life balance",
            "   • Consider meditation or yoga",
            "   • Seek support when needed",
            "",
            "6. Lifestyle Modifications:",
            "   • Quit smoking",
            "   • Limit alcohol intake",
            "   • Maintain regular sleep schedule",
            "   • Stay socially active",
            "   • Regular dental check-ups"
        ]
    },
    "high_risk": {
        "title": "Additional Measures for High-Risk Individuals",
        "measures": [
            "1. Enhanced Monitoring:",
            "   • More frequent blood sugar checks",
            "   • Regular A1C testing",
            "   • Blood pressure monitoring at home",
            "   • Weight tracking",
            "   • Symptom diary maintenance",
            "",
            "2. Medical Management:",
            "   • Regular consultations with healthcare provider",
            "   • Medication adherence",
            "   • Regular lab work",
            "   • Specialist referrals as needed",
            "   • Vaccination updates",
            "",
            "3. Dietary Modifications:",
            "   • Consult a registered dietitian",
            "   • Meal planning",
            "   • Carbohydrate counting",
            "   • Regular meal timing",
            "   • Healthy snack options",
            "",
            "4. Exercise Guidelines:",
            "   • Medical clearance before starting",
            "   • Gradual progression",
            "   • Regular activity schedule",
            "   • Exercise with a partner",
            "   • Emergency contact information",
            "",
            "5. Emergency Preparedness:",
            "   • Keep emergency contacts handy",
            "   • Wear medical identification",
            "   • Know symptoms of complications",
            "   • Have glucose tablets/snacks available",
            "   • Regular emergency plan review"
        ]
    },
    "age_specific": {
        "elderly": {
            "title": "Special Considerations for Elderly (65+)",
            "measures": [
                "1. Modified Exercise:",
                "   • Low-impact activities",
                "   • Balance exercises",
                "   • Regular walking",
                "   • Chair exercises",
                "   • Water aerobics",
                "",
                "2. Medication Management:",
                "   • Regular medication review",
                "   • Pill organizer use",
                "   • Medication reminder system",
                "   • Regular doctor consultations",
                "   • Side effect monitoring",
                "",
                "3. Fall Prevention:",
                "   • Home safety assessment",
                "   • Regular vision checks",
                "   • Proper footwear",
                "   • Assistive devices if needed",
                "   • Regular balance exercises",
                "",
                "4. Social Support:",
                "   • Regular check-ins",
                "   • Support group participation",
                "   • Caregiver communication",
                "   • Transportation assistance",
                "   • Meal delivery services if needed"
            ]
        },
        "young_adult": {
            "title": "Special Considerations for Young Adults",
            "measures": [
                "1. Lifestyle Balance:",
                "   • Work-life balance",
                "   • Stress management",
                "   • Regular sleep schedule",
                "   • Healthy social activities",
                "   • Time management",
                "",
                "2. Preventive Screening:",
                "   • Regular health check-ups",
                "   • Family planning considerations",
                "   • Mental health monitoring",
                "   • Dental care",
                "   • Vision checks",
                "",
                "3. Healthy Habits:",
                "   • Regular exercise routine",
                "   • Meal preparation",
                "   • Stress reduction techniques",
                "   • Social support network",
                "   • Health education"
            ]
        }
    }
}

def extract_number_from_text(text):
    """Extract numeric value from text input."""
    try:
        import re
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text.lower())
        if numbers:
            return float(numbers[0])
        return None
    except:
        return None

def get_next_question():
    """Get the next unanswered question."""
    if 'answers' not in session:
        session['answers'] = {}
        session['question_index'] = 0
    
    index = session['question_index']
    if index >= len(questions):
        return None
    
    # Skip pregnancy question for males
    if index == 1 and session.get('answers', {}).get('Gender') == 'male':
        session['answers']['Pregnancies'] = 0  # Set pregnancies to 0 for males
        session['question_index'] += 1
        index = session['question_index']
    
    return questions[index]

def get_follow_up_questions(field, value):
    """Get relevant follow-up questions based on the field and value."""
    if field not in follow_up_questions:
        return []
    
    questions = []
    if field == "Glucose":
        if value > 140:  # High glucose
            questions.extend(follow_up_questions["Glucose"]["high"])
        elif value < 70:  # Low glucose
            questions.extend(follow_up_questions["Glucose"]["low"])
    elif field == "BloodPressure":
        if value > 140:  # High blood pressure
            questions.extend(follow_up_questions["BloodPressure"]["high"])
        elif value < 90:  # Low blood pressure
            questions.extend(follow_up_questions["BloodPressure"]["low"])
    elif field == "BMI":
        if value > 24.9:  # High BMI
            questions.extend(follow_up_questions["BMI"]["high"])
        elif value < 18.5:  # Low BMI
            questions.extend(follow_up_questions["BMI"]["low"])
    elif field == "Age":
        if value > 65:  # Elderly
            questions.extend(follow_up_questions["Age"]["elderly"])
    
    return questions

def process_input(value, field):
    """Process and validate the input value with medically appropriate ranges."""
    try:
        if field == 'Gender':
            value = str(value).strip().lower()
            if value in ['m', 'male', 'man', 'boy']:
                return True, 'male'
            elif value in ['f', 'female', 'woman', 'girl']:
                return True, 'female'
            else:
                return False, "Please provide a valid gender (male/female)."
        
        num_value = float(value)
        # Medical validation ranges for each field
        if field == 'Pregnancies':
            if num_value < 0 or num_value > 17:  # Maximum recorded pregnancies is 17
                return False, "Please provide a valid number of pregnancies (0-17)."
        elif field == 'Glucose':
            if num_value < 40 or num_value > 400:  # Normal range: 70-140 mg/dL, but allowing wider range for fasting/after meals
                return False, "Please provide a valid glucose level (40-400 mg/dL). Normal range is 70-140 mg/dL."
        elif field == 'BloodPressure':
            if num_value < 60 or num_value > 250:  # Normal range: 90-140 mmHg systolic
                return False, "Please provide a valid blood pressure (60-250 mmHg). Normal range is 90-140 mmHg."
        elif field == 'SkinThickness':
            if num_value < 0 or num_value > 99:  # Triceps skinfold thickness in mm
                return False, "Please provide a valid skin thickness (0-99 mm). This is measured at the triceps."
        elif field == 'Insulin':
            if num_value < 0 or num_value > 846:  # Normal range: 2.6-24.9 μU/mL, but allowing wider range
                return False, "Please provide a valid insulin level (0-846 μU/mL). Normal range is 2.6-24.9 μU/mL."
        elif field == 'BMI':
            if num_value < 10 or num_value > 70:  # Normal range: 18.5-24.9, but allowing wider range
                return False, "Please provide a valid BMI (10-70). Normal range is 18.5-24.9."
        elif field == 'DiabetesPedigreeFunction':
            if num_value < 0.078 or num_value > 2.42:  # Based on Pima Indians dataset range
                return False, "Please provide a valid diabetes pedigree function value (0.078-2.42)."
        elif field == 'Age':
            if num_value < 21 or num_value > 81:  # Based on Pima Indians dataset range
                return False, "Please provide a valid age (21-81 years)."
        return True, num_value
    except ValueError:
        if field == 'Gender':
            return False, "Please provide a valid gender (male/female)."
        return False, f"Please provide a valid number for {field.lower()}."

def process_follow_up_input(value, field):
    """Process and validate follow-up question inputs."""
    # Normalize input
    value = str(value).strip().lower()
    
    # Handle common variations of yes/no
    if value in ['y', 'yes', 'yeah', 'yep', 'sure', 'okay', 'yup', 'correct', 'right']:
        value = 'yes'
    elif value in ['n', 'no', 'nope', 'nah', 'never', 'negative', 'incorrect', 'wrong']:
        value = 'no'
    
    # Debug logging
    print(f"Processing follow-up input - Field: {field}, Value: {value}")
    
    if field.startswith("Glucose"):
        if field in ["GlucoseFasting", "GlucoseSymptoms", "GlucoseHistory", "GlucoseMedication", "GlucoseFamily"]:
            return value in ["yes", "no"], value
        elif field == "GlucoseTime":
            valid_times = ["morning", "afternoon", "evening"]
            return value in valid_times, value
        elif field == "GlucoseLastMeal":
            try:
                hours = float(value)
                return 0 <= hours <= 24, hours
            except:
                return False, "Please provide a valid number of hours (0-24)."
        elif field in ["GlucoseDiet", "GlucoseExercise"]:
            if field == "GlucoseDiet":
                valid_responses = ["healthy", "moderate", "poor"]
            else:  # GlucoseExercise
                valid_responses = ["never", "occasionally", "regularly"]
            if value in valid_responses:
                return True, value
            return False, f"Please choose one of: {', '.join(valid_responses)}"
    
    elif field.startswith("BP"):
        if field in ["BPMedication", "BPStress", "BPActivity", "BPHistory", "BPFamily", "BPStanding", "BPSymptoms"]:
            return value in ["yes", "no"], value
        elif field == "BPSalt":
            valid_responses = ["low", "moderate", "high"]
            if value in valid_responses:
                return True, value
            return False, f"Please choose one of: {', '.join(valid_responses)}"
        elif field in ["BPSleep", "BPCaffeine"]:
            try:
                num = float(value)
                return num >= 0, num
            except:
                return False, "Please provide a valid number."
        elif field == "BPHydration":
            try:
                glasses = float(value)
                return 0 <= glasses <= 20, glasses
            except:
                return False, "Please provide a valid number of glasses (0-20)."
    
    elif field.startswith("BMI"):
        if field in ["BMIAppetite", "BMIWeightHistory", "BMIMedical", "BMISymptoms", "BMIFamily"]:
            return value in ["yes", "no"], value
        elif field in ["BMIActivity", "BMIDiet"]:
            if field == "BMIActivity":
                valid_responses = ["never", "occasionally", "regularly"]
            else:  # BMIDiet
                valid_responses = ["healthy", "moderate", "poor"]
            if value in valid_responses:
                return True, value
            return False, f"Please choose one of: {', '.join(valid_responses)}"
        elif field in ["BMISleep", "BMISedentary"]:
            try:
                hours = float(value)
                return 0 <= hours <= 24, hours
            except:
                return False, "Please provide a valid number of hours (0-24)."
        elif field == "BMIMealPattern":
            try:
                meals = float(value)
                return 1 <= meals <= 6, meals
            except:
                return False, "Please provide a valid number of meals (1-6)."
        elif field == "BMISnacking":
            valid_responses = ["never", "occasionally", "frequently"]
            if value in valid_responses:
                return True, value
            return False, f"Please choose one of: {', '.join(valid_responses)}"
    
    elif field.startswith("Age"):
        if field in ["AgeMobility", "AgeSupport"]:
            return value in ["yes", "no"], value
        elif field == "AgeActivity":
            valid_responses = ["sedentary", "moderate", "active"]
            if value in valid_responses:
                return True, value
            return False, f"Please choose one of: {', '.join(valid_responses)}"
        elif field == "AgeMedication":
            try:
                num = float(value)
                return num >= 0, num
            except:
                return False, "Please provide a valid number of medications."
        elif field == "AgeCheckups":
            valid_responses = ["monthly", "quarterly", "yearly", "rarely"]
            if value in valid_responses:
                return True, value
            return False, f"Please choose one of: {', '.join(valid_responses)}"
    
    # Debug logging for unhandled field
    print(f"Unhandled follow-up field: {field}")
    return False, f"Please provide a valid response for {field}. For yes/no questions, please answer with 'yes' or 'no'."

def get_range_feedback(field, value):
    """Get immediate feedback and precautions for values outside normal range."""
    feedback = []
    
    if field == "Pregnancies":
        if value > 5:  # High number of pregnancies
            feedback.append("• Note: Multiple pregnancies can increase diabetes risk.")
            feedback.append("• Regular monitoring of glucose levels is recommended.")
            feedback.append("• Consider discussing with your healthcare provider about gestational diabetes screening.")
    
    elif field == "Glucose":
        if value > 140:  # High glucose
            feedback.append("• Your glucose level is above normal range (70-140 mg/dL).")
            feedback.append("• This could indicate prediabetes or diabetes.")
            feedback.append("• Consider getting a fasting glucose test.")
            feedback.append("• Monitor for symptoms like increased thirst or frequent urination.")
        elif value < 70:  # Low glucose
            feedback.append("• Your glucose level is below normal range (70-140 mg/dL).")
            feedback.append("• This could indicate hypoglycemia.")
            feedback.append("• Be aware of symptoms like dizziness, sweating, or confusion.")
            feedback.append("• Consider eating a small snack if you feel symptoms.")
        elif value > 120:  # Borderline high
            feedback.append("• Your glucose level is slightly elevated.")
            feedback.append("• Consider monitoring your levels regularly.")
            feedback.append("• Maintain a healthy diet and regular exercise.")
    
    elif field == "BloodPressure":
        if value > 140:  # High blood pressure
            feedback.append("• Your blood pressure is above normal range (90-140 mmHg).")
            feedback.append("• This could indicate hypertension.")
            feedback.append("• Consider reducing salt intake and managing stress.")
            feedback.append("• Regular monitoring is recommended.")
        elif value < 90:  # Low blood pressure
            feedback.append("• Your blood pressure is below normal range (90-140 mmHg).")
            feedback.append("• This could indicate hypotension.")
            feedback.append("• Stay hydrated and avoid sudden position changes.")
            feedback.append("• Monitor for symptoms like dizziness or fatigue.")
        elif value > 130:  # Borderline high
            feedback.append("• Your blood pressure is slightly elevated.")
            feedback.append("• Consider monitoring it regularly.")
            feedback.append("• Maintain a healthy lifestyle with regular exercise.")
    
    elif field == "SkinThickness":
        if value > 40:  # High skin thickness
            feedback.append("• Your skin thickness measurement is elevated.")
            feedback.append("• This could be related to insulin resistance.")
            feedback.append("• Consider discussing with your healthcare provider.")
        elif value < 10:  # Low skin thickness
            feedback.append("• Your skin thickness measurement is low.")
            feedback.append("• This might indicate nutritional status.")
            feedback.append("• Consider discussing with your healthcare provider.")
    
    elif field == "Insulin":
        if value > 24.9:  # High insulin
            feedback.append("• Your insulin level is above normal range (2.6-24.9 μU/mL).")
            feedback.append("• This could indicate insulin resistance.")
            feedback.append("• Consider discussing with your healthcare provider.")
            feedback.append("• Regular exercise and healthy diet are important.")
        elif value < 2.6:  # Low insulin
            feedback.append("• Your insulin level is below normal range (2.6-24.9 μU/mL).")
            feedback.append("• This might indicate pancreatic function issues.")
            feedback.append("• Consider discussing with your healthcare provider.")
    
    elif field == "BMI":
        if value > 24.9:  # High BMI
            feedback.append("• Your BMI is above normal range (18.5-24.9).")
            if value > 30:
                feedback.append("• This indicates obesity, which increases diabetes risk.")
                feedback.append("• Consider consulting a healthcare provider for weight management.")
            else:
                feedback.append("• This indicates overweight, which can increase diabetes risk.")
            feedback.append("• Regular exercise and healthy diet are recommended.")
            feedback.append("• Consider consulting a nutritionist for dietary guidance.")
        elif value < 18.5:  # Low BMI
            feedback.append("• Your BMI is below normal range (18.5-24.9).")
            feedback.append("• This indicates underweight, which can affect health.")
            feedback.append("• Consider consulting a healthcare provider.")
            feedback.append("• Focus on healthy weight gain through proper nutrition.")
    
    elif field == "DiabetesPedigreeFunction":
        if value > 1.5:  # High pedigree function
            feedback.append("• Your diabetes pedigree function value is elevated.")
            feedback.append("• This indicates a stronger family history of diabetes.")
            feedback.append("• Regular screening and monitoring are recommended.")
            feedback.append("• Maintain a healthy lifestyle to reduce risk.")
    
    elif field == "Age":
        if value > 65:  # Elderly
            feedback.append("• As you're over 65, regular health screenings are important.")
            feedback.append("• Consider more frequent check-ups.")
            feedback.append("• Focus on maintaining a healthy lifestyle.")
        elif value < 30:  # Young adult
            feedback.append("• While you're young, early prevention is important.")
            feedback.append("• Maintain healthy habits to reduce future risk.")
    
    return feedback

@app.route('/api/process-voice', methods=['POST'])
def process_voice():
    try:
        # Get the audio file from the request
        if 'audio' not in request.files:
            print("Error: No audio file in request")
            return jsonify({
                'status': 'error',
                'message': 'No audio file received'
            })
            
        audio_file = request.files['audio']
        print(f"Received audio file: {audio_file.filename}, Content-Type: {audio_file.content_type}")
        
        # Read the audio data
        audio_data = audio_file.read()
        print(f"Audio data size: {len(audio_data)} bytes")
        
        try:
            # Convert webm/ogg to wav using pydub
            audio = AudioSegment.from_file(io.BytesIO(audio_data))
            print("Successfully loaded audio with pydub")
        except Exception as e:
            print(f"Error converting audio format: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Error converting audio format: {str(e)}'
            })
        
        try:
            # Export as WAV with specific parameters
            wav_io = io.BytesIO()
            audio.export(wav_io, format='wav', parameters=[
                "-ac", "1",  # mono
                "-ar", "16000",  # sample rate
                "-acodec", "pcm_s16le"  # codec
            ])
            wav_io.seek(0)
            print("Successfully converted to WAV format")
        except Exception as e:
            print(f"Error exporting to WAV: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Error converting to WAV format: {str(e)}'
            })
        
        try:
            # Convert speech to text
            with sr.AudioFile(wav_io) as source:
                print("Starting speech recognition...")
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                print(f"Recognized text: {text}")
        except sr.UnknownValueError:
            print("Speech recognition could not understand audio")
            return jsonify({
                'status': 'error',
                'message': "Could not understand the audio. Please speak clearly and try again."
            })
        except sr.RequestError as e:
            print(f"Error with speech recognition service: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': "Error with speech recognition service. Please try again."
            })
        except Exception as e:
            print(f"Error during speech recognition: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f"Error processing speech: {str(e)}"
            })
        
        # Get current question
        current_question = get_next_question()
        if not current_question:
            return jsonify({
                'status': 'error',
                'message': 'All questions have been answered. Please request prediction.'
            })
        
        field, question = current_question
        value = extract_number_from_text(text)
        
        if value is None:
            print(f"Could not extract number from text: {text}")
            return jsonify({
                'status': 'error',
                'message': f"I couldn't understand the number for {field.lower()}. Please try again."
            })
        
        # Process and validate the input
        is_valid, result = process_input(value, field)
        if not is_valid:
            print(f"Invalid input: {value} for field {field}")
            return jsonify({
                'status': 'error',
                'message': result
            })
        
        # Store the answer
        session['answers'][field] = result
        session['question_index'] += 1
        
        # Check for follow-up questions
        follow_ups = get_follow_up_questions(field, result)
        if follow_ups:
            session['pending_follow_ups'] = follow_ups
            next_follow_up = follow_ups.pop(0)
            session['current_follow_up'] = next_follow_up[0]
            session['pending_follow_ups'] = follow_ups
            return jsonify({
                'status': 'success',
                'message': f"Received {field}: {result}. {next_follow_up[1]}",
                'is_follow_up': True,
                'next_follow_up': next_follow_up[0]
            })
        
        # Check if all questions are answered
        if session['question_index'] >= len(questions):
            return jsonify({
                'status': 'success',
                'message': 'All questions answered. Requesting prediction...',
                'is_complete': True
            })
        
        # Get next question
        next_field, next_question = questions[session['question_index']]
        return jsonify({
            'status': 'success',
            'message': f"Received {field}: {result}. {next_question}",
            'next_feature': next_field,
            'is_complete': False
        })
        
    except Exception as e:
        print(f"Unexpected error in process_voice: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': f"Error processing voice input: {str(e)}"
        })

@app.route('/api/process-text', methods=['POST'])
def process_text():
    try:
        data = request.get_json()
        text = data.get('text', '').strip().lower()  # Normalize input
        
        # Check if we're processing a follow-up question
        if 'current_follow_up' in session and session['current_follow_up']:
            field = session['current_follow_up']
            is_valid, result = process_follow_up_input(text, field)
            
            if not is_valid:
                # Get the current follow-up question text
                follow_up_text = ""
                if field.startswith("Glucose"):
                    for category in follow_up_questions["Glucose"].values():
                        for q in category:
                            if q[0] == field:
                                follow_up_text = q[1]
                                break
                elif field.startswith("BP"):
                    for category in follow_up_questions["BloodPressure"].values():
                        for q in category:
                            if q[0] == field:
                                follow_up_text = q[1]
                                break
                elif field.startswith("BMI"):
                    for category in follow_up_questions["BMI"].values():
                        for q in category:
                            if q[0] == field:
                                follow_up_text = q[1]
                                break
                
                return jsonify({
                    'status': 'error',
                    'message': f"Please provide a valid response for: {follow_up_text}",
                    'is_follow_up': True,
                    'next_follow_up': field,
                    'current_question': field,
                    'question_text': follow_up_text
                })
            
            # Store follow-up answer
            if 'follow_up_answers' not in session:
                session['follow_up_answers'] = {}
            session['follow_up_answers'][field] = result
            
            # Get next follow-up question or return to main questions
            follow_ups = session.get('pending_follow_ups', [])
            if follow_ups:
                next_follow_up = follow_ups.pop(0)
                session['current_follow_up'] = next_follow_up[0]
                session['pending_follow_ups'] = follow_ups
                return jsonify({
                    'status': 'success',
                    'message': f"Received {field}: {result}. {next_follow_up[1]}",
                    'is_follow_up': True,
                    'next_follow_up': next_follow_up[0],
                    'current_question': next_follow_up[0],
                    'question_text': next_follow_up[1]
                })
            else:
                # Return to main questions
                session['current_follow_up'] = None
                session['pending_follow_ups'] = []
                current_question = get_next_question()
                if current_question:
                    field, question = current_question
                    return jsonify({
                        'status': 'success',
                        'message': f"Thank you for providing that information. {question}",
                        'next_feature': field,
                        'is_complete': False,
                        'is_follow_up': False,
                        'current_question': field,
                        'question_text': question
                    })
        
        # Process main question
        current_question = get_next_question()
        if not current_question:
            return jsonify({
                'status': 'error',
                'message': 'All questions have been answered. Please request prediction.',
                'is_complete': True
            })
        
        field, question = current_question
        
        # Special handling for gender
        if field == 'Gender':
            is_valid, result = process_input(text, field)  # Use text directly for gender
        else:
            value = extract_number_from_text(text)
            if value is None:
                return jsonify({
                    'status': 'error',
                    'message': f"I couldn't understand the number for {field.lower()}. Please provide a numeric value.",
                    'is_follow_up': False,
                    'current_question': field,
                    'question_text': question
                })
            is_valid, result = process_input(value, field)
        
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': result,
                'is_follow_up': False,
                'current_question': field,
                'question_text': question
            })
        
        # Store the answer
        session['answers'][field] = result
        
        # If gender is male, automatically set pregnancies to 0
        if field == 'Gender' and result == 'male':
            session['answers']['Pregnancies'] = 0
        
        # Get feedback for the value (skip for gender)
        feedback = [] if field == 'Gender' else get_range_feedback(field, result)
        
        # Prepare the response message
        if feedback:
            message = f"Received {field}: {result}.\n\nImportant Note:\n" + "\n".join(feedback) + "\n\n"
        else:
            message = f"Received {field}: {result}. "
        
        # Move to next question
        session['question_index'] += 1
        
        # Get next question (this will handle skipping pregnancy question for males)
        next_question = get_next_question()
        if not next_question:
            return jsonify({
                'status': 'success',
                'message': message + "\nAll questions answered. Requesting prediction...",
                'is_complete': True,
                'is_follow_up': False,
                'has_feedback': bool(feedback)
            })
        
        next_field, next_question_text = next_question
        
        # Check for follow-up questions
        follow_ups = get_follow_up_questions(field, result)
        if follow_ups:
            session['pending_follow_ups'] = follow_ups
            next_follow_up = follow_ups.pop(0)
            session['current_follow_up'] = next_follow_up[0]
            session['pending_follow_ups'] = follow_ups
            message += f"\n{next_follow_up[1]}"
            return jsonify({
                'status': 'success',
                'message': message,
                'is_follow_up': True,
                'next_follow_up': next_follow_up[0],
                'current_question': next_follow_up[0],
                'question_text': next_follow_up[1],
                'has_feedback': bool(feedback)
            })
        
        message += f"{next_question_text}"
        return jsonify({
            'status': 'success',
            'message': message,
            'next_feature': next_field,
            'is_complete': False,
            'is_follow_up': False,
            'current_question': next_field,
            'question_text': next_question_text,
            'has_feedback': bool(feedback)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/predict', methods=['GET'])
def predict():
    if 'answers' not in session or len(session['answers']) < len(questions):
        return jsonify({
            'status': 'error',
            'message': 'Not all questions have been answered yet.'
        })
    
    try:
        # Prepare features for prediction using pandas DataFrame
        features = pd.DataFrame([session['answers']], columns=[
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        # Get follow-up answers for personalized recommendations
        follow_up_answers = session.get('follow_up_answers', {})
        
        # Generate personalized response message
        message = f"Based on the provided information, there is a {probability:.1%} chance of diabetes risk.\n\n"
        
        if prediction == 1:
            message += "Here are your personalized recommendations based on your responses:\n\n"
            
            # Add glucose-specific recommendations
            if session['answers']['Glucose'] > 140:
                message += "• Regarding your glucose levels:\n"
                if follow_up_answers.get('GlucoseFasting') == 'no':
                    message += "  - Consider getting a fasting glucose test\n"
                if follow_up_answers.get('GlucoseSymptoms') == 'yes':
                    message += "  - Please consult a doctor about your symptoms\n"
                if follow_up_answers.get('GlucoseHistory') == 'yes':
                    message += "  - Regular monitoring of your glucose levels is important\n"
                if follow_up_answers.get('GlucoseFamily') == 'yes':
                    message += "  - Given your family history, regular screening is recommended\n"
                if follow_up_answers.get('GlucoseDiet') in ['moderate', 'poor']:
                    message += "  - Consider consulting a nutritionist for dietary guidance\n"
                if follow_up_answers.get('GlucoseExercise') in ['never', 'occasionally']:
                    message += "  - Regular exercise can help manage glucose levels\n"
                message += "  - Monitor your blood sugar regularly\n\n"
            
            # Add blood pressure-specific recommendations
            if session['answers']['BloodPressure'] > 140:
                message += "• Regarding your blood pressure:\n"
                if follow_up_answers.get('BPMedication') == 'no':
                    message += "  - Consider consulting a doctor about blood pressure management\n"
                if follow_up_answers.get('BPStress') == 'yes':
                    message += "  - Practice stress management techniques\n"
                if follow_up_answers.get('BPSalt') == 'high':
                    message += "  - Reduce your salt intake\n"
                if follow_up_answers.get('BPSleep') and float(follow_up_answers['BPSleep']) < 7:
                    message += "  - Aim for 7-8 hours of sleep per night\n"
                if follow_up_answers.get('BPCaffeine') and float(follow_up_answers['BPCaffeine']) > 2:
                    message += "  - Consider reducing caffeine intake\n"
                message += "  - Monitor your blood pressure regularly\n\n"
            
            # Add BMI-specific recommendations
            if session['answers']['BMI'] > 24.9:
                message += "• Regarding your BMI:\n"
                if follow_up_answers.get('BMIActivity') in ['never', 'occasionally']:
                    message += "  - Start a regular exercise routine\n"
                if follow_up_answers.get('BMIDiet') in ['moderate', 'poor']:
                    message += "  - Consider consulting a nutritionist\n"
                if follow_up_answers.get('BMISedentary') and float(follow_up_answers['BMISedentary']) > 8:
                    message += "  - Try to reduce sitting time and take regular breaks\n"
                if follow_up_answers.get('BMISnacking') == 'frequently':
                    message += "  - Consider healthier snacking options\n"
                message += "  - Work with a healthcare provider on a weight management plan\n\n"
            
            # Add age-specific recommendations
            if session['answers']['Age'] > 65:
                message += "• Additional recommendations for your age group:\n"
                if follow_up_answers.get('AgeActivity') == 'sedentary':
                    message += "  - Consider gentle exercises like walking or swimming\n"
                if follow_up_answers.get('AgeMobility') == 'yes':
                    message += "  - Consult a physical therapist for safe exercise options\n"
                if follow_up_answers.get('AgeMedication') and float(follow_up_answers['AgeMedication']) > 3:
                    message += "  - Regular medication review with your doctor is important\n"
                message += "  - Regular health check-ups are essential\n\n"
            
            message += "General recommendations:\n"
            message += "1. Maintain a healthy diet with low sugar and processed foods\n"
            message += "2. Exercise regularly (at least 30 minutes daily)\n"
            message += "3. Monitor blood sugar levels regularly\n"
            message += "4. Maintain a healthy weight\n"
            message += "5. Get regular check-ups with your doctor\n"
            message += "6. Avoid smoking and limit alcohol consumption\n"
            message += "7. Stay hydrated and get adequate sleep\n"
            message += "8. Manage stress through relaxation techniques\n\n"
            message += "Would you like me to provide more specific information about any of these recommendations?"
        else:
            message += "While your risk is lower, here are some personalized recommendations:\n\n"
            
            # Add preventive recommendations based on follow-up answers
            if session['answers']['Glucose'] > 120:
                message += "• Consider monitoring your glucose levels periodically\n"
            if session['answers']['BloodPressure'] > 130:
                message += "• Keep an eye on your blood pressure\n"
            if session['answers']['BMI'] > 23:
                message += "• Consider maintaining a healthy weight through diet and exercise\n"
            
            # Add age-specific preventive recommendations
            if session['answers']['Age'] > 65:
                message += "• As you're over 65, regular health screenings are important\n"
            
            message += "\nGeneral recommendations:\n"
            message += "1. Maintain a healthy lifestyle\n"
            message += "2. Get regular check-ups\n"
            message += "3. Stay physically active\n"
            message += "4. Eat a balanced diet\n"
            message += "5. Monitor your health indicators regularly\n"
            message += "6. Stay informed about diabetes prevention\n\n"
            message += "Would you like more information about preventive measures?"
        
        # Clear session
        session.clear()
        
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'probability': float(probability),
            'message': message
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset the session and start over."""
    session.clear()
    return jsonify({
        'status': 'success',
        'message': 'Session reset successfully. Starting over...',
        'next_question': questions[0][1]
    })

@app.route('/api/current-question', methods=['GET'])
def get_current_question():
    """Get the current question."""
    current_question = get_next_question()
    if not current_question:
        return jsonify({
            'status': 'success',
            'message': 'All questions have been answered.',
            'is_complete': True
        })
    
    field, question = current_question
    return jsonify({
        'status': 'success',
        'field': field,
        'question': question,
        'is_complete': False
    })

@app.route('/api/preventive-measures', methods=['POST'])
def get_preventive_measures():
    """Provide detailed preventive measures based on user's risk profile."""
    try:
        data = request.get_json()
        user_answers = session.get('answers', {})
        follow_up_answers = session.get('follow_up_answers', {})
        
        # Determine which measures to include based on user's profile
        measures_to_include = []
        
        # Always include general measures
        measures_to_include.append(preventive_measures["general"])
        
        # Check for high-risk indicators
        is_high_risk = False
        if user_answers.get('Glucose', 0) > 140:
            is_high_risk = True
        if user_answers.get('BloodPressure', 0) > 140:
            is_high_risk = True
        if user_answers.get('BMI', 0) > 30:
            is_high_risk = True
        if follow_up_answers.get('GlucoseFamily') == 'yes':
            is_high_risk = True
        
        if is_high_risk:
            measures_to_include.append(preventive_measures["high_risk"])
        
        # Add age-specific measures
        age = user_answers.get('Age', 0)
        if age > 65:
            measures_to_include.append(preventive_measures["age_specific"]["elderly"])
        elif age < 30:
            measures_to_include.append(preventive_measures["age_specific"]["young_adult"])
        
        # Compile the response
        response = "Here are detailed preventive measures based on your profile:\n\n"
        for measure_set in measures_to_include:
            response += f"{measure_set['title']}:\n"
            response += "\n".join(measure_set['measures'])
            response += "\n\n"
        
        # Add personalized recommendations based on specific answers
        if user_answers.get('BMI', 0) > 24.9:
            response += "Additional Weight Management Tips:\n"
            response += "• Consider consulting a nutritionist for personalized meal planning\n"
            response += "• Start with small, achievable exercise goals\n"
            response += "• Keep a food and activity diary\n"
            response += "• Join a support group or find an exercise buddy\n\n"
        
        if user_answers.get('BloodPressure', 0) > 130:
            response += "Additional Blood Pressure Management Tips:\n"
            response += "• Reduce sodium intake\n"
            response += "• Practice stress-reduction techniques\n"
            response += "• Monitor blood pressure at home\n"
            response += "• Limit caffeine and alcohol\n\n"
        
        if follow_up_answers.get('BMISedentary') == 'yes' or follow_up_answers.get('BMIActivity') in ['never', 'occasionally']:
            response += "Additional Activity Tips:\n"
            response += "• Start with 10-minute walks\n"
            response += "• Take the stairs instead of the elevator\n"
            response += "• Park further from your destination\n"
            response += "• Stand up and stretch every hour\n"
            response += "• Consider a standing desk\n\n"
        
        response += "Would you like more specific information about any of these areas?"
        
        return jsonify({
            'status': 'success',
            'message': response,
            'has_more_info': True
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000) 