import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load('diabetes_model.pkl')

# Test input
test_input = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Make prediction
prediction = model.predict(test_input)[0]
probability = model.predict_proba(test_input)[0][1]

print("\nTest Input:")
print("Pregnancies: 6")
print("Glucose: 148")
print("BloodPressure: 72")
print("SkinThickness: 35")
print("Insulin: 0")
print("BMI: 33.6")
print("DiabetesPedigreeFunction: 0.627")
print("Age: 50")
print("\nPrediction Results:")
print(f"Prediction (0=No Diabetes, 1=Diabetes): {prediction}")
print(f"Probability of Diabetes: {probability:.1%}")

# Additional analysis
print("\nFeature Importance:")
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
importances = model.feature_importances_
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.3f}") 