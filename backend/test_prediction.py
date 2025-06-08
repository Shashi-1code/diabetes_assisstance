import joblib
import numpy as np
import pandas as pd

def predict_diabetes(input_data):
    """
    Make a diabetes prediction using the trained model.
    
    Parameters:
    -----------
    input_data : dict
        Dictionary containing the input features
        
    Returns:
    --------
    tuple
        (prediction, probability) where prediction is 0 or 1 and probability is the confidence
    """
    # Load the model and scaler
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    
    # Convert input dictionary to array
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_array = np.array([[input_data[feature] for feature in feature_names]])
    
    # Scale the input data
    input_scaled = scaler.transform(input_array)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    return prediction, probability

if __name__ == "__main__":
    # Test input
    test_input = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,  # This will be imputed during preprocessing
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }
    
    # Make prediction
    prediction, probability = predict_diabetes(test_input)
    
    # Print results
    print("\nTest Input:")
    for feature, value in test_input.items():
        print(f"{feature}: {value}")
    
    print("\nPrediction Results:")
    print(f"Prediction (0=No Diabetes, 1=Diabetes): {prediction}")
    print(f"Probability of Diabetes: {probability:.1%}")
    
    # Print feature importance
    model = joblib.load('diabetes_model.pkl')
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    importances = model.feature_importances_
    print("\nFeature Importance:")
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.3f}") 