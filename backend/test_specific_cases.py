import pandas as pd
import numpy as np
from test_prediction import predict_diabetes

def test_specific_cases():
    # Test case 1
    case1 = {
        'Pregnancies': 15,
        'Glucose': 136,
        'BloodPressure': 70,
        'SkinThickness': 32,
        'Insulin': 110,
        'BMI': 37.1,
        'DiabetesPedigreeFunction': 0.153,
        'Age': 43
    }
    
    # Test case 2
    case2 = {
        'Pregnancies': 7,
        'Glucose': 81,
        'BloodPressure': 78,
        'SkinThickness': 40,
        'Insulin': 48,
        'BMI': 46.7,
        'DiabetesPedigreeFunction': 0.261,
        'Age': 42
    }
    
    # Get predictions
    pred1, prob1 = predict_diabetes(case1)
    pred2, prob2 = predict_diabetes(case2)
    
    print("\nTest Case 1 (Actual Outcome: 1)")
    print("Input values:")
    for feature, value in case1.items():
        print(f"{feature}: {value}")
    print(f"\nPrediction: {pred1}")
    print(f"Probability: {prob1:.1%}")
    
    print("\n" + "="*50 + "\n")
    
    print("Test Case 2 (Actual Outcome: 0)")
    print("Input values:")
    for feature, value in case2.items():
        print(f"{feature}: {value}")
    print(f"\nPrediction: {pred2}")
    print(f"Probability: {prob2:.1%}")
    
    # Load the dataset to verify these cases
    df = pd.read_csv('diabetes.csv')
    
    # Find these exact cases in the dataset
    print("\nVerifying cases in dataset:")
    
    # Function to find matching rows
    def find_matching_row(row_data):
        mask = True
        for col, val in row_data.items():
            mask = mask & (df[col] == val)
        return df[mask]
    
    case1_matches = find_matching_row(case1)
    case2_matches = find_matching_row(case2)
    
    print("\nCase 1 matches in dataset:")
    print(case1_matches)
    print("\nCase 2 matches in dataset:")
    print(case2_matches)

if __name__ == "__main__":
    test_specific_cases() 