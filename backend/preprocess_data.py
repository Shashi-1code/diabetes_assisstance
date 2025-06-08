import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def preprocess_diabetes_data(file_path='diabetes.csv'):
    """
    Preprocess the diabetes dataset by handling missing values and scaling features.
    
    Parameters:
    -----------
    file_path : str
        Path to the diabetes dataset CSV file
        
    Returns:
    --------
    tuple
        (X, y) where X is the preprocessed features and y is the target variable
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Create a copy of the original dataframe for saving processed data
    processed_df = df.copy()
    
    # Identify columns that should not have zero values
    zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Replace zeros with NaN in these columns
    for col in zero_not_allowed:
        processed_df.loc[processed_df[col] == 0, col] = np.nan
    
    # Print missing value statistics
    print("\nMissing value statistics before imputation:")
    print(processed_df[zero_not_allowed].isna().sum())
    
    # Use KNN imputation for better accuracy
    # KNN imputation uses similar patients to estimate missing values
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    
    # Fit and transform the data
    imputed_data = imputer.fit_transform(processed_df[zero_not_allowed])
    
    # Update the dataframe with imputed values
    processed_df[zero_not_allowed] = imputed_data
    
    # Print statistics after imputation
    print("\nMissing value statistics after imputation:")
    print(processed_df[zero_not_allowed].isna().sum())
    
    # Print summary statistics of imputed columns
    print("\nSummary statistics after imputation:")
    print(processed_df[zero_not_allowed].describe())
    
    # Save the processed data to a new CSV file
    processed_file_path = 'processed_diabetes.csv'
    processed_df.to_csv(processed_file_path, index=False)
    print(f"\nProcessed data saved to {processed_file_path}")
    
    # Separate features and target
    X = processed_df.drop('Outcome', axis=1)
    y = processed_df['Outcome']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for later use
    import joblib
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    return X_scaled, y

if __name__ == "__main__":
    # Test the preprocessing
    X, y = preprocess_diabetes_data()
    print("\nShape of preprocessed data:")
    print(f"Features (X): {X.shape}")
    print(f"Target (y): {y.shape}") 