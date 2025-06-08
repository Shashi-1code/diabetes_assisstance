import pandas as pd
import matplotlib.pyplot as plt

# Load the processed dataset
df = pd.read_csv('processed_diabetes.csv')

# Calculate the distribution
total_cases = len(df)
diabetes_cases = df['Outcome'].sum()
non_diabetes_cases = total_cases - diabetes_cases

# Calculate percentages
diabetes_percentage = (diabetes_cases / total_cases) * 100
non_diabetes_percentage = (non_diabetes_cases / total_cases) * 100

# Print the distribution
print("\nDiabetes Distribution in Processed Dataset:")
print(f"Total cases: {total_cases}")
print(f"Diabetes cases: {diabetes_cases} ({diabetes_percentage:.1f}%)")
print(f"Non-diabetes cases: {non_diabetes_cases} ({non_diabetes_percentage:.1f}%)")

# Print summary statistics for key features by outcome
print("\nSummary Statistics by Diabetes Status:")
key_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
for feature in key_features:
    print(f"\n{feature}:")
    print("Diabetes cases:")
    print(df[df['Outcome'] == 1][feature].describe())
    print("\nNon-diabetes cases:")
    print(df[df['Outcome'] == 0][feature].describe())

# Create a pie chart
plt.figure(figsize=(8, 6))
labels = ['Diabetes', 'Non-Diabetes']
sizes = [diabetes_cases, non_diabetes_cases]
colors = ['#ff9999', '#66b3ff']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Distribution of Diabetes Cases in Processed Dataset')
plt.savefig('processed_diabetes_distribution.png')
plt.close()

# Create modified datasets with different distributions
def create_modified_dataset(df, target_percentage, output_file):
    """
    Create a modified dataset with a specific target percentage of diabetes cases.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataset
    target_percentage : float
        Target percentage of diabetes cases (0-100)
    output_file : str
        Name of the output CSV file
    """
    # Calculate target number of cases
    total_cases = len(df)
    target_diabetes_cases = int((target_percentage / 100) * total_cases)
    
    # Get diabetes and non-diabetes cases
    diabetes_df = df[df['Outcome'] == 1]
    non_diabetes_df = df[df['Outcome'] == 0]
    
    # If we need more diabetes cases
    if target_percentage > diabetes_percentage:
        # Calculate how many more diabetes cases we need
        additional_cases_needed = target_diabetes_cases - len(diabetes_df)
        # Duplicate some diabetes cases
        additional_cases = diabetes_df.sample(n=additional_cases_needed, replace=True)
        diabetes_df = pd.concat([diabetes_df, additional_cases])
        # Remove some non-diabetes cases to maintain total
        non_diabetes_df = non_diabetes_df.sample(n=total_cases - len(diabetes_df))
    # If we need fewer diabetes cases
    else:
        # Calculate how many diabetes cases to keep
        diabetes_df = diabetes_df.sample(n=target_diabetes_cases)
        # Add non-diabetes cases to maintain total
        non_diabetes_df = non_diabetes_df.sample(n=total_cases - len(diabetes_df))
    
    # Combine the datasets
    modified_df = pd.concat([diabetes_df, non_diabetes_df])
    # Shuffle the dataset
    modified_df = modified_df.sample(frac=1, random_state=42)
    
    # Save to CSV
    modified_df.to_csv(output_file, index=False)
    
    # Print statistics
    actual_percentage = (len(modified_df[modified_df['Outcome'] == 1]) / len(modified_df)) * 100
    print(f"\nCreated {output_file} with {actual_percentage:.1f}% diabetes cases")

# Create datasets with different distributions
create_modified_dataset(df, 75, 'processed_diabetes_75_percent.csv')  # 75% diabetes cases
create_modified_dataset(df, 25, 'processed_diabetes_25_percent.csv')  # 25% diabetes cases 