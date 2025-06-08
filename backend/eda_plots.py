import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load datasets
df_raw = pd.read_csv('diabetes.csv')  # raw/original dataset
df_processed = pd.read_csv('processed_diabetes.csv')  # cleaned/processed dataset

# Output directory for plots
output_dir = 'eda_plots'
os.makedirs(output_dir, exist_ok=True)

# Define a function to generate EDA plots
def save_eda_plots(df, label):
    # Histograms
    df.hist(figsize=(12, 10), bins=20, edgecolor='black')
    plt.suptitle(f'{label} - Feature Distributions')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{label}_01_histograms.png")
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title(f"{label} - Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{label}_02_correlation_matrix.png")
    plt.close()

    # Outcome Count Plot
    sns.countplot(x='Outcome', data=df)
    plt.title(f"{label} - Outcome Distribution")
    plt.xticks([0, 1], ['No Diabetes', 'Diabetic'])
    plt.xlabel("Outcome")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{label}_03_outcome_distribution.png")
    plt.close()

    # Boxplot of Glucose by Outcome
    sns.boxplot(x='Outcome', y='Glucose', data=df)
    plt.title(f"{label} - Glucose by Outcome")
    plt.xticks([0, 1], ['No Diabetes', 'Diabetic'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{label}_04_glucose_boxplot.png")
    plt.close()

    # Scatterplot of BMI vs Age by Outcome
    sns.scatterplot(x='Age', y='BMI', hue='Outcome', data=df)
    plt.title(f"{label} - BMI vs Age Colored by Outcome")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{label}_05_bmi_vs_age_scatter.png")
    plt.close()

    # Violinplot of Insulin by Outcome
    sns.violinplot(x='Outcome', y='Insulin', data=df)
    plt.title(f"{label} - Insulin Distribution by Outcome")
    plt.xticks([0, 1], ['No Diabetes', 'Diabetic'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{label}_06_insulin_violinplot.png")
    plt.close()

# Run EDA for both datasets
save_eda_plots(df_raw, "raw")
save_eda_plots(df_processed, "processed")

print(f"✅ EDA plots saved to the '{output_dir}' folder.")
