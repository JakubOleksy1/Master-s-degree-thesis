import pandas as pd

# Define the unified schema
columns = ['age', 'gender', 'cp', 'smoker', 'thalach', 'BP', 'cholesterol', 'glucose', 'physical_activity', 'target']

# Function to standardize each file
def standardize_file(file_path, column_mapping, target_column, delimiter=',', skiprows=None):
    df = pd.read_csv(file_path, delimiter=delimiter, skiprows=skiprows)
    df = df.rename(columns=column_mapping)
    for col in columns:
        if col not in df.columns:
            df[col] = None  # Add missing columns with None values
    df = df[columns]
    df['target'] = df[target_column]
    df = df.drop(columns=[target_column])
    return df

# File 1: cardio_train.csv
file1_mapping = {
    'age': 'age',
    'gender': 'gender',
    'cholesterol': 'cholesterol',
    'thalach': None,  # No equivalent column
    'smoke': 'smoker',
    'ap_hi': 'BP',
    'gluc': 'glucose',
    'cardio': 'target',
    'cp': None,  # No equivalent column
    'active': 'physical_activity'
}
df1 = pd.read_csv('Magister/cardio_train.csv', delimiter=';')
df1 = df1.rename(columns=file1_mapping)
for col in columns:
    if col not in df1.columns:
        df1[col] = None  # Add missing columns with None values
df1 = df1[columns]

# File 2: framingham.csv
file2_mapping = {
    'age': 'age',
    'male': 'gender',
    'totChol': 'cholesterol',
    'heartRate': 'thalach',
    'currentSmoker': 'smoker',
    'sysBP': 'BP',
    'glucose': 'glucose',
    'TenYearCHD': 'target',
    'cp': None,  # No equivalent column
    'physical_activity': None  # No equivalent column
}
df2 = pd.read_csv('Magister/framingham.csv')
df2 = df2.rename(columns=file2_mapping)
for col in columns:
    if col not in df2.columns:
        df2[col] = None  # Add missing columns with None values
df2 = df2[columns]

# File 3: heartdisease.csv
file3_mapping = {
    'age': 'age',
    'sex': 'gender',
    'chol': 'cholesterol',
    'thalach': 'thalach',
    'fbs': 'smoker',  # Assuming fbs (fasting blood sugar) as smoker
    'trestbps': 'BP',
    'restecg': 'glucose',  # No equivalent column, using restecg
    'target': 'target',
    'cp': 'cp',
    'exang': 'physical_activity'  # Assuming exang (exercise induced angina) as physical activity
}
df3 = pd.read_csv('Magister/heartdisease.csv')
df3 = df3.rename(columns=file3_mapping)
for col in columns:
    if col not in df3.columns:
        df3[col] = None  # Add missing columns with None values
df3 = df3[columns]

# File 4: Heart_disease_cleveland_new.csv
df4 = pd.read_csv('Magister/Heart_disease_cleveland_new.csv')
df4 = df4.rename(columns={'target': 'target'})
for col in columns:
    if col not in df4.columns:
        df4[col] = None  # Add missing columns with None values
df4 = df4[columns]

# File 5: heart.csv
df5 = pd.read_csv('Magister/heart.csv')
df5 = df5.rename(columns={'target': 'target'})
for col in columns:
    if col not in df5.columns:
        df5[col] = None  # Add missing columns with None values
df5 = df5[columns]

# Combine all dataframes
combined_df = pd.concat([df1, df2, df3, df4, df5])

# Save the standardized data to a new CSV file
combined_df.to_csv('Magister/standardized_heart_disease_data.csv', index=False)