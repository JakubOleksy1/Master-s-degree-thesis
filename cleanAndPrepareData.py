"""
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

# Read the cardio_train_fixed.csv file
df = pd.read_csv('Magister/cardio_train_fixed.csv')

# Convert age from days to years and round up to integer
df['age'] = (df['age'] / 365).apply(np.ceil).astype(int)

# Adjust gender values by subtracting 1
df['gender'] = df['gender'] - 1

# Save the modified data to fixed2.csv
df.to_csv('Magister/cardio_train_fixed2.csv', index=False)
import pandas as pd

# Read the cardio_train_fixed2.csv file
df = pd.read_csv('Magister/cardio_train_fixed2.csv')

# Drop the 'id' column
df = df.drop(columns=['id'])

# Rename the 'gender' column to 'sex'
df = df.rename(columns={'gender': 'sex'})

# Save the modified data to cardio_train_fixed2.csv
df.to_csv('Magister/cardio_train_fixed2.csv', index=False)

import pandas as pd

# Read the framingham.csv file
df = pd.read_csv('c:/Users/jakub/Visual Studio Code sem2/Magister/framingham.csv')

# Rename the 'male' column to 'sex'
df = df.rename(columns={'male': 'sex'})

# Reorder columns to have 'age' followed by 'sex'
columns = ['age', 'sex'] + [col for col in df.columns if col not in ['age', 'sex']]
df = df[columns]

# Save the modified data to framingham.csv
df.to_csv('c:/Users/jakub/Visual Studio Code sem2/Magister/framingham.csv', index=False)

import pandas as pd

# Read the heartdisease_fixed2.csv file
df_heartdisease = pd.read_csv('c:/Users/jakub/Visual Studio Code sem2/Magister/heartdisease_fixed2.csv')

# Rename the 'num' column to 'target'
df_heartdisease = df_heartdisease.rename(columns={'num': 'target'})

# Save the modified data to heartdisease_fixed2.csv
df_heartdisease.to_csv('c:/Users/jakub/Visual Studio Code sem2/Magister/heartdisease_fixed2.csv', index=False)
import pandas as pd

# Read the framingham.csv file
df_framingham = pd.read_csv('c:/Users/jakub/Visual Studio Code sem2/Magister/framingham.csv')

# Rename the 'TenYearCHD' column to 'target'
df_framingham = df_framingham.rename(columns={'TenYearCHD': 'target'})

# Save the modified data to framingham.csv
df_framingham.to_csv('c:/Users/jakub/Visual Studio Code sem2/Magister/framingham.csv', index=False)

import pandas as pd
import numpy as np

# Read the cardio_train_fixed2.csv file
df_cardio = pd.read_csv('c:/Users/jakub/Visual Studio Code sem2/Magister/cardio_train_fixed2.csv')

# Rename columns
df_cardio = df_cardio.rename(columns={
    'ap_hi': 'trestbps',
    'cholesterol': 'chol',
    'gluc': 'fbs'
})

# Recalculate 'chol' and 'fbs'
df_cardio['chol'] = df_cardio['chol'].map({1: 200, 2: 240, 3: 280})  # Example values for recalculation
df_cardio['fbs'] = df_cardio['fbs'].map({1: 0, 2: 1, 3: 1})  # Assuming 2 and 3 as high blood sugar

# Simulate missing columns based on patterns in heart.csv
np.random.seed(42)  # For reproducibility

# Simulate 'cp' (chest pain type)
df_cardio['cp'] = np.random.choice([0, 1, 2, 3], size=len(df_cardio))

# Simulate 'restecg' (resting electrocardiographic results)
df_cardio['restecg'] = np.random.choice([0, 1, 2], size=len(df_cardio))

# Simulate 'thalach' (maximum heart rate achieved)
df_cardio['thalach'] = np.random.randint(100, 200, size=len(df_cardio))

# Simulate 'exang' (exercise induced angina)
df_cardio['exang'] = np.random.choice([0, 1], size=len(df_cardio))

# Simulate 'oldpeak' (ST depression induced by exercise relative to rest)
df_cardio['oldpeak'] = np.random.uniform(0, 6, size=len(df_cardio))
df_cardio['oldpeak'] = df_cardio['oldpeak'].round(1)

# Simulate 'slope' (the slope of the peak exercise ST segment)
df_cardio['slope'] = np.random.choice([0, 1, 2], size=len(df_cardio))

# Simulate 'ca' (number of major vessels colored by fluoroscopy)
df_cardio['ca'] = np.random.choice([0, 1, 2, 3], size=len(df_cardio))

# Simulate 'thal' (thalassemia)
df_cardio['thal'] = np.random.choice([1, 2, 3], size=len(df_cardio))

# Reorder columns
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df_cardio = df_cardio[columns]

# Save the modified data to cardio_train_fixed2.csv
df_cardio.to_csv('c:/Users/jakub/Visual Studio Code sem2/Magister/cardio_train_fixed3.csv', index=False)

import pandas as pd
import numpy as np

# Read the framingham.csv file
df = pd.read_csv('c:/Users/jakub/Visual Studio Code sem2/Magister/framingham.csv')

# Drop the specified columns
df = df.drop(columns=['education', 'currentSmoker', 'cigsPerDay'])

# Derive new columns
df['trestbps'] = df['sysBP']
df['chol'] = df['totChol']
df['fbs'] = np.where(df['glucose'] > 120, 1, 0)
df['thalach'] = df['heartRate']

# Simulate missing columns based on patterns in heart.csv
np.random.seed(42)  # For reproducibility

# Simulate 'cp' (chest pain type)
df['cp'] = np.random.choice([0, 1, 2, 3], size=len(df))

# Simulate 'restecg' (resting electrocardiographic results)
df['restecg'] = np.random.choice([0, 1, 2], size=len(df))

# Simulate 'exang' (exercise induced angina)
df['exang'] = np.random.choice([0, 1], size=len(df))

# Simulate 'oldpeak' (ST depression induced by exercise relative to rest)
df['oldpeak'] = np.random.uniform(0, 6, size=len(df))
df['oldpeak'] = df['oldpeak'].round(1)

# Simulate 'slope' (the slope of the peak exercise ST segment)
df['slope'] = np.random.choice([0, 1, 2], size=len(df))

# Simulate 'ca' (number of major vessels colored by fluoroscopy)
df['ca'] = np.random.choice([0, 1, 2, 3], size=len(df))

# Simulate 'thal' (thalassemia)
df['thal'] = np.random.choice([1, 2, 3], size=len(df))

# Reorder columns
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = df[columns]

# Save the modified data to framingham.csv
df.to_csv('c:/Users/jakub/Visual Studio Code sem2/Magister/framingha2.csv', index=False)

#import pandas as pd

#file_paths = [
   # 'Magister/cardio_train_fixed3.csv',
   # 'Magister/framingha2.csv',
   # 'Magister/heartdisease_fixed2.csv',
   # 'Magister/Heart_disease_cleveland_new.csv',
   # 'Magister/heart.csv'
#]

# Read and concatenate all CSV files
#combined_df = pd.concat([pd.read_csv(file) for file in file_paths])

# Save the combined DataFrame to a new CSV file
#combined_df.to_csv('Magister/combined_heart_data.csv', index=False)
# Open the original file in read mode and the new file in write mode
# with open('Magister/cardio_train.csv', 'r') as infile, open('Magister/cardio_train_fixed.csv', 'w') as outfile:
# Read the content of the original file
#    content = infile.read()
#     Replace all semicolons with commas
#
 #   content = content.replace(';', ',')
  #   Write the modified content to the new file
   # outfile.write(content)

#print("File has been processed and saved as 'cardio_train_fixed.csv'.")

# Open the original file in read mode and the new file in write mode
#with open('Magister/heartdisease.csv', 'r') as infile, open('Magister/heartdisease_fixed.csv', 'w') as outfile:
    # Read all lines from the original file
 #   lines = infile.readlines()
    # Write non-empty lines to the new file
  #  for line in lines:
    #    if line.strip():  # Check if the line is not empty
   #         outfile.write(line)

#print("File has been processed and saved as 'heartdisease_fixed.csv'.")


#import pandas as pd

#def convert_to_binary(input_file, output_file):
    # Read the original dataset
 #   df = pd.read_csv(input_file)
    
    # Convert the target column 'num' to binary values
  #  df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    
    # Save the modified dataset to a new file
   # df.to_csv(output_file, index=False)
    
    #print(f"File has been processed and saved as '{output_file}'.")

 #Convert the heartdisease dataset to binary and save it as heartdisease_fixed2.csv
#convert_to_binary('Magister/heartdisease.csv', 'Magister/heartdisease_fixed2.csv')

# Rename columns


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Load datasets 3 and 5
df3 = pd.read_csv(r'C:/Users/jakub/Visual Studio Code sem2/Magister/train/framingham_renamed_oversampled.csv')
df5 = pd.read_csv(r'C:/Users/jakub/Visual Studio Code sem2/Magister/train/cardio_train_fixed2_renamed_oversampled.csv')

# Perform operations on dataset 3
df3 = df3.rename(columns={
    'totChol': 'chol',
    'sysBP': 'ap_hi',
    'diaBP': 'ap_lo',
    'currentSmoker': 'smoke',
    'glucose': 'fbs'
})

# Convert continuous `fbs` to binary (e.g., >120 = 1)
df3['fbs'] = (df3['fbs'] > 120).astype(int)

# Drop incompatible columns
df3 = df3.drop(columns=[
    'education', 'cigsPerDay', 'BPMeds', 'prevalentStroke',
    'prevalentHyp', 'diabetes', 'heartRate'
])

# Add missing columns (e.g., `cp`, `restecg`, etc.) with NaN
missing_cols = ['cp', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
for col in missing_cols:
    df3[col] = np.nan  # Fill with NaN for imputation later

# Perform operations on dataset 5
df5 = df5.rename(columns={
    'cholesterol': 'chol',
    'gluc': 'fbs'
})

# Convert `fbs` to binary (if needed)
df5['fbs'] = (df5['fbs'] > 120).astype(int)

# Calculate BMI
df5['BMI'] = df5['weight'] / ((df5['height'] / 100) ** 2)

# Drop incompatible columns
df5 = df5.drop(columns=['height', 'weight', 'alco', 'active'])

# Add missing columns (e.g., `cp`, `restecg`, etc.) with NaN
for col in missing_cols:
    df5[col] = np.nan


# Ensure column alignment
common_columns = [
    'age', 'sex', 'smoke', 'chol', 'ap_hi', 'ap_lo', 'BMI', 'fbs', 'target', 
    'cp', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

df3 = df3[common_columns]
df5 = df5[common_columns]

# Combine datasets for imputation
combined = pd.concat([df3, df5], ignore_index=True)

# Separate features and target
X = combined.drop(columns=['target'])
y = combined['target']

X = combined.drop(columns=['target'])

print("Shape of X before imputation (should be 16 columns):", X.shape)
print("Columns in X before imputation:", X.columns.tolist())


X.fillna(-1, inplace=True)  # Prevents columns from being dropped
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

# Ensure imputed dataset has correct number of columns
if X_imputed.shape[1] != X.shape[1]:
    raise ValueError(f"Expected {X.shape[1]} columns, but got {X_imputed.shape[1]}. Imputer may have dropped some columns.")

# Convert back to DataFrame with correct column names
combined_imputed = pd.DataFrame(X_imputed, columns=X.columns)
combined_imputed['target'] = y


# Convert categorical columns to integers
categorical_cols = ['cp', 'restecg', 'thal', 'slope', 'ca']
for col in categorical_cols:
    combined_imputed[col] = combined_imputed[col].round().astype(int)

# Rename columns to match final format
combined_imputed = combined_imputed.rename(columns={'ap_hi': 'trestbps'})

# Select final column order (remove BMI)
final_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

combined_imputed = combined_imputed[final_columns]

# Split back into df3 and df5
df3_imputed = combined_imputed.iloc[:len(df3)]
df5_imputed = combined_imputed.iloc[len(df3):]

# Save final datasets
df3_imputed.to_csv(r'C:\\Users\\jakub\Visual Studio Code sem2\\Magister\\framingham_newnewnewn1.csv', index=False)
df5_imputed.to_csv(r'C:\\Users\\jakub\Visual Studio Code sem2\\Magister\\cardio_train_fixed2_newnewnew1.csv', index=False)

print("Datasets have been fully imputed, formatted, and saved!")

import pandas as pd
import os

# Load the CSV file
file_path = r'C:\Users\jakub\Visual Studio Code sem2\Magister\train\framingham_renamed_oversampled.csv'
df = pd.read_csv(file_path)

# Drop the specified columns
df = df.drop(columns=['education', 'cigsPerDay'])

# Rename the columns
df = df.rename(columns={'totChol': 'chol', 'sysBP': 'trestbps', 'glucose': 'fbs'})

# Convert the 'fbs' column to binary
df['fbs'] = df['fbs'].apply(lambda x: 1.0 if x > 120 else 0.0)

# Rearrange the columns
df = df[['age', 'sex', 'currentSmoker', 'chol', 'fbs', 'BPMeds', 'trestbps', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'diaBP', 'BMI', 'heartRate', 'target']]

# Define the new directory and file path
new_directory = r'C:\Users\jakub\Visual Studio Code sem2\Magister\new'
new_file_path = os.path.join(new_directory, 'framingham_modified.csv')

# Create the new directory if it doesn't exist
os.makedirs(new_directory, exist_ok=True)

# Save the modified DataFrame to a new CSV file
df.to_csv(new_file_path, index=False)

print(f"File saved to {new_file_path}")


import pandas as pd
import numpy as np
import random

# Load the dataset
file_path = r'C:\Users\jakub\Visual Studio Code sem2\Magister\train\cardio_train_fixed2_renamed_oversampled.csv'
df = pd.read_csv(file_path)

# Calculate BMI and add it to the dataset
df['BMI'] = (df['weight'] / (df['height'] / 100) ** 2).round(1)

# Rename columns
df.rename(columns={
    'ap_hi': 'trestbps',
    'cholesterol': 'chol',
    'gluc': 'fbs'
}, inplace=True)

# Function to decode cholesterol and glucose values
def decode_values(row):
    if row['chol'] == 1:
        row['chol'] = random.randint(180, 199)
    elif row['chol'] == 2:
        row['chol'] = random.randint(200, 239)
    elif row['chol'] == 3:
        row['chol'] = random.randint(240, 300)
    
    if row['fbs'] == 1:
        row['fbs'] = random.randint(70, 99)
    elif row['fbs'] == 2:
        row['fbs'] = random.randint(100, 125)
    elif row['fbs'] == 3:
        row['fbs'] = random.randint(126, 200)
    
    return row

# Apply the function to decode values
df = df.apply(decode_values, axis=1)

# Reorder columns
df = df[['age', 'sex', 'BMI', 'trestbps', 'chol', 'fbs', 'ap_lo', 'smoke', 'alco', 'active', 'height', 'weight', 'target']]

# Save the modified dataset to a new file
new_file_path = r'C:\Users\jakub\Visual Studio Code sem2\Magister\new\cardio_train_modified.csv'
df.to_csv(new_file_path, index=False)

print(f"Modified dataset saved to {new_file_path}")
"""
import pandas as pd

# List of file paths to combine
file_paths = [
    r'C:\Users\jakub\Visual Studio Code sem2\Magister\new\cardio_train_modified.csv',
    r'C:\Users\jakub\Visual Studio Code sem2\Magister\new\framingham_modified.csv',
    r'C:\Users\jakub\Visual Studio Code sem2\Magister\train\Heart_disease_cleveland_original_oversampled.csv',
    r'C:\Users\jakub\Visual Studio Code sem2\Magister\train\heart_original_oversampled.csv',
    r'C:\Users\jakub\Visual Studio Code sem2\Magister\train\heartdisease_fixed2_renamed_oversampled.csv'
]

# Load and concatenate all CSV files
combined_df = pd.concat([pd.read_csv(file) for file in file_paths])

# Remove duplicate rows
combined_df.drop_duplicates(inplace=True)

# Save the combined dataset to a new file
combined_file_path = r'C:\Users\jakub\Visual Studio Code sem2\Magister\new\combined_dataset.csv'
combined_df.to_csv(combined_file_path, index=False)

print(f"Combined dataset saved to {combined_file_path}")
