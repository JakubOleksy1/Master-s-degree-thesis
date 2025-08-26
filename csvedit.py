"""#edit datasets
import pandas as pd

# Process cardio_train.csv
df_cardio = pd.read_csv('C:/Users/jakub/Visual Studio Code sem2/Magister/originaledited/cardio_train.csv', sep=';')

# Delete the 'id' column
df_cardio.drop(columns=['id'], inplace=True)

# Convert 'age' from days to years and round to the nearest integer
df_cardio['age'] = (df_cardio['age'] / 365).round().astype(int)

# Subtract 1 from every value in the 'sex' column
df_cardio['sex'] = df_cardio['sex'] - 1

# Reorder the columns
df_cardio = df_cardio[['age', 'sex', 'ap_hi', 'chol', 'gluc', 'ap_lo', 'smoke', 'alco', 'active', 'height', 'weight', 'target']]

# Save the modified data to a new CSV file with header
df_cardio.to_csv('C:/Users/jakub/Visual Studio Code sem2/Magister/new/cardio_train_modified2.csv', index=False, sep=',')

print("Data processing complete. The modified data has been saved to 'cardio_train_modified.csv'.")

# Process framingham.csv
df_framingham = pd.read_csv('C:/Users/jakub/Visual Studio Code sem2/Magister/originaledited/framingham.csv', sep=',')

# Apply the glucose transformation
df_framingham['glucose'] = df_framingham['glucose'].apply(lambda x: 1.0 if x > 120 else 0.0)

# Reorder the columns
df_framingham = df_framingham[['age', 'sex', 'education', 'trestbps', 'chol', 'glucose', 'diaBP', 'thalach', 'BPMeds', 'currentSmoker', 'cigsPerDay', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'BMI', 'target']]

# Save the modified data to a new CSV file with header
df_framingham.to_csv('C:/Users/jakub/Visual Studio Code sem2/Magister/new/framingham_modified2.csv', index=False, sep=',')

print("Data processing complete. The modified data has been saved to 'framingham_modified.csv'.")

"""
"""
import pandas as pd
import numpy as np
import random

# Load the dataset
file_path = r'C:\Users\jakub\Visual Studio Code sem2\Magister\train\framingham_renamed_oversampled2.csv'
df = pd.read_csv(file_path)

# Rename columns
df.rename(columns={
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

# Apply the fbs transformation
df['glucose'] = df['glucose'].apply(lambda x: 1.0 if x > 120 else 0.0)

# Save the modified dataset to a new file
new_file_path = r'C:\Users\jakub\Visual Studio Code sem2\Magister\train\framingham_renamed_oversampled3.csv'
df.to_csv(new_file_path, index=False)

print(f"Modified dataset saved to {new_file_path}")
"""
import pandas as pd
import numpy as np

# Load the dataset
file_path = r'C:\Users\jakub\Visual Studio Code sem2\Magister\train\cardio_train_fixed2_renamed_oversampled3.csv'
df = pd.read_csv(file_path)

# Calculate BMI and round to one decimal place
df['BMI'] = (df['weight'] / (df['height'] / 100) ** 2).round(1)

# Infer prevalentHyp (binary) using trestbps and diaBP
# Assuming prevalentHyp is 1 if either trestbps or diaBP is above a certain threshold
df['prevalentHyp'] = df.apply(lambda row: 1 if row['trestbps'] > 140 or row['diaBP'] > 90 else 0, axis=1)

# Reorder the columns
df = df[['age', 'sex', 'trestbps', 'chol', 'fbs', 'diaBP', 'smoke', 'alco', 'active', 'height', 'weight', 'BMI', 'prevalentHyp', 'target']]

# Save the modified dataset to a new file
new_file_path = r'C:\Users\jakub\Visual Studio Code sem2\Magister\train\cardio_train_fixed2_renamed_oversampled4.csv'
df.to_csv(new_file_path, index=False)

print(f"Modified dataset saved to {new_file_path}")