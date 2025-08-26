import pandas as pd

# Load the dataset
file_path = r'C:\Users\jakub\Visual Studio Code sem2\Magister\oversampled\cardio_train5.csv'
data = pd.read_csv(file_path)

# Check how many instances have diaBP higher than 100
high_diaBP_count = data[data['diaBP'] > 100].shape[0]

# Check how many instances have height less than 100
low_height_count = data[data['height'] < 100].shape[0]

# Check how many instances have diaBP higher than 100 and height less than 100
high_diaBP_low_height_count = data[(data['diaBP'] > 100) & (data['height'] < 100)].shape[0]

print(f"Number of instances with diaBP higher than 100: {high_diaBP_count}")
print(f"Number of instances with height less than 100: {low_height_count}")
print(f"Number of instances with diaBP higher than 100 and height less than 100: {high_diaBP_low_height_count}")

# Drop rows where diaBP > 100
data_cleaned = data[data['diaBP'] <= 100]

# Drop columns smoke, alco, and active
data_cleaned = data_cleaned.drop(columns=['smoke', 'alco', 'active', 'height', 'weight'])

# Save the cleaned dataset as cardio_train7.csv
output_file_path_7 = r'C:\Users\jakub\Visual Studio Code sem2\Magister\oversampled\cardio_train8.csv'
data_cleaned.to_csv(output_file_path_7, index=False)