"""# Open the original file in read mode and the new file in write mode
with open('Magister/cardio_train.csv', 'r') as infile, open('Magister/cardio_train_fixed.csv', 'w') as outfile:
    # Read the content of the original file
    content = infile.read()
    # Replace all semicolons with commas
    content = content.replace(';', ',')
    # Write the modified content to the new file
    outfile.write(content)

print("File has been processed and saved as 'cardio_train_fixed.csv'.")
"""
"""# Open the original file in read mode and the new file in write mode
with open('Magister/heartdisease.csv', 'r') as infile, open('Magister/heartdisease_fixed.csv', 'w') as outfile:
    # Read all lines from the original file
    lines = infile.readlines()
    # Write non-empty lines to the new file
    for line in lines:
        if line.strip():  # Check if the line is not empty
            outfile.write(line)

print("File has been processed and saved as 'heartdisease_fixed.csv'.")
"""

import pandas as pd

def convert_to_binary(input_file, output_file):
    # Read the original dataset
    df = pd.read_csv(input_file)
    
    # Convert the target column 'num' to binary values
    df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    
    # Save the modified dataset to a new file
    df.to_csv(output_file, index=False)
    
    print(f"File has been processed and saved as '{output_file}'.")

# Convert the heartdisease dataset to binary and save it as heartdisease_fixed2.csv
convert_to_binary('Magister/heartdisease.csv', 'Magister/heartdisease_fixed2.csv')