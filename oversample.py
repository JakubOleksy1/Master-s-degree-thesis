import pandas as pd
import os

def balance_dataset(df, target_col='target'):
    # Split classes
    class_0 = df[df[target_col] == 0]
    class_1 = df[df[target_col] == 1]
    
    # Identify minority class
    if len(class_0) < len(class_1):
        minority, majority = class_0, class_1
    else:
        minority, majority = class_1, class_0
    
    # Oversample minority to match majority
    minority_oversampled = minority.sample(n=len(majority), 
                                          replace=True, 
                                          random_state=42)
    
    # Combine and shuffle
    balanced_df = pd.concat([majority, minority_oversampled])
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Define paths
input_files = {
    'cardio': r'C:\Users\jakub\Visual Studio Code sem2\Magister\originaledited\cardio_train_4.csv',
    'framingham': r'C:\Users\jakub\Visual Studio Code sem2\Magister\originaledited\framingham3.csv',
    'cleveland': r'C:\Users\jakub\Visual Studio Code sem2\Magister\originaledited\Heart_disease_cleveland_new.csv',
    'h': r'C:\Users\jakub\Visual Studio Code sem2\Magister\originaledited\heart.csv',
    'h1': r'C:\Users\jakub\Visual Studio Code sem2\Magister\originaledited\heart1.csv'
}

output_dir = r'C:\Users\jakub\Visual Studio Code sem2\Magister\oversampled'

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process and save
for name, path in input_files.items():
    # Load original
    df = pd.read_csv(path)
    
    # Create balanced version
    balanced = balance_dataset(df)
    
    # Save to new location
    output_path = os.path.join(output_dir, f"{name}_balanced.csv")
    balanced.to_csv(output_path, index=False)
    
    # Verify
    orig_count = df['target'].value_counts()
    new_count = balanced['target'].value_counts()
    print(f"\n{name.upper():<12} Original: 0={orig_count[0]} | 1={orig_count[1]}")
    print(f"{' ':<12} Balanced: 0={new_count[0]} | 1={new_count[1]}")
    print(f"{' ':<12} Saved to: {output_path}")