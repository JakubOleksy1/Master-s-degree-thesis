import pandas as pd
import numpy as np
import os

# Ścieżki
input_path = r"C:\Users\jakub\Visual Studio Code sem2\Magister\augmented\framingham_balanced1_augmented.csv"
output_path = r"C:\Users\jakub\Visual Studio Code sem2\Magister\augmented\framingham_balanced1_augmented.csv"

# Wczytaj dane
df_fram = pd.read_csv(input_path)
print("Oryginalny rozmiar:", len(df_fram))

# Funkcja augmentująca
def augment_framingham(data, target_size=15000, random_state=42):
    np.random.seed(random_state)
    current_size = len(data)
    needed = target_size - current_size

    numeric_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    augmented_data = []

    for _ in range(needed):
        row = data.sample(1).iloc[0].copy()

        for col in numeric_cols:
            std = data[col].std()
            noise = np.random.normal(0, std * 0.05)
            row[col] = max(0, row[col] + noise)

        # Zaokrąglenia dla integerowych kolumn
        row['age'] = int(round(row['age']))
        row['cigsPerDay'] = int(round(row['cigsPerDay']))
        row['totChol'] = int(round(row['totChol']))
        row['sysBP'] = round(row['sysBP'], 1)
        row['diaBP'] = round(row['diaBP'], 1)
        row['heartRate'] = int(round(row['heartRate']))
        row['glucose'] = int(round(row['glucose']))
        row['BMI'] = round(row['BMI'], 1)

        augmented_data.append(row)

    augmented_df = pd.DataFrame(augmented_data)
    final_df = pd.concat([data, augmented_df], ignore_index=True)

    # Usunięcie duplikatów
    final_df = final_df.drop_duplicates().reset_index(drop=True)
    return final_df

# Augmentacja
df_augmented = augment_framingham(df_fram, target_size=8500)
print("Nowy rozmiar po usunięciu duplikatów:", len(df_augmented))

# Zapis
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_augmented.to_csv(output_path, index=False)
print(f"Dane zapisane do: {output_path}")
