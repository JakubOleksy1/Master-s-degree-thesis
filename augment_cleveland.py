import pandas as pd
import numpy as np
import os

# Ścieżki
input_path = r"C:\Users\jakub\Visual Studio Code sem2\Magister\augmented\Heart_disease_cleveland_new_augmented.csv"
output_path = r"C:\Users\jakub\Visual Studio Code sem2\Magister\augmented\Heart_disease_cleveland_new_augmented.csv"

# Wczytaj dane
df_cleveland = pd.read_csv(input_path)
print("Oryginalny rozmiar:", len(df_cleveland))

# Funkcja augmentująca
def augment_cleveland(data, target_size=3000, random_state=42):
    np.random.seed(random_state)
    current_size = len(data)
    needed = target_size - current_size

    augmented_data = []

    for _ in range(needed):
        row = data.sample(1).iloc[0].copy()

        for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
            std = data[col].std()
            noise = np.random.normal(0, std * 0.05)
            row[col] = max(0, row[col] + noise)

        row['age'] = int(round(row['age']))
        row['trestbps'] = int(round(row['trestbps']))
        row['chol'] = int(round(row['chol']))
        row['thalach'] = int(round(row['thalach']))
        row['oldpeak'] = round(row['oldpeak'], 1)

        augmented_data.append(row)

    augmented_df = pd.DataFrame(augmented_data)
    final_df = pd.concat([data, augmented_df], ignore_index=True)

    # Usunięcie duplikatów
    final_df = final_df.drop_duplicates().reset_index(drop=True)
    return final_df

# Wykonanie augmentacji
df_augmented = augment_cleveland(df_cleveland, target_size=500)
print("Nowy rozmiar po usunięciu duplikatów:", len(df_augmented))

# Zapis do pliku
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_augmented.to_csv(output_path, index=False)
print(f"Dane zapisane do: {output_path}")
