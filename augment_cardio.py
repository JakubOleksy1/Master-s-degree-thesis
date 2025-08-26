import pandas as pd
import numpy as np
import os

# Ścieżki
input_path = r"C:\Users\jakub\Visual Studio Code sem2\Magister\augmented\cardio_train_augmented.csv"
output_path = r"C:\Users\jakub\Visual Studio Code sem2\Magister\augmented\cardio_train_augmented.csv"

# Wczytaj dane
df_cardio = pd.read_csv(input_path)
print("Oryginalny rozmiar:", len(df_cardio))

# Usuwamy kolumnę ID przed augmentacją
df_cardio = df_cardio.drop(columns=["id"])

# Funkcja augmentująca
def augment_cardio(data, target_size=120000, random_state=42):
    np.random.seed(random_state)
    current_size = len(data)
    needed = target_size - current_size

    numeric_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    augmented_data = []

    for _ in range(needed):
        row = data.sample(1).iloc[0].copy()

        for col in numeric_cols:
            std = data[col].std()
            noise = np.random.normal(0, std * 0.02)  # mniejszy szum przy dużych danych
            row[col] = max(0, row[col] + noise)

        # Zaokrąglenia
        row['age'] = int(round(row['age']))
        row['height'] = int(round(row['height']))
        row['weight'] = round(row['weight'], 1)
        row['ap_hi'] = int(round(row['ap_hi']))
        row['ap_lo'] = int(round(row['ap_lo']))

        augmented_data.append(row)

    augmented_df = pd.DataFrame(augmented_data)

    # Łączenie i drop duplikatów
    final_df = pd.concat([data, augmented_df], ignore_index=True)
    final_df = final_df.drop_duplicates().reset_index(drop=True)

    # Dodajmy z powrotem ID od 0 do n
    final_df.insert(0, 'id', range(len(final_df)))
    return final_df

# Augmentacja
df_augmented = augment_cardio(df_cardio, target_size=95000)
print("Nowy rozmiar po usunięciu duplikatów:", len(df_augmented))

# Zapis
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_augmented.to_csv(output_path, index=False)
print(f"Dane zapisane do: {output_path}")
