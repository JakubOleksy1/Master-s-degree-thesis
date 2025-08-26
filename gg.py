import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split

# Wczytanie danych
df = pd.read_csv(r"C:\Users\jakub\Visual Studio Code sem2\Magister\original\framingham.csv")

# Usunięcie wierszy z brakami jeśli nadal jakieś pozostały
df.dropna(inplace=True)

# Podział na zmienne X i y
X = df.drop(columns=["TenYearCHD"])
y = df["TenYearCHD"]

# Wybór metody oversamplingu
oversampling_method = "ADASYN"  # Możliwe opcje: "SMOTE", "ADASYN", "SMOTE-Tomek"

if oversampling_method == "SMOTE":
    # SMOTE z większym zakresem generowania danych
    smote = SMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=15)  # Większy zakres (k_neighbors=10)
    X_resampled, y_resampled = smote.fit_resample(X, y)
elif oversampling_method == "ADASYN":
    # ADASYN - generowanie danych w trudniejszych obszarach
    adasyn = ADASYN(sampling_strategy=1.0, random_state=42, n_neighbors=15)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
elif oversampling_method == "SMOTE-Tomek":
    # SMOTE-Tomek - oversampling + usuwanie trudnych próbek
    smote_tomek = SMOTETomek(sampling_strategy=1.0, random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
else:
    raise ValueError("Nieprawidłowa metoda oversamplingu!")

# Konwersja y_resampled na typ int
y_resampled = y_resampled.astype(int)

# Shuffle danych
df_balanced = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=["TenYearCHD"])], axis=1)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Konwersja wszystkich kolumn na typ int (jeśli możliwe)
df_balanced = df_balanced.astype(int)

# Podgląd nowego rozkładu klas
print(df_balanced["TenYearCHD"].value_counts())

# Zapis zbalansowanego datasetu
df_balanced.to_csv(r"C:\Users\jakub\Visual Studio Code sem2\Magister\original\framingham_balanced1.csv", index=False)