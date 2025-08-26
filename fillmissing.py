import pandas as pd
from sklearn.impute import SimpleImputer

#Załaduj zbiór
data = pd.read_csv(r'C:\Users\jakub\Visual Studio Code sem2\Magister\originaledited\framingham2.csv', header=None, skiprows=1)

#Sprawdzanie brakujących wartości
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

#Metoda imputacji
imputer = SimpleImputer(strategy='mean')

#Zastąpienie brakujących wartości średnią
data_imputed = pd.DataFrame(imputer.fit_transform(data))

#Zastąpienie oryginalnych nazw kolumn
print(data_imputed.head())

#Załadowanie pierwszej linii z oryginalnego pliku
with open(r'C:\Users\jakub\Visual Studio Code sem2\Magister\originaledited\framingham2.csv', 'r') as file:
    first_line = file.readline().strip()

#Zapisanie danych do nowego pliku CSV
with open(r'C:\Users\jakub\Visual Studio Code sem2\Magister\originaledited\framingham3.csv', 'w', newline='') as file:
    file.write(first_line + '\n')
    data_imputed.to_csv(file, index=False, header=False, float_format='%.1f')