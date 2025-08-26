import pandas as pd
import numpy as np

# Load the datasets
df_cardio = pd.read_csv(r'C:\Users\jakub\Visual Studio Code sem2\Magister\original\cardio_train.csv')
df_framingham = pd.read_csv(r'C:\Users\jakub\Visual Studio Code sem2\Magister\original\framingham.csv')
df_hd = pd.read_csv(r'C:\Users\jakub\Visual Studio Code sem2\Magister\original\Heart_disease_cleveland_new.csv')
df_h = pd.read_csv(r'C:\Users\jakub\Visual Studio Code sem2\Magister\original\heart.csv')
df_h1 = pd.read_csv(r'C:\Users\jakub\Visual Studio Code sem2\Magister\original\heart1.csv')

# Check for missing values in the cardio dataset
print("Cardio dataset:")
print(df_cardio.isnull().sum())

# Check for missing values in the framingham dataset
print("\nFramingham dataset:")
print(df_framingham.isnull().sum())

# Check for missing values in the heart disease cleveland dataset
print("\nHeart disease cleveland dataset:")
print(df_hd.isnull().sum())

# Check for missing values in the heart dataset
print("\nHeart dataset:")
print(df_h.isnull().sum())

# Check for missing values in the heart1 dataset
print("\nHeart1 dataset:")
print(df_h1.isnull().sum())

