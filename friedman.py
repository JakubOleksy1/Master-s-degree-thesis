import numpy as np
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare

# Dane: rangi klasyfikatorów
ranks = np.array([
    [5, 1, 3, 4, 2],  # Heart1
    [5, 1, 3, 2, 4],  # Cleveland
    [5, 1, 3, 4, 2],  # Framingham
    [4, 3, 5, 1, 2],  # Heart
    [4, 1, 5, 3, 2],  # Cardio
])

# Nazwy klasyfikatorów
classifiers = ['LR', 'RF', 'KNN', 'SVM', 'DNN']

# Test Friedmana
stat, p = friedmanchisquare(*ranks.T)
print(f"Friedman chi2 = {stat:.4f}, p = {p:.4f}")

# Test Nemenyiego
nemenyi = sp.posthoc_nemenyi_friedman(ranks)
print("Test Nemenyiego:\n", nemenyi)

# Średnie rangi
avg_ranks = np.mean(ranks, axis=0)

# Wykres
plt.figure(figsize=(8, 5))
bars = plt.bar(classifiers, avg_ranks, color='skyblue')
plt.axhline(np.mean(avg_ranks), color='red', linestyle='--', label='Średnia ogólna')
plt.title("Średnie rangi klasyfikatorów")
plt.ylabel("Średnia ranga")
plt.legend()

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, round(yval, 2), ha='center', va='bottom')

plt.tight_layout()
plt.show()
