import pandas as pd

# Read the CSV files
df_cardio = pd.read_csv(r'C:\Users\jakub\Visual Studio Code sem2\Magister\original\cardio_train.csv')
#df_framingham = pd.read_csv(r'C:\Users\jakub\Visual Studio Code sem2\Magister\originaledited\framingham3.csv')
#df_cleveland = pd.read_csv(r'C:\Users\jakub\Visual Studio Code sem2\Magister\originaledited\Heart_disease_cleveland_new.csv')
#df_h = pd.read_csv(r'C:\Users\jakub\Visual Studio Code sem2\Magister\originaledited\heart.csv')
#df_h1 = pd.read_csv(r'C:\Users\jakub\Visual Studio Code sem2\Magister\originaledited\heart1.csv')

# Count the instances for each target value
target_counts1 = df_cardio['cardio'].value_counts()
#target_counts2 = df_framingham['target'].value_counts()
#target_counts3 = df_cleveland['target'].value_counts()
#target_counts4 = df_h['target'].value_counts()
#target_counts5 = df_h1['target'].value_counts()

# Calculate the percentages
percent1_0 = (target_counts1[0] / target_counts1.sum()) * 100
percent1_1 = (target_counts1[1] / target_counts1.sum()) * 100
"""
percent2_0 = (target_counts2[0] / target_counts2.sum()) * 100
percent2_1 = (target_counts2[1] / target_counts2.sum()) * 100

percent3_0 = (target_counts3[0] / target_counts3.sum()) * 100
percent3_1 = (target_counts3[1] / target_counts3.sum()) * 100

percent4_0 = (target_counts4[0] / target_counts4.sum()) * 100
percent4_1 = (target_counts4[1] / target_counts4.sum()) * 100

percent5_0 = (target_counts5[0] / target_counts5.sum()) * 100
percent5_1 = (target_counts5[1] / target_counts5.sum()) * 100
"""

# Print the counts and percentages
print(f"CARDIO Instances with target = 0: {target_counts1[0]} ({percent1_0:.2f}%)")
print(f"CARDIO Instances with target = 1: {target_counts1[1]} ({percent1_1:.2f}%)")
"""
print(f"FR Instances with target = 0: {target_counts2[0]} ({percent2_0:.2f}%)")
print(f"FR Instances with target = 1: {target_counts2[1]} ({percent2_1:.2f}%)")

print(f"CL Instances with target = 0: {target_counts3[0]} ({percent3_0:.2f}%)")
print(f"CL Instances with target = 1: {target_counts3[1]} ({percent3_1:.2f}%)")

print(f"H Instances with target = 0: {target_counts4[0]} ({percent4_0:.2f}%)")
print(f"H Instances with target = 1: {target_counts4[1]} ({percent4_1:.2f}%)")

print(f"H1 Instances with target = 0: {target_counts5[0]} ({percent5_0:.2f}%)")
print(f"H1 Instances with target = 1: {target_counts5[1]} ({percent5_1:.2f}%)")"
"""