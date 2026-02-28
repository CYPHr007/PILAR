import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df = pd.read_csv(url)

print("=== Taille du dataset ===")
print(df.shape)

print("\n=== Aperçu des données ===")
print(df.head())

print("\n=== Nombre de pannes ===")
print(df['Machine failure'].value_counts())