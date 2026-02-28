import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df = pd.read_csv(url)

# 1. Vérifier les valeurs manquantes
print("=== Valeurs manquantes ===")
print(df.isnull().sum())

# 2. Infos générales sur les colonnes
print("\n=== Types de colonnes ===")
print(df.dtypes)

# 3. Statistiques sur les capteurs
print("\n=== Statistiques des capteurs ===")
print(df[['Air temperature [K]', 'Process temperature [K]',
          'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']].describe())

# 4. Colonnes inutiles à supprimer
print("\n=== Aperçu colonne Type ===")
print(df['Type'].value_counts())