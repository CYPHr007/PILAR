import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df = pd.read_csv(url)

# 1. Supprimer les colonnes inutiles
df = df.drop(columns=['UDI', 'Product ID'])

# 2. Encoder la colonne "Type" (L, M, H) en chiffres
df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})

# 3. Séparer les features (X) et la cible (y)
X = df.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
y = df['Machine failure']

# 4. Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Découper en train / test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("=== Données prêtes ===")
print(f"Entraînement : {X_train.shape[0]} lignes")
print(f"Test         : {X_test.shape[0]} lignes")
print(f"Pannes dans le test : {y_test.sum()} / {len(y_test)}")