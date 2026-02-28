import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df = pd.read_csv(url)

# Préparation des données
df = df.drop(columns=['UDI', 'Product ID'])
df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})
X = df.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
y = df['Machine failure']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Gérer le déséquilibre avec SMOTE
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)
print(f"Après SMOTE : {y_train_bal.value_counts().to_dict()}")

# Entraîner le modèle
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train_bal, y_train_bal)

# Évaluer
print("\n=== Résultats ===")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Normal', 'Panne']))

print("=== Matrice de confusion ===")
print(confusion_matrix(y_test, y_pred))