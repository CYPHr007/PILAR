import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df = pd.read_csv(url)

# Préparation
df = df.drop(columns=['UDI', 'Product ID'])
df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})

# Nouvelle feature : écart de température
df['ecart_temp'] = df['Process temperature [K]'] - df['Air temperature [K]']

X = df.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
y = df['Machine failure']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)

model = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train_bal, y_train_bal)

# Tester différents seuils
print("=== Comparaison des seuils ===\n")
probas = model.predict_proba(X_test)[:, 1]

for seuil in [0.5, 0.4, 0.3, 0.2]:
    y_pred = (probas >= seuil).astype(int)
    recall = recall_score(y_test, y_pred)
    fausses_alarmes = sum((y_pred == 1) & (y_test == 0))
    pannes_detectees = sum((y_pred == 1) & (y_test == 1))
    print(f"Seuil {seuil} → Recall: {recall:.0%} | Pannes détectées: {pannes_detectees}/68 | Fausses alarmes: {fausses_alarmes}")

print("\n=== Meilleur modèle avec seuil 0.3 ===")
y_pred_final = (probas >= 0.3).astype(int)
print(classification_report(y_test, y_pred_final, target_names=['Normal', 'Panne']))