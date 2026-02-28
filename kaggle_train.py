import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle

df = pd.read_csv("predictive_maintenance.csv")

# Voir les types de pannes
print("=== Types de pannes ===")
print(df['Failure Type'].value_counts())

# Préparation
df = df.drop(columns=['UDI', 'Product ID'])
df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})
df['ecart_temp'] = df['Process temperature [K]'] - df['Air temperature [K]']

X = df.drop(columns=['Target', 'Failure Type'])
y = df['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)

model = XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.03,
                      subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(X_train_bal, y_train_bal)

probas = model.predict_proba(X_test)[:, 1]
y_pred = (probas >= 0.3).astype(int)

print("\n=== Résultats ===")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Panne']))

# Sauvegarder
with open("modele_pannes.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Modèle sauvegardé !")