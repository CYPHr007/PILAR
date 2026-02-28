import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle

df = pd.read_csv("predictive_maintenance.csv")

# Créer les colonnes zones depuis Failure Type
df['TWF'] = (df['Failure Type'] == 'Tool Wear Failure').astype(int)
df['HDF'] = (df['Failure Type'] == 'Heat Dissipation Failure').astype(int)
df['PWF'] = (df['Failure Type'] == 'Power Failure').astype(int)
df['OSF'] = (df['Failure Type'] == 'Overstrain Failure').astype(int)
df['RNF'] = (df['Failure Type'] == 'Random Failures').astype(int)

df = df.drop(columns=['UDI', 'Product ID', 'Failure Type'])
df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})
df['ecart_temp'] = df['Process temperature [K]'] - df['Air temperature [K]']

X = df.drop(columns=['Target', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
y_main = df['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_main, test_size=0.2, random_state=42, stratify=y_main
)
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Modèle principal
model = XGBClassifier(n_estimators=193, max_depth=4, learning_rate=0.037,
                      subsample=0.83, colsample_bytree=0.68, min_child_weight=9, random_state=42)
model.fit(X_train_bal, y_train_bal)

# Modèles zones
modeles_zones = {}
for col in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
    y_zone = df[col]
    if y_zone.sum() < 10:
        continue
    Xt, Xv, yt, yv = train_test_split(X_scaled, y_zone, test_size=0.2, random_state=42, stratify=y_zone)
    Xb, yb = SMOTE(random_state=42).fit_resample(Xt, yt)
    m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    m.fit(Xb, yb)
    modeles_zones[col] = m
    print(f"✅ Zone {col} entraînée ({y_zone.sum()} pannes)")

with open("modele_pannes.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("modeles_zones.pkl", "wb") as f:
    pickle.dump(modeles_zones, f)

print("\n✅ Tous les modèles sauvegardés !")