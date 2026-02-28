import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import optuna
import pickle

optuna.logging.set_verbosity(optuna.logging.WARNING)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df = pd.read_csv(url)

df = df.drop(columns=['UDI', 'Product ID'])
df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})
df['ecart_temp'] = df['Process temperature [K]'] - df['Air temperature [K]']

X = df.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
y = df['Machine failure']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42
    }
    model = XGBClassifier(**params)
    model.fit(X_train_bal, y_train_bal)
    seuil = trial.suggest_float('seuil', 0.2, 0.5)
    probas = model.predict_proba(X_test)[:, 1]
    y_pred = (probas >= seuil).astype(int)
    return recall_score(y_test, y_pred)

print("🔍 Recherche des meilleurs paramètres... (environ 5 minutes)")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\n✅ Meilleur recall trouvé : {study.best_value:.0%}")
print(f"📋 Meilleurs paramètres : {study.best_params}")

# Réentraîner avec les meilleurs paramètres
best = study.best_params
seuil_optimal = best.pop('seuil')

model_final = XGBClassifier(**best, random_state=42)
model_final.fit(X_train_bal, y_train_bal)

probas = model_final.predict_proba(X_test)[:, 1]
y_pred = (probas >= seuil_optimal).astype(int)

from sklearn.metrics import classification_report
print(f"\n=== Résultats finaux (seuil {seuil_optimal:.2f}) ===")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Panne']))

# Sauvegarder le nouveau modèle
with open("modele_pannes.pkl", "wb") as f:
    pickle.dump(model_final, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Nouveau modèle sauvegardé !")