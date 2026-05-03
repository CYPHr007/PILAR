"""
Pilar — Entraînement sur données générées synthétiquement
Génère 30 000 points de données industrielles réalistes
puis entraîne et sauvegarde les modèles .pkl
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[!] XGBoost non disponible, on utilisera GradientBoosting")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[!] imbalanced-learn non disponible, SMOTE désactivé")

np.random.seed(42)
N = 30_000

print("=" * 60)
print("PILAR — Génération + Entraînement (données synthétiques)")
print("=" * 60)

# ── 1. GÉNÉRER LES DONNÉES ────────────────────────────────────
print(f"\n[1/5] Génération de {N} échantillons...")

# Type machine : 60% L, 30% M, 10% H
types = np.random.choice([0, 1, 2], size=N, p=[0.60, 0.30, 0.10])

# Température air : 298–304 K (≈ 25–31°C)
air_temp = np.random.normal(300, 2, N).clip(295, 308)

# Température process : air + 8–12 K
process_temp = air_temp + np.random.normal(10, 1.5, N).clip(5, 15)

# Vitesse rotation : 1168–2886 rpm
rot_speed = np.random.normal(1500, 400, N).clip(1000, 3000)

# Couple : 3.8–76.6 Nm (varie selon type)
torque_base = np.where(types == 0, 38, np.where(types == 1, 43, 50))
torque = (torque_base + np.random.normal(0, 10, N)).clip(3, 80)

# Usure outil : 0–253 min (L accumule plus vite)
wear_max = np.where(types == 0, 240, np.where(types == 1, 220, 200))
tool_wear = np.random.uniform(0, wear_max, N)

# ── 2. CALCULER LES PANNES SELON LES RÈGLES PHYSIQUES ────────
print("[2/5] Calcul des pannes (règles physiques industrielles)...")

ecart_temp = process_temp - air_temp
puissance   = rot_speed * torque  # feature ML (rpm × Nm)

# Puissance réelle en Watts : 2π/60 × rpm × Nm
puissance_w = rot_speed * torque * (2 * np.pi / 60)

# Tool Wear Failure : usure > 200 min ET couple élevé OU usure > 230
TWF = ((tool_wear > 200) & (torque > 60)) | (tool_wear > 235)

# Heat Dissipation Failure : vitesse < 1380 rpm ET écart_temp < 8.6 K
HDF = (rot_speed < 1380) & (ecart_temp < 8.6)

# Power Failure : puissance réelle < 3500 W OU > 9000 W
PWF = (puissance_w < 3500) | (puissance_w > 9000)

# Overstrain Failure : usure × couple > seuil selon type
osf_limit = np.where(types == 0, 11000, np.where(types == 1, 12000, 13000))
OSF = (tool_wear * torque) > osf_limit

# Random Failure : aléatoire 0.1%
RNF = np.random.random(N) < 0.001

# Panne globale : au moins une zone
failure = (TWF | HDF | PWF | OSF | RNF).astype(int)

# Ajouter du bruit réaliste : quelques faux positifs/négatifs (2%)
noise_mask = np.random.random(N) < 0.02
failure[noise_mask] = 1 - failure[noise_mask]

df = pd.DataFrame({
    'Type':                       types,
    'Air temperature [K]':        air_temp,
    'Process temperature [K]':    process_temp,
    'Rotational speed [rpm]':     rot_speed,
    'Torque [Nm]':                torque,
    'Tool wear [min]':            tool_wear,
    'ecart_temp':                 ecart_temp,
    'puissance':                  puissance,
    'TWF':  TWF.astype(int),
    'HDF':  HDF.astype(int),
    'PWF':  PWF.astype(int),
    'OSF':  OSF.astype(int),
    'RNF':  RNF.astype(int),
    'Machine failure': failure,
})

n_fail = failure.sum()
print(f"    Total : {N} lignes | Pannes : {n_fail} ({n_fail/N*100:.1f}%)")
print(f"    TWF:{TWF.sum()} HDF:{HDF.sum()} PWF:{PWF.sum()} OSF:{OSF.sum()} RNF:{RNF.sum()}")

# ── 3. PRÉPARER ───────────────────────────────────────────────
print("[3/5] Préparation des features et scaler...")

FEATURES = ['Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
            'ecart_temp', 'puissance']

X = df[FEATURES].values
y = df['Machine failure'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

if HAS_SMOTE:
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
    print(f"    SMOTE appliqué : {dict(pd.Series(y_train).value_counts())}")

# ── 4. ENTRAÎNER ──────────────────────────────────────────────
print("[4/5] Entraînement des modèles...")

candidates = {}

# Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
r = recall_score(y_test, rf.predict(X_test))
f = f1_score(y_test, rf.predict(X_test))
candidates['RandomForest'] = (rf, r, f)
print(f"    RandomForest      | Recall: {r*100:.1f}% | F1: {f*100:.1f}%")

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
gb.fit(X_train, y_train)
r = recall_score(y_test, gb.predict(X_test))
f = f1_score(y_test, gb.predict(X_test))
candidates['GradientBoosting'] = (gb, r, f)
print(f"    GradientBoosting  | Recall: {r*100:.1f}% | F1: {f*100:.1f}%")

if HAS_XGB:
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                        random_state=42, eval_metric='logloss', n_jobs=-1)
    xgb.fit(X_train, y_train)
    r = recall_score(y_test, xgb.predict(X_test))
    f = f1_score(y_test, xgb.predict(X_test))
    candidates['XGBoost'] = (xgb, r, f)
    print(f"    XGBoost           | Recall: {r*100:.1f}% | F1: {f*100:.1f}%")

# Choisir le meilleur (priorité recall)
best_name = max(candidates, key=lambda k: candidates[k][1])
best_model, best_recall, best_f1 = candidates[best_name]

print(f"\n    => Meilleur : {best_name} | Recall: {best_recall*100:.1f}% | F1: {best_f1*100:.1f}%")
print("\n    Rapport complet :")
print(classification_report(y_test, best_model.predict(X_test),
                             target_names=['Normal', 'Panne']))

# ── 5. MODÈLES ZONES ──────────────────────────────────────────
print("[5/5] Entraînement modèles zones de panne...")

ZONES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
ZONE_NAMES = {
    'TWF': 'Tool Wear Failure',
    'HDF': 'Heat Dissipation Failure',
    'PWF': 'Power Failure',
    'OSF': 'Overstrain Failure',
    'RNF': 'Random Failure',
}
modeles_zones = {}

for zone in ZONES:
    y_zone = df[zone].values
    if y_zone.sum() < 10:
        print(f"    {ZONE_NAMES[zone]:30s} | Ignoré (trop peu de cas)")
        continue

    Xz_train, Xz_test, yz_train, yz_test = train_test_split(
        X_scaled, y_zone, test_size=0.2, random_state=42,
        stratify=y_zone if y_zone.sum() > 20 else None
    )
    try:
        if HAS_SMOTE:
            Xz_train, yz_train = SMOTE(random_state=42).fit_resample(Xz_train, yz_train)
    except Exception:
        pass

    if HAS_XGB:
        mz = XGBClassifier(n_estimators=150, max_depth=5, random_state=42,
                           eval_metric='logloss', n_jobs=-1)
    else:
        mz = GradientBoostingClassifier(n_estimators=100, random_state=42)

    mz.fit(Xz_train, yz_train)
    r_z = recall_score(yz_test, mz.predict(Xz_test), zero_division=0)
    modeles_zones[zone] = mz
    print(f"    {ZONE_NAMES[zone]:30s} | Recall: {r_z*100:.1f}%")

# ── SAUVEGARDE ────────────────────────────────────────────────
print("\nSauvegarde des fichiers .pkl...")

with open("failure_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("zone_models.pkl", "wb") as f:
    pickle.dump(modeles_zones, f)

print("OK failure_model.pkl")
print("OK scaler.pkl")
print("OK zone_models.pkl")
print(f"\nModele : {best_name} | Recall : {best_recall*100:.1f}% | F1 : {best_f1*100:.1f}%")
print("Lancez 'python app.py' pour utiliser les nouveaux modeles.")
