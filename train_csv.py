"""
Pilar - Entrainement depuis donnees_industrielles_extraites_pour_modele.csv
Utilise le segment AI4I (10 000 lignes) avec donnees reelles
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

CSV_PATH = r"C:\Users\info\Documents\PILAR CONSTRUCTOR\donnees_industrielles_extraites_pour_modele.csv"

print("=" * 60)
print("PILAR - Entrainement depuis CSV industriel")
print("=" * 60)

# ── 1. CHARGEMENT ─────────────────────────────────────────────
print("\n[1/5] Chargement du CSV...")
df_raw = pd.read_csv(CSV_PATH, low_memory=False)
print(f"     {len(df_raw)} lignes brutes | {len(df_raw.columns)} colonnes")

# Garder uniquement le segment compatible Pilar
df = df_raw[df_raw['source_dataset'] == 'AI4I_2020_predictive_maintenance'].copy()
print(f"     Segment AI4I retenu : {len(df)} lignes")

# ── 2. PREPARATION ────────────────────────────────────────────
print("[2/5] Preparation des features...")

# Type machine : L=0, M=1, H=2
df['Type'] = df['product_type'].map({'L': 0, 'M': 1, 'H': 2})
df = df.dropna(subset=['Type'])
df['Type'] = df['Type'].astype(int)

# Features de base
REQUIRED = ['air_temperature_k', 'process_temperature_k',
            'rotational_speed_rpm', 'torque_nm', 'tool_wear_min']
df = df.dropna(subset=REQUIRED)

# Features derivees
df['ecart_temp'] = df['process_temperature_k'] - df['air_temperature_k']
df['puissance']  = df['rotational_speed_rpm'] * df['torque_nm']

# Label panne principale
df['failure'] = df['target_machine_failure'].fillna(0).astype(int)

# Zones depuis target_failure_type
zone_map = {
    'tool_wear_failure':        'TWF',
    'heat_dissipation_failure': 'HDF',
    'power_failure':            'PWF',
    'overstrain_failure':       'OSF',
    'random_failure':           'RNF',
}
for key, col in zone_map.items():
    df[col] = (df['target_failure_type'] == key).astype(int)

n_fail = df['failure'].sum()
print(f"     {len(df)} lignes valides | {n_fail} pannes ({n_fail/len(df)*100:.1f}%)")
for col in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
    print(f"       {col}: {df[col].sum()}")

# ── 3. FEATURES / SCALER ──────────────────────────────────────
print("[3/5] Mise a l'echelle...")

FEATURES = ['Type', 'air_temperature_k', 'process_temperature_k',
            'rotational_speed_rpm', 'torque_nm', 'tool_wear_min',
            'ecart_temp', 'puissance']

X = df[FEATURES].values
y = df['failure'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

if HAS_SMOTE:
    try:
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
        print(f"     SMOTE : {dict(pd.Series(y_train).value_counts())}")
    except Exception as e:
        print(f"     SMOTE ignore : {e}")

# ── 4. ENTRAINEMENT ───────────────────────────────────────────
print("[4/5] Entrainement des modeles...")

candidates = {}

rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
r, f = recall_score(y_test, rf.predict(X_test)), f1_score(y_test, rf.predict(X_test))
candidates['RandomForest'] = (rf, r, f)
print(f"     RandomForest     | Recall: {r*100:.1f}% | F1: {f*100:.1f}%")

gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
gb.fit(X_train, y_train)
r, f = recall_score(y_test, gb.predict(X_test)), f1_score(y_test, gb.predict(X_test))
candidates['GradientBoosting'] = (gb, r, f)
print(f"     GradientBoosting | Recall: {r*100:.1f}% | F1: {f*100:.1f}%")

if HAS_XGB:
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                        random_state=42, eval_metric='logloss', n_jobs=-1)
    xgb.fit(X_train, y_train)
    r, f = recall_score(y_test, xgb.predict(X_test)), f1_score(y_test, xgb.predict(X_test))
    candidates['XGBoost'] = (xgb, r, f)
    print(f"     XGBoost          | Recall: {r*100:.1f}% | F1: {f*100:.1f}%")

best_name = max(candidates, key=lambda k: candidates[k][1])
best_model, best_recall, best_f1 = candidates[best_name]

print(f"\n     => Meilleur : {best_name} | Recall: {best_recall*100:.1f}% | F1: {best_f1*100:.1f}%")
print(classification_report(y_test, best_model.predict(X_test),
                             target_names=['Normal', 'Panne']))

# ── 5. MODELES ZONES ──────────────────────────────────────────
print("[5/5] Modeles zones de panne...")

ZONE_NAMES = {
    'TWF': 'Tool Wear Failure',
    'HDF': 'Heat Dissipation Failure',
    'PWF': 'Power Failure',
    'OSF': 'Overstrain Failure',
    'RNF': 'Random Failure',
}
modeles_zones = {}

for zone, label in ZONE_NAMES.items():
    y_zone = df[zone].values
    if y_zone.sum() < 5:
        print(f"     {label:30s} | Ignore (< 5 cas)")
        continue

    Xz_tr, Xz_te, yz_tr, yz_te = train_test_split(
        X_scaled, y_zone, test_size=0.2, random_state=42,
        stratify=y_zone if y_zone.sum() >= 10 else None
    )
    try:
        if HAS_SMOTE:
            Xz_tr, yz_tr = SMOTE(random_state=42).fit_resample(Xz_tr, yz_tr)
    except Exception:
        pass

    mz = (XGBClassifier(n_estimators=150, max_depth=5, random_state=42,
                        eval_metric='logloss', n_jobs=-1)
          if HAS_XGB else
          GradientBoostingClassifier(n_estimators=100, random_state=42))
    mz.fit(Xz_tr, yz_tr)
    r_z = recall_score(yz_te, mz.predict(Xz_te), zero_division=0)
    modeles_zones[zone] = mz
    print(f"     {label:30s} | Recall: {r_z*100:.1f}%")

# ── SAUVEGARDE ────────────────────────────────────────────────
print("\nSauvegarde...")

with open("failure_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("zone_models.pkl", "wb") as f:
    pickle.dump(modeles_zones, f)

print(f"     failure_model.pkl  OK  ({best_name}, Recall {best_recall*100:.1f}%)")
print(f"     scaler.pkl         OK")
print(f"     zone_models.pkl  OK  ({len(modeles_zones)} zones)")
