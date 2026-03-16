"""
Pilar — Entraînement Modèle Pompe Centrifuge
Génère un dataset synthétique physiquement réaliste pour pompes centrifuges industrielles
et entraîne les modèles de détection de panne.

Features:
  vibration          (mm/s)   — vibration RMS sur le palier
  temp_palier        (°C)     — température du palier
  debit              (m³/h)   — débit volumique
  pression_entree    (bar)    — pression d'aspiration
  pression_sortie    (bar)    — pression de refoulement
  courant_moteur     (A)      — courant absorbé par le moteur
  temp_moteur        (°C)     — température du moteur
  heure_fonctionnement (h)    — heures de fonctionnement cumulées

Zones de panne:
  CAV — Cavitation
  ROL — Bearing Failure (Roulement)
  ETN — Seal Failure (Étanchéité)
  IMP — Impeller Wear
  MOT — Motor Fault

Exécutez ce script sur votre PC local pour générer modele_pannes.pkl, scaler.pkl, modeles_zones.pkl
"""

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

print("=" * 65)
print("PILAR — Entraînement Modèle Pompe Centrifuge")
print("=" * 65)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
N_NORMAL   = 18000   # échantillons en fonctionnement normal
N_FAILURES = 5400    # échantillons en mode dégradé / panne
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FEATURES = ['vibration','temp_palier','debit','pression_entree','pression_sortie',
            'courant_moteur','temp_moteur','heure_fonctionnement']
ZONES    = ['CAV','ROL','ETN','IMP','MOT']

# ── 1. GÉNÉRATION DONNÉES NORMALES ────────────────────────────────────────────
print("\n[1/8] Génération des données de fonctionnement normal...")

def normal_operating():
    """Pompe centrifuge en service normal — plages opérationnelles réalistes."""
    return {
        'vibration':          np.clip(np.random.lognormal(0.8, 0.35),  0.3,  7.0),
        'temp_palier':        np.clip(np.random.normal(65, 10),         30,   95),
        'debit':              np.clip(np.random.normal(45, 12),         5,   120),
        'pression_entree':    np.clip(np.random.normal(1.5, 0.4),       0.1,  5.0),
        'pression_sortie':    np.clip(np.random.normal(4.5, 1.2),       1.0, 18.0),
        'courant_moteur':     np.clip(np.random.normal(18, 3.5),        5,   40),
        'temp_moteur':        np.clip(np.random.normal(75, 12),         30,  110),
        'heure_fonctionnement': np.clip(np.random.exponential(4000),    0, 25000),
    }

normal_rows = [normal_operating() for _ in range(N_NORMAL)]
df_normal = pd.DataFrame(normal_rows)
df_normal['failure'] = 0
for z in ZONES:
    df_normal[z] = 0

# ── 2. GÉNÉRATION DONNÉES DÉGRADÉES PAR ZONE ──────────────────────────────────
print("[2/8] Génération des données de panne par zone...")

def cav_failure():
    """Cavitation — chute de débit, vibration élevée, pression aspiration basse."""
    vib = np.clip(np.random.normal(8.5, 2.5),  4.0, 25.0)
    return {
        'vibration':          vib,
        'temp_palier':        np.clip(np.random.normal(72, 12),     35, 110),
        'debit':              np.clip(np.random.normal(22, 10),      2,  60),
        'pression_entree':    np.clip(np.random.normal(0.4, 0.2),    0.05, 1.2),
        'pression_sortie':    np.clip(np.random.normal(3.0, 1.2),    0.5, 10),
        'courant_moteur':     np.clip(np.random.normal(20, 4),       8,  45),
        'temp_moteur':        np.clip(np.random.normal(82, 14),      40, 130),
        'heure_fonctionnement': np.clip(np.random.exponential(5000), 0, 30000),
    }

def rol_failure():
    """Roulement dégradé — vibration très élevée, temp palier critique."""
    hf = np.clip(np.random.normal(12000, 4000), 3000, 35000)
    temp_incr = np.clip((hf - 8000) / 800, 0, 50)
    return {
        'vibration':          np.clip(np.random.normal(11, 3.5),     5.0, 30),
        'temp_palier':        np.clip(np.random.normal(95 + temp_incr, 12), 55, 150),
        'debit':              np.clip(np.random.normal(40, 12),       5, 100),
        'pression_entree':    np.clip(np.random.normal(1.4, 0.4),     0.1, 5),
        'pression_sortie':    np.clip(np.random.normal(4.0, 1.3),     0.5, 16),
        'courant_moteur':     np.clip(np.random.normal(22, 4.5),      8,  50),
        'temp_moteur':        np.clip(np.random.normal(88, 14),       40, 140),
        'heure_fonctionnement': hf,
    }

def etn_failure():
    """Perte d'étanchéité — chute pression, débit réduit, vibration modérée."""
    return {
        'vibration':          np.clip(np.random.normal(4.5, 1.5),    1.0, 12),
        'temp_palier':        np.clip(np.random.normal(68, 10),       35, 100),
        'debit':              np.clip(np.random.normal(28, 10),        3,  75),
        'pression_entree':    np.clip(np.random.normal(0.8, 0.3),     0.1, 3.0),
        'pression_sortie':    np.clip(np.random.normal(2.8, 1.0),     0.3, 10),
        'courant_moteur':     np.clip(np.random.normal(16, 3.5),      5,  38),
        'temp_moteur':        np.clip(np.random.normal(78, 13),       40, 125),
        'heure_fonctionnement': np.clip(np.random.exponential(6000),  0, 30000),
    }

def imp_failure():
    """Usure roue — débit dégradé, puissance consommée élevée, vibrations."""
    hf = np.clip(np.random.normal(15000, 5000), 5000, 40000)
    return {
        'vibration':          np.clip(np.random.normal(6.5, 2.0),    2.0, 18),
        'temp_palier':        np.clip(np.random.normal(78, 12),       40, 120),
        'debit':              np.clip(np.random.normal(30, 10),        5,  80),
        'pression_entree':    np.clip(np.random.normal(1.3, 0.4),     0.1, 5),
        'pression_sortie':    np.clip(np.random.normal(3.2, 1.2),     0.5, 12),
        'courant_moteur':     np.clip(np.random.normal(25, 5),        10,  55),
        'temp_moteur':        np.clip(np.random.normal(92, 15),       45, 145),
        'heure_fonctionnement': hf,
    }

def mot_failure():
    """Défaut moteur — courant anormal, température moteur élevée."""
    return {
        'vibration':          np.clip(np.random.normal(5.0, 2.0),    1.5, 15),
        'temp_palier':        np.clip(np.random.normal(70, 12),       35, 110),
        'debit':              np.clip(np.random.normal(38, 12),        5, 110),
        'pression_entree':    np.clip(np.random.normal(1.4, 0.4),     0.1, 5),
        'pression_sortie':    np.clip(np.random.normal(4.0, 1.3),     0.5, 15),
        'courant_moteur':     np.clip(np.random.normal(32, 8),        15,  80),
        'temp_moteur':        np.clip(np.random.normal(115, 18),      65, 200),
        'heure_fonctionnement': np.clip(np.random.exponential(7000),   0, 35000),
    }

zone_generators = {
    'CAV': cav_failure,
    'ROL': rol_failure,
    'ETN': etn_failure,
    'IMP': imp_failure,
    'MOT': mot_failure,
}

n_per_zone = N_FAILURES // len(ZONES)
failure_rows = []

for zone, gen in zone_generators.items():
    rows = [gen() for _ in range(n_per_zone)]
    zone_df = pd.DataFrame(rows)
    zone_df['failure'] = 1
    for z in ZONES:
        zone_df[z] = 1 if z == zone else 0
    # Petite contamination croisée (transitions réelles)
    cross = np.random.choice(ZONES, size=len(zone_df), p=[0.05, 0.1, 0.05, 0.05, 0.75] if zone != 'ROL' else [0.05, 0.75, 0.05, 0.1, 0.05])
    for idx, cz in enumerate(cross):
        if cz != zone:
            zone_df.at[idx, cz] = 1
    failure_rows.append(zone_df)
    print(f"   {zone}: {n_per_zone} échantillons")

df_failures = pd.concat(failure_rows, ignore_index=True)

# ── 3. FUSION ──────────────────────────────────────────────────────────────────
print("\n[3/8] Fusion et mélange...")
df = pd.concat([df_normal, df_failures], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"   Total : {len(df)} lignes | {df['failure'].sum()} pannes ({df['failure'].mean()*100:.1f}%)")

# ── 4. SPLIT ───────────────────────────────────────────────────────────────────
print("\n[4/8] Split train/test (80/20)...")
X = df[FEATURES]
y = df['failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# ── 5. NORMALISATION + SMOTE ──────────────────────────────────────────────────
print("[5/8] StandardScaler + SMOTE...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

X_train_bal, y_train_bal = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_s, y_train)
print(f"   Après SMOTE: {dict(pd.Series(y_train_bal).value_counts())}")

# ── 6. MODÈLE PRINCIPAL ───────────────────────────────────────────────────────
print("\n[6/8] Entraînement du modèle principal (GradientBoosting)...")
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=RANDOM_STATE
)
model.fit(X_train_bal, y_train_bal)

y_pred = model.predict(X_test_s)
recall    = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
print(f"\n   Recall:    {recall*100:.1f}%")
print(f"   Precision: {precision*100:.1f}%")
print(f"   F1-score:  {f1*100:.1f}%")
print("\n" + classification_report(y_test, y_pred, target_names=['Normal', 'Panne']))

# ── 7. MODÈLES PAR ZONE ───────────────────────────────────────────────────────
print("[7/8] Entraînement des modèles par zone de panne...")
modeles_zones = {}

for zone in ZONES:
    y_zone = df.loc[X.index, zone]
    X_z_train, X_z_test, y_z_train, y_z_test = train_test_split(
        X, y_zone, test_size=0.2, random_state=RANDOM_STATE, stratify=y_zone
    )
    X_z_train_s = scaler.transform(X_z_train)
    X_z_test_s  = scaler.transform(X_z_test)
    try:
        X_z_bal, y_z_bal = SMOTE(random_state=RANDOM_STATE).fit_resample(X_z_train_s, y_z_train)
    except Exception:
        X_z_bal, y_z_bal = X_z_train_s, y_z_train.values
    m_zone = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                           random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0)
    m_zone.fit(X_z_bal, y_z_bal)
    modeles_zones[zone] = m_zone
    y_z_pred = m_zone.predict(X_z_test_s)
    r = recall_score(y_z_test, y_z_pred, zero_division=0)
    print(f"   {zone}: Recall {r*100:.1f}%")

# ── 8. SAUVEGARDE ─────────────────────────────────────────────────────────────
print("\n[8/8] Sauvegarde des fichiers .pkl...")

with open("modele_pannes.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("modeles_zones.pkl", "wb") as f:
    pickle.dump(modeles_zones, f)

# Mise à jour model_meta.json
import json
from datetime import datetime

meta = {
    "model_name": "GradientBoosting",
    "recall":     round(recall * 100, 1),
    "precision":  round(precision * 100, 1),
    "f1":         round(f1 * 100, 1),
    "n_train":    len(X_train_bal),
    "n_failures": int(y_train_bal.sum()),
    "failure_rate": round(float(y_train_bal.mean()) * 100, 1),
    "source":     "synthetic_pump_data",
    "domain":     "centrifugal_pump",
    "trained_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "zones":      ZONES,
    "feature_ranges": {
        feat: [round(float(X[feat].min()), 3), round(float(X[feat].max()), 3)]
        for feat in FEATURES
    }
}

with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n   modele_pannes.pkl   OK")
print("   scaler.pkl          OK")
print("   modeles_zones.pkl   OK")
print("   model_meta.json     OK")
print("\n" + "=" * 65)
print(f"Terminé — Recall {recall*100:.1f}% | Precision {precision*100:.1f}% | F1 {f1*100:.1f}%")
print("=" * 65)
