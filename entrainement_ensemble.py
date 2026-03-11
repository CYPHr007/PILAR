"""
Pilar — Entraînement Ensemble Model
Combine RandomForest + XGBoost + SVM + StackingClassifier
Lance ce script sur ton PC local pour générer les nouveaux .pkl
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PILAR — Entraînement Ensemble Model")
print("=" * 60)

# ── 1. CHARGEMENT DES DONNÉES ─────────────────────────────────
print("\n📦 Chargement des données...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df = pd.read_csv(url)
print(f"✅ {len(df)} lignes chargées | {df['Machine failure'].sum()} pannes ({df['Machine failure'].mean()*100:.1f}%)")

# ── 2. PRÉPARATION ────────────────────────────────────────────
df = df.drop(columns=['UDI', 'Product ID'])
df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})
df['ecart_temp'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['puissance'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']

FEATURES = ['Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
            'ecart_temp', 'puissance']

X = df[FEATURES]
y = df['Machine failure']

# Zones de panne individuelles
ZONES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE pour équilibrer
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)
print(f"✅ Données équilibrées avec SMOTE : {dict(pd.Series(y_train_bal).value_counts())}")

# ── 3. DÉFINIR LES MODÈLES ────────────────────────────────────
print("\n🔧 Définition des modèles...")

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, eval_metric='logloss')
gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
svm = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

# ── 4. TESTER CHAQUE MODÈLE ───────────────────────────────────
print("\n📊 Comparaison des modèles...")
print("-" * 50)

results = {}
models = {'RandomForest': rf, 'XGBoost': xgb, 'GradientBoosting': gb, 'SVM': svm, 'LogisticRegression': lr}

for name, m in models.items():
    m.fit(X_train_bal, y_train_bal)
    y_pred = m.predict(X_test)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {'model': m, 'recall': recall, 'f1': f1}
    print(f"{name:20s} | Recall: {recall*100:.1f}% | F1: {f1*100:.1f}%")

# ── 5. VOTING CLASSIFIER ──────────────────────────────────────
print("\n🗳️  Entraînement VotingClassifier...")
voting = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('gb', gb)],
    voting='soft'
)
voting.fit(X_train_bal, y_train_bal)
y_pred_v = voting.predict(X_test)
recall_v = recall_score(y_test, y_pred_v)
f1_v = f1_score(y_test, y_pred_v)
print(f"{'VotingClassifier':20s} | Recall: {recall_v*100:.1f}% | F1: {f1_v*100:.1f}%")

# ── 6. STACKING CLASSIFIER ────────────────────────────────────
print("\n🏆 Entraînement StackingClassifier...")
stacking = StackingClassifier(
    estimators=[('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
                ('xgb', XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss')),
                ('gb', GradientBoostingClassifier(n_estimators=150, random_state=42))],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5, n_jobs=-1
)
stacking.fit(X_train_bal, y_train_bal)
y_pred_s = stacking.predict(X_test)
recall_s = recall_score(y_test, y_pred_s)
f1_s = f1_score(y_test, y_pred_s)
print(f"{'StackingClassifier':20s} | Recall: {recall_s*100:.1f}% | F1: {f1_s*100:.1f}%")

# ── 7. CHOISIR LE MEILLEUR ────────────────────────────────────
all_results = {
    **{k: v['recall'] for k, v in results.items()},
    'VotingClassifier': recall_v,
    'StackingClassifier': recall_s
}
all_models = {
    **{k: v['model'] for k, v in results.items()},
    'VotingClassifier': voting,
    'StackingClassifier': stacking
}

best_name = max(all_results, key=all_results.get)
best_model = all_models[best_name]
best_recall = all_results[best_name]

print("\n" + "=" * 60)
print(f"🏆 MEILLEUR MODÈLE : {best_name}")
print(f"   Recall : {best_recall*100:.1f}%")
print("=" * 60)

# Rapport détaillé du meilleur
y_pred_best = best_model.predict(X_test)
print("\n📋 Rapport détaillé :")
print(classification_report(y_test, y_pred_best, target_names=['Normal', 'Panne']))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred_best))

# ── 8. ENTRAÎNER LES MODÈLES ZONES ───────────────────────────
print("\n🔍 Entraînement des modèles de zones de panne...")
modeles_zones = {}
ZONE_NAMES = {'TWF': 'Tool Wear Failure', 'HDF': 'Heat Dissipation Failure',
              'PWF': 'Power Failure', 'OSF': 'Overstrain Failure', 'RNF': 'Random Failure'}

for zone in ZONES:
    if zone in df.columns and df[zone].sum() > 5:
        y_zone = df[zone]
        X_z_train, X_z_test, y_z_train, y_z_test = train_test_split(
            X_scaled, y_zone, test_size=0.2, random_state=42, stratify=y_zone
        )
        try:
            X_z_bal, y_z_bal = SMOTE(random_state=42).fit_resample(X_z_train, y_z_train)
        except:
            X_z_bal, y_z_bal = X_z_train, y_z_train

        m_zone = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        m_zone.fit(X_z_bal, y_z_bal)
        modeles_zones[zone] = m_zone
        y_z_pred = m_zone.predict(X_z_test)
        print(f"  {ZONE_NAMES[zone]:30s} | Recall: {recall_score(y_z_test, y_z_pred)*100:.1f}%")

# ── 9. SAUVEGARDER ───────────────────────────────────────────
print("\n💾 Sauvegarde des fichiers...")

with open("modele_pannes.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("modeles_zones.pkl", "wb") as f:
    pickle.dump(modeles_zones, f)

print("✅ modele_pannes.pkl  — meilleur modèle sauvegardé")
print("✅ scaler.pkl         — scaler sauvegardé")
print("✅ modeles_zones.pkl  — modèles zones sauvegardés")
print("\n🚀 Remplace les anciens .pkl dans ton projet par ces nouveaux !")
print("   Sur PythonAnywhere : Files → PILAR → upload les 3 fichiers")