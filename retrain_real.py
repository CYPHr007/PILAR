"""
Pilar — Retraining sur données réelles pilar_mixed_machine_results.csv
"""
import pandas as pd, numpy as np, pickle, json, warnings, datetime
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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

CSV = r'C:/Users/info/Downloads/pilar_mixed_machine_results.csv'

print('=' * 60)
print('PILAR — Retraining sur données réelles')
print(f'XGBoost: {HAS_XGB} | SMOTE: {HAS_SMOTE}')
print('=' * 60)

# ── 1. CHARGEMENT ──────────────────────────────────────────────
print('\n[1/6] Chargement...')
df = pd.read_csv(CSV)
print(f'     {len(df)} lignes | colonnes: {list(df.columns)}')

# ── 2. NETTOYAGE ───────────────────────────────────────────────
print('[2/6] Nettoyage...')
n0 = len(df)
df = df[df['temp_air'] >= 0].copy()       # température impossible
df = df[df['vitesse'] >= 0].copy()        # vitesse impossible
print(f'     Supprimé {n0 - len(df)} lignes aberrantes -> {len(df)} lignes propres')
print(f'     Pannes : {df.panne.sum()} ({df.panne.mean()*100:.1f}%)')

# ── 3. CONVERSION UNITÉS ──────────────────────────────────────
print('[3/6] Conversion unités (°C -> K)...')
df['temp_air_k']     = df['temp_air'] + 273.15
df['temp_process_k'] = df['temp_process'] + 273.15
print(f'     temp_air:     {df.temp_air_k.min():.1f} - {df.temp_air_k.max():.1f} K')
print(f'     temp_process: {df.temp_process_k.min():.1f} - {df.temp_process_k.max():.1f} K')
print(f'     vitesse:      {df.vitesse.min():.0f} - {df.vitesse.max():.0f} rpm')
print(f'     couple:       {df.couple.min():.1f} - {df.couple.max():.1f} Nm')
print(f'     usure:        {df.usure.min():.0f} - {df.usure.max():.0f} min')

# ── 4. FEATURES ────────────────────────────────────────────────
print('[4/6] Préparation features...')
df['ecart_temp'] = df['temp_process_k'] - df['temp_air_k']
df['puissance']  = df['vitesse'] * df['couple']
df['type']       = df['type'].astype(int)

FEATURES = ['type', 'temp_air_k', 'temp_process_k', 'vitesse', 'couple', 'usure', 'ecart_temp', 'puissance']
# Noms compatibles avec COLONNES dans etape7.py
COLONNES  = ['Type', 'Air temperature [K]', 'Process temperature [K]',
             'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'ecart_temp', 'puissance']

X = df[FEATURES].values
y = df['panne'].values

scaler_new = StandardScaler()
X_sc = scaler_new.fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42, stratify=y)

if HAS_SMOTE:
    try:
        X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
        print(f'     SMOTE: {dict(pd.Series(y_tr).value_counts())}')
    except Exception as e:
        print(f'     SMOTE ignoré: {e}')

# ── 5. ENTRAINEMENT ────────────────────────────────────────────
print('[5/6] Entrainement...')
candidates = {}

rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
r = recall_score(y_te, rf.predict(X_te))
f = f1_score(y_te, rf.predict(X_te))
p = precision_score(y_te, rf.predict(X_te))
candidates['RandomForest'] = (rf, r, f, p)
print(f'     RandomForest     | Recall: {r*100:.1f}%  Precision: {p*100:.1f}%  F1: {f*100:.1f}%')

gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
gb.fit(X_tr, y_tr)
r = recall_score(y_te, gb.predict(X_te))
f = f1_score(y_te, gb.predict(X_te))
p = precision_score(y_te, gb.predict(X_te))
candidates['GradientBoosting'] = (gb, r, f, p)
print(f'     GradientBoosting | Recall: {r*100:.1f}%  Precision: {p*100:.1f}%  F1: {f*100:.1f}%')

if HAS_XGB:
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                        random_state=42, eval_metric='logloss', n_jobs=-1)
    xgb.fit(X_tr, y_tr)
    r = recall_score(y_te, xgb.predict(X_te))
    f = f1_score(y_te, xgb.predict(X_te))
    p = precision_score(y_te, xgb.predict(X_te))
    candidates['XGBoost'] = (xgb, r, f, p)
    print(f'     XGBoost          | Recall: {r*100:.1f}%  Precision: {p*100:.1f}%  F1: {f*100:.1f}%')

best_name = max(candidates, key=lambda k: candidates[k][1])
best_model, best_recall, best_f1, best_prec = candidates[best_name]

print(f'\n     => Meilleur : {best_name} | Recall: {best_recall*100:.1f}%  F1: {best_f1*100:.1f}%')
print(classification_report(y_te, best_model.predict(X_te), target_names=['Normal', 'Panne']))

# ── 5b. MODÈLES ZONES ──────────────────────────────────────────
print('     Modèles zones...')
modeles_zones_new = {}

ZONE_RULES = {
    'TWF': (df['usure'] > df['usure'].quantile(0.80)).astype(int).values,
    'HDF': (df['ecart_temp'] < df['ecart_temp'].quantile(0.20)).astype(int).values,
    'PWF': ((df['puissance'] < df['puissance'].quantile(0.05)) |
            (df['puissance'] > df['puissance'].quantile(0.95))).astype(int).values,
    'OSF': (df['couple'] > df['couple'].quantile(0.85)).astype(int).values,
    'RNF': y,
}

for zone, y_z in ZONE_RULES.items():
    if y_z.sum() < 20:
        print(f'     {zone}: ignoré (< 20 cas)')
        continue
    try:
        strat = y_z if y_z.sum() >= 20 else None
        Xz_tr, Xz_te, yz_tr, yz_te = train_test_split(X_sc, y_z, test_size=0.2,
                                                        random_state=42, stratify=strat)
        if HAS_SMOTE:
            try:
                Xz_tr, yz_tr = SMOTE(random_state=42).fit_resample(Xz_tr, yz_tr)
            except Exception:
                pass
        mz = (XGBClassifier(n_estimators=150, max_depth=5, random_state=42,
                            eval_metric='logloss', n_jobs=-1)
              if HAS_XGB else
              GradientBoostingClassifier(n_estimators=100, random_state=42))
        mz.fit(Xz_tr, yz_tr)
        r_z = recall_score(yz_te, mz.predict(Xz_te), zero_division=0)
        modeles_zones_new[zone] = mz
        print(f'     {zone}: Recall {r_z*100:.1f}%  (n={y_z.sum()} positifs)')
    except Exception as e:
        print(f'     {zone}: ignoré ({e})')

# ── 6. SAUVEGARDE ──────────────────────────────────────────────
print('[6/6] Sauvegarde...')
with open('modele_pannes.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler_new, f)
with open('modeles_zones.pkl', 'wb') as f:
    pickle.dump(modeles_zones_new, f)

model_meta = {
    'model_name': best_name,
    'recall':    round(best_recall * 100, 1),
    'precision': round(best_prec   * 100, 1),
    'f1':        round(best_f1     * 100, 1),
    'n_train':   int(len(df)),
    'n_failures': int(y.sum()),
    'failure_rate': round(float(y.mean()) * 100, 1),
    'source': 'pilar_mixed_machine_results.csv',
    'trained_at': datetime.datetime.now().isoformat()[:19],
    'zones': list(modeles_zones_new.keys()),
    'feature_ranges': {
        'temp_air_k':     [round(float(df.temp_air_k.min()),    1), round(float(df.temp_air_k.max()),    1)],
        'temp_process_k': [round(float(df.temp_process_k.min()),1), round(float(df.temp_process_k.max()),1)],
        'vitesse':        [round(float(df.vitesse.min()),        0), round(float(df.vitesse.max()),        0)],
        'couple':         [round(float(df.couple.min()),         1), round(float(df.couple.max()),         1)],
        'usure':          [round(float(df.usure.min()),          0), round(float(df.usure.max()),          0)],
    }
}
with open('model_meta.json', 'w') as f:
    json.dump(model_meta, f, indent=2, ensure_ascii=False)

print(f'     modele_pannes.pkl  OK  ({best_name}, Recall {best_recall*100:.1f}%)')
print(f'     scaler.pkl         OK')
print(f'     modeles_zones.pkl  OK  ({len(modeles_zones_new)} zones)')
print(f'     model_meta.json    OK')
print('\n' + '='*60)
print('RETRAINING TERMINÉ')
print('='*60)
