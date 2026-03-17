"""
Pilar — Retraining pompe centrifuge
Features : vibration, temp_palier, debit, pression_entree, pression_sortie,
           courant_moteur, temp_moteur, heure_fonctionnement
Zones    : CAV (Cavitation) | ROL (Bearing Failure) | ETN (Seal Failure)
           IMP (Impeller Wear) | MOT (Motor Fault)
"""
import sys, os, pandas as pd, numpy as np, pickle, json, warnings, datetime
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

# ── CONFIG ──────────────────────────────────────────────────────────────────
# Chemin vers ton CSV — modifie si nécessaire, ou passe en argument : python retrain_real.py mon_fichier.csv
CSV = sys.argv[1] if len(sys.argv) > 1 else r'C:/Users/info/Downloads/pilar_mixed_machine_results.csv'

COLONNES = [
    'vibration', 'temp_palier', 'debit',
    'pression_entree', 'pression_sortie',
    'courant_moteur', 'temp_moteur', 'heure_fonctionnement'
]

# Patterns de détection auto des colonnes (noms alternatifs acceptés)
COL_PATTERNS = {
    'vibration':           ['vibration','vib','vibration_mm','vib_mms','vibration_mmps','accel','acceleration','vibr'],
    'temp_palier':         ['temp_palier','bearing_temp','palier_temp','t_palier','tpalier','bearing_temperature','temp_roulement'],
    'debit':               ['debit','flow','flow_rate','flowrate','debit_m3h','flow_m3h','caudal'],
    'pression_entree':     ['pression_entree','inlet_pressure','pressure_in','p_in','pe','suction_pressure','pression_aspiration'],
    'pression_sortie':     ['pression_sortie','outlet_pressure','discharge_pressure','pressure_out','p_out','ps','pression_refoulement'],
    'courant_moteur':      ['courant_moteur','motor_current','current_motor','im','courant_a','current_a','ampere_moteur','motor_amp'],
    'temp_moteur':         ['temp_moteur','motor_temp','motor_temperature','tm','temperature_moteur','t_moteur'],
    'heure_fonctionnement':['heure_fonctionnement','run_hours','operating_hours','runtime','heures','hours','hf','total_hours'],
}

TARGET_PATTERNS = ['panne','failure','fault','label','target','defaut','defaillance','anomalie','anomaly','broken']

print('=' * 65)
print('PILAR — Retraining pompe centrifuge')
print(f'XGBoost : {HAS_XGB} | SMOTE : {HAS_SMOTE}')
print('=' * 65)

# ── 1. CHARGEMENT ────────────────────────────────────────────────────────────
print(f'\n[1/7] Chargement : {CSV}')
if not os.path.exists(CSV):
    print(f'  ERREUR : fichier introuvable — {CSV}')
    print('  Usage  : python retrain_real.py chemin/vers/ton_fichier.csv')
    sys.exit(1)

df = pd.read_csv(CSV)
print(f'  {len(df)} lignes | {len(df.columns)} colonnes')
print(f'  Colonnes : {list(df.columns)}')

# ── 2. DÉTECTION AUTO DES COLONNES ──────────────────────────────────────────
print('\n[2/7] Détection des colonnes...')
cols_lower = {c.lower().strip(): c for c in df.columns}

def find_col(patterns):
    for p in patterns:
        if p in cols_lower:
            return cols_lower[p]
    return None

col_map = {}
for feat, pats in COL_PATTERNS.items():
    found = find_col(pats)
    if found:
        col_map[feat] = found
        print(f'  {feat:25s} -> {found}')
    else:
        print(f'  {feat:25s} -> NON TROUVÉ (sera imputé avec médiane)')

target_col = find_col(TARGET_PATTERNS)
if not target_col:
    # Essaie les colonnes binaires 0/1
    for c in df.columns:
        if df[c].dropna().isin([0, 1]).all() and df[c].sum() > 0:
            target_col = c
            break
if not target_col:
    print('\n  ERREUR : colonne cible (panne/failure) introuvable.')
    print('  Colonnes disponibles :', list(df.columns))
    sys.exit(1)
print(f'\n  Colonne cible         -> {target_col}')

# ── 3. CONSTRUCTION DU DATAFRAME ────────────────────────────────────────────
print('\n[3/7] Construction features...')
feat_df = pd.DataFrame(index=df.index)
for feat in COLONNES:
    if feat in col_map:
        feat_df[feat] = pd.to_numeric(df[col_map[feat]], errors='coerce')
    else:
        feat_df[feat] = np.nan

y = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int).values

# Imputation par médiane pour les colonnes manquantes / NaN
medians = {}
for c in COLONNES:
    med = feat_df[c].median()
    if np.isnan(med):
        # Valeurs physiques par défaut si aucune donnée
        defaults = {'vibration':2.5,'temp_palier':65.0,'debit':45.0,
                    'pression_entree':1.5,'pression_sortie':4.5,
                    'courant_moteur':18.0,'temp_moteur':75.0,'heure_fonctionnement':5000.0}
        med = defaults[c]
    medians[c] = round(float(med), 3)
    feat_df[c] = feat_df[c].fillna(med)

# Supprime les lignes complètement aberrantes
feat_df = feat_df.clip(lower=0)
feat_df['vibration']    = feat_df['vibration'].clip(upper=100)
feat_df['temp_palier']  = feat_df['temp_palier'].clip(upper=200)
feat_df['temp_moteur']  = feat_df['temp_moteur'].clip(upper=300)
feat_df['debit']        = feat_df['debit'].clip(upper=2000)

# Aligne y
feat_df['__target__'] = y
feat_df = feat_df.dropna(subset=['__target__'])
y = feat_df['__target__'].astype(int).values
feat_df = feat_df.drop(columns=['__target__'])

X = feat_df[COLONNES].values
print(f'  Dataset final : {len(X)} lignes | Pannes : {y.sum()} ({y.mean()*100:.1f}%)')
for c in COLONNES:
    print(f'  {c:30s} : [{feat_df[c].min():.2f} — {feat_df[c].max():.2f}]  médiane={medians[c]}')

# ── 4. SCALING + SPLIT ───────────────────────────────────────────────────────
print('\n[4/7] Scaling + split...')
scaler_new = StandardScaler()
X_sc = scaler_new.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42, stratify=y)

if HAS_SMOTE and y_tr.sum() >= 6:
    try:
        X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
        print(f'  SMOTE : {dict(pd.Series(y_tr).value_counts())}')
    except Exception as e:
        print(f'  SMOTE ignoré : {e}')

# ── 5. ENTRAÎNEMENT MODÈLE PRINCIPAL ────────────────────────────────────────
print('\n[5/7] Entraînement modèle principal...')
candidates = {}

rf = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
r, f, p = recall_score(y_te, rf.predict(X_te), zero_division=0), f1_score(y_te, rf.predict(X_te), zero_division=0), precision_score(y_te, rf.predict(X_te), zero_division=0)
candidates['RandomForest'] = (rf, r, f, p)
print(f'  RandomForest     | Recall {r*100:.1f}%  Precision {p*100:.1f}%  F1 {f*100:.1f}%')

gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
gb.fit(X_tr, y_tr)
r, f, p = recall_score(y_te, gb.predict(X_te), zero_division=0), f1_score(y_te, gb.predict(X_te), zero_division=0), precision_score(y_te, gb.predict(X_te), zero_division=0)
candidates['GradientBoosting'] = (gb, r, f, p)
print(f'  GradientBoosting | Recall {r*100:.1f}%  Precision {p*100:.1f}%  F1 {f*100:.1f}%')

if HAS_XGB:
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                        scale_pos_weight=(len(y_tr)-y_tr.sum())/(y_tr.sum()+1),
                        random_state=42, eval_metric='logloss', n_jobs=-1, verbosity=0)
    xgb.fit(X_tr, y_tr)
    r, f, p = recall_score(y_te, xgb.predict(X_te), zero_division=0), f1_score(y_te, xgb.predict(X_te), zero_division=0), precision_score(y_te, xgb.predict(X_te), zero_division=0)
    candidates['XGBoost'] = (xgb, r, f, p)
    print(f'  XGBoost          | Recall {r*100:.1f}%  Precision {p*100:.1f}%  F1 {f*100:.1f}%')

best_name = max(candidates, key=lambda k: candidates[k][1])
best_model, best_recall, best_f1, best_prec = candidates[best_name]
print(f'\n  => Meilleur : {best_name}  (Recall {best_recall*100:.1f}%  F1 {best_f1*100:.1f}%)')
print(classification_report(y_te, best_model.predict(X_te), target_names=['Normal','Panne'], zero_division=0))

# ── 6. MODÈLES ZONES (règles physique pompe) ────────────────────────────────
print('[6/7] Modèles zones...')

# Règles physiques pompe centrifuge
q = {}
for c in COLONNES:
    q[c] = {p: np.percentile(feat_df[c].values, p) for p in [5,10,15,20,80,85,90,95]}

ZONE_RULES = {
    # CAV — Cavitation : débit trop faible + vibration élevée
    'CAV': ((feat_df['debit'].values <= q['debit'][20]) &
            (feat_df['vibration'].values >= q['vibration'][80])).astype(int),
    # ROL — Usure roulements : temp palier haute + vibration élevée
    'ROL': ((feat_df['temp_palier'].values >= q['temp_palier'][85]) |
            ((feat_df['temp_palier'].values >= q['temp_palier'][80]) &
             (feat_df['vibration'].values >= q['vibration'][80]))).astype(int),
    # ETN — Fuite joint : pression sortie anormalement basse par rapport à l'entrée
    'ETN': (feat_df['pression_sortie'].values <= q['pression_sortie'][10]).astype(int),
    # IMP — Usure roue : débit bas pour pression normale (rendement dégradé)
    'IMP': ((feat_df['debit'].values <= q['debit'][15]) &
            (feat_df['pression_sortie'].values >= q['pression_sortie'][20])).astype(int),
    # MOT — Défaut moteur : courant élevé + temp moteur élevée
    'MOT': ((feat_df['courant_moteur'].values >= q['courant_moteur'][85]) |
            ((feat_df['courant_moteur'].values >= q['courant_moteur'][80]) &
             (feat_df['temp_moteur'].values >= q['temp_moteur'][80]))).astype(int),
}

modeles_zones_new = {}
for zone, y_z in ZONE_RULES.items():
    n_pos = y_z.sum()
    if n_pos < 10:
        print(f'  {zone} : ignoré (seulement {n_pos} cas positifs)')
        continue
    try:
        strat = y_z if n_pos >= 10 else None
        Xz_tr, Xz_te, yz_tr, yz_te = train_test_split(X_sc, y_z, test_size=0.2,
                                                        random_state=42, stratify=strat)
        if HAS_SMOTE and yz_tr.sum() >= 6:
            try:
                Xz_tr, yz_tr = SMOTE(random_state=42).fit_resample(Xz_tr, yz_tr)
            except Exception:
                pass
        mz = (XGBClassifier(n_estimators=150, max_depth=5, random_state=42,
                            scale_pos_weight=(len(yz_tr)-yz_tr.sum())/(yz_tr.sum()+1),
                            eval_metric='logloss', n_jobs=-1, verbosity=0)
              if HAS_XGB else
              GradientBoostingClassifier(n_estimators=100, random_state=42))
        mz.fit(Xz_tr, yz_tr)
        r_z = recall_score(yz_te, mz.predict(Xz_te), zero_division=0)
        modeles_zones_new[zone] = mz
        print(f'  {zone} : Recall {r_z*100:.1f}%  ({n_pos} positifs)')
    except Exception as e:
        print(f'  {zone} : ignoré ({e})')

# ── 7. SAUVEGARDE ────────────────────────────────────────────────────────────
print('\n[7/7] Sauvegarde...')
with open('modele_pannes.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler_new, f)
with open('modeles_zones.pkl', 'wb') as f:
    pickle.dump(modeles_zones_new, f)

meta = {
    'model_name':    best_name,
    'recall':        round(best_recall * 100, 1),
    'precision':     round(best_prec   * 100, 1),
    'f1':            round(best_f1     * 100, 1),
    'n_train':       int(len(X)),
    'n_failures':    int(y.sum()),
    'failure_rate':  round(float(y.mean()) * 100, 1),
    'colonnes':      COLONNES,
    'zones':         list(modeles_zones_new.keys()),
    'feature_medians': medians,
    'source':        os.path.basename(CSV),
    'trained_at':    datetime.datetime.now().isoformat()[:19],
}
with open('model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f'  modele_pannes.pkl  OK  ({best_name})')
print(f'  scaler.pkl         OK')
print(f'  modeles_zones.pkl  OK  ({list(modeles_zones_new.keys())})')
print(f'  model_meta.json    OK')

print('\n' + '=' * 65)
print('MÉDIANES À COPIER DANS etape7.py -> FEATURE_MEDIANS :')
print(f"FEATURE_MEDIANS = {json.dumps(medians, ensure_ascii=False)}")
print('=' * 65)
print('RETRAINING TERMINÉ')
print('=' * 65)
