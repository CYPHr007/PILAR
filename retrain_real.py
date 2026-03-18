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
    'vibration':           ['vibration','vib','vibration_mm','vib_mms','vibration_mmps','vibrations_mm_s',
                            'vibrations','accel','acceleration','vibr','vs1','vs2'],
    'temp_palier':         ['temp_palier','bearing_temp','palier_temp','t_palier','tpalier',
                            'bearing_temperature','temp_roulement','temperature_palier_c',
                            'temperature_palier','ts1','ts2','ts3','ts4','bearing_t'],
    'debit':               ['debit','flow','flow_rate','flowrate','debit_m3h','flow_m3h',
                            'debit_l_min','debit_lmin','fs1','fs2','caudal'],
    'pression_entree':     ['pression_entree','inlet_pressure','pressure_in','p_in','pe',
                            'suction_pressure','pression_aspiration','pression_entree_bar',
                            'ps1','ps2_in','pressure_inlet'],
    'pression_sortie':     ['pression_sortie','outlet_pressure','discharge_pressure','pressure_out',
                            'p_out','ps','pression_refoulement','pression_sortie_bar',
                            'ps2','ps3','ps4','ps5','ps6','pressure_outlet','discharge_p'],
    'courant_moteur':      ['courant_moteur','motor_current','current_motor','im','courant_a',
                            'courant_moteur_a','current_a','ampere_moteur','motor_amp','eps1','current'],
    'temp_moteur':         ['temp_moteur','motor_temp','motor_temperature','tm',
                            'temperature_moteur','t_moteur','motor_t','temp_moteur_c'],
    'heure_fonctionnement':['heure_fonctionnement','run_hours','operating_hours','runtime',
                            'heures','hours','hf','total_hours','cycle_id'],
}

TARGET_PATTERNS = ['panne','failure','fault','label','target','defaut','defaillance','anomalie','anomaly','broken',
                   'etat_pompe_code','etat_pompe','status','machine_status','pump_status','condition']

# Colonnes à convertir d'unité avant entraînement (pour rester cohérent avec le live monitor)
# débit : L/min -> m³/h  (× 0.06)
DEBIT_LMIN_PATTERNS = ['l_min','lmin','l/min','litre_min','litres_min']

# Proxy puissance → courant : si courant_moteur absent mais puissance_moteur_w disponible
# I_proxy = P_w / P_median * 18.0 (médiane courant physique pompe)
POWER_PROXY_PATTERNS = ['puissance_moteur_w','motor_power_w','puissance_w','power_w',
                        'puissance_moteur','motor_power','eps1','electrical_power']
POWER_PROXY_CURRENT_MEDIAN = 18.0  # A — valeur de référence médiane

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
        vals = df[c].dropna()
        if set(vals.unique()).issubset({0, 1}) and vals.sum() > 0:
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
        raw_col = col_map[feat]
        feat_df[feat] = pd.to_numeric(df[raw_col], errors='coerce')
        # Conversion L/min -> m³/h pour le débit (cohérence avec live monitor)
        if feat == 'debit' and any(p in raw_col.lower() for p in DEBIT_LMIN_PATTERNS):
            feat_df[feat] = feat_df[feat] * 0.06
            print(f'  debit: conversion L/min -> m³/h (×0.06)')
    else:
        feat_df[feat] = np.nan

# Conversion cible -> binaire 0/1
raw_target = df[target_col]
if raw_target.dtype == object or str(raw_target.dtype) == 'category':
    # Texte : 'panne'/'panne_proche' -> 1, 'bon_fonctionnement'/'normal'/0 -> 0
    PANNE_WORDS = ['panne','failure','fault','defaut','broken','anomal','leak','wear','error']
    y = raw_target.astype(str).str.lower().apply(
        lambda v: 1 if any(w in v for w in PANNE_WORDS) else 0
    ).values.astype(int)
else:
    numeric = pd.to_numeric(raw_target, errors='coerce').fillna(0)
    unique_vals = sorted(numeric.dropna().unique())
    if set(unique_vals).issubset({0, 1}):
        y = numeric.astype(int).values
    else:
        # Multi-classe : 0 = normal, tout le reste = anomalie
        y = (numeric > 0).astype(int).values
print(f'  Distribution cible    : normal={int((y==0).sum())}  panne={int((y==1).sum())}  ({y.mean()*100:.1f}% pannes)')

# Imputation initiale par médiane pour les colonnes non-NaN (avant dérivations)
_defaults = {'vibration':0.6,'temp_palier':45.0,'debit':0.4,
             'pression_entree':1.5,'pression_sortie':108.0,
             'courant_moteur':4.7,'temp_moteur':50.0,'heure_fonctionnement':1100.0}
for c in COLONNES:
    med = feat_df[c].median()
    feat_df[c] = feat_df[c].fillna(_defaults[c] if np.isnan(med) else med)

# ── courant_moteur : formule 3-phase si courant absent ───────────────────────
# I = P / (sqrt(3) * V * eta * cos_phi)  — moteur industriel standard 400V
_MOTOR_V, _MOTOR_ETA, _MOTOR_COSPHI = 400.0, 0.92, 0.85
_MOTOR_DENOM = 1.732 * _MOTOR_V * _MOTOR_ETA * _MOTOR_COSPHI  # 539.8

if feat_df['courant_moteur'].isna().all() or feat_df['courant_moteur'].nunique() <= 1:
    pwr_col = find_col(POWER_PROXY_PATTERNS)
    if pwr_col:
        pwr = pd.to_numeric(df[pwr_col], errors='coerce')
        if not pwr.isna().all():
            feat_df['courant_moteur'] = (pwr / _MOTOR_DENOM).values
            i_min = feat_df['courant_moteur'].min()
            i_max = feat_df['courant_moteur'].max()
            print(f'  courant_moteur: 3-phase formula I=P/(sqrt3*V*eta*cosphi) depuis {pwr_col} [{i_min:.3f}—{i_max:.3f} A]')

# ── temp_moteur : modele thermique (palier + echauffement electrique + refroidissement) ──
# T_motor = T_bearing + k_load*(P-P_nom)/P_nom + k_cool*(1 - cooler_pct/100)
# k_load=4°C per sigma, k_cool=18°C for fully degraded cooler
_COOL_COL  = next((c for c in df.columns if 'cooler_condition' in c.lower()), None)
_PWR_COL2  = find_col(POWER_PROXY_PATTERNS)
if _COOL_COL is not None and _PWR_COL2 is not None:
    pwr2 = pd.to_numeric(df[_PWR_COL2], errors='coerce').fillna(df[_PWR_COL2].median() if _PWR_COL2 else 0)
    cool = pd.to_numeric(df[_COOL_COL], errors='coerce').fillna(100.0)
    pwr_std = pwr2.std() if pwr2.std() > 0 else 1.0
    power_excess   = (pwr2 - pwr2.mean()) / pwr_std      # normalised motor load
    cooler_penalty = (1.0 - cool / 100.0).clip(0, 1)     # 0=perfect, 1=broken
    heat_rise = (power_excess * 4.0 + cooler_penalty * 18.0).clip(lower=0.0)
    feat_df['temp_moteur'] = feat_df['temp_palier'] + heat_rise.values
    tm_min = feat_df['temp_moteur'].min(); tm_max = feat_df['temp_moteur'].max()
    print(f'  temp_moteur: thermal model T_bearing + load_rise + cool_penalty [{tm_min:.1f}—{tm_max:.1f} C]')

# ── pression_entree : loi hydraulique depuis debit si toutes NaN ─────────────
# p_in = p_min + (Q/Q_max)^0.5 * (p_max - p_min)  — profil de courbe centrifuge
# p_min=0.8 bar (cavitation threshold), p_max=2.0 bar (nominal suction head)
if feat_df['pression_entree'].isna().all() or feat_df['pression_entree'].nunique() <= 1:
    q = feat_df['debit']
    q_max = q.max() if q.max() > 0 else 1.0
    feat_df['pression_entree'] = (0.8 + (q / q_max).clip(0, 1) ** 0.5 * 1.2).values
    pe_min = feat_df['pression_entree'].min(); pe_max = feat_df['pression_entree'].max()
    print(f'  pression_entree: hydraulic model p=0.8+1.2*(Q/Q_max)^0.5 [{pe_min:.3f}—{pe_max:.3f} bar]')

# Calcul des médianes APRÈS toutes les dérivations (valeurs réelles)
medians = {}
for c in COLONNES:
    medians[c] = round(float(feat_df[c].median()), 3)

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

# Utilise cv_fold_5 (pli 4 comme test) si disponible, sinon split aléatoire stratifié
cv_fold_col = None
for c in df.columns:
    if 'cv_fold' in c.lower():
        cv_fold_col = c
        break

if cv_fold_col is not None:
    folds = df[cv_fold_col].values[:len(X_sc)]
    test_mask = folds == folds.max()  # dernier pli = test
    X_tr, X_te = X_sc[~test_mask], X_sc[test_mask]
    y_tr, y_te = y[~test_mask], y[test_mask]
    print(f'  Split via {cv_fold_col} (pli {int(folds.max())} = test) — train: {len(X_tr)}, test: {len(X_te)}')
else:
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42, stratify=y)
    print(f'  Split aléatoire stratifié — train: {len(X_tr)}, test: {len(X_te)}')

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

# ── 8. RUL MODEL (NASA C-MAPSS FD001) ────────────────────────────────────────
# Cherche le fichier CMAPSS : argv[2], même dossier, ou ~/Downloads
_CMAPSS_NAME = 'nasa_cmapss_fd001_status_cv.csv'
_cmapss_candidates = [
    sys.argv[2] if len(sys.argv) > 2 else '',
    os.path.join(os.path.dirname(os.path.abspath(CSV)), _CMAPSS_NAME),
    os.path.join(os.path.expanduser('~'), 'Downloads', _CMAPSS_NAME),
]
cmapss_path = next((p for p in _cmapss_candidates if p and os.path.exists(p)), None)

if cmapss_path:
    print(f'\n[8/8] RUL model — NASA C-MAPSS FD001 : {cmapss_path}')
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    cm = pd.read_csv(cmapss_path)
    print(f'  {len(cm)} lignes | {cm["unit_number"].nunique()} engines | RUL 0–{cm["rul_cycles"].max()}')

    def _minmax_remap(series, tmin, tmax):
        mn, mx = float(series.min()), float(series.max())
        if mx == mn:
            return pd.Series([(tmin + tmax) / 2.0] * len(series), index=series.index)
        return (series - mn) / (mx - mn) * (tmax - tmin) + tmin

    # Map CMAPSS engine sensors → pump feature space (same column names, pump-compatible ranges)
    # Scaler trained on these ranges will match pump readings at inference time
    rul_feat = pd.DataFrame({
        'vibration':            _minmax_remap(cm['bleed_enthalpy_htbleed'], 0.30, 1.20),
        'temp_palier':          _minmax_remap(cm['temp_t24_r'],              35.0, 60.0),
        'debit':                _minmax_remap(cm['fan_speed_nf_rpm'],        0.10, 0.42),
        'pression_entree':      1.5,
        'pression_sortie':      _minmax_remap(cm['pressure_p30_psia'],       95.0, 130.0),
        'courant_moteur':       _minmax_remap(cm['core_speed_nc_rpm'],       15.0, 22.0),
        'temp_moteur':          _minmax_remap(cm['temp_t50_r'],              35.0, 80.0),
        'heure_fonctionnement': cm['time_in_cycles'].astype(float),
    })
    rul_target = cm['rul_cycles'].values.astype(float)

    # GroupKFold split by unit (fold max = test) — no data leakage across units
    _rul_cv = 'cv_fold_5' if 'cv_fold_5' in cm.columns else None
    if _rul_cv:
        _folds = cm[_rul_cv].values
        _test_m = _folds == _folds.max()
        print(f'  Split via {_rul_cv} — train: {(~_test_m).sum()}, test: {_test_m.sum()}')
    else:
        _test_m = np.random.RandomState(42).rand(len(rul_feat)) < 0.2

    Xr = rul_feat[COLONNES].values
    rul_scaler_new = StandardScaler()
    Xr_sc = rul_scaler_new.fit_transform(Xr)
    Xr_tr, Xr_te = Xr_sc[~_test_m], Xr_sc[_test_m]
    yr_tr, yr_te = rul_target[~_test_m], rul_target[_test_m]

    rul_mdl = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    rul_mdl.fit(Xr_tr, yr_tr)
    yr_pred = np.clip(rul_mdl.predict(Xr_te), 0, None)
    mae  = mean_absolute_error(yr_te, yr_pred)
    rmse = mean_squared_error(yr_te, yr_pred) ** 0.5
    print(f'  MAE = {mae:.1f} cycles  |  RMSE = {rmse:.1f} cycles')
    _rul_scale = 5000.0 / 361.0
    print(f'  Scale: rul_hours = cycles x {_rul_scale:.2f}  (PUMP_MTBF=5000h / CMAPSS_MAX=361 cycles)')

    with open('rul_model.pkl',  'wb') as f: pickle.dump(rul_mdl,        f)
    with open('rul_scaler.pkl', 'wb') as f: pickle.dump(rul_scaler_new, f)
    print('  rul_model.pkl   OK  (GradientBoostingRegressor)')
    print('  rul_scaler.pkl  OK')

    # Update meta
    meta['rul_model'] = 'GradientBoostingRegressor'
    meta['rul_mae_cycles'] = round(float(mae), 1)
    meta['rul_rmse_cycles'] = round(float(rmse), 1)
    meta['rul_source'] = os.path.basename(cmapss_path)
    meta['rul_scale_factor'] = round(5000.0 / 361.0, 4)
    with open('model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
else:
    print(f'\n[8/8] NASA C-MAPSS non trouvé — RUL model ignoré')
    print(f'  Fichier attendu : {_CMAPSS_NAME}  (~/Downloads ou argv[2])')

print('\n' + '=' * 65)
print('RETRAINING TERMINÉ')
print('=' * 65)
