"""
PILAR — Model Evaluation Script
Loads production models and evaluates on an independent test set.
"""
import pickle, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.metrics import (classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score)

# ── Load models ────────────────────────────────────────────────────────────────
with open('modele_pannes.pkl','rb') as f: model = pickle.load(f)
with open('scaler.pkl','rb') as f: scaler = pickle.load(f)
with open('modeles_zones.pkl','rb') as f: zones_models = pickle.load(f)
with open('model_meta.json') as f: meta = json.load(f)

FEATURES = meta['colonnes']
ZONES    = meta['zones']
print('Model:', meta['model_name'])
print('Source:', meta.get('source','?'))
print('Trained:', meta.get('trained_at','?'), '|', meta['n_train'], 'training samples')
print()

# ── Generate independent test data (seed 99, different from training seed 42) ──
np.random.seed(99)
N_NORMAL, N_FAIL = 3000, 900

def normal_op():
    return {
        'vibration':            np.clip(np.random.lognormal(0.8, 0.35), 0.3, 7.0),
        'temp_palier':          np.clip(np.random.normal(65, 10),        30,  95),
        'debit':                np.clip(np.random.normal(45, 12),        5,  120),
        'pression_entree':      np.clip(np.random.normal(1.5, 0.4),     0.1, 5.0),
        'pression_sortie':      np.clip(np.random.normal(4.5, 1.2),     1.0, 18.0),
        'courant_moteur':       np.clip(np.random.normal(18, 3.5),      5,   40),
        'temp_moteur':          np.clip(np.random.normal(75, 12),        30, 110),
        'heure_fonctionnement': np.clip(np.random.exponential(4000),     0, 25000),
    }

def cav():
    return {
        'vibration':            np.clip(np.random.normal(8.5, 2.5),   4.0, 25.0),
        'temp_palier':          np.clip(np.random.normal(72, 12),       35, 110),
        'debit':                np.clip(np.random.normal(22, 10),        2,  60),
        'pression_entree':      np.clip(np.random.normal(0.4, 0.2),  0.05, 1.2),
        'pression_sortie':      np.clip(np.random.normal(3.0, 1.2),   0.5,  10),
        'courant_moteur':       np.clip(np.random.normal(20, 4),        8,  45),
        'temp_moteur':          np.clip(np.random.normal(82, 14),       40, 130),
        'heure_fonctionnement': np.clip(np.random.exponential(5000),    0, 30000),
    }

def rol():
    hf = np.clip(np.random.normal(12000, 4000), 3000, 35000)
    return {
        'vibration':            np.clip(np.random.normal(11, 3.5),    5.0, 30),
        'temp_palier':          np.clip(np.random.normal(95 + max((hf-8000)/800, 0), 12), 55, 150),
        'debit':                np.clip(np.random.normal(40, 12),       5, 100),
        'pression_entree':      np.clip(np.random.normal(1.4, 0.4),   0.1,   5),
        'pression_sortie':      np.clip(np.random.normal(4.0, 1.3),   0.5,  16),
        'courant_moteur':       np.clip(np.random.normal(22, 4.5),      8,  50),
        'temp_moteur':          np.clip(np.random.normal(88, 14),       40, 140),
        'heure_fonctionnement': hf,
    }

def etn():
    return {
        'vibration':            np.clip(np.random.normal(4.5, 1.5),   1.0,  12),
        'temp_palier':          np.clip(np.random.normal(68, 10),       35, 100),
        'debit':                np.clip(np.random.normal(28, 10),        3,  75),
        'pression_entree':      np.clip(np.random.normal(0.8, 0.3),   0.1, 3.0),
        'pression_sortie':      np.clip(np.random.normal(2.8, 1.0),   0.3,  10),
        'courant_moteur':       np.clip(np.random.normal(16, 3.5),      5,  38),
        'temp_moteur':          np.clip(np.random.normal(78, 13),       40, 125),
        'heure_fonctionnement': np.clip(np.random.exponential(6000),    0, 30000),
    }

def imp():
    hf = np.clip(np.random.normal(15000, 5000), 5000, 40000)
    return {
        'vibration':            np.clip(np.random.normal(6.5, 2.0),   2.0,  18),
        'temp_palier':          np.clip(np.random.normal(78, 12),       40, 120),
        'debit':                np.clip(np.random.normal(30, 10),        5,  80),
        'pression_entree':      np.clip(np.random.normal(1.3, 0.4),   0.1,   5),
        'pression_sortie':      np.clip(np.random.normal(3.2, 1.2),   0.5,  12),
        'courant_moteur':       np.clip(np.random.normal(25, 5),       10,  55),
        'temp_moteur':          np.clip(np.random.normal(92, 15),       45, 145),
        'heure_fonctionnement': hf,
    }

def mot():
    return {
        'vibration':            np.clip(np.random.normal(5.0, 2.0),  1.5,  15),
        'temp_palier':          np.clip(np.random.normal(70, 12),      35, 110),
        'debit':                np.clip(np.random.normal(38, 12),       5, 110),
        'pression_entree':      np.clip(np.random.normal(1.4, 0.4),  0.1,   5),
        'pression_sortie':      np.clip(np.random.normal(4.0, 1.3),  0.5,  15),
        'courant_moteur':       np.clip(np.random.normal(32, 8),      15,  80),
        'temp_moteur':          np.clip(np.random.normal(115, 18),     65, 200),
        'heure_fonctionnement': np.clip(np.random.exponential(7000),   0, 35000),
    }

generators = {'CAV': cav, 'ROL': rol, 'ETN': etn, 'IMP': imp, 'MOT': mot}

df_n = pd.DataFrame([normal_op() for _ in range(N_NORMAL)])
df_n['failure'] = 0
for z in ZONES: df_n[z] = 0

rows = []
for z, g in generators.items():
    d = pd.DataFrame([g() for _ in range(N_FAIL // 5)])
    d['failure'] = 1
    for zz in ZONES: d[zz] = (1 if zz == z else 0)
    rows.append(d)

df = pd.concat([df_n] + rows, ignore_index=True).sample(frac=1, random_state=99).reset_index(drop=True)
X = df[FEATURES].values
y = df['failure'].values
Xs = scaler.transform(X)

print('=== TEST SET ===')
print(f'Total: {len(df)}  |  Normal: {(y==0).sum()}  |  Failures: {(y==1).sum()} ({y.mean()*100:.1f}%)')
print()

# ── Main classifier ────────────────────────────────────────────────────────────
y_pred = model.predict(Xs)
y_prob = model.predict_proba(Xs)[:, 1]

rec  = recall_score(y, y_pred)
prec = precision_score(y, y_pred)
f1   = f1_score(y, y_pred)
auc  = roc_auc_score(y, y_prob)
ap   = average_precision_score(y, y_prob)
cm   = confusion_matrix(y, y_pred)
fp_rate = cm[0,1] / (cm[0,0] + cm[0,1]) * 100
fn_rate = cm[1,0] / (cm[1,0] + cm[1,1]) * 100

print('=== MAIN CLASSIFIER ===')
print(f'  Recall:           {rec*100:.1f}%   (% of failures caught — key safety metric)')
print(f'  Precision:        {prec*100:.1f}%   (% of alerts that are real failures)')
print(f'  F1-score:         {f1*100:.1f}%')
print(f'  ROC-AUC:          {auc:.4f}')
print(f'  Avg Precision:    {ap:.4f}')
print(f'  False Positive rate: {fp_rate:.1f}%  (normal ops flagged as failure)')
print(f'  False Negative rate: {fn_rate:.1f}%  (failures missed)')
print()
print('  Confusion matrix:')
print(f'                   Pred Normal  Pred Failure')
print(f'  True Normal         {cm[0,0]:5d}         {cm[0,1]:5d}')
print(f'  True Failure        {cm[1,0]:5d}         {cm[1,1]:5d}')
print()
print(classification_report(y, y_pred, target_names=['Normal','Failure']))

# ── Zone classifiers ────────────────────────────────────────────────────────────
print('=== ZONE CLASSIFIERS ===')
print(f'  {"Zone":<6}  {"Recall":>8}  {"Precision":>10}  {"F1":>7}  {"Pos samples":>12}')
print('  ' + '-' * 52)
for zone, zm in zones_models.items():
    y_z = df[zone].values
    n_pos = int(y_z.sum())
    if n_pos < 5:
        print(f'  {zone:<6}  too few samples ({n_pos})')
        continue
    y_z_pred = zm.predict(Xs)
    rz  = recall_score(y_z, y_z_pred, zero_division=0)
    prz = precision_score(y_z, y_z_pred, zero_division=0)
    f1z = f1_score(y_z, y_z_pred, zero_division=0)
    print(f'  {zone:<6}  {rz*100:>7.1f}%  {prz*100:>9.1f}%  {f1z*100:>6.1f}%  {n_pos:>12d}')
print()

# ── Feature importance ──────────────────────────────────────────────────────────
if hasattr(model, 'feature_importances_'):
    print('=== FEATURE IMPORTANCE ===')
    imp_pairs = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    for feat, score in imp_pairs:
        bar = '#' * int(score * 60)
        print(f'  {feat:<28} {score:.4f}  {bar}')
    print()

# ── Threshold sensitivity ───────────────────────────────────────────────────────
print('=== THRESHOLD SENSITIVITY (default=0.5) ===')
print(f'  {"Threshold":>10}  {"Recall":>8}  {"Precision":>10}  {"Alert rate":>12}')
for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    yp = (y_prob >= thr).astype(int)
    r  = recall_score(y, yp, zero_division=0)
    p  = precision_score(y, yp, zero_division=0)
    ar = yp.mean() * 100
    marker = ' <-- default' if thr == 0.5 else ''
    print(f'  {thr:>10.1f}  {r*100:>7.1f}%  {p*100:>9.1f}%  {ar:>10.1f}%{marker}')
print()

# ── RUL model ──────────────────────────────────────────────────────────────────
print('=== RUL MODEL ===')
print(f'  Type:    {meta.get("rul_model","?")}')
print(f'  Source:  {meta.get("rul_source","?")}')
mae_c  = meta.get('rul_mae_cycles', 0)
rmse_c = meta.get('rul_rmse_cycles', 0)
scale  = meta.get('rul_scale_factor', 1)
print(f'  MAE:     {mae_c} cycles  =  {mae_c*scale:.0f} hours')
print(f'  RMSE:    {rmse_c} cycles  =  {rmse_c*scale:.0f} hours')
print(f'  Scale:   1 cycle = {scale:.1f} hours')
print()

# ── MOT / ROL / IMP sub-model stats from meta ──────────────────────────────────
print('=== ZONE SUB-MODEL METADATA (from model_meta.json) ===')
for prefix in ['mot', 'rol', 'imp']:
    mtype = meta.get(f'{prefix}_model', '?')
    cvr   = meta.get(f'{prefix}_cv_recall', '?')
    cvstd = meta.get(f'{prefix}_cv_recall_std', '?')
    npos  = meta.get(f'{prefix}_n_pos', '?')
    print(f'  {prefix.upper()}  {mtype}  |  CV Recall: {cvr}% ± {cvstd}%  |  n_pos: {npos}')
print()

print('=== SUMMARY ===')
verdict = 'GOOD' if rec >= 0.95 and prec >= 0.80 else ('ACCEPTABLE' if rec >= 0.90 else 'NEEDS IMPROVEMENT')
print(f'  Overall verdict: {verdict}')
print(f'  Recall {rec*100:.1f}% — model catches {"nearly all" if rec>=0.99 else "most" if rec>=0.95 else "most"} failures')
print(f'  Precision {prec*100:.1f}% — {100-prec*100:.1f}% false alarm rate')
if fn_rate > 5:
    print(f'  WARNING: {fn_rate:.1f}% missed failures — consider lowering threshold')
if fp_rate > 20:
    print(f'  WARNING: {fp_rate:.1f}% false positives — may cause alert fatigue')
