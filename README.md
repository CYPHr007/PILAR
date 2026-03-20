---
title: Pilar
emoji: 🔧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Pilar — Predictive Maintenance for Industrial Pumps

AI-powered failure prediction for centrifugal pumps. Analyzes 8 sensor signals, classifies 5 failure zones, estimates remaining useful life (RUL), and alerts teams before breakdown.

---

## What it does

| Feature | Detail |
|---|---|
| **Failure prediction** | RandomForest — binary (normal / anomaly) |
| **Zone classification** | 5 zones: CAV · ROL · ETN · IMP · MOT |
| **Anomaly detection** | Isolation Forest (unsupervised) |
| **RUL forecast** | GradientBoosting on NASA C-MAPSS, ~13.85 h/cycle |
| **SHAP explanations** | Top-3 contributing features per prediction |
| **AI assistant** | Claude Haiku with pump-domain knowledge base |
| **Fleet management** | Multi-machine dashboard with quick-analyse |
| **Auto-retrain** | Fires when 150 confirmed analyses accumulate |
| **Alerts** | Email on anomaly detection |
| **Weekly PDF reports** | Auto-generated, sent every Monday 08:00 UTC |

---

## Sensors (8 features)

| Field | Unit | Median | Description |
|---|---|---|---|
| `vibration` | mm/s | 0.61 | Bearing housing vibration |
| `temp_palier` | °C | 44.8 | Bearing temperature |
| `debit` | m³/h | 0.395 | Flow rate |
| `pression_entree` | bar | 1.99 | Inlet pressure |
| `pression_sortie` | bar | 107.7 | Outlet pressure |
| `courant_moteur` | A | 4.58 | Motor current (3-phase RMS) |
| `temp_moteur` | °C | 58.0 | Motor temperature |
| `heure_fonctionnement` | h | 1103 | Total run hours |

All 8 are optional per request — missing fields are imputed from medians.

---

## Failure zones

| Code | Name | Key signals |
|---|---|---|
| CAV | Cavitation | Flow ↓ + vibration ↑ |
| ROL | Bearing Failure | Bearing temp ↑ + vibration ↑ |
| ETN | Seal Failure | Outlet pressure ↓ |
| IMP | Impeller Wear | Flow ↓ at normal pressure |
| MOT | Motor Fault | Current ↑ + motor temp ↑ |

---

## Run locally

```bash
pip install -r requirements.txt
python etape7.py
```

Open `http://localhost:5000`.

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | No | PostgreSQL URL (defaults to SQLite) |
| `SECRET_KEY` | Yes (prod) | Flask session secret |
| `SMTP_HOST` | No | Email alert server |
| `SMTP_USER` | No | Email username |
| `SMTP_PASS` | No | Email password |
| `ANTHROPIC_API_KEY` | No | Enables AI chat assistant |
| `RAILWAY_ENVIRONMENT` | Auto | Set by Railway — enables HTTPS cookie |

---

## Retrain the model

### From your own pump data
```bash
python retrain_real.py your_pump_data.csv
```
Accepts any CSV with pump sensor columns (auto-detected by name patterns).

### From NLN-EMP (real motor current data — improves MOT zone)
```bash
# 1. Download 20.8 GB archive from:
#    https://data.4tu.nl/datasets/2b61183e-c14f-4131-829b-cc4822c369d0/4

# 2. Install extraction dependency
pip install py7zr

# 3. Extract working sample (~1-5 GB)
python extract_nln_emp_sample.py "archive.7z" ./nln_sample

# 4. Retrain MOT zone
python retrain_nln_emp.py ./nln_sample

# 5. Push updated model
git add modeles_zones.pkl model_meta.json && git push
```

---

## Project structure

```
etape7.py              # Main Flask app (routes, ML, HTML templates)
config.py              # All tunable constants — edit here
retrain_real.py        # General-purpose retraining pipeline
retrain_nln_emp.py     # NLN-EMP motor-current MOT zone retraining
retrain_kaggle.py      # Kaggle dataset retraining pipeline
extract_nln_emp_sample.py  # NLN-EMP archive extraction helper
Dockerfile             # Production container
nixpacks.toml          # Railway deployment config
requirements.txt       # Python dependencies
```

> **Architecture note:** `etape7.py` is intentionally monolithic for Railway deployment simplicity.
> The planned split is: `config.py` (done) → `models.py` → `ml/` → `routes/` → `templates/`.

---

## Known technical debt

| Issue | Location | Fix |
|---|---|---|
| Analysis DB columns use CNC-era names (`temp_air`, `vitesse`, `couple`, `usure`, `temp_process`) | `Analysis` model | DB migration to rename columns |
| All HTML embedded as Python strings | `etape7.py` | Move to `templates/` directory |
| In-memory rate limiting reset on restart | `_login_attempts`, `_api_calls` | Replace with Redis |
| Auto-retrain blocks a gunicorn worker thread | `_auto_retrain()` | Move to Celery / RQ worker |
| SHAP explainer re-initialised per request | `_compute_shap()` | Cache on model load |

---

## Scalability limits (current architecture)

| Threshold | Bottleneck | Fix |
|---|---|---|
| ~500 req/s | Single gunicorn process, SQLite | Add workers, switch to PostgreSQL |
| ~1 000 users | In-memory session state | Redis session store |
| ~5 000 analyses | Auto-retrain runs inline | Background queue (Celery/RQ) |
| ~10 000 DAU | 7 000-line monolith slows deploys | Split into modules |
| Restart | Rate-limit memory lost | Redis rate limiter |

---

## Training data

| Model | Source | Notes |
|---|---|---|
| Main classifier | UCI Hydraulic Systems (CC BY 4.0) | 2205 cycles, centrifugal pump |
| RUL regressor | NASA C-MAPSS FD001 | Mapped to pump domain, ×13.85 h/cycle |
| MOT zone | Power→current proxy | Improves with NLN-EMP data |
| Isolation Forest | User analyses | Trains on confirmed normal readings |

---

## API

```bash
POST /api/v1/analyze
X-Api-Key: <your_key>
Content-Type: application/json

{
  "machine_id": "PUMP-01",
  "vibration": 2.5,
  "pression_sortie": 115,
  "courant_moteur": 4.8
}
```

Full docs at `/api/docs` (login required).
