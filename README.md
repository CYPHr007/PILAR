# PILAR — Predictive Maintenance for Any Industrial Machine

AI-powered failure prediction for any industrial machine. PILAR analyzes telemetry from pumps, compressors, fans, conveyors, robotic systems, and more — classifies likely fault patterns, estimates remaining useful life (RUL), and alerts teams before breakdown.

---

## What it does

| Feature | Detail |
|---|---|
| **Failure prediction** | RandomForest - binary (normal / anomaly) |
| **Pattern classification** | Default packaged model includes 5 fault zones |
| **Anomaly detection** | Isolation Forest (unsupervised) |
| **RUL forecast** | GradientBoosting on NASA C-MAPSS, about 13.85 h/cycle |
| **SHAP explanations** | Top-3 contributing features per prediction |
| **AI assistant** | Claude Haiku with maintenance-domain knowledge base |
| **Fleet management** | Multi-machine dashboard with quick analysis |
| **Auto-retrain** | Fires when 150 confirmed analyses accumulate |
| **Alerts** | Email on anomaly detection |
| **Weekly PDF reports** | Auto-generated, sent every Monday 08:00 UTC |

---

## Default telemetry schema

The bundled model currently ships with 8 default telemetry fields:

| Field | Unit | Median | Description |
|---|---|---|---|
| `vibration` | mm/s | 0.61 | Bearing housing vibration |
| `temp_palier` | C | 44.8 | Bearing temperature |
| `debit` | m3/h | 0.395 | Flow rate |
| `pression_entree` | bar | 1.99 | Inlet pressure |
| `pression_sortie` | bar | 107.7 | Outlet pressure |
| `courant_moteur` | A | 4.58 | Motor current (3-phase RMS) |
| `temp_moteur` | C | 58.0 | Motor temperature |
| `heure_fonctionnement` | h | 1103 | Total run hours |

All inputs are optional per request. Missing fields are imputed from medians.

These defaults come from the packaged example model and datasets. The product itself is not pump-specific: PILAR is intended for broader machine monitoring and can be retrained on client-specific equipment data.

---

## Default fault zones

| Code | Name | Default interpretation |
|---|---|---|
| CAV | Cavitation / hydraulic anomaly | Flow down + vibration up |
| ROL | Bearing failure | Bearing temp up + vibration up |
| ETN | Sealing / pressure anomaly | Outlet pressure down |
| IMP | Rotating-part wear | Flow down at normal pressure |
| MOT | Motor fault | Current up + motor temp up |

These labels belong to the current packaged model, not to the product boundary.

---

## Run locally

```bash
pip install -r requirements.txt
python app.py
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
| `RAILWAY_ENVIRONMENT` | Auto | Set by Railway - enables HTTPS cookie |
| `PILAR_ENABLE_SCHEDULER` | No | `1` by default; set to `0` to disable scheduled jobs in this process |

---

## Scheduler leadership

Weekly PDF reports and weekly auto-retrain use a single-leader scheduler lock, so multi-worker Gunicorn deployments do not run the same cron job twice.

- PostgreSQL deployments use a database advisory lock
- SQLite and desktop deployments use a local file lock
- `/api/health` exposes scheduler state

To run scheduled jobs in a dedicated process instead of the web workers:

```bash
PILAR_ENABLE_SCHEDULER=0 gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
python pilar_scheduler.py
```

---

## Retrain the model

### From your own machine data

```bash
python retrain_real.py your_machine_data.csv
```

Accepts CSV telemetry columns auto-detected by name patterns.

### From NLN-EMP (real motor current data - improves MOT zone)

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

```text
app.py                 # Main Flask app (routes, ML, HTML templates)
config.py              # All tunable constants - edit here
retrain_real.py        # General-purpose retraining pipeline
retrain_nln_emp.py     # NLN-EMP motor-current MOT zone retraining
retrain_kaggle.py      # Kaggle dataset retraining pipeline
extract_nln_emp_sample.py  # NLN-EMP archive extraction helper
Dockerfile             # Production container
nixpacks.toml          # Railway deployment config
requirements.txt       # Python dependencies
```

> **Architecture note:** `app.py` is intentionally monolithic for Railway deployment simplicity. The planned split is: `config.py` (done) -> `models.py` -> `ml/` -> `routes/` -> `templates/`.

---

## Known technical debt

| Issue | Location | Fix |
|---|---|---|
| Analysis DB columns use CNC-era names (`temp_air`, `vitesse`, `couple`, `usure`, `temp_process`) | `Analysis` model | DB migration to rename columns |
| All HTML embedded as Python strings | `app.py` | Move to `templates/` directory |
| In-memory rate limiting reset on restart | `_login_attempts`, `_api_calls` | Replace with Redis |
| Auto-retrain blocks a gunicorn worker thread | `_auto_retrain()` | Move to Celery / RQ worker |
| SHAP explainer re-initialized per request | `_compute_shap()` | Cache on model load |

---

## Scalability limits (current architecture)

| Threshold | Bottleneck | Fix |
|---|---|---|
| ~500 req/s | Single gunicorn process, SQLite | Add workers, switch to PostgreSQL |
| ~1,000 users | In-memory session state | Redis session store |
| ~5,000 analyses | Auto-retrain runs inline | Background queue (Celery/RQ) |
| ~10,000 DAU | 7,000-line monolith slows deploys | Split into modules |
| Restart | Rate-limit memory lost | Redis rate limiter |

---

## Training data

| Model | Source | Notes |
|---|---|---|
| Main classifier | UCI Hydraulic Systems (CC BY 4.0) | Default packaged seed dataset |
| RUL regressor | NASA C-MAPSS FD001 | Generic RUL baseline adapted to the app |
| MOT zone | Power-to-current proxy | Improves with NLN-EMP data |
| Isolation Forest | User analyses | Trains on confirmed normal readings |

The shipped defaults come from these datasets, but the product can be retrained for other machine types.

---

## API

```bash
POST /api/v1/analyze
X-Api-Key: <your_key>
Content-Type: application/json

{
  "machine_id": "MACHINE-01",
  "vibration": 2.5,
  "pression_sortie": 115,
  "courant_moteur": 4.8
}
```

Full docs at `/api/docs` (login required).
