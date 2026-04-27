# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

PILAR — AI-powered predictive maintenance for any industrial machine (pumps, compressors, fans, turbines, agitators, conveyors, robotic systems). Flask monolith (`app.py`, ~10K lines) with embedded HTML templates, ML prediction (GradientBoosting + XGBoost, Platt-calibrated), a 3-agent AutoGen pipeline running on local Ollama, and a Windows desktop build (PyInstaller + Inno Setup).

## Commands

```bash
# Run the app (web) — use py -3.14 on dev machine (3.12-32bit lacks ML packages)
py -3.14 app.py                  # starts Flask on http://localhost:5000

# Run the app (desktop — Windows only)
py -3.14 launcher.py             # Flask + system tray + pywebview

# Syntax check (no comprehensive test suite — always verify after edits)
py -3.14 -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"

# Run existing tests
py -3.14 -m pytest tests/

# Build Windows desktop installer
py -3.14 -m PyInstaller --clean --noconfirm pilar.spec   # → dist/pilar/
build.bat                                                 # full build (PyInstaller + Inno Setup)

# Retrain ML model (generates .pkl files — restart app.py to load)
py -3.14 train_universal.py              # full retrain from 23 machine profiles
py -3.14 train_universal.py --extra-csv real_data.csv  # merge real data

# Create/update Ollama agent models
ollama create pilar-diag -f modelfiles/pilar_diag.modelfile
ollama create pilar-maintenance -f modelfiles/pilar_maintenance.modelfile
ollama create pilar-alert -f modelfiles/pilar_alert.modelfile

# Release flow
gh release create vX.Y.Z --title "PILAR vX.Y.Z" --notes "..."
gh release upload vX.Y.Z dist/PILAR_Setup_X.Y.Z.exe
```

## Architecture

**Monolith pattern**: `app.py` contains all Flask routes, SQLAlchemy models (14 tables), and some embedded HTML. A `templates/` directory also exists with standalone Jinja2 templates (`machine_space.html`, `twin.html`, `account.html`, `adapter.html`, etc.). Both patterns coexist — older routes use embedded strings, newer pages use `render_template()`.

**Extracted modules** (imported by `app.py`):
- `pilar_ml.py` — ML pipeline: model loading (`init_models`), feature building (`build_model_input`), prediction (`predict_risk`), SHAP, RUL, anomaly scoring. Stateless functions operating on module-level model references.
- `pilar_validators.py` — Input validation (`validate_sensor_input`), CSV validation, CSRF token generation/verification.
- `pilar_email.py` — Email sending with SMTP/relay fallback (`send_email`, `send_alert_email`, `send_escalation_email`, `send_verify_email`, `send_reset_email`).
- `pilar_monitor.py` — Background CSV file watcher (`start_monitor`, `stop_monitor`). Fuzzy-matches CSV column headers to sensor fields.
- `pilar_logging.py` — Centralized logging config. All modules use `from pilar_logging import get_logger`.
- `pilar_calibrator.py` — `PlattModel` wrapper class. Must be importable before loading `failure_model.pkl` (pickle needs to resolve the class).

**Config centralization**: All tunable constants live in `config.py` (thresholds, feature lists, sensor bounds, domain KB, IsolationForest params, MOT fallback thresholds, batch limits). Never hardcode magic numbers in `app.py`.

**ML pipeline**: 6 pickle files loaded at startup via `pilar_ml.init_models()`:
- `failure_model.pkl` — binary failure classifier (GradientBoosting, Platt-calibrated via `PlattModel`)
- `zone_models.pkl` — per-zone classifiers (CAV, ROL, ETN, IMP, MOT)
- `isolation_forest.pkl` — unsupervised anomaly detector (auto-retrained from normal samples)
- `rul_model.pkl` + `rul_scaler.pkl` — remaining useful life (NASA C-MAPSS → pump hours via `RUL_SCALE_FACTOR`)
- `scaler.pkl` — StandardScaler (12 features: 8 raw + 4 derived: dp, eta_proxy, thermal_load, wear_index)

**Training pipeline** — two scripts, both output the same pkl files:
- `train_universal.py` — synthetic training from 23 machine profiles. 3-tier failure severity, sensor jitter, SMOTE, RF+XGBoost+GradientBoosting ensemble, Platt calibration. Accepts `--extra-csv` to blend real data.
- `train_kaggle.py` — trains on Kaggle pump sensor real data (`data/sensor.csv`). Use `--keep-zones` to preserve existing `zone_models.pkl` (Kaggle has no zone labels). Currently the active production model source.

**Agent system** (`agents/`):
- `orchestrator.py` — runs 3 AutoGen 0.4 agents: Diagnostic → Maintenance → Alert
- Each calls Ollama locally. Custom models defined in `modelfiles/`
- Falls back to rule-based functions when Ollama is unavailable
- `sla_tracker.py` tracks response time SLAs
- `pilar_bootstrap.py` handles first-run Ollama detection and model pulling

**Desktop build**: `launcher.py` starts Flask in-process, opens pywebview/Edge window, shows pystray tray icon. Single-instance lock on port 19847. `pilar.spec` bundles everything via PyInstaller. `pilar_installer.iss` creates Windows installer (per-user, no admin).

## Key Conventions

- **Python version**: Dev machine uses `py -3.14`. The 32-bit Python 3.12 lacks numpy/sklearn. Always use `py -3.14` for running ML code.
- **Sensor value ranges**: `config.py` `SENSOR_BOUNDS` and `FEATURE_MEDIANS` now reflect Kaggle real data scales (e.g. `pression_sortie` 0–880, `debit` 0–3, `courant_moteur` 0–6) — not traditional industrial units. Update both when switching training sources.
- **Language**: UI is bilingual French/English. Sensor feature names are French (`debit`, `pression_sortie`, `courant_moteur`) — do not rename them as they match the ML model's training columns (`COLONNES` in config.py). These names come from the default hydraulic training profile; other machine types may use different sensor mappings via the adapter layer.
- **Version**: bump in 3 places: `config.py` (`APP_VERSION`), `build.bat`, `pilar_installer.iss` (`AppVersion`)
- **8 core sensor features** = `COLONNES` = `CORE_FEATURES` in config.py. The universal model adds 4 derived features (dp, eta_proxy, thermal_load, wear_index) for 12 total. Detection is automatic via `scaler.n_features_in_`.
- **5 failure zones**: CAV, ROL, ETN, IMP, MOT — defined in `config.py` as `FAILURE_ZONES`
- Logging: use `from pilar_logging import get_logger; logger = get_logger("pilar.xxx")` — never bare `print()` for operational messages.
- Rate limiting is in-memory (`defaultdict`-based) — resets on restart.
- The main app file was historically called `etape7.py`. Some references in README/Procfile may use the old name.

## Database

SQLite by default (`pilar.db` next to the exe in frozen mode), PostgreSQL when `DATABASE_URL` is set. 14 SQLAlchemy models defined inline in `app.py`. No Alembic — migrations are manual `ALTER TABLE` in `app.py`'s startup block (lines ~261-336). Legacy column names in `Analysis` table: `temp_air`=bearing temp, `temp_process`=motor temp, `vitesse`=flow, `couple`=outlet pressure, `usure`=run hours.

## Environment

See `.env.example` for all variables. Key ones: `SECRET_KEY`, `DATABASE_URL`, `ANTHROPIC_API_KEY`, `GMAIL_ADDRESS`/`GMAIL_APP_PASSWORD`, `SUPER_USER_EMAIL`.

Dev machine: Python 3.14, Windows 11, no GPU (CPU-only Ollama ~80-180s/agent), ~4.2 GB RAM.
