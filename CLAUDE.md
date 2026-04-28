# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

PILAR — AI-powered predictive maintenance for any industrial machine (pumps, compressors, fans, turbines, conveyors, robotic systems). Two products share this codebase:

- **PILAR Standard** — Flask web/desktop app with dashboard, ML predictions, alerts, fleet management
- **PILAR Embedded** — Standalone edge inference engine for deployment inside robots and industrial machines

## Commands

```bash
# Run the app
py -3.14 app.py                   # Flask on http://localhost:5000
py -3.14 launcher.py              # Desktop: Flask + system tray + pywebview

# Syntax check — always run after editing app.py or config.py
py -3.14 -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"

# Tests
py -3.14 -m pytest tests/

# PILAR Embedded — edge inference CLI
echo '{"vibration":2.5,"courant_moteur":4.8}' | py -3.14 pilar_embedded.py
py -3.14 pilar_embedded.py --input reading.json                        # one-shot from file
py -3.14 pilar_embedded.py --input sensor.json --watch --interval 5    # poll file
py -3.14 pilar_embedded.py --adapter adapters/examples/modbus_tcp.json # Modbus source
py -3.14 pilar_embedded.py --machine-config oem/examples/robot_arm.json --daemon --port 7000

# OEM certification — train a model on machine-specific data
py -3.14 -m oem.certify \
  --data  machine_data.csv \
  --config oem/examples/robot_arm.json \
  --out   ./certified/robot_arm/

# ML retraining
py -3.14 train_kaggle.py --keep-zones   # real data (active production source)
py -3.14 train_universal.py             # synthetic 23-profile model
py -3.14 train_universal.py --extra-csv real_data.csv

# Windows desktop installer
build.bat                               # PyInstaller + Inno Setup → dist/PILAR_Setup_*.exe
```

## Architecture — Standard App

**Monolith pattern**: `app.py` (~10K lines) contains all Flask routes, SQLAlchemy models (14 tables), and embedded HTML. `templates/` holds newer Jinja2 pages. Both patterns coexist — older routes use inline strings, newer ones use `render_template()`.

**Extracted modules** (all imported by `app.py`):
- `pilar_ml.py` — entire ML pipeline: `init_models()`, `build_model_input()`, `predict_risk()`, SHAP, RUL, IsolationForest. Stateless except for module-level model globals set by `init_models`.
- `pilar_calibrator.py` — `PlattModel` wrapper. **Must be imported before unpickling `failure_model.pkl`** — pickle needs to resolve the class name.
- `config.py` — all tunable constants: thresholds, `COLONNES`, `FEATURE_MEDIANS`, `SENSOR_BOUNDS`, `FAILURE_ZONES`, domain KB. Never hardcode these in `app.py`.
- `pilar_validators.py` — sensor input validation, CSV column validation, CSRF.
- `pilar_email.py` — SMTP with relay fallback for alerts, escalation, verification emails.
- `pilar_monitor.py` — background CSV file watcher; fuzzy-matches column headers to sensor names.
- `pilar_logging.py` — centralized logging. Always `from pilar_logging import get_logger`. Cross-platform log dir (respects `PILAR_LOG_DIR` env var).

**ML pipeline** — 6 pkl files loaded at startup via `pilar_ml.init_models(app_dir)`:

| File | Model | Notes |
|---|---|---|
| `failure_model.pkl` | GradientBoosting + PlattModel | Binary failure classifier |
| `zone_models.pkl` | Per-zone classifiers × 5 | CAV, ROL, ETN, IMP, MOT |
| `isolation_forest.pkl` | IsolationForest | Auto-retrained from normal samples |
| `rul_model.pkl` + `rul_scaler.pkl` | GBT regressor | NASA C-MAPSS → hours via `RUL_SCALE_FACTOR` |
| `scaler.pkl` | StandardScaler | 8 or 12 features; `scaler.n_features_in_` determines which |

The model accepts 8 raw features (`COLONNES`) or 12 (adds dp, eta_proxy, thermal_load, wear_index). Auto-detected from `scaler.n_features_in_`.

**Agent system** (`agents/`): 3 agents (Diagnostic → Maintenance → Alert) backed by Qwen3 4B via `llm_engine.py`. Every agent has a full rule-based fallback — the LLM path is optional and not required for production use.

**Database**: SQLite default (`pilar.db`), PostgreSQL when `DATABASE_URL` is set. 14 models inline in `app.py`. No Alembic — manual `ALTER TABLE` in the startup block (~lines 261–336). Legacy `Analysis` column aliases: `temp_air`=bearing temp, `temp_process`=motor temp, `vitesse`=flow, `couple`=outlet pressure, `usure`=run hours.

## Architecture — Embedded Stack

PILAR Embedded (`pilar_embedded.py`) runs the ML pipeline without Flask, SQLAlchemy, or any server dependency. Designed for edge compute (Raspberry Pi, Jetson, industrial PCs).

```
pilar_embedded.py       CLI entry point — 4 modes: stdin, --watch, --adapter, --daemon
adapters/               Pluggable sensor readers
  base.py               BaseAdapter + load_adapter(config.json) factory
  file_adapter.py       JSON / JSONL / CSV file watcher
  http_adapter.py       REST poller (dot-notation field_map)
  modbus_adapter.py     Modbus/TCP and Modbus/RTU (requires pymodbus)
  gpio_adapter.py       MCP3008 ADC via SPI (Raspberry Pi, requires spidev)
  examples/             Ready-to-copy JSON configs for each adapter
oem/                    OEM machine configuration + certification
  machine_config.py     MachineConfig — nominals, custom zone labels, thresholds
  certify.py            Training pipeline for machine-specific models
  examples/             robot_arm.json, centrifugal_pump.json, conveyor.json
```

**Output**: one JSON line per reading to stdout:
```json
{"timestamp":"...","machine_id":"...","risk_score":78.4,"status":"critical",
 "zone":"ROL","zone_label":"Bearing Failure","rul_hours":38,"anomaly_score":62,
 "confidence":87,"top_sensors":[...],"missing_sensors":[...],"all_zones":[...]}
```

**Machine config** (`oem/MachineConfig`) flows into `predict_risk()` as `machine_context`:
- Overrides global `FEATURE_MEDIANS` with machine-specific nominals for imputation
- Disables irrelevant failure zones (e.g. CAV/IMP disabled for robots)
- Applies custom zone labels ("Joint Bearing Failure" instead of "Bearing Failure")
- Sets per-machine risk and zone alert thresholds

**Embedded dependencies** (much lighter than the full app):
```
requirements_embedded.txt — numpy, pandas, scikit-learn, shap (optional)
# Adapters add: pymodbus (Modbus), spidev (GPIO/Raspberry Pi)
```

## Key Conventions

- **Python version**: `py -3.14` on dev machine. 32-bit Python 3.12 has no numpy/sklearn.
- **Sensor names are French** and match ML training columns — do not rename. The 8 names in `COLONNES` are the contract between the app, the ML models, and any adapter/OEM config.
- **Sensor value scales** in `SENSOR_BOUNDS` and `FEATURE_MEDIANS` reflect the Kaggle training data (e.g. `debit` 0–3, `pression_sortie` 0–880). Update both when switching training sources.
- **PILAR is machine-agnostic** — the pump profile is the *default*, not the product boundary. OEM configs define what normal looks like for any machine type.
- **Version bump**: 3 places — `config.py` (`APP_VERSION`), `build.bat`, `pilar_installer.iss`.
- **Rate limiting** is in-memory — resets on restart.

## Environment

Key env vars: `SECRET_KEY`, `DATABASE_URL`, `ANTHROPIC_API_KEY`, `GMAIL_ADDRESS`/`GMAIL_APP_PASSWORD`, `SUPER_USER_EMAIL`, `PILAR_LOG_DIR` (embedded log path).

Dev machine: Python 3.14, Windows 11, no GPU, ~4.2 GB RAM.
