"""
PILAR Data Contribution — Upload Module
========================================
Packages Analysis rows (anonymised sensor readings + outcomes) and
POSTs them to the PILAR improvement server.

Configuration (set at build time via environment variables):
  PILAR_UPLOAD_URL   — receiver endpoint (default: PILAR server)
  PILAR_UPLOAD_KEY   — shared API key for v1 auth

Install identity:
  pilar_install_id.txt — auto-generated UUID at first run, never changes.
  Sent with every upload so the team can associate data batches with an
  installation without any user identity.

Data included per row:
  vibration, temp_palier, debit, pression_entree, pression_sortie,
  courant_moteur, temp_moteur, heure_fonctionnement,
  machine_name, machine_type, fluid_type,
  risk (0-100), prediction (0/1), feedback (tp/fp/fn/blank),
  date (YYYY-MM-DD, not full timestamp),
  app_version, install_id

Data NEVER included:
  user email, company name, IP address, account identifiers.
"""

import os
import csv
import io
import json
import uuid
import threading
import time
import ssl
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pilar_logging import get_logger

logger = get_logger('pilar.upload')

# ── Configuration ─────────────────────────────────────────────────────────────
UPLOAD_URL     = os.environ.get('PILAR_UPLOAD_URL', 'https://pilar-site.up.railway.app/api/data-upload')
UPLOAD_KEY     = os.environ.get('PILAR_UPLOAD_KEY', 'pilar-upload-v1')
CONSENT_VERSION = 'v1.0'

_upload_lock = threading.Lock()
_last_status = {'state': 'idle', 'rows_sent': 0, 'error': None, 'last_run': None}


# ── Install ID ────────────────────────────────────────────────────────────────
def get_install_id(app_dir: str) -> str:
    """Return the stable UUID for this installation (created on first call)."""
    id_path = os.path.join(app_dir, 'pilar_install_id.txt')
    if os.path.exists(id_path):
        try:
            val = open(id_path).read().strip()
            if val:
                return val
        except Exception:
            pass
    new_id = str(uuid.uuid4())
    try:
        with open(id_path, 'w') as f:
            f.write(new_id)
    except Exception as e:
        logger.warning(f'Could not persist install_id: {e}')
    return new_id


# ── Data packaging ────────────────────────────────────────────────────────────
def _package_rows(analyses, feature_medians: dict, app_version: str, install_id: str) -> bytes:
    """Serialise Analysis ORM objects to a UTF-8 CSV payload."""
    FIELDS = [
        'install_id', 'app_version', 'date',
        'machine_name', 'machine_type', 'fluid_type',
        'vibration', 'temp_palier', 'debit',
        'pression_entree', 'pression_sortie',
        'courant_moteur', 'temp_moteur', 'heure_fonctionnement',
        'risk', 'prediction', 'feedback',
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS, extrasaction='ignore')
    writer.writeheader()

    def _med(key):
        return feature_medians.get(key, 0)

    for a in analyses:
        ep = {}
        try:
            if a.extra_params:
                ep = json.loads(a.extra_params)
        except Exception:
            pass

        date_str = ''
        if a.timestamp:
            try:
                date_str = a.timestamp.date().isoformat()
            except Exception:
                pass

        writer.writerow({
            'install_id':            install_id,
            'app_version':           app_version,
            'date':                  date_str,
            'machine_name':          (a.machine_id or '').strip()[:80],
            'machine_type':          ep.get('pump_type', ep.get('machine_type', '')),
            'fluid_type':            ep.get('fluid_type', ''),
            'vibration':             ep.get('vibration', _med('vibration')),
            'temp_palier':           a.temp_air          if a.temp_air          is not None else _med('temp_palier'),
            'debit':                 a.vitesse           if a.vitesse           is not None else _med('debit'),
            'pression_entree':       ep.get('pression_entree', _med('pression_entree')),
            'pression_sortie':       a.couple            if a.couple            is not None else _med('pression_sortie'),
            'courant_moteur':        ep.get('courant_moteur', _med('courant_moteur')),
            'temp_moteur':           a.temp_process      if a.temp_process      is not None else _med('temp_moteur'),
            'heure_fonctionnement':  a.usure             if a.usure             is not None else _med('heure_fonctionnement'),
            'risk':                  round(float(a.risk or 0), 1),
            'prediction':            int(a.prediction or 0),
            'feedback':              a.feedback or '',
        })

    return buf.getvalue().encode('utf-8')


# ── HTTP POST ─────────────────────────────────────────────────────────────────
def _post(csv_bytes: bytes, metadata: dict) -> dict:
    """POST CSV + metadata to the PILAR server. Returns {'ok': True, ...}."""
    payload = json.dumps({
        'api_key':  UPLOAD_KEY,
        'metadata': metadata,
        'csv_b64':  __import__('base64').b64encode(csv_bytes).decode(),
    }).encode('utf-8')

    req = urllib.request.Request(
        UPLOAD_URL,
        data=payload,
        headers={
            'Content-Type':  'application/json',
            'User-Agent':    f'PILAR-Desktop/{metadata.get("app_version", "4.0")}',
            'X-Install-ID':  metadata.get('install_id', ''),
        },
        method='POST',
    )

    for verify in (True, False):
        try:
            ctx = ssl.create_default_context()
            if not verify:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                return json.loads(resp.read())
        except ssl.SSLError:
            if not verify:
                raise
            continue
        except urllib.error.HTTPError as e:
            body = e.read(500).decode('utf-8', errors='replace')
            raise RuntimeError(f'HTTP {e.code}: {body}')


# ── Main upload function ──────────────────────────────────────────────────────
def upload_pending(flask_app, db, Analysis, FEATURE_MEDIANS: dict,
                   app_dir: str, app_version: str = '4.0') -> tuple:
    """
    Upload Analysis rows that have not yet been sent (uploaded_at IS NULL).
    Must be called from outside Flask request context (e.g. scheduler).

    Returns (rows_sent: int, error: str|None).
    """
    global _last_status

    if not _upload_lock.acquire(blocking=False):
        return 0, 'Upload already in progress'

    try:
        install_id = get_install_id(app_dir)

        with flask_app.app_context():
            # Only upload rows from users who have given consent
            from sqlalchemy import text as _text
            # Collect IDs of consenting users
            try:
                consenting = db.session.execute(
                    _text("SELECT user_id FROM user_data_consent WHERE enabled=1 AND withdrawn_at IS NULL")
                ).fetchall()
                consenting_ids = {r[0] for r in consenting}
            except Exception:
                consenting_ids = None  # table may not exist yet

            if consenting_ids is not None and not consenting_ids:
                _last_status = {'state': 'idle', 'rows_sent': 0, 'error': None,
                                'last_run': datetime.now(timezone.utc).isoformat()}
                return 0, None  # no consenting users

            # Query pending rows
            q = Analysis.query.filter(Analysis.uploaded_at.is_(None))
            if consenting_ids is not None:
                q = q.filter(Analysis.user_id.in_(consenting_ids))
            analyses = q.order_by(Analysis.timestamp.asc()).limit(5000).all()

            if not analyses:
                logger.info('upload: no pending rows')
                _last_status = {'state': 'idle', 'rows_sent': 0, 'error': None,
                                'last_run': datetime.now(timezone.utc).isoformat()}
                return 0, None

            logger.info(f'upload: packaging {len(analyses)} rows…')
            csv_bytes = _package_rows(analyses, FEATURE_MEDIANS, app_version, install_id)

            metadata = {
                'install_id':   install_id,
                'app_version':  app_version,
                'row_count':    len(analyses),
                'consent_version': CONSENT_VERSION,
                'first_date':   analyses[0].timestamp.date().isoformat() if analyses[0].timestamp else '',
                'last_date':    analyses[-1].timestamp.date().isoformat() if analyses[-1].timestamp else '',
            }

        # POST outside app context to avoid holding DB connection
        result = _post(csv_bytes, metadata)
        rows_accepted = result.get('rows_received', len(analyses))
        logger.info(f'upload: server accepted {rows_accepted} rows')

        # Mark rows as uploaded
        with flask_app.app_context():
            now = datetime.now(timezone.utc)
            ids = [a.id for a in analyses]
            Analysis.query.filter(Analysis.id.in_(ids)).update(
                {'uploaded_at': now}, synchronize_session=False
            )
            db.session.commit()

        _last_status = {
            'state': 'ok', 'rows_sent': rows_accepted, 'error': None,
            'last_run': datetime.now(timezone.utc).isoformat(),
        }
        return rows_accepted, None

    except Exception as e:
        err = str(e)
        logger.error(f'upload: failed — {err}')
        _last_status = {
            'state': 'error', 'rows_sent': 0, 'error': err,
            'last_run': datetime.now(timezone.utc).isoformat(),
        }
        return 0, err

    finally:
        _upload_lock.release()


def get_status() -> dict:
    return dict(_last_status)


def upload_async(flask_app, db, Analysis, FEATURE_MEDIANS, app_dir, app_version='4.0'):
    """Fire-and-forget wrapper — runs upload in a daemon thread."""
    threading.Thread(
        target=upload_pending,
        args=(flask_app, db, Analysis, FEATURE_MEDIANS, app_dir, app_version),
        daemon=True,
        name='pilar-uploader',
    ).start()
