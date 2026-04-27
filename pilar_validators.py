# -*- coding: utf-8 -*-
"""
PILAR -- Input validation and CSRF protection.
"""
import secrets
import hmac
from typing import Optional, Tuple, Dict, List, Any
from config import CORE_FEATURES, SENSOR_BOUNDS, OPTIONAL_FIELDS


def validate_sensor_input(
    data: dict,
) -> Tuple[Optional[dict], Optional[dict], Optional[str], Optional[str]]:
    """
    Validate and normalize sensor input from any source (form, API, CSV).

    Returns (params, extra, machine_id, error_message).
    On success error_message is None; on failure params is None.
    """
    if not data:
        return None, None, None, "Empty input data"

    machine_id = str(data.get("machine_id", "") or "")[:100] or None

    # Coerce core features to float; None means "missing, will be imputed"
    for field in CORE_FEATURES:
        if field in data and data[field] is not None:
            try:
                data[field] = float(data[field])
            except (TypeError, ValueError):
                data[field] = None
        else:
            data[field] = None

    if all(data.get(f) is None for f in CORE_FEATURES):
        return None, None, None, "At least one sensor field is required"

    # Physical bounds check
    for fld, (lo, hi) in SENSOR_BOUNDS.items():
        v = data.get(fld)
        if v is not None and not (lo <= v <= hi):
            return None, None, None, f"Value out of range: {fld}={v} (expected [{lo},{hi}])"

    # Collect optional fields
    extra: Dict[str, float] = {}
    for field in OPTIONAL_FIELDS:
        if field in data and data[field] is not None:
            try:
                extra[field] = round(float(data[field]), 3)
            except (TypeError, ValueError):
                pass

    return data, extra, machine_id, None


def validate_csv_size(content: bytes, max_mb: float = 10.0) -> Optional[str]:
    """Return error string if CSV content exceeds max size, else None."""
    size_mb = len(content) / (1024 * 1024)
    if size_mb > max_mb:
        return f"File too large: {size_mb:.1f} MB (max {max_mb} MB)"
    return None


def validate_csv_columns(
    headers: List[str], required_patterns: Dict[str, List[str]]
) -> Dict[str, int]:
    """
    Map CSV headers to expected sensor fields using fuzzy matching.
    Returns {field_name: column_index} for matched columns.
    """
    import re

    def _norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9]", "_", s)
        s = re.sub(r"_+", "_", s)
        return s.strip("_")

    norm = [_norm(h) for h in headers]
    used: set = set()
    col_map: Dict[str, int] = {}

    for field, pats in required_patterns.items():
        best_idx, best_score = -1, 0
        for j, h in enumerate(norm):
            if j in used:
                continue
            score = 0
            for pi, p in enumerate(pats):
                if h == p:
                    score = 100 - pi
                    break
                elif p in h and score < 70 - pi:
                    score = 70 - pi
            if score > best_score:
                best_score, best_idx = score, j
        if best_idx >= 0:
            col_map[field] = best_idx
            used.add(best_idx)

    return col_map


# ── CSRF Protection ──────────────────────────────────────────────────────────

def generate_csrf_token(session: dict) -> str:
    """Generate and store a CSRF token in the session."""
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_hex(32)
    return session["_csrf_token"]


def validate_csrf_token(session: dict, token: Optional[str]) -> bool:
    """Validate a CSRF token against the session."""
    expected = session.get("_csrf_token")
    if not expected or not token:
        return False
    return hmac.compare_digest(expected, token)
