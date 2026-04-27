# -*- coding: utf-8 -*-
"""
PILAR -- Email sending (SMTP + relay fallback).
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from typing import Optional, List, Dict
from pilar_logging import get_logger

logger = get_logger("pilar.email")

GMAIL = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_PWD = os.environ.get("GMAIL_APP_PASSWORD", "")
_EMAIL_RELAY_URL = os.environ.get("EMAIL_RELAY_URL", "")
_EMAIL_RELAY_KEY = os.environ.get("EMAIL_RELAY_KEY", "")


def _send_via_relay(to_addr: str, subject: str, html: str) -> bool:
    """Send email through relay service. Returns True on success."""
    if not _EMAIL_RELAY_URL:
        return False
    try:
        import urllib.request as _ur
        import json as _j
        import urllib.error as _ue

        payload = _j.dumps({"to": to_addr, "subject": subject, "html": html}).encode()
        req = _ur.Request(
            _EMAIL_RELAY_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-Relay-Key": _EMAIL_RELAY_KEY,
            },
        )
        with _ur.urlopen(req, timeout=10) as resp:
            result = _j.loads(resp.read())
            return bool(result.get("ok"))
    except Exception as e:
        logger.error("Email relay error: %s", e)
        return False


def send_email(to_addr: str, subject: str, html: str) -> bool:
    """Send email: try SMTP first, fall back to relay. Returns True on success."""
    if GMAIL and GMAIL_PWD:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"PILAR <{GMAIL}>"
            msg["To"] = to_addr
            msg.attach(MIMEText(html, "html", "utf-8"))
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as srv:
                srv.login(GMAIL, GMAIL_PWD)
                srv.sendmail(GMAIL, [to_addr], msg.as_string())
            logger.info("Email sent to %s via SMTP", to_addr)
            return True
        except Exception as e:
            logger.warning("SMTP failed (%s) — trying relay", e)

    ok = _send_via_relay(to_addr, subject, html)
    if ok:
        logger.info("Email sent to %s via relay", to_addr)
    else:
        logger.error("Email delivery FAILED for %s (both SMTP and relay)", to_addr)
    return ok


def send_verify_email(email: str, token: str, base_url: Optional[str] = None) -> bool:
    base = (
        base_url or os.environ.get("APP_URL", "http://127.0.0.1:5000")
    ).rstrip("/")
    link = f"{base}/verify-email/{token}"
    html = f"""<div style="font-family:sans-serif;background:#07090f;color:#e2e8f0;padding:40px;border-radius:8px;max-width:520px">
<h2 style="color:#14b8a6;letter-spacing:3px;margin-bottom:16px">PILAR</h2>
<p style="margin-bottom:20px">Confirmez votre adresse email pour activer votre compte PILAR.</p>
<a href="{link}" style="display:inline-block;padding:12px 28px;background:#0d9488;color:#fff;border-radius:6px;text-decoration:none;font-weight:700;font-size:15px">V\u00e9rifier mon email</a>
<p style="margin-top:24px;color:#64748b;font-size:12px">Lien valide 24h. Si vous n'avez pas cr\u00e9\u00e9 de compte, ignorez cet email.</p>
<p style="color:#334155;font-size:11px;margin-top:8px">Ou copiez ce lien : <a href="{link}" style="color:#0d9488">{link}</a></p>
<p style="color:#1e2433;font-size:10px;margin-top:24px;border-top:1px solid #1e2433;padding-top:12px">PILAR \u2014 Predictive Maintenance Platform</p>
</div>"""
    ok = send_email(email, "PILAR \u2014 V\u00e9rifiez votre email", html)
    if ok:
        logger.info("Verification email sent to %s", email)
    else:
        logger.error("Verification email FAILED for %s (token: %s)", email, token)
    return ok


def send_reset_email(email: str, token: str, base_url: Optional[str] = None) -> bool:
    base = (
        base_url or os.environ.get("APP_URL", "http://127.0.0.1:5000")
    ).rstrip("/")
    link = f"{base}/reset-password/{token}"
    html = f"""<div style="font-family:sans-serif;background:#07090f;color:#e2e8f0;padding:40px;border-radius:8px;max-width:520px">
<h2 style="color:#14b8a6;letter-spacing:3px;margin-bottom:16px">PILAR</h2>
<p style="margin-bottom:20px">Vous avez demand\u00e9 la r\u00e9initialisation de votre mot de passe.</p>
<a href="{link}" style="display:inline-block;padding:12px 28px;background:#0d9488;color:#fff;border-radius:6px;text-decoration:none;font-weight:700;font-size:15px">R\u00e9initialiser mon mot de passe</a>
<p style="margin-top:24px;color:#64748b;font-size:12px">Lien valide 1h. Si vous n'\u00eates pas \u00e0 l'origine de cette demande, ignorez cet email.</p>
<p style="color:#334155;font-size:11px;margin-top:8px">Ou copiez ce lien : <a href="{link}" style="color:#0d9488">{link}</a></p>
<p style="color:#1e2433;font-size:10px;margin-top:24px;border-top:1px solid #1e2433;padding-top:12px">PILAR \u2014 Predictive Maintenance Platform</p>
</div>"""
    ok = send_email(email, "PILAR \u2014 R\u00e9initialisation de mot de passe", html)
    if ok:
        logger.info("Reset email sent to %s", email)
    else:
        logger.error("Reset email FAILED for %s (token: %s)", email, token)
    return ok


def send_alert_email(
    email_to: str,
    probabilite: float,
    zones_risque: List[Dict],
    data: dict,
    ack_token: Optional[str] = None,
) -> bool:
    """Send failure alert email. Returns True on success."""
    severity = "CRITICAL" if probabilite >= 75 else "HIGH"
    sc = "#dc2626"
    zones_rows = "".join(
        f'<tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#94a3b8;font-size:12px;">{z["nom"]}</td>'
        f'<td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#dc2626;font-weight:700;">{z["proba"]}%</td></tr>'
        for z in zones_risque
    ) or '<tr><td colspan="2" style="padding:8px 12px;color:#64748b;">No specific zone identified</td></tr>'

    _base_url = os.environ.get("APP_URL", "http://127.0.0.1:5000")
    ack_row = ""
    if ack_token:
        ack_row = (
            f'<tr><td style="padding:0 28px 24px;"><a href="{_base_url}/alert/ack/{ack_token}" '
            'style="display:inline-block;padding:10px 22px;background:#0d9488;color:#fff;'
            'font-size:11px;font-weight:700;letter-spacing:2px;text-decoration:none;border-radius:4px;">'
            "ACKNOWLEDGE ALERT</a></td></tr>"
        )

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html><html><body style="margin:0;background:#07090f;font-family:Segoe UI,Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#07090f;padding:40px 0;"><tr><td align="center">
<table width="520" cellpadding="0" cellspacing="0" style="background:#0e1118;border:1px solid #1e2433;border-radius:8px;">
<tr><td style="padding:24px 28px;border-bottom:1px solid #1e2433;">
<table width="100%" cellpadding="0" cellspacing="0"><tr>
<td><div style="font-size:11px;font-weight:700;letter-spacing:4px;color:#14b8a6;text-transform:uppercase;">PILAR</div></td>
<td align="right"><span style="padding:4px 10px;background:rgba(220,38,38,0.12);border:1px solid #dc2626;border-radius:3px;color:#dc2626;font-size:10px;font-weight:700;letter-spacing:2px;">FAILURE ALERT</span></td>
</tr></table></td></tr>
<tr><td style="padding:28px;">
<div style="font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:6px;">Failure Probability</div>
<div style="font-size:52px;font-weight:800;color:{sc};line-height:1;">{probabilite}<span style="font-size:22px;color:#64748b;">%</span></div>
<div style="margin-top:8px;"><span style="padding:3px 10px;background:rgba(220,38,38,0.1);border:1px solid {sc};border-radius:3px;font-size:10px;font-weight:700;color:{sc};">SEVERITY: {severity}</span></div></td></tr>
<tr><td style="padding:0 28px 24px;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#07090f;border:1px solid #1e2433;border-radius:6px;">
<tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#64748b;font-size:11px;">Vibration</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{data.get("vibration")} mm/s</td></tr>
<tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#64748b;font-size:11px;">Bearing temperature</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{data.get("temp_palier")} C</td></tr>
<tr><td style="padding:8px 12px;border-bottom:1px solid #1e2433;color:#64748b;font-size:11px;">Flow rate</td><td style="padding:8px 12px;border-bottom:1px solid #1e2433;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{data.get("debit")} m3/h</td></tr>
<tr><td style="padding:8px 12px;color:#64748b;font-size:11px;">Motor temperature</td><td style="padding:8px 12px;text-align:right;color:#e2e8f0;font-weight:600;font-size:11px;">{data.get("temp_moteur")} C</td></tr>
</table></td></tr>
<tr><td style="padding:0 28px 24px;">
<div style="font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:10px;">Failure Zones</div>
<table width="100%" cellpadding="0" cellspacing="0" style="background:#07090f;border:1px solid #1e2433;border-radius:6px;">{zones_rows}</table></td></tr>
{ack_row}
<tr><td style="padding:16px 28px;border-top:1px solid #1e2433;background:#0a0d16;">
<div style="font-size:10px;color:#64748b;">Pilar - {now_str} UTC</div></td></tr>
</table></td></tr></table></body></html>"""

    ok = send_email(email_to, f"Pilar Alert \u2014 Risk {probabilite}% | {severity}", html)
    if ok:
        logger.info("Alert email sent to %s (risk=%s%%)", email_to, probabilite)
    else:
        logger.error("Alert email FAILED for %s (risk=%s%%)", email_to, probabilite)
    return ok


def send_escalation_email(
    email_to: str,
    probabilite: float,
    zones_risque: List[Dict],
    machine_id_str: str,
) -> bool:
    """Send escalation email when primary alert is unacknowledged."""
    zones_rows = "".join(
        f'<tr><td style="padding:8px 12px;color:#94a3b8;font-size:12px;">{z["nom"]}</td>'
        f'<td style="text-align:right;padding:8px 12px;color:#dc2626;font-weight:700;">{z["proba"]}%</td></tr>'
        for z in zones_risque
    ) or '<tr><td colspan="2" style="padding:8px 12px;color:#64748b;">No specific zone</td></tr>'

    machine_label = f" ({machine_id_str})" if machine_id_str else ""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html><html><body style="margin:0;background:#07090f;font-family:Segoe UI,Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="padding:40px 0;"><tr><td align="center">
<table width="520" cellpadding="0" cellspacing="0" style="background:#0e1118;border:1px solid #7c2d12;border-radius:8px;">
<tr><td style="padding:24px 28px;border-bottom:1px solid #7c2d12;background:#1c0a04;">
<div style="font-size:11px;font-weight:700;letter-spacing:4px;color:#f97316;text-transform:uppercase;">PILAR \u2014 ESCALATION ALERT</div>
<div style="font-size:12px;color:#94a3b8;margin-top:6px;">Primary contact did not acknowledge this alert within 30 minutes.</div></td></tr>
<tr><td style="padding:28px;">
<div style="font-size:9px;letter-spacing:2px;color:#64748b;text-transform:uppercase;margin-bottom:6px;">Failure Probability{machine_label}</div>
<div style="font-size:52px;font-weight:800;color:#f97316;line-height:1;">{probabilite}<span style="font-size:22px;color:#64748b;">%</span></div></td></tr>
<tr><td style="padding:0 28px 28px;"><table width="100%">{zones_rows}</table></td></tr>
<tr><td style="padding:16px 28px;border-top:1px solid #1e2433;background:#0a0d16;">
<div style="font-size:10px;color:#64748b;">Pilar Escalation - {now_str} UTC</div></td></tr>
</table></td></tr></table></body></html>"""

    ok = send_email(
        email_to, f"Pilar ESCALATION \u2014 Risk {probabilite}% unacknowledged", html
    )
    if ok:
        logger.info("Escalation email sent to %s", email_to)
    else:
        logger.error("Escalation email FAILED for %s", email_to)
    return ok
