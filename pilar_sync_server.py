"""
PILAR Sync Server
=================
Central relay for PILAR desktop clients:
  - Auth via X-Client-ID / X-Client-Secret headers
  - POST /api/sync          — clients push analyses + notes
  - GET  /api/sync/pull     — clients pull chat messages
  - GET  /api/sync/status   — health / ping
  - GET  /api/chat/messages — get chat messages (with ?room= and ?since= filters)
  - POST /api/chat/messages — post a chat message

Deploy on Railway (or any host):
  Environment variables:
    SECRET_KEY         — Flask session key
    DATABASE_URL       — PostgreSQL URL (optional, falls back to SQLite)
    ALLOWED_CLIENTS    — JSON: {"client_id": "client_secret", ...}
                         e.g. '{"office_pc":"s3cr3t1","factory_pc":"s3cr3t2"}'
    PORT               — port to listen on (default 5001)
"""

import os
import json
import uuid
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, request, jsonify, g
from flask_sqlalchemy import SQLAlchemy

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

db_url = (
    os.environ.get("DATABASE_URL")
    or os.environ.get("DATABASE_PUBLIC_URL")
    or "sqlite:///pilar_sync.db"
)
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True, "pool_recycle": 280}
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(32).hex())

db = SQLAlchemy(app)

# ── Allowed clients ────────────────────────────────────────────────────────────
# Loaded once at startup from env var ALLOWED_CLIENTS (JSON dict)
# Example: '{"office_pc": "secret1", "factory_pc": "secret2"}'
_RAW = os.environ.get("ALLOWED_CLIENTS", "{}")
try:
    ALLOWED_CLIENTS: dict[str, str] = json.loads(_RAW)
except Exception:
    ALLOWED_CLIENTS = {}
    print("[SyncServer] WARNING: ALLOWED_CLIENTS is invalid JSON — all requests will be rejected")

print(f"[SyncServer] Loaded {len(ALLOWED_CLIENTS)} client(s): {list(ALLOWED_CLIENTS.keys())}")


# ── Models ─────────────────────────────────────────────────────────────────────
class SyncedAnalysis(db.Model):
    __tablename__ = "synced_analysis"
    id          = db.Column(db.Integer, primary_key=True)
    client_id   = db.Column(db.String(64), nullable=False, index=True)
    remote_id   = db.Column(db.Integer, nullable=True)
    machine_id  = db.Column(db.String(64), nullable=True)
    machine_type= db.Column(db.String(64), nullable=True)
    risk        = db.Column(db.Integer,  nullable=True)
    prediction  = db.Column(db.String(32), nullable=True)
    zones       = db.Column(db.String(128), nullable=True)
    confidence  = db.Column(db.Integer, nullable=True)
    feedback    = db.Column(db.String(32), nullable=True)
    extra_json  = db.Column(db.Text, nullable=True)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)
    received_at = db.Column(db.DateTime, default=datetime.utcnow)


class ChatMessage(db.Model):
    __tablename__ = "chat_message"
    id          = db.Column(db.Integer, primary_key=True)
    public_id   = db.Column(db.String(36), default=lambda: str(uuid.uuid4()), unique=True, index=True)
    client_id   = db.Column(db.String(64), nullable=False, index=True)
    sender_name = db.Column(db.String(128), nullable=True)
    room        = db.Column(db.String(64), default="general", index=True)
    content     = db.Column(db.Text, nullable=True)
    image_data  = db.Column(db.Text, nullable=True)   # base64
    image_mime  = db.Column(db.String(32), nullable=True)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow, index=True)


with app.app_context():
    db.create_all()
    print("[SyncServer] Database ready")


# ── Auth middleware ────────────────────────────────────────────────────────────
def require_client(f):
    """Decorator: validate X-Client-ID and X-Client-Secret headers."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        client_id     = request.headers.get("X-Client-ID", "").strip()
        client_secret = request.headers.get("X-Client-Secret", "").strip()

        if not client_id or not client_secret:
            return jsonify({"error": "Missing credentials"}), 401

        expected = ALLOWED_CLIENTS.get(client_id)
        if not expected or expected != client_secret:
            return jsonify({"error": "Invalid credentials"}), 403

        g.client_id = client_id
        return f(*args, **kwargs)
    return wrapper


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/api/sync/status", methods=["GET"])
@require_client
def sync_status():
    """Ping — used by clients to check connectivity."""
    return jsonify({
        "ok":        True,
        "server":    "PILAR Sync Server",
        "client_id": g.client_id,
        "time":      datetime.utcnow().isoformat(),
    })


@app.route("/api/sync", methods=["POST"])
@require_client
def sync_push():
    """Receive analyses and notes from a client."""
    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    analyses = payload.get("analyses", [])
    notes    = payload.get("notes", [])   # stored as analyses with type=note for now
    saved    = 0

    for a in analyses:
        try:
            ts = None
            try:
                ts = datetime.fromisoformat(a.get("timestamp", ""))
            except Exception:
                ts = datetime.utcnow()

            row = SyncedAnalysis(
                client_id   = g.client_id,
                remote_id   = a.get("remote_id"),
                machine_id  = str(a.get("machine_id", ""))[:64],
                machine_type= str(a.get("machine_type", ""))[:64],
                risk        = int(a.get("risk", 0)),
                prediction  = str(a.get("prediction", ""))[:32],
                zones       = str(a.get("zones", ""))[:128],
                confidence  = int(a.get("confidence", 100)),
                feedback    = str(a.get("feedback", ""))[:32],
                extra_json  = json.dumps(a.get("extra_params", {})),
                timestamp   = ts,
            )
            db.session.add(row)
            saved += 1
        except Exception as e:
            print(f"[SyncServer] Analysis parse error: {e}")
            continue

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

    print(f"[SyncServer] [{g.client_id}] Pushed {saved} analyses, {len(notes)} notes")
    return jsonify({"ok": True, "saved": saved})


@app.route("/api/sync/pull", methods=["GET"])
@require_client
def sync_pull():
    """Return chat messages since a given timestamp."""
    since_str = request.args.get("since")
    since     = None
    if since_str:
        try:
            since = datetime.fromisoformat(since_str)
        except Exception:
            pass

    q = ChatMessage.query
    if since:
        q = q.filter(ChatMessage.created_at > since)
    # Return messages from all OTHER clients (not this one) — and this client too for full history
    messages = q.order_by(ChatMessage.created_at.asc()).limit(200).all()

    return jsonify({
        "messages": [
            {
                "id":          m.public_id,
                "client_id":   m.client_id,
                "sender_name": m.sender_name or "",
                "room":        m.room or "general",
                "content":     m.content or "",
                "image_data":  m.image_data or "",
                "image_mime":  m.image_mime or "",
                "created_at":  m.created_at.isoformat(),
            }
            for m in messages
        ]
    })


@app.route("/api/chat/messages", methods=["GET"])
@require_client
def chat_get():
    """Get chat messages with optional ?room= and ?since= filters."""
    room      = request.args.get("room", "general")
    since_str = request.args.get("since")
    since     = None
    if since_str:
        try:
            since = datetime.fromisoformat(since_str)
        except Exception:
            pass

    q = ChatMessage.query.filter_by(room=room)
    if since:
        q = q.filter(ChatMessage.created_at > since)
    messages = q.order_by(ChatMessage.created_at.asc()).limit(100).all()

    return jsonify({
        "messages": [
            {
                "id":          m.public_id,
                "client_id":   m.client_id,
                "sender_name": m.sender_name or "",
                "room":        m.room,
                "content":     m.content or "",
                "image_data":  m.image_data or "",
                "image_mime":  m.image_mime or "",
                "created_at":  m.created_at.isoformat(),
            }
            for m in messages
        ]
    })


@app.route("/api/chat/messages", methods=["POST"])
@require_client
def chat_post():
    """Post a chat message from a client."""
    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    content    = str(payload.get("content", ""))[:4000]
    room       = str(payload.get("room", "general"))[:64]
    sender     = str(payload.get("sender_name", g.client_id))[:128]
    image_data = payload.get("image_data") or None
    image_mime = payload.get("image_mime") or None

    if not content and not image_data:
        return jsonify({"error": "Empty message"}), 400

    msg = ChatMessage(
        client_id   = g.client_id,
        sender_name = sender,
        room        = room,
        content     = content,
        image_data  = image_data,
        image_mime  = image_mime,
    )
    db.session.add(msg)
    db.session.commit()
    print(f"[SyncServer] [{g.client_id}] Chat: {sender}: {content[:60]}")

    return jsonify({"ok": True, "id": msg.public_id})


@app.route("/api/health", methods=["GET"])
def health():
    """Public health check (no auth needed)."""
    return jsonify({"status": "ok", "server": "PILAR Sync Server"})


@app.route("/", methods=["GET"])
def index():
    n_analyses = SyncedAnalysis.query.count()
    n_messages = ChatMessage.query.count()
    n_clients  = len(ALLOWED_CLIENTS)
    return jsonify({
        "name":       "PILAR Sync Server",
        "status":     "running",
        "clients":    n_clients,
        "analyses":   n_analyses,
        "messages":   n_messages,
        "time":       datetime.utcnow().isoformat(),
    })


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"[SyncServer] Starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
