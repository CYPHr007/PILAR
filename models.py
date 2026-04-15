"""SQLAlchemy models for PILAR — extracted from app.py."""
from datetime import datetime, timezone
from extensions import db
from config import DEFAULT_THRESHOLD


class Team(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(200), default='My Team')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class TeamMember(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    team_id   = db.Column(db.Integer, nullable=False)
    user_id   = db.Column(db.Integer, nullable=False)
    role      = db.Column(db.String(20), default='member')  # 'leader' or 'member'
    is_kicked = db.Column(db.Boolean, default=False)
    joined_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class User(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    email          = db.Column(db.String(200), unique=True, nullable=False)
    password_hash  = db.Column(db.String(256), nullable=False)
    email_verified = db.Column(db.Boolean, default=True)
    verify_token   = db.Column(db.String(64))
    api_key        = db.Column(db.String(64), unique=True)
    plan           = db.Column(db.String(20), default='free')
    plan_expires_at= db.Column(db.DateTime, nullable=True)
    plan_note      = db.Column(db.String(300), nullable=True)
    is_admin       = db.Column(db.Boolean, default=False)
    is_banned      = db.Column(db.Boolean, default=False)
    team_id        = db.Column(db.Integer, nullable=True)
    onboarded           = db.Column(db.Boolean, default=False)
    machine_quota       = db.Column(db.Integer, default=3)
    reset_token         = db.Column(db.String(64), nullable=True)
    reset_token_expires = db.Column(db.DateTime, nullable=True)
    created_at          = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class BannedEmail(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    email      = db.Column(db.String(200), unique=True, nullable=False)
    reason     = db.Column(db.String(300), nullable=True)
    banned_at  = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class Settings(db.Model):
    id      = db.Column(db.Integer, primary_key=True)
    key     = db.Column(db.String(120))
    value   = db.Column(db.String(500))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)


class Analysis(db.Model):
    # Legacy CNC-era column names mapped to pump-domain fields:
    #   temp_air=bearing temp, temp_process=motor temp, vitesse=flow,
    #   couple=outlet pressure, usure=run hours.
    id           = db.Column(db.Integer, primary_key=True)
    timestamp    = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    machine_type = db.Column(db.String(10))
    temp_air     = db.Column(db.Float)
    temp_process = db.Column(db.Float)
    vitesse      = db.Column(db.Float)
    couple       = db.Column(db.Float)
    usure        = db.Column(db.Float)
    risk         = db.Column(db.Float)
    prediction   = db.Column(db.Integer)
    zones        = db.Column(db.String(500))
    mail_sent    = db.Column(db.Boolean, default=False)
    user_id      = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    extra_params = db.Column(db.Text)
    confidence   = db.Column(db.Integer, default=100)
    machine_id   = db.Column(db.String(100))
    feedback     = db.Column(db.String(10))  # 'tp'=confirmed failure, 'fp'=false positive


class SavedFile(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, nullable=True)
    team_id    = db.Column(db.Integer, nullable=True)
    machine_id = db.Column(db.Integer, nullable=True)
    filename   = db.Column(db.String(200), nullable=False)
    content    = db.Column(db.Text, nullable=False)
    row_count  = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class TeamMessage(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    team_id    = db.Column(db.Integer, nullable=False)
    user_id    = db.Column(db.Integer, nullable=False)
    user_email = db.Column(db.String(200))
    content    = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class DiscoveredParam(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    name         = db.Column(db.String(100))
    label        = db.Column(db.String(200))
    unit_guess   = db.Column(db.String(20))
    impact       = db.Column(db.Float, default=0.0)
    n_samples    = db.Column(db.Integer, default=0)
    samples_json = db.Column(db.Text)
    risks_json   = db.Column(db.Text)
    created_at   = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at   = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    user_id      = db.Column(db.Integer, nullable=True)


class Machine(db.Model):
    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name             = db.Column(db.String(200), nullable=False)
    description      = db.Column(db.String(500))
    machine_type     = db.Column(db.String(10), default='M')
    pump_type        = db.Column(db.String(50), default='centrifuge')
    fluid_type       = db.Column(db.String(50), default='eau')
    roue_material    = db.Column(db.String(50), default='inox_316')
    threshold        = db.Column(db.Float, default=DEFAULT_THRESHOLD)
    location         = db.Column(db.String(200))
    install_date     = db.Column(db.Date, nullable=True)
    serial_number    = db.Column(db.String(100))
    nominal_flow     = db.Column(db.Float)
    nominal_pressure = db.Column(db.Float)
    power_kw         = db.Column(db.Float)
    nominal_current  = db.Column(db.Float)
    nominal_vibration= db.Column(db.Float)
    alert_email      = db.Column(db.String(200))
    escalation_email = db.Column(db.String(200))
    is_active        = db.Column(db.Boolean, default=True)
    created_at       = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    asset_type       = db.Column(db.String(50))
    brand            = db.Column(db.String(100))
    model_name       = db.Column(db.String(100))
    age_years        = db.Column(db.Float, default=0)
    environment      = db.Column(db.String(50))
    criticality      = db.Column(db.String(20), default='medium')
    last_maintenance = db.Column(db.DateTime)


class MachineNote(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    machine_id = db.Column(db.Integer, db.ForeignKey('machine.id'), nullable=False)
    user_id    = db.Column(db.Integer, nullable=False)
    user_email = db.Column(db.String(200))
    content    = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class AlertLog(db.Model):
    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, nullable=True)
    machine_id_str   = db.Column(db.String(100))
    analysis_id      = db.Column(db.Integer, nullable=True)
    email_to         = db.Column(db.String(200))
    probabilite      = db.Column(db.Float)
    sent_at          = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    acked_at         = db.Column(db.DateTime, nullable=True)
    ack_token        = db.Column(db.String(64), unique=True)
    escalated_at     = db.Column(db.DateTime, nullable=True)
    escalation_email = db.Column(db.String(200))


class MachineRequest(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, nullable=True)
    name         = db.Column(db.String(200))
    manufacturer = db.Column(db.String(200))
    rpm_range    = db.Column(db.String(100))
    torque_range = db.Column(db.String(100))
    description  = db.Column(db.Text)
    submitted_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    status       = db.Column(db.String(20), default='pending')


class MachineBaseline(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    machine_id   = db.Column(db.Integer, db.ForeignKey('machine.id'), nullable=False, index=True)
    feature      = db.Column(db.String(50), nullable=False)
    mean         = db.Column(db.Float)
    std          = db.Column(db.Float)
    min_normal   = db.Column(db.Float)
    max_normal   = db.Column(db.Float)
    sample_count = db.Column(db.Integer, default=0)
    computed_at  = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class MachineModel(db.Model):
    id               = db.Column(db.Integer, primary_key=True)
    machine_id       = db.Column(db.Integer, db.ForeignKey('machine.id'), nullable=False, unique=True, index=True)
    model_blob       = db.Column(db.LargeBinary)
    scaler_blob      = db.Column(db.LargeBinary)
    f1_score         = db.Column(db.Float)
    training_samples = db.Column(db.Integer, default=0)
    version          = db.Column(db.Integer, default=1)
    last_trained     = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class MaintenanceEvent(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    machine_id     = db.Column(db.Integer, db.ForeignKey('machine.id'), nullable=False, index=True)
    event_type     = db.Column(db.String(50))
    description    = db.Column(db.Text)
    parts_replaced = db.Column(db.String(500))
    cost           = db.Column(db.Float)
    timestamp      = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class SyncQueue(db.Model):
    """Queues local analyses/notes for uploading to the sync server."""
    __tablename__ = "sync_queue"
    id         = db.Column(db.Integer, primary_key=True)
    data_type  = db.Column(db.String(20), default="analysis")
    data_json  = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    synced_at  = db.Column(db.DateTime, nullable=True)

    @property
    def is_synced(self):
        return self.synced_at is not None


class LocalChatMessage(db.Model):
    """Local storage for chat messages (online: pulled from sync server; offline: created locally)."""
    __tablename__ = "local_chat_message"
    id          = db.Column(db.Integer, primary_key=True)
    remote_id   = db.Column(db.Integer, nullable=True)
    client_id   = db.Column(db.String(64))
    sender_name = db.Column(db.String(200))
    room        = db.Column(db.String(100), default="general")
    content     = db.Column(db.Text)
    image_data  = db.Column(db.Text)
    image_mime  = db.Column(db.String(50))
    created_at  = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    is_local    = db.Column(db.Boolean, default=False)
    synced_at   = db.Column(db.DateTime, nullable=True)
