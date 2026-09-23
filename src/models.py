"""
Database models (SQLAlchemy ORM, SQLite by default).
See database/schema.sql for the equivalent SQL and documentation/DATA_DICTIONARY.md.
"""
import json
from datetime import datetime

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from .extensions import db


def now():
    return datetime.now()


class JSONText(db.TypeDecorator):
    """Stores Python dict/list as JSON text."""
    impl = db.Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return None if value is None else json.dumps(value)

    def process_result_value(self, value, dialect):
        return None if value is None else json.loads(value)


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    user_code = db.Column(db.String(20), unique=True, index=True)        # USR-00001
    username = db.Column(db.String(50), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    full_name = db.Column(db.String(120))
    phone = db.Column(db.String(30))
    organization = db.Column(db.String(120))
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user")
    is_active_flag = db.Column(db.Boolean, default=True)
    failed_logins = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=now)
    last_login = db.Column(db.DateTime)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)

    @property
    def is_active(self):
        return bool(self.is_active_flag)

    def has_role(self, *roles):
        return self.role == "admin" or self.role in roles


class LiveSession(db.Model):
    __tablename__ = "live_sessions"
    id = db.Column(db.Integer, primary_key=True)
    session_code = db.Column(db.String(30), unique=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    started_at = db.Column(db.DateTime, default=now)
    ended_at = db.Column(db.DateTime)
    windows = db.Column(db.Integer, default=0)
    user = db.relationship("User")


class AudioEvent(db.Model):
    """One analysed recording (uploaded file OR one live microphone window)."""
    __tablename__ = "audio_events"
    id = db.Column(db.Integer, primary_key=True)
    audio_id = db.Column(db.String(40), unique=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), index=True)
    source = db.Column(db.String(20), default="upload")                  # upload | batch | live | evaluation
    live_session_id = db.Column(db.Integer, db.ForeignKey("live_sessions.id"))
    actual_class = db.Column(db.String(60))                             # ground truth (evaluation only)

    # file metadata
    original_filename = db.Column(db.String(255))
    stored_path = db.Column(db.String(500))
    format = db.Column(db.String(10))
    file_size = db.Column(db.Integer)
    duration = db.Column(db.Float)
    sample_rate = db.Column(db.Integer)
    channels = db.Column(db.Integer)
    bit_depth = db.Column(db.Integer)
    uploaded_at = db.Column(db.DateTime, default=now, index=True)
    sha256 = db.Column(db.String(64), index=True)
    fingerprint = db.Column(JSONText)
    duplicate_of_id = db.Column(db.Integer, db.ForeignKey("audio_events.id"))
    near_duplicate_of_id = db.Column(db.Integer, db.ForeignKey("audio_events.id"))
    near_duplicate_score = db.Column(db.Float)

    # quality + visuals
    quality_label = db.Column(db.String(20))
    quality = db.Column(JSONText)
    noise_level_db = db.Column(db.Float)
    waveform_path = db.Column(db.String(500))
    spectrogram_path = db.Column(db.String(500))

    # model outputs (preserved even after a reviewer override)
    python_prediction = db.Column(db.String(60))
    python_confidence = db.Column(db.Float)
    python_scores = db.Column(JSONText)
    python_margin = db.Column(db.Float)
    python_segment_index = db.Column(db.Integer)
    gtm_prediction = db.Column(db.String(60))
    gtm_confidence = db.Column(db.Float)
    gtm_scores = db.Column(JSONText)
    gtm_margin = db.Column(db.Float)
    gtm_status = db.Column(db.String(20), default="pending")            # pending | done | unavailable | error

    # comparison + decision
    class_match = db.Column(db.Boolean)
    consistency_status = db.Column(db.String(30))
    confidence_diff = db.Column(db.Float)
    combined_scores = db.Column(JSONText)
    overlap_detected = db.Column(db.Boolean, default=False)
    overlap_classes = db.Column(JSONText)
    uncertain = db.Column(db.Boolean, default=False)
    uncertain_reasons = db.Column(JSONText)
    repeated_count = db.Column(db.Integer, default=1)
    final_category = db.Column(db.String(60), index=True)
    final_confidence = db.Column(db.Float)
    confidence_level = db.Column(db.String(10))
    severity = db.Column(db.String(20), index=True)
    alert_status = db.Column(db.String(30))
    recommended_action = db.Column(db.String(300))
    manual_review_required = db.Column(db.Boolean, default=False)
    review_reasons = db.Column(JSONText)
    status = db.Column(db.String(30), default="Uploaded", index=True)
    decision_trace = db.Column(JSONText)                                # step-by-step explanation

    # reviewer outcome
    reviewed_category = db.Column(db.String(60))
    reviewer_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    reviewed_at = db.Column(db.DateTime)
    overridden = db.Column(db.Boolean, default=False)

    python_model_version = db.Column(db.String(60))
    gtm_model_version = db.Column(db.String(60))
    processing_ms = db.Column(db.Integer)
    processed_at = db.Column(db.DateTime)

    user = db.relationship("User", foreign_keys=[user_id])
    reviewer = db.relationship("User", foreign_keys=[reviewer_id])
    segments = db.relationship("Segment", backref="event", cascade="all, delete-orphan", order_by="Segment.index")
    alerts = db.relationship("Alert", backref="event", cascade="all, delete-orphan")
    reviews = db.relationship("Review", backref="event", cascade="all, delete-orphan")
    live_session = db.relationship("LiveSession")

    @property
    def display_category(self):
        return self.reviewed_category or self.final_category or "—"


class Segment(db.Model):
    __tablename__ = "segments"
    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.Integer, db.ForeignKey("audio_events.id"), index=True)
    index = db.Column(db.Integer)
    start_s = db.Column(db.Float)
    end_s = db.Column(db.Float)
    audio_path = db.Column(db.String(500))       # 44.1 kHz WAV (GTM input + reviewer playback)
    rms_db = db.Column(db.Float)
    python_scores = db.Column(JSONText)
    python_prediction = db.Column(db.String(60))
    python_confidence = db.Column(db.Float)
    gtm_scores = db.Column(JSONText)
    gtm_prediction = db.Column(db.String(60))
    gtm_confidence = db.Column(db.Float)


class Alert(db.Model):
    __tablename__ = "alerts"
    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.Integer, db.ForeignKey("audio_events.id"), index=True)
    category = db.Column(db.String(60))
    severity = db.Column(db.String(20))
    message = db.Column(db.String(300))
    recommended_action = db.Column(db.String(300))
    audience = db.Column(JSONText)                # roles that should see it
    status = db.Column(db.String(20), default="Active", index=True)   # Active | Acknowledged | Escalated | Dismissed
    created_at = db.Column(db.DateTime, default=now, index=True)
    handled_by_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    handled_at = db.Column(db.DateTime)
    handled_by = db.relationship("User")
    actions = db.relationship("AlertAction", backref="alert", cascade="all, delete-orphan", order_by="AlertAction.at")


class AlertAction(db.Model):
    __tablename__ = "alert_actions"
    id = db.Column(db.Integer, primary_key=True)
    alert_id = db.Column(db.Integer, db.ForeignKey("alerts.id"), index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    action = db.Column(db.String(20))
    note = db.Column(db.String(500))
    at = db.Column(db.DateTime, default=now)
    user = db.relationship("User")


class Review(db.Model):
    __tablename__ = "reviews"
    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.Integer, db.ForeignKey("audio_events.id"), index=True)
    reviewer_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    original_category = db.Column(db.String(60))        # automatic result, preserved
    decided_category = db.Column(db.String(60))
    decision = db.Column(db.String(20))                  # Confirmed | Corrected
    override = db.Column(db.Boolean, default=False)
    comment = db.Column(db.Text)
    recommended_action = db.Column(db.String(300))
    created_at = db.Column(db.DateTime, default=now)
    reviewer = db.relationship("User")


class AuditLog(db.Model):
    __tablename__ = "audit_logs"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    username = db.Column(db.String(50))
    action = db.Column(db.String(40), index=True)
    entity = db.Column(db.String(40))
    entity_id = db.Column(db.String(60))
    details = db.Column(db.Text)
    ip = db.Column(db.String(45))
    at = db.Column(db.DateTime, default=now, index=True)


class SystemNotification(db.Model):
    """Anomaly alerts for administrators (failed logins, model failures, ...)."""
    __tablename__ = "system_notifications"
    id = db.Column(db.Integer, primary_key=True)
    kind = db.Column(db.String(40))
    message = db.Column(db.String(400))
    at = db.Column(db.DateTime, default=now, index=True)
    is_read = db.Column(db.Boolean, default=False)


class Setting(db.Model):
    __tablename__ = "settings"
    key = db.Column(db.String(60), primary_key=True)
    value = db.Column(JSONText)
    updated_at = db.Column(db.DateTime, default=now, onupdate=now)


class ModelVersion(db.Model):
    __tablename__ = "model_versions"
    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(db.String(10))      # python | gtm
    version = db.Column(db.String(60))
    details = db.Column(JSONText)
    registered_at = db.Column(db.DateTime, default=now)
