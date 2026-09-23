"""Audit trail + administrator anomaly notifications."""
from datetime import timedelta

from flask import request, has_request_context
from flask_login import current_user

from ..extensions import db
from ..models import AuditLog, SystemNotification, AudioEvent, now


def log(action: str, entity: str = "", entity_id="", details: str = "", user=None, commit=True):
    u = user
    if u is None and has_request_context() and getattr(current_user, "is_authenticated", False):
        u = current_user
    entry = AuditLog(user_id=getattr(u, "id", None), username=getattr(u, "username", None),
                     action=action, entity=entity, entity_id=str(entity_id or ""), details=(details or "")[:2000],
                     ip=request.remote_addr if has_request_context() else None)
    db.session.add(entry)
    if commit:
        db.session.commit()
    return entry


def notify(kind: str, message: str, dedupe_minutes: int = 10):
    """Create an admin notification (skips an identical one raised in the last few minutes)."""
    since = now() - timedelta(minutes=dedupe_minutes)
    if SystemNotification.query.filter(SystemNotification.kind == kind, SystemNotification.message == message,
                                       SystemNotification.at >= since).first():
        return
    db.session.add(SystemNotification(kind=kind, message=message))
    db.session.commit()


# ---------------------------------------------------------------------------
# Anomaly checks (SRS lxxviii)
# ---------------------------------------------------------------------------
def check_failed_uploads(user):
    since = now() - timedelta(minutes=10)
    n = AuditLog.query.filter(AuditLog.action == "upload_rejected", AuditLog.user_id == user.id,
                              AuditLog.at >= since).count()
    if n >= 3:
        notify("failed_uploads", f"User {user.username} had {n} rejected uploads in 10 minutes.")


def check_failed_logins(username: str):
    since = now() - timedelta(minutes=15)
    n = AuditLog.query.filter(AuditLog.action == "login_failed", AuditLog.details.contains(username),
                              AuditLog.at >= since).count()
    if n >= 3:
        notify("failed_logins", f"{n} failed login attempts for '{username}' in 15 minutes.")


def check_after_event():
    recent = AudioEvent.query.filter(AudioEvent.final_category.isnot(None)) \
        .order_by(AudioEvent.id.desc()).limit(10).all()
    if len(recent) >= 10:
        low = sum(1 for e in recent if (e.final_confidence or 0) < 0.5)
        if low >= 6:
            notify("low_confidence_spike", f"{low} of the last 10 detections had confidence below 0.50.")
    since = now() - timedelta(minutes=10)
    from ..models import Alert
    crit = Alert.query.filter(Alert.severity == "Critical", Alert.created_at >= since).count()
    if crit >= 10:
        notify("excessive_critical", f"{crit} critical alerts in the last 10 minutes – check for false alarms.")
