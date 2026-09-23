"""Alert list, acknowledgement / dismissal / escalation and alert history."""
from flask import Blueprint, render_template, request, redirect, url_for, flash, abort, jsonify
from flask_login import login_required, current_user

from ..extensions import db
from ..models import Alert, AlertAction, now
from ..security import roles_required, can_handle_alert
from ..services import audit

bp = Blueprint("alerts", __name__, url_prefix="/alerts")
ACTIONS = {"acknowledge": "Acknowledged", "dismiss": "Dismissed", "escalate": "Escalated"}


def visible_alerts():
    q = Alert.query
    if current_user.role != "admin":
        q = q.filter(Alert.audience.contains(f'"{current_user.role}"'))
    return q


@bp.route("/")
@login_required
@roles_required("security", "maintenance")
def index():
    status = request.args.get("status", "Active")
    q = visible_alerts()
    if status != "all":
        q = q.filter(Alert.status == status)
    alerts = q.order_by(Alert.created_at.desc()).limit(300).all()
    return render_template("alerts.html", alerts=alerts, status=status)


@bp.route("/<int:alert_id>/<action>", methods=["POST"])
@login_required
@roles_required("security", "maintenance")
def act(alert_id, action):
    a = db.session.get(Alert, alert_id) or abort(404)
    if action not in ACTIONS or not can_handle_alert(current_user, a):
        abort(403)
    note = request.form.get("note", "").strip()[:500]
    a.status = ACTIONS[action]
    a.handled_by_id, a.handled_at = current_user.id, now()
    if action == "escalate" and a.severity != "Critical":
        a.severity = "Critical"
        a.event.severity = "Critical"
    a.actions.append(AlertAction(user_id=current_user.id, action=ACTIONS[action], note=note))
    db.session.commit()
    audit.log(f"alert_{action}", "alert", a.id, f"{a.category} ({a.event.audio_id}) {note}")
    if request.headers.get("X-Requested-With") == "fetch":
        return jsonify({"ok": True, "status": a.status})
    flash(f"Alert #{a.id} {ACTIONS[action].lower()}.", "success")
    return redirect(request.referrer or url_for("alerts.index"))


@bp.route("/history")
@login_required
@roles_required("security", "maintenance")
def history():
    actions = AlertAction.query.join(Alert).order_by(AlertAction.at.desc()).limit(500).all()
    alerts = visible_alerts().order_by(Alert.created_at.desc()).limit(500).all()
    return render_template("alert_history.html", actions=actions, alerts=alerts)


@bp.route("/poll")
@login_required
def poll():
    """Lightweight poll for the navbar badge / toast of new alerts."""
    if current_user.role not in ("admin", "security", "maintenance"):
        return jsonify({"active": 0, "latest": None})
    q = visible_alerts().filter(Alert.status == "Active")
    latest = q.order_by(Alert.id.desc()).first()
    return jsonify({"active": q.count(),
                    "latest": {"id": latest.id, "message": latest.message, "severity": latest.severity,
                               "url": url_for("main.event_detail", audio_id=latest.event.audio_id)} if latest else None})
