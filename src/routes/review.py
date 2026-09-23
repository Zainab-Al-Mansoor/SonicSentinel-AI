"""Manual review queue: playback, confirm / correct, comments, override (original outputs preserved)."""
from flask import Blueprint, render_template, request, redirect, url_for, flash, abort
from flask_login import login_required, current_user

from config.settings import CLASSES, SEVERITY_LEVELS
from ..extensions import db
from ..models import AudioEvent, Review, Alert, now
from ..security import roles_required
from ..services import audit
from ..services.analysis import gtm_info
from ..services.rules import load_rules
from python_models.inference import top_k

bp = Blueprint("review", __name__, url_prefix="/review")


@bp.route("/")
@login_required
@roles_required("reviewer")
def queue():
    show = request.args.get("show", "pending")
    q = AudioEvent.query
    q = q.filter(AudioEvent.status == "Manual Review") if show == "pending" else q.filter(AudioEvent.reviewer_id.isnot(None))
    events = q.order_by(AudioEvent.uploaded_at.desc()).limit(300).all()
    return render_template("review_queue.html", events=events, show=show)


@bp.route("/<audio_id>", methods=["GET", "POST"])
@login_required
@roles_required("reviewer")
def review(audio_id):
    ev = AudioEvent.query.filter_by(audio_id=audio_id).first_or_404()
    if request.method == "POST":
        f = request.form
        decided = f.get("category")
        if decided not in CLASSES + ["Unknown"]:
            flash("Choose a valid category.", "error")
            return redirect(url_for("review.review", audio_id=audio_id))
        original = ev.final_category
        corrected = decided != original
        override = corrected or f.get("severity") not in (None, "", ev.severity)
        r = Review(event_id=ev.id, reviewer_id=current_user.id, original_category=original,
                   decided_category=decided, decision="Corrected" if corrected else "Confirmed", override=override,
                   comment=f.get("comment", "").strip(), recommended_action=f.get("recommended_action", "").strip())
        db.session.add(r)
        # Model outputs (python_*, gtm_*, final_category) stay untouched – only reviewer fields change.
        ev.reviewed_category = decided
        ev.reviewer_id, ev.reviewed_at, ev.overridden = current_user.id, now(), override
        if f.get("severity") in SEVERITY_LEVELS:
            ev.severity = f["severity"]
        if r.recommended_action:
            ev.recommended_action = r.recommended_action
        ev.status = "Closed" if f.get("close") else "Reviewed"
        # reviewer can raise an alert that the rules did not raise
        if f.get("raise_alert") and not ev.alerts:
            rule = load_rules().get(decided, {})
            ev.alerts.append(Alert(category=decided, severity=ev.severity,
                                   message=f"{ev.severity} – {decided} confirmed by reviewer",
                                   recommended_action=ev.recommended_action, audience=rule.get("audience", ["security"])))
            ev.alert_status = "Alert Generated"
        db.session.commit()
        audit.log("review_override" if override else "review", "event", ev.audio_id,
                  f"{original} -> {decided}: {r.comment[:200]}")
        flash(f"Review saved ({r.decision}).", "success")
        return redirect(url_for("review.queue"))
    return render_template("review_detail.html", ev=ev, top_k=top_k, gtm=gtm_info())
