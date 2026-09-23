"""Administrator pages: analytics, users, settings, alert rules, audit, notifications, models, export."""
import io
import json
from datetime import timedelta

import pandas as pd
from flask import Blueprint, render_template, request, redirect, url_for, flash, abort, Response, send_file
from flask_login import login_required, current_user

from config.settings import (ROLES, CLASSES, SEVERITY_LEVELS, QUALITY_LEVELS, DEFAULT_RUNTIME_SETTINGS,
                             EXPORT_DIR, BASE_DIR)
from ..extensions import db
from ..models import User, AudioEvent, Alert, Review, AuditLog, SystemNotification, ModelVersion, now
from ..security import roles_required
from ..services import audit
from ..services.analysis import gtm_info, python_model
from ..services.analytics import summary, comparison_dataframe
from ..services.retention import apply_retention
from ..services.rules import load_rules, save_rules, validate_rule, RULE_FIELDS, default_rule
from ..services.settings_service import get_settings, update_settings

bp = Blueprint("admin", __name__, url_prefix="/admin")


@bp.before_request
@login_required
@roles_required("admin")
def _guard():
    pass


@bp.route("/")
def dashboard():
    days = request.args.get("days", 30, type=int)
    events = AudioEvent.query.filter(AudioEvent.uploaded_at >= now() - timedelta(days=days)).all()
    try:
        pm = python_model()
        py = {"version": pm.version, "algorithm": pm.algorithm, "metrics": pm.metrics}
    except Exception as exc:
        py = {"error": str(exc)}
    return render_template("admin/dashboard.html", s=summary(events), days=days, py=py, gtm=gtm_info(),
                           notes=SystemNotification.query.order_by(SystemNotification.at.desc()).limit(8).all())


# ---- users --------------------------------------------------------------
@bp.route("/users", methods=["GET", "POST"])
def users():
    if request.method == "POST":
        u = db.session.get(User, request.form.get("user_id", type=int)) or abort(404)
        if u.id == current_user.id and request.form.get("action") in ("disable", "role"):
            flash("You cannot change your own role or disable yourself.", "error")
        elif request.form.get("action") == "role" and request.form.get("role") in ROLES:
            u.role = request.form["role"]
            audit.log("user_role", "user", u.user_code, f"role -> {u.role}", commit=False)
        elif request.form.get("action") in ("disable", "enable"):
            u.is_active_flag = request.form["action"] == "enable"
            u.failed_logins, u.locked_until = 0, None
            audit.log("user_" + request.form["action"], "user", u.user_code, "", commit=False)
        db.session.commit()
        return redirect(url_for("admin.users"))
    return render_template("admin/users.html", users=User.query.order_by(User.id).all())


# ---- settings / thresholds ----------------------------------------------
@bp.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        vals = {k: request.form.get(k) for k in DEFAULT_RUNTIME_SETTINGS if not isinstance(DEFAULT_RUNTIME_SETTINGS[k], bool)}
        vals.update({k: request.form.get(k) == "on" for k, v in DEFAULT_RUNTIME_SETTINGS.items() if isinstance(v, bool)})
        try:
            s = update_settings(vals)
            audit.log("settings_update", "settings", "", json.dumps(s))
            flash("Settings saved.", "success")
        except ValueError:
            flash("Invalid number in settings.", "error")
        return redirect(url_for("admin.settings"))
    model_seg = None
    try:
        model_seg = python_model().settings
    except Exception:
        pass
    return render_template("admin/settings.html", s=get_settings(), defaults=DEFAULT_RUNTIME_SETTINGS, model_seg=model_seg)


@bp.route("/retention", methods=["POST"])
def retention():
    res = apply_retention()
    audit.log("retention", "data", "", json.dumps(res))
    flash(f"Retention applied: {res['audio_files_removed']} audio files and {res['records_removed']} records removed.", "success")
    return redirect(url_for("admin.settings"))


# ---- alert rules --------------------------------------------------------
@bp.route("/rules", methods=["GET", "POST"])
def rules():
    rules = load_rules()
    if request.method == "POST":
        cat = request.form.get("category")
        if request.form.get("action") == "add":
            if cat and cat not in rules:
                rules[cat] = default_rule(cat)
                save_rules(rules)
                audit.log("rule_add", "alert_rule", cat)
                flash(f"Rule added for '{cat}'.", "success")
            return redirect(url_for("admin.rules"))
        if cat not in rules:
            abort(404)
        new = dict(rules[cat])
        for k, typ in RULE_FIELDS.items():
            if typ is bool:
                new[k] = request.form.get(k) == "on"
            elif typ is list:
                new[k] = request.form.getlist(k)
            elif k in request.form:
                try:
                    new[k] = typ(request.form[k]) if typ is not str else request.form[k].strip()
                except ValueError:
                    flash(f"Invalid value for {k}.", "error")
                    return redirect(url_for("admin.rules"))
        errs = validate_rule(new)
        if errs:
            flash(f"{cat}: " + "; ".join(errs), "error")
        else:
            rules[cat] = new
            save_rules(rules)
            audit.log("rule_update", "alert_rule", cat, json.dumps(new))
            flash(f"Rule for '{cat}' saved.", "success")
        return redirect(url_for("admin.rules") + f"#rule-{cat.replace(' ', '-')}")
    return render_template("admin/rules.html", rules=rules, QUALITY_LEVELS=QUALITY_LEVELS, roles=ROLES)


# ---- audit / notifications / models -------------------------------------
@bp.route("/audit")
def audit_log():
    q = AuditLog.query
    if request.args.get("action"):
        q = q.filter(AuditLog.action == request.args["action"])
    if request.args.get("user"):
        q = q.filter(AuditLog.username.contains(request.args["user"]))
    page = request.args.get("page", 1, type=int)
    pag = q.order_by(AuditLog.at.desc()).paginate(page=page, per_page=50, error_out=False)
    actions = [a[0] for a in db.session.query(AuditLog.action).distinct().all()]
    return render_template("admin/audit.html", pag=pag, actions=sorted(actions), args=request.args)


@bp.route("/notifications", methods=["GET", "POST"])
def notifications():
    if request.method == "POST":
        SystemNotification.query.update({"is_read": True})
        db.session.commit()
        return redirect(url_for("admin.notifications"))
    return render_template("admin/notifications.html",
                           notes=SystemNotification.query.order_by(SystemNotification.at.desc()).limit(300).all())


@bp.route("/models")
def models():
    import pathlib
    reports = BASE_DIR / "reports"
    comp = pd.read_csv(reports / "python_model_comparison.csv").to_dict("records") \
        if (reports / "python_model_comparison.csv").exists() else []
    test = json.loads((reports / "python_test_metrics.json").read_text()) \
        if (reports / "python_test_metrics.json").exists() else None
    return render_template("admin/models.html", versions=ModelVersion.query.order_by(ModelVersion.id.desc()).all(),
                           comp=comp, test=test, gtm=gtm_info(),
                           has_cm=(reports / "confusion_matrix_python.png").exists())


@bp.route("/models/confusion-matrix.png")
def confusion_png():
    p = BASE_DIR / "reports" / "confusion_matrix_python.png"
    return send_file(p) if p.exists() else abort(404)


# ---- evaluation + export -------------------------------------------------
@bp.route("/evaluation")
def evaluation():
    evs = AudioEvent.query.filter(AudioEvent.actual_class.isnot(None)).order_by(AudioEvent.id).all()
    df, s = comparison_dataframe(evs)
    return render_template("admin/evaluation.html", s=s, rows=df.to_dict("records") if not df.empty else [],
                           gtm=gtm_info())


EXPORTS = {
    "events": lambda: pd.DataFrame([{
        "audio_id": e.audio_id, "filename": e.original_filename, "source": e.source, "user": e.user.username if e.user else "",
        "uploaded_at": e.uploaded_at, "duration": e.duration, "format": e.format, "sample_rate": e.sample_rate,
        "channels": e.channels, "quality": e.quality_label, "python_prediction": e.python_prediction,
        "python_confidence": e.python_confidence, "gtm_prediction": e.gtm_prediction, "gtm_confidence": e.gtm_confidence,
        "confidence_diff": e.confidence_diff, "consistency": e.consistency_status, "final_category": e.final_category,
        "final_confidence": e.final_confidence, "severity": e.severity, "alert_status": e.alert_status,
        "status": e.status, "manual_review": e.manual_review_required, "reviewed_category": e.reviewed_category,
        "python_model_version": e.python_model_version, "gtm_model_version": e.gtm_model_version,
        "processing_ms": e.processing_ms} for e in AudioEvent.query.order_by(AudioEvent.id).all()]),
    "alerts": lambda: pd.DataFrame([{
        "alert_id": a.id, "audio_id": a.event.audio_id, "category": a.category, "severity": a.severity,
        "status": a.status, "created_at": a.created_at, "handled_by": a.handled_by.username if a.handled_by else "",
        "handled_at": a.handled_at, "message": a.message} for a in Alert.query.order_by(Alert.id).all()]),
    "reviews": lambda: pd.DataFrame([{
        "review_id": r.id, "audio_id": r.event.audio_id, "reviewer": r.reviewer.username if r.reviewer else "",
        "original_category": r.original_category, "decided_category": r.decided_category, "decision": r.decision,
        "override": r.override, "comment": r.comment, "created_at": r.created_at} for r in Review.query.all()]),
    "audit": lambda: pd.DataFrame([{
        "at": l.at, "user": l.username, "action": l.action, "entity": l.entity, "entity_id": l.entity_id,
        "details": l.details, "ip": l.ip} for l in AuditLog.query.order_by(AuditLog.id).all()]),
    "comparison": lambda: comparison_dataframe(
        AudioEvent.query.filter(AudioEvent.final_category.isnot(None)).order_by(AudioEvent.id).all())[0],
    "evaluation": lambda: comparison_dataframe(
        AudioEvent.query.filter(AudioEvent.actual_class.isnot(None)).order_by(AudioEvent.id).all())[0],
}


@bp.route("/export/<name>.<fmt>")
def export(name, fmt):
    if name not in EXPORTS or fmt not in ("csv", "xlsx"):
        abort(404)
    df = EXPORTS[name]()
    audit.log("export", "data", name, f"{len(df)} rows as {fmt}")
    stamp = now().strftime("%Y%m%d-%H%M")
    if fmt == "csv":
        return Response(df.to_csv(index=False), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=sonicsentinel_{name}_{stamp}.csv"})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name=name[:30])
        if name in ("comparison", "evaluation"):
            evs = AudioEvent.query.filter(AudioEvent.actual_class.isnot(None) if name == "evaluation"
                                          else AudioEvent.final_category.isnot(None)).all()
            _, s = comparison_dataframe(evs)
            pd.DataFrame([{"metric": k, "value": json.dumps(v) if isinstance(v, dict) else v} for k, v in s.items()]) \
                .to_excel(xw, index=False, sheet_name="summary")
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f"sonicsentinel_{name}_{stamp}.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
