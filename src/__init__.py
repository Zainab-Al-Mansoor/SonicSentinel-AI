"""SonicSentinel AI – Flask application factory."""
import os
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from flask_login import current_user

from config.settings import BASE_DIR, DATABASE_PATH, SECRET_KEY, MAX_UPLOAD_MB, CLASSES, SEVERITY_LEVELS, ROLES
from .extensions import db, login_manager
from .security import csrf_token, check_csrf


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
    app.config.update(
        SECRET_KEY=SECRET_KEY,
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{DATABASE_PATH}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        MAX_CONTENT_LENGTH=MAX_UPLOAD_MB * 1024 * 1024 * 20,   # batch requests carry several files
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        WARM_UP=True,
    )
    if test_config:
        app.config.update(test_config)

    db.init_app(app)
    login_manager.init_app(app)

    from .models import User

    @login_manager.user_loader
    def load_user(uid):
        return db.session.get(User, int(uid))

    from .routes.auth import bp as auth_bp
    from .routes.main import bp as main_bp
    from .routes.api import bp as api_bp
    from .routes.alerts import bp as alerts_bp
    from .routes.review import bp as review_bp
    from .routes.admin import bp as admin_bp
    for bp in (auth_bp, main_bp, api_bp, alerts_bp, review_bp, admin_bp):
        app.register_blueprint(bp)

    app.before_request(check_csrf)

    @app.context_processor
    def inject():
        from .models import SystemNotification, Alert, AudioEvent
        ctx = {"csrf_token": csrf_token, "CLASSES": CLASSES, "SEVERITY_LEVELS": SEVERITY_LEVELS, "ROLES": ROLES}
        if current_user.is_authenticated:
            try:
                ctx["nav_active_alerts"] = Alert.query.filter_by(status="Active").count() \
                    if current_user.role in ("admin", "security", "maintenance") else 0
                ctx["nav_notifications"] = SystemNotification.query.filter_by(is_read=False).count() \
                    if current_user.role == "admin" else 0
                ctx["nav_reviews"] = AudioEvent.query.filter_by(status="Manual Review").count() \
                    if current_user.role in ("admin", "reviewer") else 0
            except Exception:
                pass
        return ctx

    @app.template_filter("pct")
    def pct(v):
        return "—" if v is None else f"{v * 100:.1f}%"

    @app.template_filter("dt")
    def dt(v):
        return "—" if v is None else v.strftime("%Y-%m-%d %H:%M:%S")

    # ---- error handling (SRS lxxvii) ---------------------------------------
    def _err(code, title, message):
        if request.path.startswith("/api/"):
            return jsonify({"error": message}), code
        return render_template("errors/error.html", code=code, title=title, message=message), code

    @app.errorhandler(400)
    def bad_request(e):
        return _err(400, "Bad request", getattr(e, "description", "The request could not be processed."))

    @app.errorhandler(401)
    def unauth(e):
        return _err(401, "Login required", "Please log in to continue.")

    @app.errorhandler(403)
    def forbidden(e):
        return _err(403, "Access denied", "Your role does not have permission for this page.")

    @app.errorhandler(404)
    def not_found(e):
        return _err(404, "Not found", "The page or record you requested does not exist.")

    @app.errorhandler(413)
    def too_large(e):
        return _err(413, "File too large", f"Uploads are limited to {MAX_UPLOAD_MB} MB per file.")

    @app.errorhandler(500)
    def server_error(e):
        db.session.rollback()
        return _err(500, "Something went wrong", "An internal error occurred. It has been logged.")

    with app.app_context():
        db.create_all()
        from database.init_db import seed_defaults
        seed_defaults()
        from .services.retention import apply_retention
        try:
            apply_retention()
        except Exception:
            pass
        if app.config.get("WARM_UP") and not app.config.get("TESTING"):
            from .services.analysis import warm_up
            warm_up()
    return app
