"""CSRF protection, role checks and password policy."""
import re
import secrets
from functools import wraps

from flask import session, request, abort, jsonify
from flask_login import current_user


def csrf_token() -> str:
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)
    return session["_csrf"]


def check_csrf():
    """Called before every POST/PUT/DELETE. Forms send `csrf_token`, fetch() sends X-CSRFToken."""
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        sent = request.form.get("csrf_token") or request.headers.get("X-CSRFToken")
        if not sent or not secrets.compare_digest(sent, session.get("_csrf", "")):
            if request.path.startswith("/api/"):
                return jsonify({"error": "CSRF token missing or invalid. Reload the page."}), 400
            abort(400, description="Security token missing or expired. Please reload the page and try again.")


def roles_required(*roles):
    """Allow the listed roles (admin is always allowed)."""
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            if not current_user.is_authenticated:
                abort(401)
            if not current_user.has_role(*roles):
                abort(403)
            return fn(*a, **kw)
        return wrapper
    return deco


def password_problems(pw: str) -> list[str]:
    p = []
    if len(pw) < 8:
        p.append("at least 8 characters")
    if not re.search(r"[A-Za-z]", pw) or not re.search(r"\d", pw):
        p.append("letters and numbers")
    return p


def can_view_event(user, ev) -> bool:
    return user.role in ("admin", "reviewer", "security", "maintenance") or ev.user_id == user.id


def can_handle_alert(user, alert) -> bool:
    if user.role == "admin":
        return True
    return user.role in (alert.audience or []) or (user.role == "security" and not alert.audience)
