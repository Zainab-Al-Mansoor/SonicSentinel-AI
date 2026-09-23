"""Registration, login / logout, profile management."""
from datetime import timedelta

from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user

from config.settings import ROLES
from ..extensions import db
from ..models import User, now
from ..security import password_problems
from ..services import audit

bp = Blueprint("auth", __name__)
SELF_REGISTER_ROLES = ["user", "reviewer", "security", "maintenance"]   # admin is assigned by an admin
MAX_FAILED = 5
LOCK_MINUTES = 5


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    form = request.form
    if request.method == "POST":
        username = form.get("username", "").strip().lower()
        email = form.get("email", "").strip().lower()
        pw = form.get("password", "")
        role = form.get("role", "user")
        errors = []
        if not (3 <= len(username) <= 30) or not username.replace("_", "").replace(".", "").isalnum():
            errors.append("Username must be 3–30 letters/numbers (._ allowed).")
        if "@" not in email or "." not in email.split("@")[-1]:
            errors.append("Enter a valid email address.")
        if pw != form.get("confirm", ""):
            errors.append("Passwords do not match.")
        probs = password_problems(pw)
        if probs:
            errors.append("Password needs " + " and ".join(probs) + ".")
        if role not in SELF_REGISTER_ROLES:
            errors.append("Invalid role.")
        if User.query.filter((User.username == username) | (User.email == email)).first():
            errors.append("Username or email is already registered.")
        if errors:
            for e in errors:
                flash(e, "error")
        else:
            from database.init_db import next_user_code
            u = User(user_code=next_user_code(), username=username, email=email,
                     full_name=form.get("full_name", "").strip(), organization=form.get("organization", "").strip(),
                     role=role)
            u.set_password(pw)
            db.session.add(u)
            db.session.commit()
            audit.log("register", "user", u.user_code, f"role={role}", user=u)
            flash(f"Account created. Your User ID is {u.user_code}. Please log in.", "success")
            return redirect(url_for("auth.login"))
    roles = {k: v for k, v in ROLES.items() if k in SELF_REGISTER_ROLES}
    return render_template("auth/register.html", roles=roles, form=form)


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        pw = request.form.get("password", "")
        u = User.query.filter((User.username == username) | (User.email == username)).first()
        if u and u.locked_until and u.locked_until > now():
            flash(f"Account temporarily locked after repeated failed logins. Try again later.", "error")
        elif u and u.check_password(pw) and u.is_active:
            u.failed_logins, u.locked_until, u.last_login = 0, None, now()
            db.session.commit()
            login_user(u, remember=bool(request.form.get("remember")))
            audit.log("login", "user", u.user_code, "", user=u)
            nxt = request.args.get("next")
            return redirect(nxt if nxt and nxt.startswith("/") and not nxt.startswith("//") else url_for("main.dashboard"))
        else:
            if u:
                u.failed_logins = (u.failed_logins or 0) + 1
                if u.failed_logins >= MAX_FAILED:
                    u.locked_until = now() + timedelta(minutes=LOCK_MINUTES)
                db.session.commit()
            audit.log("login_failed", "user", "", f"username={username}", user=None)
            audit.check_failed_logins(username)
            flash("Invalid username or password." if not (u and not u.is_active) else "This account is disabled.", "error")
    return render_template("auth/login.html")


@bp.route("/logout")
@login_required
def logout():
    audit.log("logout", "user", current_user.user_code)
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))


@bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        f = request.form
        if f.get("action") == "password":
            if not current_user.check_password(f.get("current", "")):
                flash("Current password is incorrect.", "error")
            elif f.get("new") != f.get("confirm"):
                flash("New passwords do not match.", "error")
            elif password_problems(f.get("new", "")):
                flash("Password needs " + " and ".join(password_problems(f.get("new", ""))) + ".", "error")
            else:
                current_user.set_password(f["new"])
                db.session.commit()
                audit.log("password_change", "user", current_user.user_code)
                flash("Password updated.", "success")
        else:
            email = f.get("email", "").strip().lower()
            if email != current_user.email and User.query.filter_by(email=email).first():
                flash("That email is already used by another account.", "error")
            else:
                current_user.full_name = f.get("full_name", "").strip()
                current_user.email = email or current_user.email
                current_user.phone = f.get("phone", "").strip()
                current_user.organization = f.get("organization", "").strip()
                db.session.commit()
                audit.log("profile_update", "user", current_user.user_code)
                flash("Profile updated.", "success")
        return redirect(url_for("auth.profile"))
    return render_template("auth/profile.html")
