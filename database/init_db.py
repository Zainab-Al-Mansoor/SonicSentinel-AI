"""
Database initialisation + demo / evaluator accounts.

    python -m database.init_db            # create tables + default accounts
    python -m database.init_db --reset    # DELETE everything and start again

Default accounts (CHANGE THE PASSWORDS after first login):
    admin / Admin@12345          (Administrator)
    reviewer / Review@12345      (Audio reviewer)
    security / Secure@12345      (Security operator)
    maintenance / Maint@12345    (Maintenance operator)
    evaluator / Eval@12345       (Normal user – for competition evaluators)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_USERS = [
    ("admin", "admin@sonicsentinel.local", "System Administrator", "admin", "Admin@12345"),
    ("reviewer", "reviewer@sonicsentinel.local", "Audio Reviewer", "reviewer", "Review@12345"),
    ("security", "security@sonicsentinel.local", "Security Operator", "security", "Secure@12345"),
    ("maintenance", "maintenance@sonicsentinel.local", "Maintenance Operator", "maintenance", "Maint@12345"),
    ("evaluator", "evaluator@sonicsentinel.local", "Competition Evaluator", "user", "Eval@12345"),
]


def next_user_code():
    from src.models import User
    last = User.query.order_by(User.id.desc()).first()
    return f"USR-{(last.id + 1 if last else 1):05d}"


def seed_defaults():
    from src.extensions import db
    from src.models import User
    if User.query.count() == 0:
        for username, email, name, role, pw in DEFAULT_USERS:
            u = User(user_code=next_user_code(), username=username, email=email, full_name=name, role=role)
            u.set_password(pw)
            db.session.add(u)
            db.session.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    from src import create_app
    from src.extensions import db
    app = create_app({"WARM_UP": False})
    with app.app_context():
        if args.reset:
            db.drop_all()
            db.create_all()
            seed_defaults()
            print("Database reset.")
        print("Database ready. Accounts:")
        for u, _, _, role, pw in DEFAULT_USERS:
            print(f"  {u:<12} {pw:<14} ({role})")


if __name__ == "__main__":
    main()
