"""Runtime settings: defaults from config/settings.py, overridden by the DB (admin page)."""
from config.settings import DEFAULT_RUNTIME_SETTINGS
from ..extensions import db
from ..models import Setting


def get_settings() -> dict:
    s = dict(DEFAULT_RUNTIME_SETTINGS)
    for row in Setting.query.all():
        if row.key in s:
            s[row.key] = row.value
    return s


def update_settings(values: dict) -> dict:
    for k, v in values.items():
        if k not in DEFAULT_RUNTIME_SETTINGS:
            continue
        default = DEFAULT_RUNTIME_SETTINGS[k]
        if isinstance(default, bool):
            v = v in (True, "on", "true", "1", 1)
        elif isinstance(default, int):
            v = int(float(v))
        elif isinstance(default, float):
            v = float(v)
        row = db.session.get(Setting, k)
        if row:
            row.value = v
        else:
            db.session.add(Setting(key=k, value=v))
    db.session.commit()
    return get_settings()
