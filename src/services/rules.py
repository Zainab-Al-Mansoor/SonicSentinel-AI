"""Load / save / evaluate the configurable alert rules (alert_rules/alert_rules.json)."""
import json

from config.settings import ALERT_RULES_PATH, SEVERITY_LEVELS, QUALITY_LEVELS

RULE_FIELDS = {
    "severity": str, "generate_alert": bool, "min_confidence": float, "top2_margin": float,
    "required_consecutive": int, "strong_confidence": float, "require_model_agreement": bool,
    "min_quality": str, "manual_review_if_unconfirmed": bool, "escalate_after_repeats": int,
    "escalate_to": str, "recommended_action": str, "audience": list,
}


def load_rules() -> dict:
    with open(ALERT_RULES_PATH, encoding="utf-8") as fh:
        return json.load(fh)["rules"]


def save_rules(rules: dict):
    with open(ALERT_RULES_PATH, encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["rules"] = rules
    with open(ALERT_RULES_PATH, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)


def default_rule(category: str) -> dict:
    return {"severity": "Low", "generate_alert": False, "min_confidence": 0.6, "top2_margin": 0.1,
            "required_consecutive": 1, "strong_confidence": 0, "require_model_agreement": False,
            "min_quality": "Poor", "manual_review_if_unconfirmed": True, "escalate_after_repeats": 0,
            "escalate_to": "", "recommended_action": f"Review the '{category}' event.", "audience": []}


def validate_rule(rule: dict) -> list[str]:
    errs = []
    if rule.get("severity") not in SEVERITY_LEVELS:
        errs.append("invalid severity")
    if rule.get("escalate_to") and rule["escalate_to"] not in SEVERITY_LEVELS:
        errs.append("invalid escalate_to")
    if rule.get("min_quality") not in QUALITY_LEVELS[:3]:
        errs.append("min_quality must be Good, Acceptable or Poor")
    for k in ("min_confidence", "top2_margin", "strong_confidence"):
        if not 0 <= float(rule.get(k, 0)) <= 1:
            errs.append(f"{k} must be between 0 and 1")
    if int(rule.get("required_consecutive", 1)) < 1:
        errs.append("required_consecutive must be >= 1")
    return errs


def quality_ok(actual: str, minimum: str) -> bool:
    """'Good' >= 'Acceptable' >= 'Poor' >= 'Unusable'"""
    order = {q: i for i, q in enumerate(QUALITY_LEVELS)}
    return order.get(actual, 3) <= order.get(minimum, 2)


def evaluate_rule(rule: dict, *, confidence: float, margin: float, quality: str,
                  models_agree: bool, repeated: int) -> dict:
    """Check every condition of one rule and say whether the event is CONFIRMED."""
    checks = {
        "confidence": confidence >= rule["min_confidence"],
        "top2_margin": margin >= rule["top2_margin"],
        "quality": quality_ok(quality, rule["min_quality"]),
        "model_agreement": models_agree or not rule["require_model_agreement"],
    }
    strong = rule.get("strong_confidence", 0) or 0
    single_ok = strong > 0 and models_agree and confidence >= strong
    checks["repeated_detection"] = repeated >= rule["required_consecutive"] or single_ok
    severity = rule["severity"]
    escalated = False
    if rule.get("escalate_after_repeats") and rule.get("escalate_to") and repeated >= rule["escalate_after_repeats"]:
        if SEVERITY_LEVELS.index(rule["escalate_to"]) > SEVERITY_LEVELS.index(severity):
            severity, escalated = rule["escalate_to"], True
    return {"confirmed": all(checks.values()), "checks": checks, "severity": severity,
            "escalated": escalated, "single_detection_accepted": single_ok and repeated < rule["required_consecutive"]}
