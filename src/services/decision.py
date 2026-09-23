"""
Model comparison + final sound-event decision (pure functions, no database).

Inputs : Python scores, GTM scores (or None), audio quality, noise level,
         a function giving the repeated-detection count, rules, settings.
Output : everything the SRS "Final Sound Event Decision" needs, plus a
         human-readable `trace` explaining each step.
"""
from config.settings import CLASSES, BACKGROUND_CLASS, UNKNOWN_CLASS, SEVERITY_LEVELS
from python_models.inference import top_k, top_margin
from .rules import evaluate_rule, default_rule, quality_ok


def compare_models(py: dict, gtm: dict | None, s: dict) -> dict:
    """Prediction and confidence comparison between the two independent models."""
    py_cls, py_conf = top_k(py, 1)[0]
    out = {"python_prediction": py_cls, "python_confidence": py_conf, "python_margin": top_margin(py),
           "python_top3": top_k(py, 3)}
    if not gtm:
        out.update({"gtm_prediction": None, "gtm_confidence": None, "gtm_margin": None, "gtm_top3": [],
                    "class_match": None, "confidence_diff": None,
                    "consistency_status": "Uncertain Result"})
        return out
    g_cls, g_conf = top_k(gtm, 1)[0]
    diff = abs(py_conf - g_conf)          # |Python top-class conf − GTM top-class conf|
    match = py_cls == g_cls
    th = s["min_confidence"]
    if py_conf < th and g_conf < th:
        status = "Uncertain Result"
    elif match and diff <= s["acceptable_match_diff"] and min(py_conf, g_conf) >= th:
        status = "Acceptable Match"
    elif match:
        status = "Weak Match"
    else:
        status = "Model Disagreement"
    out.update({"gtm_prediction": g_cls, "gtm_confidence": g_conf, "gtm_margin": top_margin(gtm),
                "gtm_top3": top_k(gtm, 3), "class_match": match, "confidence_diff": diff,
                "consistency_status": status})
    return out


def combine(py: dict, gtm: dict | None) -> dict:
    """Average of both models' confidence vectors (Python only if GTM unavailable)."""
    if not gtm:
        return {c: float(py.get(c, 0.0)) for c in CLASSES}
    return {c: (float(py.get(c, 0.0)) + float(gtm.get(c, 0.0))) / 2 for c in CLASSES}


def confidence_level(c: float, s: dict) -> str:
    return "High" if c >= 0.85 else ("Medium" if c >= s["min_confidence"] else "Low")


def decide(py: dict, gtm: dict | None, *, quality: str, noise_db: float | None,
           repeated_for, rules: dict, settings: dict) -> dict:
    s = settings
    trace = []
    cmp = compare_models(py, gtm, s)
    trace.append(f"Python → {cmp['python_prediction']} ({cmp['python_confidence']:.2f}); "
                 + (f"GTM → {cmp['gtm_prediction']} ({cmp['gtm_confidence']:.2f}); "
                    f"|Δconf| = {cmp['confidence_diff']:.2f} → {cmp['consistency_status']}"
                    if gtm else "GTM result unavailable → Uncertain Result"))

    comb = combine(py, gtm)
    best, conf = top_k(comb, 1)[0]
    margin = top_margin(comb)
    trace.append(f"Combined scores → {best} ({conf:.2f}), top-2 margin {margin:.2f}")

    category = best
    if conf < s["unknown_threshold"]:
        category = UNKNOWN_CLASS
        trace.append(f"Combined confidence {conf:.2f} < unknown threshold {s['unknown_threshold']:.2f} → Unknown")

    # ---- overlap / uncertainty ------------------------------------------
    strong = [c for c, v in comb.items() if c != BACKGROUND_CLASS and v >= s["overlap_threshold"]]
    overlap = len(strong) >= 2
    if overlap:
        trace.append(f"Overlapping sounds: {', '.join(strong)} all ≥ {s['overlap_threshold']:.2f}")

    uncertain_reasons = []
    if conf < s["min_confidence"]:
        uncertain_reasons.append("Confidence below threshold")
    if margin < s["top2_margin"]:
        uncertain_reasons.append("Top classes have similar confidence")
    if quality in ("Poor", "Unusable"):
        uncertain_reasons.append(f"Audio quality is {quality}")
    if gtm and not cmp["class_match"]:
        uncertain_reasons.append("Models disagree")
    if not gtm:
        uncertain_reasons.append("GTM prediction unavailable")
    if overlap:
        uncertain_reasons.append("Overlapping sounds detected")
    uncertain = bool(uncertain_reasons)

    # ---- rules -----------------------------------------------------------
    rule = rules.get(category) or default_rule(category)
    repeated = repeated_for(category) if category != UNKNOWN_CLASS else 1
    agree = bool(gtm) and cmp["class_match"] and cmp["python_prediction"] == category
    ev = evaluate_rule(rule, confidence=conf, margin=margin, quality=quality,
                       models_agree=agree, repeated=repeated)
    severity = ev["severity"]
    trace.append(f"Rule '{category}': " + ", ".join(f"{k}={'✔' if v else '✘'}" for k, v in ev["checks"].items())
                 + f" (repeated {repeated}×)")

    alert_status = "No Alert"
    action = rule["recommended_action"]
    review_reasons = []

    if category == BACKGROUND_CLASS and noise_db is not None and noise_db > s["background_noise_limit_db"]:
        severity = rule.get("noise_alert_severity", "Medium")
        action = rule.get("noise_alert_action", action)
        alert_status = "Alert Generated"
        trace.append(f"Noise level {noise_db:.1f} dBFS > limit {s['background_noise_limit_db']:.1f} → noise alert")
    elif rule["generate_alert"]:
        if ev["confirmed"]:
            alert_status = "Alert Generated"
            trace.append(f"All conditions met → {severity} alert" + (" (escalated)" if ev["escalated"] else ""))
        else:
            alert_status = "Pending Confirmation"
            failed = [k for k, v in ev["checks"].items() if not v]
            trace.append("Not confirmed (" + ", ".join(failed) + ") → no automatic alert")
            if rule.get("manual_review_if_unconfirmed") and conf >= s["unknown_threshold"]:
                review_reasons.append("Critical event without sufficient confirmation"
                                      if severity in ("High", "Critical") else "Event not confirmed by rules")

    # possible false alarm: an alert for a class the other model barely supports
    if alert_status == "Alert Generated" and gtm and category != BACKGROUND_CLASS:
        weakest = min(py.get(category, 0), gtm.get(category, 0))
        if weakest < 0.20:
            review_reasons.append("Possible false alarm (one model gives this class < 0.20)")

    # ---- manual-review routing (SRS Step 17) ---------------------------
    if gtm and not cmp["class_match"]:
        review_reasons.append("Different predictions from the two models")
    if conf < s["min_confidence"]:
        review_reasons.append("Low confidence")
    if quality in ("Poor", "Unusable"):
        review_reasons.append("Poor audio quality")
    if margin < s["top2_margin"]:
        review_reasons.append("Similar top-class confidence scores")
    if overlap:
        review_reasons.append("Overlapping sounds")
    if category == UNKNOWN_CLASS:
        review_reasons.append("Unsupported / unknown sound pattern")
    # Background noise that is clearly background noise needs no human.
    if category == BACKGROUND_CLASS and alert_status == "No Alert":
        review_reasons = [r for r in review_reasons if r in ("Different predictions from the two models",)]
    review_reasons = list(dict.fromkeys(review_reasons))
    manual = bool(review_reasons)

    if manual:
        status = "Manual Review"
    elif alert_status == "Alert Generated":
        status = "Alert Generated"
    elif uncertain:
        status = "Uncertain"
    else:
        status = "Classified"
    trace.append(f"Final: {category} | severity {severity} | {alert_status} | status {status}")

    return {
        **cmp,
        "combined_scores": comb, "final_category": category, "final_confidence": conf,
        "combined_margin": margin, "confidence_level": confidence_level(conf, s),
        "overlap_detected": overlap, "overlap_classes": strong if overlap else [],
        "uncertain": uncertain, "uncertain_reasons": uncertain_reasons,
        "repeated_count": repeated, "severity": severity, "alert_status": alert_status,
        "recommended_action": action, "manual_review_required": manual, "review_reasons": review_reasons,
        "status": status, "rule_checks": ev["checks"], "audience": rule.get("audience", []),
        "trace": trace,
    }


def severity_rank(sev: str) -> int:
    return SEVERITY_LEVELS.index(sev) if sev in SEVERITY_LEVELS else 0
