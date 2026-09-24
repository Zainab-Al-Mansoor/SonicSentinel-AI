"""Unit tests: model comparison, uncertainty, overlap, alert rules, manual-review routing."""
from config.settings import CLASSES, DEFAULT_RUNTIME_SETTINGS
from src.services.decision import decide, compare_models
from src.services.rules import load_rules, validate_rule, quality_ok, evaluate_rule
from python_models.inference import aggregate_scores

S = dict(DEFAULT_RUNTIME_SETTINGS)
RULES = load_rules()


def scores(main, conf, second=None, conf2=0.0):
    rest = (1 - conf - conf2) / (len(CLASSES) - (2 if second else 1))
    d = {c: rest for c in CLASSES}
    d[main] = conf
    if second:
        d[second] = conf2
    return d


def run(py, gtm, quality="Good", repeated=1, noise=-60.0):
    return decide(py, gtm, quality=quality, noise_db=noise, repeated_for=lambda c: repeated, rules=RULES, settings=S)


def test_acceptable_match_confirmed_critical_alert():
    d = run(scores("Gunshot", 0.92), scores("Gunshot", 0.90))
    assert d["consistency_status"] == "Acceptable Match"
    assert d["alert_status"] == "Alert Generated" and d["severity"] == "Critical"
    assert not d["manual_review_required"] and d["status"] == "Alert Generated"


def test_confidence_difference_formula():
    c = compare_models(scores("Glass Breaking", 0.9), scores("Glass Breaking", 0.6), S)
    assert abs(c["confidence_diff"] - 0.3) < 1e-9 and c["consistency_status"] == "Weak Match"


def test_gunshot_needs_confirmation():
    # agree but below strong confidence and only one detection -> not confirmed -> manual review
    d = run(scores("Gunshot", 0.75), scores("Gunshot", 0.72), repeated=1)
    assert d["alert_status"] == "Pending Confirmation" and d["manual_review_required"]
    # the same event repeated in 2 consecutive windows -> confirmed
    d2 = run(scores("Gunshot", 0.75), scores("Gunshot", 0.72), repeated=2)
    assert d2["alert_status"] == "Alert Generated"


def test_model_disagreement_routes_to_review():
    d = run(scores("Gunshot", 0.8), scores("Glass Breaking", 0.7))
    assert d["consistency_status"] == "Model Disagreement"
    assert "Different predictions from the two models" in d["review_reasons"]
    assert d["status"] == "Manual Review"


def test_low_confidence_and_unknown():
    d = run(scores("Animal Sound", 0.3), scores("Vehicle Horn", 0.3))
    assert d["final_category"] == "Unknown" and d["manual_review_required"]
    assert d["consistency_status"] == "Uncertain Result"


def test_overlapping_sounds():
    d = run(scores("Glass Breaking", 0.5, "Alarm or Siren", 0.4), scores("Glass Breaking", 0.5, "Alarm or Siren", 0.4))
    assert d["overlap_detected"] and "Overlapping sounds" in d["review_reasons"]


def test_similar_top_two():
    d = run(scores("Panic Scream", 0.45, "Aggression", 0.40), scores("Panic Scream", 0.45, "Aggression", 0.40))
    assert "Similar top-class confidence scores" in d["review_reasons"]


def test_poor_quality_blocks_critical_alert():
    d = run(scores("Glass Breaking", 0.95), scores("Glass Breaking", 0.95), quality="Poor", repeated=3)
    assert d["alert_status"] != "Alert Generated" and "Poor audio quality" in d["review_reasons"]


def test_background_noise_quiet_vs_loud():
    quiet = run(scores("Background Noise", 0.9), scores("Background Noise", 0.9), noise=-50)
    assert quiet["alert_status"] == "No Alert" and not quiet["manual_review_required"]
    loud = run(scores("Background Noise", 0.9), scores("Background Noise", 0.9), noise=-10)
    assert loud["alert_status"] == "Alert Generated" and loud["severity"] == "Medium"


def test_non_critical_event_logged_only():
    d = run(scores("Vehicle Horn", 0.9), scores("Vehicle Horn", 0.9))
    assert d["alert_status"] == "No Alert" and d["severity"] == "Low" and d["status"] == "Classified"


def test_aggression_escalates_when_repeated():
    d = run(scores("Aggression", 0.9), scores("Aggression", 0.9), repeated=2)
    assert d["severity"] == "Critical"


def test_gtm_unavailable_is_uncertain():
    d = run(scores("Machinery Fault", 0.9), None, repeated=2)
    assert d["consistency_status"] == "Uncertain Result" and "GTM prediction unavailable" in d["uncertain_reasons"]


def test_possible_false_alarm():
    py = scores("Panic Scream", 0.99)
    gtm = scores("Panic Scream", 0.10, "Aggression", 0.8)
    d = run(py, gtm)
    assert d["manual_review_required"]


def test_rule_validation_and_quality_order():
    assert validate_rule(RULES["Gunshot"]) == []
    bad = dict(RULES["Gunshot"], severity="Huge", min_confidence=2)
    assert len(validate_rule(bad)) == 2
    assert quality_ok("Good", "Acceptable") and not quality_ok("Poor", "Acceptable")


def test_strong_confidence_single_detection():
    r = evaluate_rule(RULES["Gunshot"], confidence=0.9, margin=0.8, quality="Good", models_agree=True, repeated=1)
    assert r["confirmed"] and r["single_detection_accepted"]


def test_aggregation_keeps_short_event():
    bg = scores("Background Noise", 0.9)
    shot = scores("Gunshot", 0.8)
    agg, idx = aggregate_scores([bg, bg, shot, bg], 0.6)
    assert idx == 2 and max(agg, key=agg.get) == "Gunshot"
    agg2, idx2 = aggregate_scores([bg, bg], 0.6)
    assert idx2 == -1 and max(agg2, key=agg2.get) == "Background Noise"


def test_python_weight_changes_the_combined_score():
    from src.services.decision import combine
    py = {c: 0.0 for c in CLASSES}; py["Gunshot"] = 0.9; py["Background Noise"] = 0.1
    gtm = {c: 0.0 for c in CLASSES}; gtm["Glass Breaking"] = 0.6; gtm["Gunshot"] = 0.4
    assert abs(combine(py, gtm)["Gunshot"] - 0.65) < 1e-9                  # default: plain average
    assert abs(combine(py, gtm, 0.8)["Gunshot"] - 0.8) < 1e-9
    assert combine(py, None, 0.2)["Gunshot"] == 0.9                         # no GTM -> Python only
    assert abs(combine(py, gtm, 7)["Gunshot"] - 0.9) < 1e-9                 # clamped to 0..1


def _scores(**kw):
    d = {c: 0.0 for c in CLASSES}
    for k, v in kw.items():
        d[k.replace("_", " ")] = v
    return d


def _decide(py, gtm):
    s = dict(DEFAULT_RUNTIME_SETTINGS)
    return decide(py, gtm, quality="Good", noise_db=None, repeated_for=lambda c: 3, rules=load_rules(), settings=s)


def test_lookalike_check_distinguishes_clear_gunshot():
    d = _decide(_scores(Gunshot=0.95, Background_Noise=0.05), _scores(Gunshot=0.9, Background_Noise=0.1))
    chk = d["lookalike_checks"][0]
    assert chk["passed"] and "fireworks" in chk["lookalike"]
    assert not any(r.startswith("Possible look-alike") for r in d["review_reasons"])
    assert any(t.startswith("Look-alike check") and "distinguished" in t for t in d["trace"])


def test_lookalike_check_flags_close_call_for_review():
    d = _decide(_scores(Gunshot=0.55, Background_Noise=0.45), _scores(Gunshot=0.5, Background_Noise=0.5))
    assert d["final_category"] == "Gunshot"
    assert not d["lookalike_checks"][0]["passed"]
    assert d["manual_review_required"] and any("fireworks" in r for r in d["review_reasons"])


def test_alarm_vs_horn_and_hidden_event_behind_background():
    d = _decide(_scores(Alarm_or_Siren=0.52, Vehicle_Horn=0.44), _scores(Alarm_or_Siren=0.5, Vehicle_Horn=0.46))
    assert any("vehicle horn" in r for r in d["review_reasons"])
    d = _decide(_scores(Background_Noise=0.55, Gunshot=0.45), _scores(Background_Noise=0.5, Gunshot=0.4))
    assert d["final_category"] == "Background Noise"
    assert any(r.startswith("Possible Gunshot") for r in d["review_reasons"]) and d["manual_review_required"]
