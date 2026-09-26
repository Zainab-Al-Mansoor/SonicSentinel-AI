"""Analytics for the admin dashboard and the model-comparison report."""
from collections import Counter

import numpy as np
import pandas as pd

from config.settings import CLASSES
from ..models import AudioEvent, Alert, Review


def summary(events: list[AudioEvent]) -> dict:
    ev = [e for e in events if e.final_category]
    n = len(ev)
    alerts = Alert.query.all()
    handled = [a for a in alerts if a.handled_at]
    resp = [(a.handled_at - a.created_at).total_seconds() for a in handled]
    reviews = Review.query.all()
    fp = Counter(r.original_category for r in reviews if r.decision == "Corrected")     # model said X, wrong
    fn = Counter(r.decided_category for r in reviews if r.decision == "Corrected")      # truly Y, missed
    for e in ev:                                                                         # evaluation ground truth
        if e.actual_class and e.final_category != e.actual_class:
            fp[e.final_category] += 1
            fn[e.actual_class] += 1
    return {
        "total": len(events), "classified": n,
        "critical_alerts": sum(1 for a in alerts if a.severity == "Critical"),
        "active_alerts": sum(1 for a in alerts if a.status == "Active"),
        "avg_confidence": float(np.mean([e.final_confidence for e in ev])) if ev else 0.0,
        "disagreements": sum(1 for e in ev if e.consistency_status == "Model Disagreement"),
        "agreement_rate": (sum(1 for e in ev if e.class_match) / max(1, sum(1 for e in ev if e.class_match is not None))),
        "poor_quality": sum(1 for e in events if e.quality_label in ("Poor", "Unusable")),
        "manual_review": sum(1 for e in events if e.status == "Manual Review"),
        "reviewed": sum(1 for e in events if e.reviewer_id),
        "alert_response_avg_s": float(np.mean(resp)) if resp else None,
        "alert_status": Counter(a.status for a in alerts),
        "false_positives": dict(fp), "false_negatives": dict(fn),
        "severity": Counter(e.severity for e in ev),
    }


def _top2(d: dict):
    t = sorted((d or {}).items(), key=lambda kv: -kv[1])
    t += [("-", 0.0)] * (2 - len(t))
    return t[:2]


def _explain(e: AudioEvent, py: dict, g: dict, final: str) -> str:
    """Plain-language explanation for a MAJOR disagreement: different classes, or the same class with a large
    confidence gap (|Δ| ≥ 0.40)."""
    pt, gt = _top2(py), _top2(g)
    extra = f" Quality {e.quality_label}" + ("; overlapping sounds." if e.overlap_detected else ".")
    if e.class_match is False:
        who = ""
        if e.actual_class:
            pr, gr = e.python_prediction == e.actual_class, e.gtm_prediction == e.actual_class
            who = (" Python was right." if pr and not gr else " GTM was right." if gr and not pr
                   else " Neither model was right." if not (pr or gr) else "")
        return (f"Different classes: Python favoured {pt[0][0]} ({pt[0][1]:.2f}, 2nd {pt[1][0]} {pt[1][1]:.2f}); "
                f"GTM favoured {gt[0][0]} ({gt[0][1]:.2f}, 2nd {gt[1][0]} {gt[1][1]:.2f}). "
                f"Final decision {final}.{who}{extra}")
    if e.class_match and e.confidence_diff is not None and e.confidence_diff >= 0.40:
        weaker, strong = ("GTM", "Python") if (e.gtm_confidence or 0) < (e.python_confidence or 0) else ("Python", "GTM")
        return (f"Same class ({e.python_prediction}) but a large confidence gap of {e.confidence_diff:.2f}: "
                f"{weaker} was much less sure than {strong} (1-second GTM window vs 2-second Python segment, "
                f"or the sound is only partly in the window).{extra}")
    return ""


def comparison_dataframe(events: list[AudioEvent]) -> tuple[pd.DataFrame, dict]:
    """Rows required by SRS deliverable 6 (Model Prediction and Confidence Comparison Report)."""
    rows = []
    for e in events:
        if not e.final_category:
            continue
        py, g, comb = e.python_scores or {}, e.gtm_scores or {}, e.combined_scores or {}
        final = e.reviewed_category or e.final_category
        ct = _top2(comb)
        row = {"Audio ID": e.audio_id, "Filename": e.original_filename, "Actual class": e.actual_class or "",
               "Python predicted class": e.python_prediction}
        row.update({f"Python conf: {c}": round(py.get(c, 0.0), 4) for c in CLASSES})
        row["GTM predicted class"] = e.gtm_prediction
        row.update({f"GTM conf: {c}": round(g.get(c, 0.0), 4) for c in CLASSES})
        row.update({
            "Class match": "Yes" if e.class_match else ("No" if e.class_match is False else "n/a"),
            "Top-class confidence difference": round(e.confidence_diff, 4) if e.confidence_diff is not None else None,
            "Python top-two margin": round(e.python_margin or 0, 4),
            "GTM top-two margin": round(e.gtm_margin, 4) if e.gtm_margin is not None else None,
            "Final top-two margin": round(ct[0][1] - ct[1][1], 4) if comb else None,
            "Consistency status": e.consistency_status, "Audio quality": e.quality_label, "Severity": e.severity,
            "Alert status": e.alert_status, "Manual review": "Yes" if e.manual_review_required else "No",
            "Final decision": final,
            "Final confidence": round(e.final_confidence, 4) if e.final_confidence is not None else None,
            "Correct": ("Yes" if final == e.actual_class else "No") if e.actual_class else "",
            "Result": ("Correct" if final == e.actual_class else "Incorrect") if e.actual_class else "",
            "Python correct": ("Yes" if e.python_prediction == e.actual_class else "No") if e.actual_class else "",
            "GTM correct": ("Yes" if e.gtm_prediction == e.actual_class else "No") if e.actual_class and e.gtm_prediction else "",
            "Explanation of disagreement": _explain(e, py, g, final),
        })
        rows.append(row)
    df = pd.DataFrame(rows)
    s = {}
    if not df.empty:
        labelled = df[df["Actual class"] != ""]
        counts = labelled["Actual class"].value_counts().to_dict() if len(labelled) else {}
        per_class = {}
        for c in CLASSES:
            sub = labelled[labelled["Actual class"] == c]
            if len(sub):
                per_class[c] = {"recordings": int(len(sub)),
                                "python_accuracy": float((sub["Python correct"] == "Yes").mean()),
                                "gtm_accuracy": float((sub["GTM correct"] == "Yes").mean()),
                                "final_accuracy": float((sub["Correct"] == "Yes").mean()),
                                "agreement": float((sub["Class match"] == "Yes").mean())}
        dis = df[df["Class match"] == "No"]
        s = {
            "records": len(df), "with_ground_truth": len(labelled),
            "per_class_counts": counts,
            "meets_srs_size": bool(len(labelled) >= 100 and all(counts.get(c, 0) >= 10 for c in CLASSES)),
            "python_accuracy": float((labelled["Python correct"] == "Yes").mean()) if len(labelled) else None,
            "gtm_accuracy": float((labelled["GTM correct"] == "Yes").mean()) if len(labelled) else None,
            "final_accuracy": float((labelled["Correct"] == "Yes").mean()) if len(labelled) else None,
            "agreement_rate": float((df["Class match"] == "Yes").mean()),
            "disagreements": int(len(dis)),
            "disagreements_python_right": int(((dis["Python correct"] == "Yes") & (dis["GTM correct"] != "Yes")).sum()),
            "disagreements_gtm_right": int(((dis["GTM correct"] == "Yes") & (dis["Python correct"] != "Yes")).sum()),
            "mean_confidence_difference": float(df["Top-class confidence difference"].dropna().mean())
            if df["Top-class confidence difference"].notna().any() else None,
            "manual_review_rate": float((df["Manual review"] == "Yes").mean()),
            "alerts_generated": int((df["Alert status"] == "Alert Generated").sum()),
            "per_class": per_class,
        }
    return df, s
