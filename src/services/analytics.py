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


def comparison_dataframe(events: list[AudioEvent]) -> tuple[pd.DataFrame, dict]:
    """Rows required by SRS deliverable 6 (Model Prediction and Confidence Comparison Report)."""
    rows = []
    for e in events:
        if not e.final_category:
            continue
        py, g = e.python_scores or {}, e.gtm_scores or {}
        explanation = ""
        if e.class_match is False:
            pt = sorted(py.items(), key=lambda kv: -kv[1])[:2]
            gt = sorted(g.items(), key=lambda kv: -kv[1])[:2]
            explanation = (f"Python favoured {pt[0][0]} ({pt[0][1]:.2f}, 2nd {pt[1][0]} {pt[1][1]:.2f}); "
                           f"GTM favoured {gt[0][0]} ({gt[0][1]:.2f}, 2nd {gt[1][0]} {gt[1][1]:.2f}). "
                           f"Quality {e.quality_label}" + ("; overlapping sounds" if e.overlap_detected else ""))
        row = {"Audio ID": e.audio_id, "Filename": e.original_filename, "Actual class": e.actual_class or "",
               "Python predicted class": e.python_prediction}
        row.update({f"Python conf: {c}": round(py.get(c, 0.0), 4) for c in CLASSES})
        row["GTM predicted class"] = e.gtm_prediction
        row.update({f"GTM conf: {c}": round(g.get(c, 0.0), 4) for c in CLASSES})
        final = e.reviewed_category or e.final_category
        row.update({
            "Class match": "Yes" if e.class_match else ("No" if e.class_match is False else "n/a"),
            "Top-class confidence difference": round(e.confidence_diff, 4) if e.confidence_diff is not None else None,
            "Python top-two margin": round(e.python_margin or 0, 4),
            "GTM top-two margin": round(e.gtm_margin, 4) if e.gtm_margin is not None else None,
            "Consistency status": e.consistency_status, "Audio quality": e.quality_label, "Severity": e.severity,
            "Alert status": e.alert_status, "Manual review": "Yes" if e.manual_review_required else "No",
            "Final decision": final,
            "Correct": ("Yes" if final == e.actual_class else "No") if e.actual_class else "",
            "Python correct": ("Yes" if e.python_prediction == e.actual_class else "No") if e.actual_class else "",
            "GTM correct": ("Yes" if e.gtm_prediction == e.actual_class else "No") if e.actual_class and e.gtm_prediction else "",
            "Explanation of disagreement": explanation,
        })
        rows.append(row)
    df = pd.DataFrame(rows)
    s = {}
    if not df.empty:
        labelled = df[df["Actual class"] != ""]
        s = {
            "records": len(df), "with_ground_truth": len(labelled),
            "per_class_counts": labelled["Actual class"].value_counts().to_dict() if len(labelled) else {},
            "python_accuracy": float((labelled["Python correct"] == "Yes").mean()) if len(labelled) else None,
            "gtm_accuracy": float((labelled["GTM correct"] == "Yes").mean()) if len(labelled) else None,
            "final_accuracy": float((labelled["Correct"] == "Yes").mean()) if len(labelled) else None,
            "agreement_rate": float((df["Class match"] == "Yes").mean()),
            "mean_confidence_difference": float(df["Top-class confidence difference"].dropna().mean())
            if df["Top-class confidence difference"].notna().any() else None,
            "manual_review_rate": float((df["Manual review"] == "Yes").mean()),
        }
    return df, s
