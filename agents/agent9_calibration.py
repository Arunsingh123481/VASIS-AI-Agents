# agents/agent9_calibration.py
# Confidence and trust calibrator agent — pure Python

from config import CONFIDENCE_THRESHOLD
from console_helper import print_msg

# Weight schema depending on query intent
# gap_weight   : penalty per context gap found during RE-MSE expansion
# conflict_weight: penalty per high-severity contradiction found
# gap_cap      : maximum total gap penalty (intent-specific — causal/procedural span many pages)
# conflict_cap : maximum total conflict penalty
WEIGHTS = {
    # ── Original intent types (from Agent1 LLM routing) ──
    "factual":      {"gap": 0.15, "conflict": 0.25, "gap_cap": 0.25, "conflict_cap": 0.40},
    "comparative":  {"gap": 0.10, "conflict": 0.08, "gap_cap": 0.20, "conflict_cap": 0.25},
    "definitional": {"gap": 0.05, "conflict": 0.30, "gap_cap": 0.15, "conflict_cap": 0.30},
    "procedural":   {"gap": 0.06, "conflict": 0.15, "gap_cap": 0.15, "conflict_cap": 0.25},
    "causal":       {"gap": 0.06, "conflict": 0.10, "gap_cap": 0.15, "conflict_cap": 0.20},
    # ── Smart routing query types ──
    "summary":      {"gap": 0.02, "conflict": 0.05, "gap_cap": 0.08, "conflict_cap": 0.10},
    "methodology":  {"gap": 0.06, "conflict": 0.15, "gap_cap": 0.15, "conflict_cap": 0.25},
    "results":      {"gap": 0.10, "conflict": 0.20, "gap_cap": 0.20, "conflict_cap": 0.35},
    "limitations":  {"gap": 0.05, "conflict": 0.08, "gap_cap": 0.10, "conflict_cap": 0.15},
    "bibliography": {"gap": 0.01, "conflict": 0.02, "gap_cap": 0.05, "conflict_cap": 0.05},
    "explanation":  {"gap": 0.05, "conflict": 0.15, "gap_cap": 0.15, "conflict_cap": 0.25},
    "novelty":      {"gap": 0.06, "conflict": 0.08, "gap_cap": 0.15, "conflict_cap": 0.20},
    "timeline":     {"gap": 0.04, "conflict": 0.10, "gap_cap": 0.10, "conflict_cap": 0.20},
    "verification": {"gap": 0.12, "conflict": 0.25, "gap_cap": 0.25, "conflict_cap": 0.40},
    "deep_research":{"gap": 0.08, "conflict": 0.10, "gap_cap": 0.20, "conflict_cap": 0.25},
    # ── Fallback ──
    "unknown":      {"gap": 0.15, "conflict": 0.20, "gap_cap": 0.25, "conflict_cap": 0.35},
}

def calibrate(
    validation: dict,
    contradiction: dict,
    expansion: dict,
    temporal: dict,
    intent: str
) -> dict:
    """Calculate mathematical confidence and trust metrics based on validation, gaps, and contradictions."""
    base = float(validation.get("confidence_score", 0.5))
    w    = WEIGHTS.get(intent, WEIGHTS["unknown"])

    # Strict floor for out-of-scope or rejected validations
    if base == 0.0 or validation.get("verdict") == "ungrounded" or validation.get("recommendation") == "reject":
        print_msg("[Agent9] Out-of-scope or ungrounded validation detected. Forcing score to 0.0 and trust to low.")
        return {
            "base_score":       0.0,
            "calibrated_score": 0.0,
            "trust_level":      "low",
            "gap_penalty":      0.0,
            "conflict_penalty": 0.0,
            "temporal_bonus":   0.0,
            "requery_penalty":  0.0
        }

    # Gap Penalty — uses per-intent cap so causal/procedural queries aren't crushed
    gap_cap = w.get("gap_cap", 0.25)
    gap_penalty = min(gap_cap, float(expansion.get("gap_count", 0)) * w["gap"])

    # Conflict Penalty — uses per-intent cap
    conflict_cap = w.get("conflict_cap", 0.35)
    high_severity_conflicts = 0
    for c in contradiction.get("llm_contradictions", []):
        if isinstance(c, dict):
            try:
                if c.get("severity") == "high":
                    high_severity_conflicts += 1
            except Exception:
                pass
    # Add structural triple conflicts to severity counting
    high_severity_conflicts += len(contradiction.get("triple_conflicts", []))
    conflict_penalty = min(conflict_cap, high_severity_conflicts * w["conflict"])

    # Temporal Bonus (rewarding chronology alignment)
    temporal_bonus = 0.05 if temporal.get("time_ordered") else 0.0

    # Requery Penalty (capped at 0.10 to prevent cascading punishment)
    requery_count = int(validation.get("requery_count", 0))
    requery_penalty = min(0.10, 0.05 * requery_count)

    # Calibrate final score
    calibrated = base - gap_penalty - conflict_penalty + temporal_bonus - requery_penalty
    calibrated = max(0.0, min(1.0, calibrated))

    # Evaluate trust level string
    if calibrated >= 0.75:
        trust = "high"
    elif calibrated >= CONFIDENCE_THRESHOLD:
        trust = "medium"
    else:
        trust = "low"

    print_msg(f"[Agent9] Calibration: base_score={base:.2f} -> calibrated_score={calibrated:.2f} | trust={trust}")

    return {
        "base_score":       base,
        "calibrated_score": round(calibrated, 3),
        "trust_level":      trust,
        "gap_penalty":      gap_penalty,
        "conflict_penalty": conflict_penalty,
        "temporal_bonus":   temporal_bonus,
        "requery_penalty":  requery_penalty
    }
