# agents/agent8_temporal.py
# Temporal analyzer agent — uses local Qwen Coder (via router)

import re
from llm.router import generate_json
from console_helper import print_msg

SYSTEM = ("You are a temporal reasoning agent. "
          "Return ONLY valid JSON.")

PATTERNS = [
    r'\b(as of|previously|currently|updated|revised)\b',
    r'\b(since|until|before|after|during)\b',
    r'\b(20\d{2}|19\d{2})\b',
    r'\b(january|february|march|april|may|june|july|'
    r'august|september|october|november|december)\b',
    r'\b(latest|recent|current|former|previous|new|old)\b'
]

def analyze(narrative: str, atoms: list[dict]) -> dict:
    """Detect chronologies or timeline statements and order narrative chronologically."""
    has_temporal = any(re.search(p, narrative, re.IGNORECASE) for p in PATTERNS)
    
    if not has_temporal:
        return {
            "temporal_content_found": False,
            "ordered_narrative": narrative,
            "temporal_markers": [],
            "time_ordered": False
        }

    prompt = f"""Identify temporal markers and reorder
facts chronologically (oldest to newest).
Return JSON:
{{
  "temporal_markers": [
    {{"text":"...","type":"date|period|relative"}}
  ],
  "has_conflicting_timeframes": true or false,
  "chronological_summary": "facts oldest to newest",
  "most_recent_fact": "most current fact"
}}
Text: {narrative[:3000]}"""

    try:
        result = generate_json("agent8_temporal", prompt, system=SYSTEM)
        if not isinstance(result, dict):
            result = {}
    except Exception:
        result = {}

    # Set default values if keys are missing or None
    if result.get("temporal_markers") is None:
        result["temporal_markers"] = []
    if result.get("has_conflicting_timeframes") is None:
        result["has_conflicting_timeframes"] = False
    if result.get("chronological_summary") is None:
        result["chronological_summary"] = narrative
    if result.get("most_recent_fact") is None:
        result["most_recent_fact"] = ""

    ordered = result.get("chronological_summary") or narrative
    markers = result.get("temporal_markers", [])

    print_msg(f"[Agent8] Chronological analyze: found={has_temporal} | markers_count={len(markers)} | conflicting={result['has_conflicting_timeframes']}")
    
    return {
        "temporal_content_found": True,
        "ordered_narrative":      ordered,
        "temporal_markers":       markers,
        "has_conflicting_timeframes": result.get("has_conflicting_timeframes", False),
        "most_recent_fact":       result.get("most_recent_fact", ""),
        "time_ordered":           True
    }
