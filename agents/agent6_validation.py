# agents/agent6_validation.py
# Factual grounding validation agent — uses local DeepSeek (via router)

from llm.router import generate_json
from utils.exceptions import RequerySignal
from console_helper import print_msg
from config import CONFIDENCE_THRESHOLD

SYSTEM = ("You are a factual validation agent. "
          "Return ONLY valid JSON.")


def _sanitize_str_field(value: str, allowed: list, default: str) -> str:
    """Take the first token from a pipe-delimited LLM response and validate it.
    e.g. 'grounded|partially_grounded|ungrounded' -> 'grounded'"""
    if not isinstance(value, str):
        return default
    first = value.split("|")[0].strip().lower()
    return first if first in allowed else default


def validate(answer: str, narrative: str, query: str, entities: list = None) -> dict:
    """Check every claim in the answer against context. Triggers requery signal on low confidence."""
    # Entity-based topical relevance check: if query entities are specified
    # but none are found in the narrative context, flag as out-of-scope.
    if entities:
        narrative_lower = narrative.lower()
        overlap = [ent for ent in entities if ent.lower() in narrative_lower]
        if not overlap:
            print_msg(f"[Agent6] Entity-based topical relevance check failed. No query entities {entities} found in narrative context.")
            return {
                "grounded_claims": [],
                "ungrounded_claims": [f"Query entities {entities} are not present in the document context."],
                "confidence_score": 0.0,
                "verdict": "ungrounded",
                "recommendation": "reject",
                "refined_query": query
            }

    prompt = f"""Check every claim in answer against context.
Query: {query}
Context: {narrative[:3000]}
Answer: {answer}
Return JSON with SINGLE values (not pipe-separated lists):
{{
  "grounded_claims": ["list of claims found in context"],
  "ungrounded_claims": ["list of claims NOT in context"],
  "confidence_score": 0.85,
  "verdict": "grounded",
  "recommendation": "accept",
  "refined_query": null
}}
verdict must be exactly one of: grounded, partially_grounded, ungrounded
recommendation must be exactly one of: accept, requery, reject

IMPORTANT: If the answer correctly states that the document does not contain, mention, or discuss certain information (a safe refusal), and you verify that the context indeed does not contain that information, you MUST classify that statement/refusal as a grounded claim. In such cases of correct refusal, the recommendation should be 'accept' and the verdict 'grounded' (do NOT trigger a requery or reject for honest, correct refusals)."""

    try:
        result = generate_json("agent6_validation", prompt, system=SYSTEM)
        if not isinstance(result, dict):
            result = {}
    except Exception:
        result = {}

    # Set default values if keys are missing or None
    if result.get("grounded_claims") is None:
        result["grounded_claims"] = []
    if result.get("ungrounded_claims") is None:
        result["ungrounded_claims"] = []
    if result.get("confidence_score") is None:
        result["confidence_score"] = 0.5

    # Sanitize pipe-delimited multi-value strings the LLM sometimes returns
    result["verdict"] = _sanitize_str_field(
        result.get("verdict", ""),
        ["grounded", "partially_grounded", "ungrounded"],
        "partially_grounded"
    )
    result["recommendation"] = _sanitize_str_field(
        result.get("recommendation", ""),
        ["accept", "requery", "reject"],
        "accept"
    )
    if result.get("refined_query") is None:
        result["refined_query"] = query

    # Confidence floor guard: never requery if confidence already meets threshold
    confidence = float(result.get("confidence_score", 0.5))
    if confidence >= CONFIDENCE_THRESHOLD and result["recommendation"] == "requery":
        print_msg(f"[Agent6] Confidence {confidence:.2f} >= threshold {CONFIDENCE_THRESHOLD} — overriding requery to accept.")
        result["recommendation"] = "accept"

    print_msg(f"[Agent6] verdict={result['verdict']} | confidence={confidence:.2f} | decision={result['recommendation']}")

    # Raise requery signal only if validation genuinely fails
    if result["recommendation"] == "requery":
        refined = result.get("refined_query") or query
        print_msg(f"[Agent6] LOW CONFIDENCE AUDIT. Raising RequerySignal with refined search: '{refined}'")
        raise RequerySignal(
            reason=f"Grounding validation audit failed: confidence={confidence:.2f}",
            refined_query=refined
        )

    return result
