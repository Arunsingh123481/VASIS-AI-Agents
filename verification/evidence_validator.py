"""
Evidence Validator — Step 5 of the Verification Retrieval Pipeline.

Applies an entity-level gate before an atom can be used as evidence.

Rules
-----
1. Evidence MUST contain the main claim entity.
2. Evidence MUST contain at least 1 of the required relation/target terms.
3. If fewer than MIN_OVERLAP terms are satisfied → reject.
4. Return (valid_evidence, rejection_log).
"""

from typing import Dict, List, Tuple


# Minimum number of required-terms that must match in addition to entity
MIN_REQUIRED_TERM_HITS = 1


def _atom_text(atom: Dict) -> str:
    return (atom.get("text") or "").lower()


def validate_evidence(
    atoms: List[Dict],
    claim_entities: Dict,
    min_final_score: float = 0.0,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter atoms to only those that pass strict entity validation.

    Rules:
    1. Only show supporting evidence if it shares the main claim entity
       and at least one relation/target term with the claim.
    """
    main_entity = (claim_entities.get("main_entity") or "").lower().strip()
    entity_parts = [w for w in main_entity.split() if len(w) > 3]

    # Collect relation/target terms
    relation_target_terms = []
    if claim_entities.get("relation"):
        relation_target_terms.append(claim_entities["relation"].lower().strip())
    
    if claim_entities.get("target"):
        stop = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "with", "by", "its", "only", "that", "this"}
        for w in claim_entities["target"].split():
            w_clean = "".join(c for c in w.lower() if c.isalnum())
            if w_clean and w_clean not in stop and len(w_clean) > 2:
                relation_target_terms.append(w_clean)

    for t in claim_entities.get("required_terms", []):
        relation_target_terms.append(t.lower())

    relation_target_terms = list(set([t for t in relation_target_terms if t and t != main_entity]))

    valid: List[Dict] = []
    rejected: List[Dict] = []

    for atom in atoms:
        text = _atom_text(atom)
        reasons = []

        # ── Gate 1: main entity must appear ──────────────────────────────
        if main_entity:
            entity_found = (
                main_entity in text
                or any(part in text for part in entity_parts)
            )
            if not entity_found:
                reasons.append(f"missing main entity '{main_entity}'")

        # ── Gate 2: at least one relation/target term must appear ────────
        if relation_target_terms:
            term_hits = [t for t in relation_target_terms if t in text]
            if not term_hits:
                reasons.append(
                    f"missing relation/target terms (need 1 of: {relation_target_terms[:5]})"
                )

        # ── Gate 3: final score threshold ────────────────────────────────
        if min_final_score > 0:
            score = atom.get("_final_score", atom.get("_prescore", 0.0))
            if score < min_final_score:
                reasons.append(f"score {score:.3f} < threshold {min_final_score:.3f}")

        if reasons:
            a = atom.copy()
            a["_rejection_reason"] = "; ".join(reasons)
            rejected.append(a)
        else:
            valid.append(atom)

    return valid, rejected
