"""
Paper-Level Filter — Step 2 of the Verification Retrieval Pipeline.

Before doing atom-level search, identify which sessions (papers) are
most likely to contain evidence for the given claim.

Strategy:
  - Score each session's atoms for entity/keyword overlap.
  - Return ranked list of (session_id, rag, score).
  - Always keep top-N even if score is 0 (fallback).
"""

from typing import Dict, List, Tuple, Any


def _text_keyword_score(text: str, keywords: List[str]) -> float:
    """Count how many of `keywords` appear in `text` (case-insensitive)."""
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def filter_papers(
    sessions: Dict[str, Any],          # { session_id: rag_instance }
    claim_entities: Dict,               # output of claim_extractor.extract_claim_entities()
    top_n: int = 3,
) -> List[Tuple[str, Any, float]]:
    """
    Rank sessions by relevance to the claim entities.

    Returns:
        List of (session_id, rag_instance, relevance_score) sorted desc.
        At minimum returns all sessions if fewer than top_n exist.
    """
    keywords = claim_entities.get("all_keywords", [])
    main_entity = (claim_entities.get("main_entity") or "").lower()
    required_terms = [t.lower() for t in claim_entities.get("required_terms", [])]

    ranked: List[Tuple[str, Any, float]] = []

    for sid, rag in sessions.items():
        atoms = getattr(rag, "atoms", [])
        if not atoms:
            ranked.append((sid, rag, 0.0))
            continue

        # Sample up to 200 atoms to keep scoring fast
        sample = atoms[:200]
        combined_text = " ".join(a.get("text", "") for a in sample).lower()

        # Main entity hit (heavy weight)
        entity_score = 3.0 if main_entity and main_entity in combined_text else 0.0

        # Required terms coverage
        term_hits = sum(1 for t in required_terms if t in combined_text)
        term_score = term_hits * 1.5

        # General keyword density
        kw_hits = _text_keyword_score(combined_text, keywords)
        kw_score = min(kw_hits * 0.5, 5.0)  # cap to avoid bias toward huge papers

        total = entity_score + term_score + kw_score
        ranked.append((sid, rag, total))

    # Sort descending by score
    ranked.sort(key=lambda x: x[2], reverse=True)

    # Always return at least top_n (or all if fewer)
    n = max(top_n, 1)
    return ranked[:n] if len(ranked) >= n else ranked
