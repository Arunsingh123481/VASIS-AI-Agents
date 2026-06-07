# agents/agent5_expansion.py
# Phase 12 Adaptive Stopping RE-MSE — Pure Python stateful expansion

from config import (ADAPTIVE_MAX_RADIUS,
                    ADAPTIVE_MIN_RELEVANCE,
                    ADAPTIVE_MIN_NEW_ATOMS,
                    ADAPTIVE_RADIUS_STEP,
                    MAX_NARRATIVE_CHARS,
                    GAP_THRESHOLD)
from console_helper import print_msg
import re

# Stop words to clean up terms
STOP_WORDS = {
    "what", "how", "the", "a", "an", "is", "are", "was", "were", 
    "in", "on", "at", "to", "for", "of", "with", "and", "or", "from", "by"
}

def _get_clean_words(text: str) -> set[str]:
    """Lowercase, remove punctuation, and extract unique tokens excluding stopwords."""
    words = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}


def _score_relevance(nb_text: str, anchor_words: set[str], query_words: set[str]) -> float:
    """Evaluate lexical overlap relevance between neighbor atom and combined anchor/query terms."""
    nb_words = _get_clean_words(nb_text)
    if not nb_words:
        return 0.0
        
    combined_targets = anchor_words | query_words
    if not combined_targets:
        return 0.0

    # Jaccard-like overlap score
    intersection = nb_words & combined_targets
    score = len(intersection) / len(combined_targets)
    return score


def expand(anchor_atoms: list[dict], atom_store, query: str = "", is_bibliography: bool = False) -> dict:
    """
    Perform patented RE-MSE stateful multi-segment expansion.
    Uses Adaptive Stopping (Phase 12) to dynamically grow expansion radius based on neighbor relevance.
    """
    # Seed expansion state cache with anchors
    state = {int(a["atom_id"]): a for a in anchor_atoms}
    
    # Pre-calculate term filters
    query_words = _get_clean_words(query)
    anchor_words = set()
    for a in anchor_atoms:
        anchor_words.update(_get_clean_words(a["text"]))

    radius = 1
    consecutive_empty = 0
    passes_run = 0

    print_msg(f"[Agent5] Starting Adaptive Stopping RE-MSE expansion. Anchors: {list(state.keys())}")

    # Stateful multi-pass expansion loop
    while radius <= ADAPTIVE_MAX_RADIUS:
        new_atoms_added = 0
        passes_run += 1
        
        # Iterate over all accumulated atoms in the State Cache
        current_state_ids = list(state.keys())
        for atom_id in current_state_ids:
            # Query in-memory atom store for neighbors within the current radius
            for nb in atom_store.get_neighbours(atom_id, radius):
                nb_id = int(nb["atom_id"])
                if nb_id not in state:
                    # Evaluate relevance to query/anchors
                    relevance = _score_relevance(nb["text"], anchor_words, query_words)
                    
                    if relevance >= ADAPTIVE_MIN_RELEVANCE:
                        state[nb_id] = nb
                        new_atoms_added += 1
                        
        if new_atoms_added < ADAPTIVE_MIN_NEW_ATOMS:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                # Early stop: Jaccard overlap drops off or neighbors are irrelevant
                print_msg(f"  -> Adaptive expansion stopped early at radius {radius} (no new relevant neighbors found).")
                break
        else:
            consecutive_empty = 0
            
        radius += ADAPTIVE_RADIUS_STEP

    # Gather expanded list and reconstruct text
    sorted_atoms = sorted(state.values(), key=lambda a: int(a["atom_id"]))
    
    parts = []
    gaps  = 0
    for i, atom in enumerate(sorted_atoms):
        if i > 0:
            diff = int(atom["atom_id"]) - int(sorted_atoms[i-1]["atom_id"])
            if diff > GAP_THRESHOLD:
                if not is_bibliography:
                    parts.append("\n[CONTEXT GAP — passages omitted]\n")
                gaps += 1
        parts.append(atom["text"])

    narrative = " ".join(parts)
    if not is_bibliography and len(narrative) > MAX_NARRATIVE_CHARS:
        narrative = narrative[:MAX_NARRATIVE_CHARS] + "\n...[truncated for length]"

    pages = sorted(set(int(a.get("page_number", a.get("page_num", 1))) for a in sorted_atoms))
    
    print_msg(f"[Agent5] Expansion finished after {passes_run} passes. Reconstructed {len(sorted_atoms)} atoms across {len(pages)} pages (found {gaps} context gaps).")
    
    return {
        "narrative":        narrative,
        "atom_ids_used":    [int(a["atom_id"]) for a in sorted_atoms],
        "pages_referenced": pages,
        "gap_count":        gaps,
        "atom_count":       len(sorted_atoms)
    }
