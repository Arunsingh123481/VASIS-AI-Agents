# agents/agent4_retrieval.py
# Scoped vectorless retriever agent — pure Python

from db.bm25_index import BM25Index
from db.triple_store import TripleStore
from config import TOP_K_ANCHORS
from console_helper import print_msg

def retrieve(
    query: str,
    selected_nodes: list[str],
    tree: list[dict],
    atom_store,
    bm25_index: BM25Index,
    triple_store: TripleStore,
    warm_atom_ids: list[int] = None,
    routed: dict = None
) -> list[dict]:
    """Retrieve top anchor atoms from validated sections using BM25 & structural triple matches."""

    # ── Bibliography Fast-Path ──────────────────────────────────────────────
    # If Agent1 flagged this as a references/citations query, use a 3-tier
    # search strategy rather than blindly targeting the last N pages.
    #
    # Tier 1 — Tree-node scan: look for a section explicitly labelled
    #           "references", "bibliography", or "works cited" in the tree.
    #           Academic papers always have this as a named section.
    #
    # Tier 2 — Atom keyword scan: look for atoms that contain dense citation
    #           patterns (numbers in square brackets like [1], [2] followed by
    #           author/year text).  Catches inline reference lists.
    #
    # Tier 3 — Last-N-pages fallback: only if tiers 1 and 2 both fail.
    #
    # If all tiers fail, we return an empty list with a special sentinel so the
    # orchestrator can emit a grounded "no formal bibliography found" answer
    # instead of hallucinating references from figure-caption numbers.
    if routed and (routed.get("is_bibliography_query") or routed.get("is_bibliography")):
        BIB_TITLE_KEYWORDS = {
            "references", "bibliography", "works cited", "reference list",
            "citations", "cited works", "literature"
        }

        # ── Tier 1: Tree node title/summary/topics keyword scan ─────────────
        bib_node = None
        for node in tree:
            node_text = " ".join([
                node.get("title", ""),
                node.get("summary", ""),
                " ".join(node.get("key_topics", []))
            ]).lower()
            if any(kw in node_text for kw in BIB_TITLE_KEYWORDS):
                bib_node = node
                break  # take the first (and usually only) match

        if bib_node:
            bib_atoms = atom_store.get_in_page_range(bib_node["start_page"], bib_node["end_page"])
            if bib_atoms:
                for a in bib_atoms:
                    a["combined_score"] = 1.0
                print_msg(
                    f"[Agent4] Bibliography mode Tier-1: found references section "
                    f"'{bib_node['title']}' (pages {bib_node['start_page']}–{bib_node['end_page']}). "
                    f"Returning {len(bib_atoms)} atoms."
                )
                return bib_atoms

        # ── Tier 1.5 & Tier 2: End-of-document scanning ─────────────────────
        import re as _re
        all_atoms = sorted(atom_store.all(), key=lambda x: int(x["atom_id"]))
        search_start = int(len(all_atoms) * 0.70)  # Only look at the last 30% of the document
        tail_candidates = all_atoms[search_start:]

        # Tier 1.5: Look for an explicit standalone "References" header atom
        header_pattern = _re.compile(r'^\s*(?:\d+\.?\s*)?(references|bibliography|works cited|literature cited)\s*$', _re.IGNORECASE)
        for i, atom in enumerate(tail_candidates):
            if header_pattern.match(atom.get("text", "")):
                bib_atoms = tail_candidates[i+1:]
                if bib_atoms:
                    for a in bib_atoms: a["combined_score"] = 1.0
                    print_msg(f"[Agent4] Bibliography mode Tier-1.5: found explicit references header at atom {atom['atom_id']}. Returning {len(bib_atoms)} atoms to end of document.")
                    return bib_atoms

        # Tier 2: Atom-level dense citation pattern scan
        # Pattern: [N] | N. Author | (Author, Year) | Author, A. (Year).
        _CITE_PATTERN = _re.compile(r'^\s*\[\d+\]\s+\w|^\s*\(\w+,\s*\d{4}\)|^\s*\d+\.\s+\w|^\s*[A-Z][A-Za-z\-]+,.*?\((?:19|20)\d{2}\)')
        
        cite_indices = [
            i for i, a in enumerate(tail_candidates)
            if _CITE_PATTERN.search(a.get("text", ""))
        ]
        
        if len(cite_indices) >= 3:
            # Found a cluster of citations. Return everything from the FIRST citation to the end of the document!
            # This ensures we don't drop references that missed the strict regex.
            first_idx = cite_indices[0]
            bib_atoms = tail_candidates[first_idx:]
            
            for a in bib_atoms:
                a["combined_score"] = 1.0
                
            print_msg(
                f"[Agent4] Bibliography mode Tier-2: found citation block in tail section. "
                f"Extracting contiguous block of {len(bib_atoms)} atoms to end of document."
            )
            return bib_atoms

        # ── Tier 3: Last-N-pages fallback ───────────────────────────────────
        tail_atoms = atom_store.get_last_n_pages(n=2)
        if tail_atoms:
            print_msg(
                f"[Agent4] Bibliography mode Tier-3 (fallback): returning {len(tail_atoms)} atoms "
                f"from last 2 pages. No dedicated references section found in tree."
            )
            # Attach a sentinel so the orchestrator knows these may not be real refs
            for a in tail_atoms:
                a["_bib_fallback"] = True
            return tail_atoms

        print_msg("[Agent4] Bibliography mode: no atoms found at any tier — document may have no references section.")
        return []


    # ── Standard Retrieval ──────────────────────────────────────────────────
    node_map = {n["node_id"]: n for n in tree}
    scoped   = []
    
    # Restrict indexing scope to validated sections
    for nid in selected_nodes:
        node = node_map.get(nid)
        if node:
            scoped.extend(
                int(a["atom_id"]) for a in
                atom_store.get_in_page_range(
                    node["start_page"], node["end_page"]
                )
            )
            
    scoped = list(set(scoped))
    # Fallback to all atoms if no nodes are loaded/valid
    if not scoped:
        scoped = [int(a["atom_id"]) for a in atom_store.all()]
        
    # Warm start injects pre-learned successful atoms
    if warm_atom_ids:
        scoped = list(set(scoped + [int(x) for x in warm_atom_ids]))

    # Heuristic 1: Scoped BM25
    bm25_results = bm25_index.search_scoped(query, scoped, TOP_K_ANCHORS * 2)
    bm25 = {int(r["atom_id"]): float(r["bm25_score"]) for r in bm25_results}
    max_b = max(bm25.values()) if bm25 else 1.0

    # Heuristic 2: Scoped Triples Match
    t_ids = triple_store.get_atoms_for_query(query, scoped, TOP_K_ANCHORS * 2)
    triple = {
        int(aid): float((TOP_K_ANCHORS * 2 - i) / (TOP_K_ANCHORS * 2))
        for i, aid in enumerate(t_ids)
    }

    # Normalize and merge scores: 60% BM25 + 40% Triples
    combined = {}
    all_candidate_ids = set(bm25.keys()) | set(triple.keys())
    for aid in all_candidate_ids:
        # If the max BM25 score is very low (e.g. < 1.5), do not scale it up to 1.0.
        # This preserves the weak absolute match strength and flags out-of-scope queries.
        b_norm = (bm25.get(aid, 0.0) / max_b) if max_b >= 1.5 else (bm25.get(aid, 0.0) / 1.5)
        b_score = b_norm * 0.6
        t_score = triple.get(aid, 0.0) * 0.4
        combined[aid] = b_score + t_score

    # Rank and retrieve top-k anchors
    top_ids = sorted(combined, key=combined.get, reverse=True)[:TOP_K_ANCHORS]
    anchors = atom_store.get_many(top_ids)
    
    for a in anchors:
        a["combined_score"] = combined.get(int(a["atom_id"]), 0.0)

    print_msg(f"[Agent4] Vectorless retrieval complete. Selected {len(anchors)} anchor atoms from scoped candidate set.")
    return anchors
