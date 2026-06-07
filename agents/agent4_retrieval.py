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


    # ── Standard Retrieval (Weighted RRF: Vector + BM25 + Graph) ───────────
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

    # Number of candidates to fetch per layer before fusion
    FETCH_PER_LAYER = TOP_K_ANCHORS * 3

    # ── Layer 1: BM25 Lexical Search ─────────────────────────────────────
    bm25_results = bm25_index.search_scoped(query, scoped, FETCH_PER_LAYER)
    # Ranked list of atom IDs (position = rank)
    bm25_ranked = [int(r["atom_id"]) for r in bm25_results if r.get("bm25_score", 0) > 0]

    # ── Layer 2: Triple / Graph Match ────────────────────────────────────
    graph_ranked = [int(aid) for aid in
                    triple_store.get_atoms_for_query(query, scoped, FETCH_PER_LAYER)]

    # ── Layer 3: Dense Vector Similarity ─────────────────────────────────
    vector_ranked = _vector_rank(query, scoped, atom_store, FETCH_PER_LAYER)

    # ── Weighted Reciprocal Rank Fusion ──────────────────────────────────
    #
    # RRF_Score(d) = Σ  W_m / (k + r_m(d))
    #                m∈M
    #
    # Weights reflect layer strengths:
    #   Vector  = 1.0  (baseline — catches synonyms and semantic intent)
    #   BM25    = 1.2  (boosted — exact keyword / ID matching is precise)
    #   Graph   = 1.5  (heavily boosted — deterministic causal triples are
    #                    high-confidence when they match)
    #
    # k = 64 (industry standard smoothing constant)

    K = 64
    WEIGHTS = {"vector": 1.0, "bm25": 1.2, "graph": 1.5}

    rrf_scores: dict[int, float] = {}

    def _apply_rrf(ranked_ids: list[int], weight: float):
        for rank_idx, aid in enumerate(ranked_ids):
            rank = rank_idx + 1  # 1-based
            if aid not in rrf_scores:
                rrf_scores[aid] = 0.0
            rrf_scores[aid] += weight / (K + rank)

    _apply_rrf(vector_ranked, WEIGHTS["vector"])
    _apply_rrf(bm25_ranked,   WEIGHTS["bm25"])
    _apply_rrf(graph_ranked,  WEIGHTS["graph"])

    # ── Out-of-scope detection ───────────────────────────────────────────
    # If the best RRF score is extremely low, the query likely has no
    # meaningful match in the document.
    if rrf_scores:
        max_rrf = max(rrf_scores.values())
        # A well-matched atom appearing in all 3 layers at rank 1 would score
        # ~(1.0 + 1.2 + 1.5) / 65 ≈ 0.057.  A threshold of 0.005 flags truly
        # unmatched queries without false-flagging weak single-layer hits.
        if max_rrf < 0.005:
            print_msg("[Agent4] RRF scores uniformly near-zero — query may be out-of-scope.")
            return []

    # Rank and retrieve top-k anchors
    top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:TOP_K_ANCHORS]
    anchors = atom_store.get_many(top_ids)
    
    for a in anchors:
        a["combined_score"] = rrf_scores.get(int(a["atom_id"]), 0.0)

    print_msg(
        f"[Agent4] Weighted RRF retrieval complete (V={len(vector_ranked)}, "
        f"B={len(bm25_ranked)}, G={len(graph_ranked)} candidates). "
        f"Selected {len(anchors)} anchor atoms."
    )
    return anchors


# ── Vector similarity helper ────────────────────────────────────────────────

_embed_model = None

def _get_embed_model():
    """Lazy-load the embedding model (shared singleton)."""
    global _embed_model
    if _embed_model is None:
        import os, warnings
        warnings.filterwarnings("ignore")
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _vector_rank(query: str, scoped_ids: list[int], atom_store, top_k: int) -> list[int]:
    """Rank scoped atoms by dense cosine similarity to the query.
    Returns a list of atom IDs sorted by descending similarity.
    """
    import numpy as np

    scoped_atoms = atom_store.get_many(scoped_ids)
    if not scoped_atoms:
        return []

    texts = [a.get("text", "") for a in scoped_atoms]
    aids  = [int(a["atom_id"]) for a in scoped_atoms]

    model = _get_embed_model()
    q_emb = model.encode(query, convert_to_tensor=False)
    a_embs = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)

    # Cosine similarity
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)
    similarities = []
    for emb in a_embs:
        a_norm = emb / (np.linalg.norm(emb) + 1e-10)
        similarities.append(float(np.dot(q_norm, a_norm)))

    # Sort by similarity descending
    ranked = sorted(zip(aids, similarities), key=lambda x: x[1], reverse=True)
    return [aid for aid, _ in ranked[:top_k]]

