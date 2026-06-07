"""
Cross-Encoder Reranker — Step 4 of the Verification Retrieval Pipeline.

Uses BGE-Reranker-v2-m3 (BAAI/bge-reranker-v2-m3) when available.
Falls back to a lightweight keyword-overlap scorer if the model can't be loaded
(e.g. first run, no GPU memory, offline environment).

Applies the final composite score:
  final_score = 0.45 * reranker_score
              + 0.25 * bm25_score
              + 0.20 * entity_score
              + 0.10 * vector_score
"""

import os
import warnings
from typing import List, Dict

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

_reranker = None
_reranker_failed = False   # don't retry after a load failure


def _get_reranker():
    global _reranker, _reranker_failed
    if _reranker is not None:
        return _reranker
    if _reranker_failed:
        return None
    try:
        from sentence_transformers import CrossEncoder
        print("[reranker] Loading BAAI/bge-reranker-v2-m3 …")
        _reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            max_length=512,
            device="cpu",        # safe default; GPU picked up automatically if available
        )
        print("[reranker] Reranker loaded.")
        return _reranker
    except Exception as e:
        print(f"[reranker] Could not load cross-encoder ({e}). Using keyword fallback.")
        _reranker_failed = True
        return None


def _keyword_fallback_score(claim: str, text: str) -> float:
    """Simple keyword overlap fraction as reranker substitute."""
    import re
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "are", "was", "were"}
    c_words = set(re.findall(r'[a-z]+', claim.lower())) - stop
    t_words = set(re.findall(r'[a-z]+', text.lower())) - stop
    if not c_words:
        return 0.5
    return len(c_words & t_words) / len(c_words)


def rerank(
    claim: str,
    candidates: List[Dict],
) -> List[Dict]:
    """
    Rerank `candidates` (output of hybrid_retrieve) against `claim`.

    Each candidate must have:
      _vector_score, _bm25_score, _entity_score, _prescore   (from hybrid_retriever)

    Returns candidates sorted by final composite score (desc),
    each annotated with _reranker_score and _final_score.
    """
    if not candidates:
        return []

    model = _get_reranker()
    texts = [c.get("text", "") for c in candidates]

    # ── Reranker scores ───────────────────────────────────────────────────
    if model is not None:
        pairs = [[claim, t] for t in texts]
        try:
            raw_scores = model.predict(pairs)
            # Sigmoid to bring into [0, 1]
            import math
            def _sigmoid(x):
                return 1.0 / (1.0 + math.exp(-float(x)))
            reranker_scores = [_sigmoid(s) for s in raw_scores]
        except Exception as e:
            print(f"[reranker] Inference failed ({e}). Using keyword fallback.")
            reranker_scores = [_keyword_fallback_score(claim, t) for t in texts]
    else:
        reranker_scores = [_keyword_fallback_score(claim, t) for t in texts]

    # ── Composite final score ─────────────────────────────────────────────
    reranked = []
    for i, atom in enumerate(candidates):
        a = atom.copy()
        rr = reranker_scores[i]
        a["_reranker_score"] = rr
        a["_final_score"] = (
            0.45 * rr
            + 0.25 * a.get("_bm25_score",   0.0)
            + 0.20 * a.get("_entity_score",  0.0)
            + 0.10 * a.get("_vector_score",  0.0)
        )
        reranked.append(a)

    reranked.sort(key=lambda x: x["_final_score"], reverse=True)
    return reranked
