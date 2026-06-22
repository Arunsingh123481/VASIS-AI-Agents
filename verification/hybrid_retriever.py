"""
Hybrid Retriever — Step 3 of the Verification Retrieval Pipeline.

Combines three retrieval signals:
  1. Dense vector similarity  (sentence-transformers / all-MiniLM-L6-v2)
  2. BM25 keyword search      (pure-python BM25 — no extra deps)
  3. Entity overlap score     (claim entity coverage in each atom)

Final composite score:
  final_score = 0.45 * reranker_score   (filled in by reranker step)
              + 0.25 * bm25_score       (normalised)
              + 0.20 * entity_score     (normalised)
              + 0.10 * vector_score     (normalised)

Before the reranker runs we use:
  prescore = 0.45 * vector_score + 0.30 * bm25_score + 0.25 * entity_score
to pick top-20 candidates for the reranker.
"""

import math
import re
from typing import Dict, List


# ── BM25 (parameter-free, no third-party lib) ────────────────────────────────

def _tokenise(text: str) -> List[str]:
    return re.findall(r'[a-z0-9]+', text.lower())


class _BM25:
    """Minimal BM25 implementation over a fixed corpus."""
    K1 = 1.5
    B  = 0.75

    def __init__(self, corpus: List[str]):
        self.docs = [_tokenise(d) for d in corpus]
        N = len(self.docs)
        avgdl = sum(len(d) for d in self.docs) / max(N, 1)
        # IDF for each term (across corpus)
        df: Dict[str, int] = {}
        for doc in self.docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self.idf: Dict[str, float] = {
            term: math.log((N - f + 0.5) / (f + 0.5) + 1)
            for term, f in df.items()
        }
        self.avgdl = avgdl

    def score(self, query_terms: List[str], doc_idx: int) -> float:
        doc = self.docs[doc_idx]
        dl = len(doc)
        from collections import Counter
        tf_map = Counter(doc)
        s = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue
            tf = tf_map.get(term, 0)
            idf = self.idf[term]
            num = tf * (self.K1 + 1)
            denom = tf + self.K1 * (1 - self.B + self.B * dl / max(self.avgdl, 1))
            s += idf * (num / denom)
        return s


# ── Vector similarity ─────────────────────────────────────────────────────────

_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        import os
        import warnings
        warnings.filterwarnings("ignore")
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _cosine(a, b) -> float:
    import numpy as np
    na = a / (np.linalg.norm(a) + 1e-10)
    nb = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(na, nb))


# ── Entity overlap ────────────────────────────────────────────────────────────

def _entity_overlap(text: str, keywords: List[str]) -> float:
    """Fraction of required keywords present in atom text."""
    if not keywords:
        return 0.0
    t = text.lower()
    hits = sum(1 for kw in keywords if kw in t)
    return hits / len(keywords)


# ── Normalisation helper ──────────────────────────────────────────────────────

def _normalise(values: List[float]) -> List[float]:
    mn, mx = min(values, default=0.0), max(values, default=1.0)
    span = mx - mn
    if span < 1e-9:
        return [0.5] * len(values)
    return [(v - mn) / span for v in values]


# ── Public API ────────────────────────────────────────────────────────────────

def hybrid_retrieve(
    claim: str,
    atoms: List[Dict],
    claim_entities: Dict,
    top_k: int = 20,
) -> List[Dict]:
    """
    Run hybrid retrieval over `atoms` for `claim`.

    Returns top_k atoms sorted by composite prescore (desc),
    each annotated with:
      - _vector_score
      - _bm25_score
      - _entity_score
      - _prescore
    """
    if not atoms:
        return []

    texts = [a.get("text", "") for a in atoms]
    keywords = claim_entities.get("all_keywords", [])
    query_tokens = _tokenise(claim)

    # ── Vector scores ─────────────────────────────────────────────────────
    model = _get_embed_model()
    q_emb = model.encode(claim, convert_to_tensor=False)
    a_embs = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
    vec_scores = [_cosine(q_emb, e) for e in a_embs]

    # ── BM25 scores ───────────────────────────────────────────────────────
    bm25 = _BM25(texts)
    bm25_raw = [bm25.score(query_tokens, i) for i in range(len(atoms))]

    # ── Entity overlap scores ─────────────────────────────────────────────
    ent_raw = [_entity_overlap(t, keywords) for t in texts]

    # ── Normalise all three ───────────────────────────────────────────────
    vec_norm  = _normalise(vec_scores)
    bm25_norm = _normalise(bm25_raw)
    ent_norm  = _normalise(ent_raw)

    # ── Composite prescore (before reranker) ──────────────────────────────
    prescores = [
        0.45 * v + 0.30 * b + 0.25 * e
        for v, b, e in zip(vec_norm, bm25_norm, ent_norm)
    ]

    # ── Annotate and sort ─────────────────────────────────────────────────
    candidates = []
    for i, atom in enumerate(atoms):
        a = atom.copy()
        a["_vector_score"] = vec_norm[i]
        a["_bm25_score"]   = bm25_norm[i]
        a["_entity_score"] = ent_norm[i]
        a["_prescore"]     = prescores[i]
        candidates.append(a)

    candidates.sort(key=lambda x: x["_prescore"], reverse=True)
    return candidates[:top_k]
