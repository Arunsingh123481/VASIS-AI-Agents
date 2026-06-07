# db/bm25_index.py
# BM25-based keyword indexing for vectorless retrieval, with scoping capabilities

from rank_bm25 import BM25Okapi

class BM25Index:
    def __init__(self, atoms: list[dict]):
        self.atoms = atoms
        self.ids   = [int(a["atom_id"]) for a in atoms]
        self.bm25  = BM25Okapi(
            [a["text"].lower().split() for a in atoms]
        )

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Perform a lexical search on all indexing atoms."""
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
        lookup = {int(a["atom_id"]): a for a in self.atoms}
        
        return [
            dict(lookup[aid], bm25_score=round(float(s), 4))
            for aid, s in ranked if s > 0
        ]

    def search_scoped(self, query: str, atom_ids: list[int], top_k: int = 10) -> list[dict]:
        """Perform a BM25 lexical search restricted to a specific set of atom IDs."""
        id_set = set(int(aid) for aid in atom_ids)
        scoped = [a for a in self.atoms if int(a["atom_id"]) in id_set]
        
        return BM25Index(scoped).search(query, top_k) if scoped else []
