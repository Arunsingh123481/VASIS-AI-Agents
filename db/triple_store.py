# db/triple_store.py
# Factual triple storage, inconsistency locator, and semantic matching

from collections import defaultdict

class TripleStore:
    def __init__(self, triples: list[dict]):
        self.triples = triples
        self._by_subject = defaultdict(list)
        self._by_atom    = defaultdict(list)
        self._causal     = []

        for t in triples:
            s = (t.get("subject") or "").lower()
            self._by_subject[s].append(t)
            
            atom_id = t.get("atom_id")
            if atom_id is not None:
                self._by_atom[int(atom_id)].append(t)
                
            causal_type = t.get("causal_type")
            if causal_type and causal_type not in (None, "null", ""):
                self._causal.append(t)

    def get_causal_triples(self, scoped: list[int] = None) -> list[dict]:
        """Get causal relation triples, optionally restricted to a set of atom IDs."""
        if scoped:
            ids = set(int(x) for x in scoped)
            return [t for t in self._causal if t.get("atom_id") is not None and int(t["atom_id"]) in ids]
        return self._causal

    def find_conflicts(self, scoped_atom_ids: list[int] = None) -> list[dict]:
        """Detect conflicting objects for same subject-relation pairs."""
        triples = self.triples
        if scoped_atom_ids:
            ids = set(int(x) for x in scoped_atom_ids)
            triples = [t for t in triples if t.get("atom_id") is not None and int(t["atom_id"]) in ids]
            
        sr_map = defaultdict(list)
        for t in triples:
            key = ((t.get("subject") or "").lower().strip(), (t.get("relation") or "").lower().strip())
            sr_map[key].append(t)
            
        conflicts = []
        for (s, r), g in sr_map.items():
            objs = list({(t.get("object") or "").lower().strip() for t in g if t.get("object")})
            if len(objs) > 1:
                conflicts.append({
                    "subject": s,
                    "relation": r,
                    "conflicting_objects": objs,
                    "source_triples": g
                })
        return conflicts

    def get_atoms_for_query(self, query: str, scoped: list[int] = None, top_k: int = 10) -> list[int]:
        """Rank and return atom IDs matching terms in query via factual triples."""
        terms   = set(query.lower().split())
        triples = self.triples
        if scoped:
            ids = set(int(x) for x in scoped)
            triples = [t for t in triples if t.get("atom_id") is not None and int(t["atom_id"]) in ids]
            
        scores = defaultdict(int)
        for t in triples:
            text = f"{t.get('subject') or ''} {t.get('relation') or ''} {t.get('object') or ''}".lower()
            m = sum(1 for term in terms if term in text)
            if m and t.get("atom_id") is not None:
                scores[int(t["atom_id"])] += m
                
        return [
            aid for aid, _ in
            sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        ]
