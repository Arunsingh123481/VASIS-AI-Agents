# db/causal_store.py
# Graph representations of causal chains, enabling multi-hop reasoning

from collections import defaultdict, deque

class CausalStore:
    def __init__(self, triples: list[dict]):
        self.graph   = defaultdict(list)
        self.reverse = defaultdict(list)

        for t in triples:
            causal_type = t.get("causal_type")
            if causal_type and causal_type not in (None, "null", ""):
                src = (t.get("subject") or "").lower().strip()
                tgt = (t.get("object") or "").lower().strip()
                rel = (causal_type or "").lower().strip()
                
                if src and tgt:
                    self.graph[src].append({
                        "relation": rel,
                        "target":   tgt,
                        "atom_id":  t.get("atom_id"),
                        "doc_id":   t.get("doc_id")
                    })
                    self.reverse[tgt].append({
                        "relation": rel,
                        "source":   src,
                        "atom_id":  t.get("atom_id")
                    })

    def traverse_chain(self, start: str, max_hops: int = 3) -> list[dict]:
        """Breadth-first traversal of causal chains starting from an entity."""
        visited = set()
        queue   = deque([{
            "node": start.lower().strip(),
            "path": [start.lower().strip()],
            "atom_ids": [],
            "hops": 0
        }])
        chains = []
        
        while queue:
            curr = queue.popleft()
            node = curr["node"]
            if node in visited or curr["hops"] >= max_hops:
                continue
            visited.add(node)
            
            for edge in self.graph.get(node, []):
                tgt = edge["target"]
                new_path = curr["path"] + [tgt]
                new_ids  = curr["atom_ids"] + [edge["atom_id"]]
                
                chains.append({
                    "path":     new_path,
                    "terminal": tgt,
                    "hops":     curr["hops"] + 1,
                    "atom_ids": new_ids
                })
                
                if tgt not in visited:
                    queue.append({
                        "node":     tgt,
                        "path":     new_path,
                        "atom_ids": new_ids,
                        "hops":     curr["hops"] + 1
                    })
        return chains

    def all_entities(self) -> list[str]:
        """List all unique subject and object entities in the causal graph."""
        entities = set(self.graph.keys())
        entities.update(self.reverse.keys())
        return list(entities)
