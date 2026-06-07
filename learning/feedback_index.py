# learning/feedback_index.py
# Feedback index warm starter to bypass retrieval phase based on historical queries

from config import SIMILARITY_THRESHOLD, CONFIDENCE_THRESHOLD
from learning.experience_store import load_experiences
from console_helper import print_msg

class FeedbackIndex:
    def __init__(self):
        # Load experiences on startup
        self.experiences = load_experiences()
        print_msg(f"[FeedbackIndex] Loaded {len(self.experiences)} historical query experiences.")

    def _similarity(self, q1: list[str], q2: list[str]) -> float:
        """Calculate simple Jaccard similarity of cleaned tokens."""
        s1 = set(q1)
        s2 = set(q2)
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    def find_similar(self, question: str, doc_id: str, top_k: int = 3) -> list[dict]:
        """Search historical logs for similar questions matching current document and high confidence."""
        q_words = [w for w in question.lower().split() if len(w) > 2]
        scored  = []
        
        for exp in self.experiences:
            if exp.get("doc_id") != doc_id:
                continue
            if float(exp.get("confidence", 0.0)) < CONFIDENCE_THRESHOLD:
                continue
                
            sim = self._similarity(q_words, exp.get("question_words", []))
            if sim >= SIMILARITY_THRESHOLD:
                scored.append((sim, exp))
                
        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:top_k]]

    def get_warm_start(self, question: str, doc_id: str) -> dict | None:
        """Analyze query and extract a warm start (atom/node IDs) if highly similar experience is located."""
        similar = self.find_similar(question, doc_id, 1)
        if not similar:
            return None
            
        best = similar[0]
        print_msg(f"[FeedbackIndex] [bold green]WARM START DETECTED![/bold green] Found highly similar historical query. Re-using cached index targets (confidence={best['confidence']:.2f}).")
        
        return {
            "warm_atom_ids": [int(x) for x in best.get("useful_atom_ids", [])],
            "warm_node_ids": [str(x) for x in best.get("useful_node_ids", [])],
            "past_confidence": float(best.get("confidence", 0.0)),
            "past_trust": str(best.get("trust_level", "low"))
        }
