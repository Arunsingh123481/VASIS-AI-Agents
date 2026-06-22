"""
Tests for vault.py, mocked so they run without Ollama/a live model.

The critical thing under test is the atom_id collision fix: two papers
both have an atom_id=0, atom_id=1, etc. (since atomic_decomposer numbers
atoms locally per document). VaultSession.compare() must not let that
collision either (a) miss a real cross-document conflict, or (b) invent a
fake one just because two unrelated atoms from different papers happen to
share a number.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from vault import VaultSession, is_comparison_question


# ── is_comparison_question ──────────────────────────────────────────────────

@pytest.mark.parametrize("q,expected", [
    ("Do these papers contradict each other on accuracy?", True),
    ("Is there a disagreement between paper A and paper B?", True),
    ("How does the methodology compare across these papers?", True),
    ("Are the two papers consistent on dataset size?", True),
    ("What is the main contribution of this paper?", False),
    ("Summarize section 3", False),
    ("What dataset was used?", False),
])
def test_is_comparison_question(q, expected):
    assert is_comparison_question(q) == expected


# ── Fakes ────────────────────────────────────────────────────────────────────

class FakeRag:
    """Stands in for PageIndexREMSE. Holds atoms numbered 0..N-1, exactly
    like the real decomposer does per-document, and triples tagged with
    the SAME doc_id on every triple (as the real pipeline does)."""

    def __init__(self, doc_id, triples, query_answer="answer", narrative="narrative"):
        self.doc_id = doc_id
        self.triples = triples
        self._query_answer = query_answer
        self._narrative = narrative
        self.model = "fake"

    def ingest(self, pdf_path, force_reindex=False):
        pass

    def query(self, question, **kwargs):
        # Every triple's atom_id is "used" for simplicity in these tests.
        used_atom_ids = sorted({t["atom_id"] for t in self.triples})
        ordered_atoms = [{"atom_id": aid} for aid in used_atom_ids]
        return {
            "answer": self._query_answer,
            "narrative": self._narrative,
            "ordered_atoms": ordered_atoms,
        }

    def get_stats(self):
        return {"doc_id": self.doc_id, "pdf_path": "fake.pdf", "tree_nodes": 1,
                "total_atoms": len(self.triples), "total_triples": len(self.triples),
                "model": "fake", "ready": True}


def make_session_with(papers: dict) -> VaultSession:
    """papers: label -> FakeRag"""
    session = VaultSession()
    for label, rag in papers.items():
        session.papers[label] = rag
        session._order.append(label)
    return session


# ── Real cross-document conflict, despite colliding atom_ids ───────────────

def test_compare_detects_real_cross_doc_conflict_with_colliding_atom_ids():
    # Both papers have atom_id=0. Paper A says the model has 7B params;
    # Paper B says the SAME model has 13B params. This should be flagged.
    paper_a = FakeRag("doc_aaa", [
        {"atom_id": 0, "doc_id": "doc_aaa", "subject": "the model", "relation": "has", "object": "7b parameters"},
    ])
    paper_b = FakeRag("doc_bbb", [
        {"atom_id": 0, "doc_id": "doc_bbb", "subject": "the model", "relation": "has", "object": "13b parameters"},
    ])
    session = make_session_with({"paper_a": paper_a, "paper_b": paper_b})

    result = session.compare("Do these papers contradict each other on model size?")

    assert "error" not in result
    assert result["structural_conflict_found"] is True
    assert len(result["cross_doc_conflicts"]) == 1
    conflict = result["cross_doc_conflicts"][0]
    assert conflict["subject"] == "the model"
    # Must correctly attribute each value to its REAL doc_id, not the
    # synthetic remapped id used internally.
    assert conflict["per_document"] == {"doc_aaa": "7b parameters", "doc_bbb": "13b parameters"}


# ── No false positive purely from atom_id collision ─────────────────────────

def test_compare_no_false_conflict_from_atom_id_collision_alone():
    # Both papers have atom_id=0, but about UNRELATED subjects. If the
    # remap didn't work, a buggy merge could still accidentally treat them
    # as comparable — this confirms it doesn't.
    paper_a = FakeRag("doc_aaa", [
        {"atom_id": 0, "doc_id": "doc_aaa", "subject": "dataset x", "relation": "has", "object": "10000 samples"},
    ])
    paper_b = FakeRag("doc_bbb", [
        {"atom_id": 0, "doc_id": "doc_bbb", "subject": "dataset y", "relation": "has", "object": "50000 samples"},
    ])
    session = make_session_with({"paper_a": paper_a, "paper_b": paper_b})

    result = session.compare("Do these papers contradict each other?")

    assert result["structural_conflict_found"] is False
    assert result["cross_doc_conflicts"] == []


# ── Remapped atom_ids are actually unique in the merged store ──────────────

def test_merged_atom_ids_are_globally_unique():
    paper_a = FakeRag("doc_aaa", [
        {"atom_id": 0, "doc_id": "doc_aaa", "subject": "s", "relation": "r", "object": "o1"},
        {"atom_id": 1, "doc_id": "doc_aaa", "subject": "s2", "relation": "r2", "object": "o2"},
    ])
    paper_b = FakeRag("doc_bbb", [
        {"atom_id": 0, "doc_id": "doc_bbb", "subject": "s3", "relation": "r3", "object": "o3"},
        {"atom_id": 1, "doc_id": "doc_bbb", "subject": "s4", "relation": "r4", "object": "o4"},
    ])
    session = make_session_with({"paper_a": paper_a, "paper_b": paper_b})

    # Reach into the merge logic the same way compare() does, to assert
    # uniqueness directly rather than only indirectly via conflict results.
    per_paper = session._ask_each_paper("placeholder question")
    merged = []
    for label, result in per_paper.items():
        rag = session.papers[label]
        d_idx = session.doc_index(label)
        used = {a["atom_id"] for a in result["ordered_atoms"]}
        for t in rag.triples:
            if t["atom_id"] in used:
                r = dict(t)
                r["atom_id"] = d_idx * 1_000_000 + int(t["atom_id"])
                merged.append(r)

    ids = [t["atom_id"] for t in merged]
    assert len(ids) == len(set(ids)), "remapped atom_ids collided across documents"


# ── Need at least 2 papers ──────────────────────────────────────────────────

def test_compare_requires_two_papers():
    paper_a = FakeRag("doc_aaa", [{"atom_id": 0, "doc_id": "doc_aaa", "subject": "s", "relation": "r", "object": "o"}])
    session = make_session_with({"paper_a": paper_a})
    result = session.compare("Do these contradict?")
    assert "error" in result


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
