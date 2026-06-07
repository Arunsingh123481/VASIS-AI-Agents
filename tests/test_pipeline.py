"""
Test Suite — Unit tests for PageIndex-RE-MSE components.
Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from ingest.atomic_decomposer import decompose_to_atoms, cross_reference_atoms_to_tree, count_tokens
from reconstruction.stitcher import stitch, get_provenance


# ── Sample Data ──────────────────────────────────────────────────────────────

SAMPLE_PAGES = [
    {"page_num": 1, "text": "The company reported strong revenue growth in fiscal year 2023. Total revenue reached $5.2 billion, up 18% year over year. Operating margins improved significantly due to cost reduction initiatives."},
    {"page_num": 2, "text": "Research and development expenses increased by 12% to $800 million. The company launched three new product lines in Q3. Customer acquisition costs decreased by 8% compared to the prior year."},
    {"page_num": 3, "text": "The balance sheet remains strong with $2.1 billion in cash and equivalents. Long-term debt stands at $1.5 billion. The company repurchased $300 million in shares during the year."},
]

SAMPLE_TREE = [
    {"node_id": "0000", "doc_id": "test", "title": "Financial Overview", "summary": "Revenue and margins", "start_page": 1, "end_page": 1, "key_topics": ["revenue"], "page_count": 1, "prev_node_id": None, "next_node_id": "0001"},
    {"node_id": "0001", "doc_id": "test", "title": "R&D and Products", "summary": "Research spending and launches", "start_page": 2, "end_page": 2, "key_topics": ["R&D"], "page_count": 1, "prev_node_id": "0000", "next_node_id": "0002"},
    {"node_id": "0002", "doc_id": "test", "title": "Balance Sheet", "summary": "Cash and debt position", "start_page": 3, "end_page": 3, "key_topics": ["cash", "debt"], "page_count": 1, "prev_node_id": "0001", "next_node_id": None},
]


# ── Token Counter Tests ──────────────────────────────────────────────────────

def test_count_tokens_basic():
    text = "Hello world this is a test"
    count = count_tokens(text)
    assert count > 0
    assert isinstance(count, int)


def test_count_tokens_empty():
    assert count_tokens("") == 0


# ── Atomic Decomposer Tests ──────────────────────────────────────────────────

def test_decompose_creates_atoms():
    atoms = decompose_to_atoms(SAMPLE_PAGES, "test_doc", target_tokens=20)
    assert len(atoms) > 0


def test_atoms_have_required_fields():
    atoms = decompose_to_atoms(SAMPLE_PAGES, "test_doc", target_tokens=20)
    required = {"atom_id", "doc_id", "page_num", "text", "token_count", "prev_atom_id", "next_atom_id", "section_node_id"}
    for atom in atoms:
        for field in required:
            assert field in atom, f"Missing field: {field}"


def test_atoms_sequential_ids():
    atoms = decompose_to_atoms(SAMPLE_PAGES, "test_doc", target_tokens=20)
    for i, atom in enumerate(atoms):
        assert atom["atom_id"] == i


def test_atoms_bidirectional_pointers():
    atoms = decompose_to_atoms(SAMPLE_PAGES, "test_doc", target_tokens=20)
    for i, atom in enumerate(atoms):
        if i > 0:
            assert atom["prev_atom_id"] == atoms[i-1]["atom_id"]
        else:
            assert atom["prev_atom_id"] is None
        if i < len(atoms) - 1:
            assert atom["next_atom_id"] == atoms[i+1]["atom_id"]
        else:
            assert atom["next_atom_id"] is None


def test_atoms_doc_id():
    atoms = decompose_to_atoms(SAMPLE_PAGES, "my_doc_id", target_tokens=20)
    for atom in atoms:
        assert atom["doc_id"] == "my_doc_id"


def test_cross_reference():
    atoms = decompose_to_atoms(SAMPLE_PAGES, "test_doc", target_tokens=20)
    atoms = cross_reference_atoms_to_tree(atoms, SAMPLE_TREE)
    for atom in atoms:
        assert atom["section_node_id"] is not None, f"Atom {atom['atom_id']} on page {atom['page_num']} has no section ref"


# ── Stitcher Tests ────────────────────────────────────────────────────────────

def _make_atoms(count=10):
    return [
        {"atom_id": i, "doc_id": "test", "page_num": 1 + i // 3, "text": f"Atom text number {i} with some content here.", "token_count": 10, "section_node_id": "0000"}
        for i in range(count)
    ]


def test_stitch_basic():
    atoms = _make_atoms(5)
    narrative, ordered = stitch(atoms)
    assert len(narrative) > 0
    assert len(ordered) == 5


def test_stitch_deduplicates():
    atoms = _make_atoms(5)
    # Add duplicates
    dupes = atoms + atoms[:3]
    narrative, ordered = stitch(dupes)
    assert len(ordered) == 5  # deduped back to 5


def test_stitch_sorted_order():
    atoms = _make_atoms(10)
    import random
    shuffled = atoms.copy()
    random.shuffle(shuffled)
    narrative, ordered = stitch(shuffled)
    for i in range(len(ordered) - 1):
        assert ordered[i]["atom_id"] < ordered[i+1]["atom_id"]


def test_stitch_respects_max_chars():
    atoms = _make_atoms(200)
    narrative, ordered = stitch(atoms, max_chars=500)
    assert len(narrative) <= 600  # some tolerance for truncation message


def test_stitch_empty():
    narrative, ordered = stitch([])
    assert narrative == ""
    assert ordered == []


def test_get_provenance():
    atoms = _make_atoms(5)
    provenance = get_provenance(atoms, SAMPLE_TREE, "Selected based on revenue query")
    assert "reasoning_path" in provenance
    assert "sections_used" in provenance
    assert "atoms_used" in provenance
    assert "total_atoms" in provenance
    assert provenance["total_atoms"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
