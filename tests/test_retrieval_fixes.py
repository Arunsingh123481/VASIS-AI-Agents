"""
Test suite for retrieval & grounding fixes.
Tests the core logic WITHOUT requiring Ollama/LLMs to be running.

Run with: python tests/test_retrieval_fixes.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_fix1_extract_retrieval_queries():
    """Fix 1: Sub-query extraction strips instructional prefixes."""
    from grounding_fix import extract_retrieval_queries

    # Test 1: Raw topic with "Write a paper on..." prefix
    queries = extract_retrieval_queries(
        "Write a research paper on the solutions of the limitations "
        "of attention is all you need"
    )
    print(f"  Input:  'Write a research paper on the solutions of the limitations of attention is all you need'")
    print(f"  Output: {queries}")

    assert len(queries) >= 2, f"Expected >=2 queries, got {len(queries)}"
    # The raw instructional prefix should be stripped
    assert not any(q.lower().startswith("write") for q in queries), \
        "Queries should NOT start with 'write' — instructional prefix not stripped!"
    # Should contain the core subject
    joined = " ".join(queries).lower()
    assert "attention" in joined, "Core subject 'attention' should be in queries"
    print("  [OK] PASSED: Instructional prefix stripped, noun-phrase queries generated\n")

    # Test 2: Short factual query should pass through
    queries2 = extract_retrieval_queries("transformer architecture self-attention")
    print(f"  Input:  'transformer architecture self-attention'")
    print(f"  Output: {queries2}")
    assert len(queries2) >= 1, "Should produce at least 1 query"
    print("  [OK] PASSED: Short queries pass through\n")

    # Test 3: Various instructional prefixes
    for prefix in [
        "Explain the key findings of",
        "Summarize the methodology of",
        "Describe the architecture of",
        "Generate a paper about",
    ]:
        topic = f"{prefix} convolutional neural networks"
        qs = extract_retrieval_queries(topic)
        assert not any(q.lower().startswith(prefix.split()[0].lower()) for q in qs), \
            f"Prefix '{prefix}' was not stripped from queries: {qs}"
    print("  [OK] PASSED: All instructional prefixes stripped correctly\n")


def test_fix2_anchor_threshold():
    """Fix 2: Anchor threshold lowered from 0.1 to 0.05 in routing rules."""
    from agent_routing_rules import AGENT10_REVIEW_RULES

    agent4_rules = AGENT10_REVIEW_RULES.get("agent4_retrieval", {})
    pass_if = agent4_rules.get("pass_if", [])

    print(f"  Agent4 pass_if rules: {pass_if}")

    # Check that the threshold condition says 0.05, not 0.1
    threshold_rule = [r for r in pass_if if "anchor" in r.lower() and "score" in r.lower()]
    assert len(threshold_rule) == 1, f"Expected exactly 1 anchor score rule, got {threshold_rule}"
    assert "0.05" in threshold_rule[0], \
        f"Threshold should be 0.05 but found: {threshold_rule[0]}"
    assert "0.1" not in threshold_rule[0].replace("0.05", ""), \
        f"Old threshold 0.1 still present: {threshold_rule[0]}"
    print("  [OK] PASSED: Anchor threshold is 0.05 in routing rules\n")


def test_fix2_evaluate_condition():
    """Fix 2: The condition evaluator in agent10 uses 0.05 threshold."""
    from agents.agent10_super import _evaluate_condition

    # Test: anchor with score 0.07 (above 0.05, below old 0.1)
    mock_anchors = [{"combined_score": 0.07}]
    result = _evaluate_condition(
        "at least 1 anchor has score > 0.05",
        mock_anchors,
        {}
    )
    print(f"  Anchor score 0.07 with threshold 0.05: {'PASS' if result else 'FAIL'}")
    assert result is True, "Anchor with score 0.07 should PASS with threshold 0.05"
    print("  [OK] PASSED: Score 0.07 passes new 0.05 threshold\n")

    # Test: anchor with score 0.03 (below 0.05) should fail
    mock_anchors_low = [{"combined_score": 0.03}]
    result_low = _evaluate_condition(
        "at least 1 anchor has score > 0.05",
        mock_anchors_low,
        {}
    )
    print(f"  Anchor score 0.03 with threshold 0.05: {'PASS' if result_low else 'FAIL'}")
    assert result_low is False, "Anchor with score 0.03 should FAIL with threshold 0.05"
    print("  [OK] PASSED: Score 0.03 correctly fails 0.05 threshold\n")


def test_fix3_citation_forcing_prompt():
    """Fix 3: Agent 13 system prompt contains citation forcing rules."""
    from agents.agent13_paper_writer import SYSTEM_WRITER

    print(f"  SYSTEM_WRITER length: {len(SYSTEM_WRITER)} chars")

    assert "HARD CITATION RULE" in SYSTEM_WRITER, \
        "SYSTEM_WRITER should contain 'HARD CITATION RULE'"
    assert "citation tag" in SYSTEM_WRITER.lower() or "citation_key" in SYSTEM_WRITER.lower(), \
        "SYSTEM_WRITER should mention citation tags"
    assert "85%" in SYSTEM_WRITER, \
        "SYSTEM_WRITER should mention 85% target"
    assert "[A:page_" in SYSTEM_WRITER, \
        "SYSTEM_WRITER should contain example atom citation"
    assert "WRONG" in SYSTEM_WRITER or "rejected" in SYSTEM_WRITER.lower(), \
        "SYSTEM_WRITER should show wrong example"
    print("  [OK] PASSED: Citation forcing prompt contains all required elements\n")


def test_fix4_citation_injector():
    """Fix 5: Post-processing citation injector tags ungrounded sentences."""
    from grounding_fix import inject_missing_citations

    # Test paper with some grounded and some ungrounded sentences
    test_paper = (
        "The Transformer uses self-attention mechanisms [A:page_3_id42]. "
        "This allows parallel computation of sequence representations. "
        "The multi-head attention mechanism computes queries keys and values. "
        "Positional encoding provides sequence order information [A:page_4_id55]."
    )

    test_atoms = [
        {"atom_id": 42, "page": 3, "text": "self-attention mechanism replaces recurrence"},
        {"atom_id": 55, "page": 4, "text": "positional encoding sinusoidal functions"},
        {"atom_id": 60, "page": 5, "text": "parallel computation sequence representations"},
        {"atom_id": 70, "page": 6, "text": "multi-head attention queries keys values"},
    ]

    result = inject_missing_citations(
        paper_text=test_paper,
        atoms=test_atoms,
        web_sources=[],
    )

    print(f"  Tagged before: {result['tagged_before']}")
    print(f"  Tagged after:  {result['tagged_after']}")
    print(f"  Paper preview: {result['paper_text'][:200]}...")

    assert result["tagged_after"] >= result["tagged_before"], \
        "Injector should tag MORE sentences, not fewer"
    assert result["tagged_after"] > 2, \
        f"Expected >2 tagged sentences after injection, got {result['tagged_after']}"
    print("  [OK] PASSED: Citation injector tags ungrounded sentences\n")


def test_fix5_retrieval_fix_module():
    """Fix 5: retrieval_fix.py is importable and has expected functions."""
    import retrieval_fix

    assert hasattr(retrieval_fix, "re_index"), "retrieval_fix should have re_index()"
    assert hasattr(retrieval_fix, "patch_threshold"), "retrieval_fix should have patch_threshold()"
    assert hasattr(retrieval_fix, "audit_index"), "retrieval_fix should have audit_index()"
    assert hasattr(retrieval_fix, "reindex_vault"), "retrieval_fix should have reindex_vault()"
    assert hasattr(retrieval_fix, "print_comparison"), "retrieval_fix should have print_comparison()"

    # Check tuned params
    assert retrieval_fix.TUNED_PARAMS["max_pages_per_node"] == 2
    assert retrieval_fix.TUNED_PARAMS["max_tokens_per_node"] == 6000
    assert retrieval_fix.ANCHOR_SCORE_THRESHOLD_PATCHED == 0.05

    print("  [OK] PASSED: retrieval_fix.py importable with correct parameters\n")


def test_merge_retrieved_atoms():
    """Utility: merge_retrieved_atoms deduplicates correctly."""
    from grounding_fix import merge_retrieved_atoms

    list1 = [
        {"atom_id": 1, "text": "fact A", "combined_score": 0.5},
        {"atom_id": 2, "text": "fact B", "combined_score": 0.3},
    ]
    list2 = [
        {"atom_id": 2, "text": "fact B", "combined_score": 0.8},  # higher score
        {"atom_id": 3, "text": "fact C", "combined_score": 0.4},
    ]

    merged = merge_retrieved_atoms([list1, list2])
    print(f"  Input: {len(list1)} + {len(list2)} atoms")
    print(f"  Merged: {len(merged)} unique atoms")

    assert len(merged) == 3, f"Expected 3 unique atoms, got {len(merged)}"
    # Atom 2 should have the higher score (0.8)
    atom2 = [a for a in merged if int(a["atom_id"]) == 2][0]
    assert atom2["combined_score"] == 0.8, \
        f"Atom 2 should keep higher score 0.8, got {atom2['combined_score']}"
    print("  [OK] PASSED: merge_retrieved_atoms deduplicates and keeps best scores\n")


# --- MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  VASIS AI — Retrieval & Grounding Fixes Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("Fix 1: Sub-query extraction",      test_fix1_extract_retrieval_queries),
        ("Fix 2: Anchor threshold (rules)",   test_fix2_anchor_threshold),
        ("Fix 2: Anchor threshold (eval)",    test_fix2_evaluate_condition),
        ("Fix 3: Citation forcing prompt",    test_fix3_citation_forcing_prompt),
        ("Fix 4: Citation injector",          test_fix4_citation_injector),
        ("Fix 5: retrieval_fix module",       test_fix5_retrieval_fix_module),
        ("Util: merge_retrieved_atoms",       test_merge_retrieved_atoms),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"--- {name} ───")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] FAILED: {e}\n")
            failed += 1

    print("=" * 60)
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED [OK]")
    else:
        print(f"  {passed} passed, {failed} FAILED [X]")
    print("=" * 60)
