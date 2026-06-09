import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agents.agent10_super import SuperAgent

# Sample data to initialize stores
SAMPLE_TREE = [
    {"node_id": "0000", "doc_id": "test_doc", "title": "Section 1", "summary": "Intro to Attention", "start_page": 1, "end_page": 2, "key_topics": ["attention"], "page_count": 2, "prev_node_id": None, "next_node_id": None}
]

SAMPLE_ATOMS = [
    {"atom_id": 0, "doc_id": "test_doc", "page_num": 1, "text": "Self-attention mechanism relates different positions of a single sequence.", "token_count": 10, "prev_atom_id": None, "next_atom_id": 1, "section_node_id": "0000"},
    {"atom_id": 1, "doc_id": "test_doc", "page_num": 1, "text": "It has been highly successful in sequence modeling.", "token_count": 8, "prev_atom_id": 0, "next_atom_id": None, "section_node_id": "0000"}
]

SAMPLE_TRIPLES = [
    {"subject": "self-attention mechanism", "relation": "relates", "object": "different positions of a single sequence", "atom_id": 0}
]

@pytest.fixture
def mock_stores(mocker):
    from db.atom_store import AtomStore
    from db.bm25_index import BM25Index
    from db.triple_store import TripleStore
    from db.causal_store import CausalStore
    from learning.feedback_index import FeedbackIndex

    return {
        "tree": SAMPLE_TREE,
        "atom_store": AtomStore(SAMPLE_ATOMS),
        "bm25_index": BM25Index(SAMPLE_ATOMS),
        "triple_store": TripleStore(SAMPLE_TRIPLES),
        "causal_store": CausalStore(SAMPLE_TRIPLES),
        "feedback_index": mocker.Mock(spec=FeedbackIndex)
    }

def test_super_agent_initial_plan(mock_stores, mocker):
    # Patch the generate_json function in the agents.agent10_super module namespace where it is used
    mocker.patch("agents.agent10_super.generate_json", return_value={
        "initial_steps": ["agent1_router", "agent3_navigator", "agent4_retrieval", "agent5_expansion"],
        "skip_reasons": {},
        "complexity": "simple"
    })
    
    super_agent = SuperAgent(
        tree=mock_stores["tree"],
        atom_store=mock_stores["atom_store"],
        bm25_index=mock_stores["bm25_index"],
        triple_store=mock_stores["triple_store"],
        causal_store=mock_stores["causal_store"],
        feedback_index=mock_stores["feedback_index"]
    )
    
    plan = super_agent._plan("What is self-attention?")
    assert plan.steps == ["agent1_router", "agent3_navigator", "agent4_retrieval", "agent5_expansion"]
    assert plan.query == "What is self-attention?"

def test_super_agent_full_execution(mock_stores, mocker):
    # Mock LLM calls in agents
    mocker.patch("agents.agent1_router.route", return_value={
        "intent": "factual",
        "rewritten_query": "Explain self-attention mechanism.",
        "key_entities": ["self-attention"],
        "is_complex": False,
        "original_query": "Explain self-attention.",
        "is_bibliography_query": False,
        "confidence": 0.9
    })
    
    mocker.patch("agents.agent3_navigator.navigate", return_value={
        "selected_nodes": ["0000"],
        "reasoning": "Matches Section 1",
        "confidence": 0.85
    })
    
    mocker.patch("agents.agent4_retrieval.retrieve", return_value=SAMPLE_ATOMS)
    
    mocker.patch("agents.agent5_expansion.expand", return_value={
        "narrative": "Self-attention mechanism relates different positions of a single sequence. It has been highly successful in sequence modeling.",
        "atom_ids_used": [0, 1],
        "pages_referenced": [1],
        "gap_count": 0,
        "atom_count": 2
    })
    
    mocker.patch("agents.agent6_validation.validate", return_value={
        "verdict": "grounded",
        "confidence": 0.95,
        "decision": "accept"
    })
    
    # Mock detect() in agent7_contradiction
    mocker.patch("agents.agent7_contradiction.detect", return_value={
        "triple_conflicts": [],
        "cross_doc_conflicts": [],
        "llm_contradictions": [],
        "contradictions_found": False,
        "llm_contradictions_found": False,
        "consistency_score": 1.0
    })
    
    mocker.patch("agents.agent8_temporal.analyze", return_value={
        "has_timeline": False,
        "markers_count": 0,
        "conflicting": False
    })
    
    mocker.patch("agents.agent9_calibration.calibrate", return_value={
        "calibrated_score": 0.95,
        "trust_level": "high"
    })
    
    mocker.patch("agents.agent11_synthesis.synthesize", return_value={
        "novel_connections": []
    })

    # Mock the cooperative quality reviewer inside agent10_super
    mocker.patch("agents.agent10_super.generate", return_value="The work is complete and usable.")
    mocker.patch("agents.agent10_super.generate_json", return_value={
        "score": 0.95,
        "grade": "A",
        "issues": [],
        "is_usable": True,
        "recommendation": "proceed"
    })
    
    # Mock LLM generate call in superagent for generating the final answer
    mocker.patch("llm.router.generate", return_value="Self-attention is a mechanism that relates different positions of a sequence.")
    
    mock_stores["feedback_index"].get_warm_start.return_value = None

    super_agent = SuperAgent(
        tree=mock_stores["tree"],
        atom_store=mock_stores["atom_store"],
        bm25_index=mock_stores["bm25_index"],
        triple_store=mock_stores["triple_store"],
        causal_store=mock_stores["causal_store"],
        feedback_index=mock_stores["feedback_index"]
    )
    
    result = super_agent.execute("Explain self-attention.", doc_id="test_doc")
    
    assert result is not None
    assert result["answer"] == "Self-attention is a mechanism that relates different positions of a sequence."
    assert result["trust_level"] == "high"
    assert result["confidence"] == 0.95
