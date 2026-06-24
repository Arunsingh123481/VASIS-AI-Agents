# ============================================================
# CRDB SMART AGENT ROUTING RULES
# File: agent_routing_rules.py
#
# PURPOSE:
# Define which agents run for each query type.
# Agent 10 (Super Agent) reads these rules and
# only dispatches agents that are actually needed.
#
# BEFORE (all queries run all agents):
# Summary query → 11 agents → 6 minutes
#
# AFTER (smart routing):
# Summary query → 3 agents → 45 seconds
# ============================================================

# ── QUERY TYPE DEFINITIONS ───────────────────────────────────
# Each query type has:
#   required_agents  → MUST run (core pipeline)
#   optional_agents  → run only if needed
#   skip_agents      → never run for this type
#   agent10_checks   → what Super Agent verifies
#   max_requery      → how many retries allowed
#   description      → plain English explanation

ROUTING_RULES = {

    # ── 1. SUMMARY ─────────────────────────────────────────
    # "Summarise this paper"
    # "Give me an overview"
    # "What is this paper about?"
    "summary": {
        "required_agents": [
            "agent3_navigator",   # find all sections
            "agent5_expansion",   # expand full document
            "agent11_synthesis",  # find key themes
        ],
        "optional_agents": [
        ],
        "skip_agents": [
            "agent1_router",      # no routing needed
            "agent2_decomposer",  # not complex
            "agent4_retrieval",   # full doc not BM25
            "agent6_validation",  # no specific claims
            "agent7_contradiction", # not needed
            "agent8_temporal",    # skip for summary
            "agent9_calibration", # not needed
            "agent12_websearch",  # no web needed
        ],
        "agent10_checks": [
            "did_agent3_cover_all_sections",
            "is_narrative_complete",
            "does_summary_mention_key_topics",
        ],
        "max_requery": 0,
        "min_atoms_needed": 20,
        "use_full_document": True,
        "description": "Full document overview — "
                       "covers all sections"
    },

    # ── 2. FACTUAL ─────────────────────────────────────────
    # "What is the accuracy reported?"
    # "What dataset was used?"
    # "What is the learning rate?"
    "factual": {
        "required_agents": [
            "agent1_router",      # understand exact fact
            "agent3_navigator",   # find right section
            "agent4_retrieval",   # find exact atom
            "agent5_expansion",   # get context
            "agent6_validation",  # verify the fact
            "agent9_calibration", # score confidence
        ],
        "optional_agents": [
            "agent2_decomposer",  # only if multi-fact
        ],
        "skip_agents": [
            "agent7_contradiction", # single fact
            "agent8_temporal",    # skip for factual
            "agent10_super",      # handled by pipeline
            "agent11_synthesis",  # no synthesis needed
            "agent12_websearch",  # vault only
        ],
        "agent10_checks": [
            "is_fact_grounded_in_atom",
            "does_page_number_match",
            "is_confidence_above_0.7",
        ],
        "max_requery": 2,
        "min_atoms_needed": 3,
        "use_full_document": False,
        "description": "Precise fact lookup — "
                       "single verified answer"
    },

    # ── 3. COMPARATIVE ─────────────────────────────────────
    # "Compare X and Y"
    # "What is better — A or B?"
    # "How does this differ from other methods?"
    "comparative": {
        "required_agents": [
            "agent1_router",        # detect both entities
            "agent2_decomposer",    # split into sub-queries
            "agent3_navigator",     # find sections for each
            "agent4_retrieval",     # retrieve for each
            "agent5_expansion",     # expand both contexts
            "agent7_contradiction", # find disagreements
            "agent9_calibration",   # honest confidence
        ],
        "optional_agents": [
            "agent6_validation",    # verify key claims
            "agent11_synthesis",    # find novel connections
        ],
        "skip_agents": [
            "agent8_temporal",      # skip for comparative
            "agent12_websearch",    # vault only
        ],
        "agent10_checks": [
            "did_both_sides_get_retrieved",
            "are_differences_clearly_stated",
            "is_contradiction_score_reported",
        ],
        "max_requery": 1,
        "min_atoms_needed": 8,
        "use_full_document": False,
        "description": "Side-by-side comparison — "
                       "both perspectives retrieved"
    },

    # ── 4. METHODOLOGY ─────────────────────────────────────
    # "What method was used?"
    # "How was the experiment designed?"
    # "What is the training procedure?"
    "methodology": {
        "required_agents": [
            "agent1_router",      # identify method terms
            "agent3_navigator",   # find methods section
            "agent5_expansion",   # expand method context
            "agent6_validation",  # verify method claims
            "agent9_calibration", # score
        ],
        "optional_agents": [
            "agent2_decomposer",  # if multi-step method
        ],
        "skip_agents": [
            "agent4_retrieval",     # skip BM25, use full section
            "agent7_contradiction", # methods rarely conflict
            "agent8_temporal",      # skip for methodology
            "agent11_synthesis",    # no synthesis needed
            "agent12_websearch",    # vault only
        ],
        "agent10_checks": [
            "is_methods_section_referenced",
            "are_steps_in_correct_order",
            "is_confidence_above_0.65",
        ],
        "max_requery": 1,
        "min_atoms_needed": 5,
        "use_full_document": False,
        "description": "Method extraction — "
                       "steps and procedures"
    },

    # ── 5. RESULTS ─────────────────────────────────────────
    # "What results were achieved?"
    # "What is the performance?"
    # "What are the benchmark scores?"
    "results": {
        "required_agents": [
            "agent1_router",        # identify metrics
            "agent3_navigator",     # find results section
            "agent5_expansion",     # expand result context
            "agent6_validation",    # verify numbers
            "agent7_contradiction", # check conflicting nums
            "agent9_calibration",   # score
        ],
        "optional_agents": [
            "agent2_decomposer",    # if multiple metrics
        ],
        "skip_agents": [
            "agent4_retrieval",     # skip BM25, use full section
            "agent8_temporal",      # skip for results
            "agent11_synthesis",    # no synthesis
            "agent12_websearch",    # vault only
        ],
        "agent10_checks": [
            "are_numbers_grounded_in_atoms",
            "do_results_have_page_citation",
            "is_confidence_above_0.7",
            "any_conflicting_numbers_detected",
        ],
        "max_requery": 2,
        "min_atoms_needed": 5,
        "use_full_document": False,
        "description": "Results extraction — "
                       "numbers and benchmarks verified"
    },

    # ── 6. LIMITATIONS ─────────────────────────────────────
    # "What are the limitations?"
    # "What does this paper fail at?"
    # "What are the weaknesses?"
    "limitations": {
        "required_agents": [
            "agent1_router",        # find limitation terms
            "agent3_navigator",     # find limitations section
            "agent5_expansion",     # expand
            "agent7_contradiction", # conflicts = limitations
            "agent9_calibration",   # score
        ],
        "optional_agents": [
            "agent6_validation",    # verify claims
            "agent11_synthesis",    # novel limitation chains
        ],
        "skip_agents": [
            "agent2_decomposer",    # usually simple
            "agent4_retrieval",     # skip BM25, use full section
            "agent8_temporal",      # rarely time-based
            "agent12_websearch",    # vault only
        ],
        "agent10_checks": [
            "are_limitations_from_paper_not_inferred",
            "is_contradiction_score_included",
        ],
        "max_requery": 1,
        "min_atoms_needed": 3,
        "use_full_document": False,
        "description": "Limitation extraction — "
                       "what paper admits failing"
    },

    # ── 7. BIBLIOGRAPHY ────────────────────────────────────
    # "What are the references?"
    # "List all citations"
    # "What papers does this cite?"
    "bibliography": {
        "required_agents": [
            "agent4_retrieval",     # last-section mode
            "agent5_expansion",     # no-gap mode
        ],
        "optional_agents": [],
        "skip_agents": [
            "agent1_router",        # already detected
            "agent2_decomposer",    # not complex
            "agent3_navigator",     # force last section
            "agent6_validation",    # ref format check only
            "agent7_contradiction", # refs cant contradict
            "agent8_temporal",      # not needed
            "agent9_calibration",   # not needed
            "agent11_synthesis",    # not needed
            "agent12_websearch",    # not needed
        ],
        "agent10_checks": [
            "did_last_section_get_retrieved",
            "are_gaps_disabled",
            "reference_count_above_5",
        ],
        "max_requery": 0,
        "min_atoms_needed": 10,
        "use_full_document": False,
        "is_bibliography": True,    # triggers gap-free mode
        "description": "Reference extraction — "
                       "full list, no gaps"
    },

    # ── 8. EXPLANATION ─────────────────────────────────────
    # "Explain concept X"
    # "What does Y mean?"
    # "How does Z work?"
    "explanation": {
        "required_agents": [
            "agent1_router",      # identify concept
            "agent3_navigator",   # find where explained
            "agent5_expansion",   # full explanation context
            "agent6_validation",  # verify explanation
            "agent9_calibration", # score
        ],
        "optional_agents": [
            "agent11_synthesis",  # if novel insight
        ],
        "skip_agents": [
            "agent2_decomposer",    # single concept
            "agent4_retrieval",     # skip BM25, use full section
            "agent7_contradiction", # explanation not conflict
            "agent8_temporal",      # skip for explanation
            "agent12_websearch",    # vault only
        ],
        "agent10_checks": [
            "is_explanation_grounded",
            "does_explanation_cover_definition",
            "is_confidence_above_0.65",
        ],
        "max_requery": 1,
        "min_atoms_needed": 4,
        "use_full_document": False,
        "description": "Concept explanation — "
                       "clear grounded definition"
    },

    # ── 9. NOVELTY FINDER ──────────────────────────────────
    # "What is novel in this paper?"
    # "What gap does this fill?"
    # "What is the contribution?"
    "novelty": {
        "required_agents": [
            "agent1_router",        # find contribution terms
            "agent3_navigator",     # intro + conclusion
            "agent5_expansion",     # expand
            "agent7_contradiction", # gaps = contradictions
            "agent11_synthesis",    # KEY — find novel chains
            "agent9_calibration",   # score
        ],
        "optional_agents": [
            "agent12_websearch",    # find if gap is real
        ],
        "skip_agents": [
            "agent2_decomposer",    # usually single
            "agent4_retrieval",     # skip BM25, use full section
            "agent6_validation",    # novel claims ungrounded
            "agent8_temporal",      # skip for novelty
        ],
        "agent10_checks": [
            "did_agent11_find_chains",
            "is_novelty_score_above_0.6",
            "was_web_search_triggered",
        ],
        "max_requery": 0,
        "min_atoms_needed": 10,
        "use_full_document": False,
        "description": "Novelty detection — "
                       "gaps and contributions"
    },

    # ── 10. TIMELINE ───────────────────────────────────────
    # "How did X evolve?"
    # "What happened first?"
    # "Chronological development of Y"
    "timeline": {
        "required_agents": [
            "agent1_router",      # find time entities
            "agent3_navigator",   # all sections with dates
            "agent5_expansion",   # expand
            "agent8_temporal",    # KEY — order by time
            "agent9_calibration", # score
        ],
        "optional_agents": [
            "agent6_validation",  # verify date claims
            "agent2_decomposer",  # if multi-era
        ],
        "skip_agents": [
            "agent4_retrieval",     # skip BM25, use full section
            "agent7_contradiction", # timeline not conflict
            "agent11_synthesis",    # no synthesis
            "agent12_websearch",    # vault only
        ],
        "agent10_checks": [
            "did_agent8_find_temporal_markers",
            "is_chronological_order_correct",
            "are_dates_grounded",
        ],
        "max_requery": 0,
        "min_atoms_needed": 5,
        "use_full_document": False,
        "description": "Timeline extraction — "
                       "chronological ordering"
    },

    # ── 11. VERIFICATION ───────────────────────────────────
    # "Is this claim true?"
    # "Verify: X causes Y"
    # "Is it correct that Z?"
    "verification": {
        "required_agents": [
            "agent1_router",        # identify claim
            "agent3_navigator",     # find evidence
            "agent4_retrieval",     # claim atoms
            "agent5_expansion",     # context
            "agent6_validation",    # KEY — verify claim
            "agent7_contradiction", # find counter-evidence
            "agent9_calibration",   # honest score
        ],
        "optional_agents": [
            "agent12_websearch",    # if not in vault
        ],
        "skip_agents": [
            "agent2_decomposer",    # single claim
            "agent8_temporal",      # usually not needed
            "agent11_synthesis",    # not synthesis
        ],
        "agent10_checks": [
            "is_claim_grounded_or_ungrounded",
            "any_counter_evidence_found",
            "is_verdict_clearly_stated",
        ],
        "max_requery": 1,
        "min_atoms_needed": 3,
        "use_full_document": False,
        "description": "Claim verification — "
                       "true/false with evidence"
    },

    # ── 12. DEEP RESEARCH ──────────────────────────────────
    # "Give me everything about X"
    # Complex multi-part research queries
    # Full analysis needed
    "deep_research": {
        "required_agents": [
            "agent1_router",
            "agent2_decomposer",
            "agent3_navigator",
            "agent4_retrieval",
            "agent5_expansion",
            "agent6_validation",
            "agent7_contradiction",
            "agent8_temporal",
            "agent9_calibration",
            "agent11_synthesis",
        ],
        "optional_agents": [
            "agent12_websearch",  # if gap found
        ],
        "skip_agents": [],
        "agent10_checks": [
            "all_agents_scored_above_0.5",
            "pipeline_grade_above_C",
            "novel_connections_found",
        ],
        "max_requery": 2,
        "min_atoms_needed": 20,
        "use_full_document": False,
        "description": "Full analysis — all agents — "
                       "maximum depth"
    },

    # ── 13. PAPER WRITING ──────────────────────────────────────
    # "Write a research paper on X"
    # "Write a review article about Y"
    "paper_writing": {
        "required_agents": [
            "agent1_router",
            "agent3_navigator",
            "agent4_retrieval",
            "agent5_expansion",
            "agent11_synthesis",
            "agent12_websearch",
        ],
        "optional_agents": [
            "agent6_validation",
        ],
        "skip_agents": [
            "agent2_decomposer",
            "agent7_contradiction",
            "agent8_temporal",
            "agent9_calibration",
        ],
        "agent10_checks": [
            "did_agent12_find_sources",
            "is_narrative_complete",
        ],
        "max_requery": 0,
        "min_atoms_needed": 10,
        "use_full_document": True,
        "description": "Paper writing — full paper "
                       "with web evidence"
    },

    # ── 14. IMPLEMENTATION GUIDE ───────────────────────────────
    # "How to implement this?"
    # "Guide me to build X"
    "implementation_guide": {
        "required_agents": [
            "agent3_navigator",
            "agent4_retrieval",
            "agent5_expansion",
            "agent11_synthesis",
            "agent12_websearch",
        ],
        "optional_agents": [
            "agent8_temporal",
        ],
        "skip_agents": [
            "agent1_router",
            "agent2_decomposer",
            "agent6_validation",
            "agent7_contradiction",
            "agent9_calibration",
        ],
        "agent10_checks": [
            "does_guide_have_code",
            "are_datasets_specified",
            "is_plan_week_by_week",
            "are_pitfalls_listed",
        ],
        "max_requery": 0,
        "min_atoms_needed": 10,
        "use_full_document": False,
        "description": "Implementation guide — "
                       "code, datasets, plan, pitfalls"
    },
}

# ── QUERY TYPE DETECTION RULES ───────────────────────────────
# Agent 1 uses these patterns to detect query type
# before deciding which agents to dispatch

QUERY_DETECTION_PATTERNS = {

    "summary": [
        "summarize", "summarise", "summary",
        "overview", "what is this paper about",
        "brief description", "give me an overview",
        "what does this paper do", "abstract",
        "main points", "key points", "tldr",
        "in brief", "shortly describe"
    ],

    "bibliography": [
        "references", "bibliography", "citations",
        "cited", "cited works", "reference list",
        "what papers", "list all references",
        "cite", "works cited", "sources used"
    ],

    "comparative": [
        "compare", "comparison", "versus", "vs",
        "difference between", "better than",
        "worse than", "how does X differ",
        "contrast", "which is better",
        "advantages and disadvantages"
    ],

    "results": [
        "results", "performance", "accuracy",
        "score", "benchmark", "metric",
        "achieved", "obtained", "reported",
        "percentage", "f1", "bleu", "rouge",
        "precision", "recall", "evaluation"
    ],

    "methodology": [
        "method", "methodology", "approach",
        "how was", "procedure", "algorithm",
        "training", "architecture", "design",
        "implementation", "technique", "process",
        "how does it work", "pipeline"
    ],

    "limitations": [
        "limitation", "weakness", "drawback",
        "problem", "issue", "fail", "cannot",
        "challenge", "shortcoming", "constraint",
        "disadvantage", "what it lacks",
        "future work", "not addressed"
    ],

    "novelty": [
        "novel", "contribution", "new",
        "innovation", "gap", "original",
        "first time", "propose", "introduces",
        "unlike previous", "state of the art",
        "what is unique", "research gap"
    ],

    "explanation": [
        "explain", "what is", "define",
        "how does", "meaning of", "describe",
        "clarify", "elaborate", "what does",
        "what are", "tell me about"
    ],

    "timeline": [
        "timeline", "history", "evolution",
        "chronolog", "over time", "developed",
        "first", "then", "later", "before",
        "since", "progress", "milestones"
    ],

    "verification": [
        "is it true", "verify", "check",
        "is it correct", "does the paper say",
        "confirm", "validate", "fact check",
        "is the claim", "does it state"
    ],

    "factual": [
        "what is the", "how many", "which",
        "when", "who", "where", "what value",
        "what number", "how much",
        "exact", "specific"
    ],

    "paper_writing": [
        "write a paper", "write a research paper",
        "write paper", "draft a paper",
        "write a review paper", "write a review article",
        "write an article", "paper on",
        "research paper on", "review paper on",
        "short communication", "case study on",
        "perspective article", "technical note on",
        "systematic review on", "write for journal",
        "conference paper", "submit to",
        "generate a paper", "author a paper",
        "write a paper on", "write me a paper",
        "produce a paper", "create a research paper",
        "write a survey", "write a survey on",
        "write a literature review",
    ],

    "implementation_guide": [
        "how to implement", "implement this",
        "guide me", "guide me to implement",
        "guide me to build", "guide me through",
        "show me how", "show me how to implement",
        "how do i build", "code for",
        "step by step", "implementation plan",
        "help me implement", "how to build",
        "development guide", "coding guide",
        "implementation guide", "help me code",
        "how to code", "teach me to implement",
        "build this", "build a", "create an implementation",
        "practical guide", "hands-on guide",
    ],
}

# ── AGENT 10 REVIEW RULES ────────────────────────────────────
# After each agent runs, Agent 10 checks these conditions
# If check FAILS → Agent 10 can retry or skip to next

AGENT10_REVIEW_RULES = {

    "agent1_router": {
        "pass_if": [
            "intent is not 'unknown'",
            "rewritten_query length > 5 words",
            "confidence > 0.6",
        ],
        "on_fail": "retry_once",
        "max_retries": 1,
    },

    "agent2_decomposer": {
        "pass_if": [
            "sub_queries is a non-empty list",
            "each sub_query length > 3 words",
            "sub_query count <= MAX_SUB_QUERIES",
        ],
        "on_fail": "skip",    # use original query
        "max_retries": 0,
    },

    "agent3_navigator": {
        "pass_if": [
            "selected_nodes is not empty",
            "confidence > 0.5",
            "at least 1 valid node_id returned",
        ],
        "on_fail": "use_all_sections",
        "max_retries": 1,
    },

    "agent4_retrieval": {
        "pass_if": [
            "anchors count >= 2",
            "at least 1 anchor has score > 0.1",
        ],
        "on_fail": "expand_scope_to_full_doc",
        "max_retries": 1,
    },

    "agent5_expansion": {
        "pass_if": [
            "atom_count >= min_atoms_needed",
            "narrative length > 200 chars",
            "gap_count <= 5",
        ],
        "on_fail": "increase_radius",
        "max_retries": 0,
    },

    "agent6_validation": {
        "pass_if": [
            "verdict in ['grounded', 'partially_grounded']",
            "confidence_score > 0.3",
        ],
        "on_fail": "trigger_requery",
        "max_retries": 2,
    },

    "agent7_contradiction": {
        "pass_if": [
            "consistency_score returned",
            "contradiction_details is a list",
        ],
        "on_fail": "skip",    # not critical
        "max_retries": 0,
    },

    "agent8_temporal": {
        "pass_if": [
            "temporal_markers is a list",
            "ordered_narrative is not empty",
        ],
        "on_fail": "use_original_narrative",
        "max_retries": 0,
    },

    "agent9_calibration": {
        "pass_if": [
            "calibrated_score between 0 and 1",
            "trust_level in ['high','medium','low']",
        ],
        "on_fail": "use_base_score",
        "max_retries": 0,
    },

    "agent11_synthesis": {
        "pass_if": [
            "synthesis_performed is boolean",
            "novel_connections is a list",
        ],
        "on_fail": "skip",    # synthesis optional
        "max_retries": 0,
    },

    "agent12_websearch": {
        "pass_if": [
            "results count > 0",
            "search_query not empty",
        ],
        "on_fail": "skip",    # web optional
        "max_retries": 1,
    },

    "agent13_paper_writer": {
        "pass_if": [
            "sections is not empty",
            "word_count > 500",
        ],
        "on_fail": "retry_once",
        "max_retries": 1,
    },

    "agent14_impl_guide": {
        "pass_if": [
            "guide is not empty",
            "breakdown exists",
        ],
        "on_fail": "skip",
        "max_retries": 0,
    },
}

# ── AGENT SPEED ESTIMATES ────────────────────────────────────
# Approximate time per agent on CPU
# Used by Agent 10 to estimate total time before running

AGENT_SPEED_ESTIMATES = {
    "agent1_router":        {"cpu_secs": 15, "gpu_secs": 3},
    "agent2_decomposer":    {"cpu_secs": 12, "gpu_secs": 2},
    "agent3_navigator":     {"cpu_secs": 20, "gpu_secs": 4},
    "agent4_retrieval":     {"cpu_secs": 1,  "gpu_secs": 1},
    "agent5_expansion":     {"cpu_secs": 1,  "gpu_secs": 1},
    "agent6_validation":    {"cpu_secs": 45, "gpu_secs": 8},
    "agent7_contradiction": {"cpu_secs": 30, "gpu_secs": 6},
    "agent8_temporal":      {"cpu_secs": 25, "gpu_secs": 5},
    "agent9_calibration":   {"cpu_secs": 1,  "gpu_secs": 1},
    "agent10_super":        {"cpu_secs": 20, "gpu_secs": 4},
    "agent11_synthesis":    {"cpu_secs": 25, "gpu_secs": 5},
    "agent12_websearch":    {"cpu_secs": 15, "gpu_secs": 15},
    "agent13_paper_writer": {"cpu_secs": 300, "gpu_secs": 120},
    "agent14_impl_guide":   {"cpu_secs": 300, "gpu_secs": 120},
}

def estimate_total_time(query_type: str,
                         use_gpu: bool = False) -> dict:
    """
    Estimate total time before running query.
    Shown to user before pipeline starts.
    """
    rules    = ROUTING_RULES.get(
        query_type, ROUTING_RULES["factual"]
    )
    agents   = (rules["required_agents"] +
                rules["optional_agents"])
    key      = "gpu_secs" if use_gpu else "cpu_secs"
    total    = sum(
        AGENT_SPEED_ESTIMATES.get(a, {}).get(key, 10)
        for a in agents
    )
    return {
        "query_type":    query_type,
        "agents_needed": len(agents),
        "estimated_secs": total,
        "description":   rules["description"]
    }
