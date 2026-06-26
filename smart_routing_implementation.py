# ============================================================
# SMART AGENT ROUTING IMPLEMENTATION
# 2 files to update in your existing project:
#
# File 1: agent1_router.py  → detect query type
# File 2: agent10_super.py  → dispatch only needed agents
# ============================================================


# ============================================================
# FILE 1: agent1_router.py
# ADD this function to your existing agent1_router.py
# ============================================================

"""
ADD THIS ENTIRE FUNCTION to your agent1_router.py
Place it BEFORE your existing route() function
"""

from agent_routing_rules import (
    ROUTING_RULES,
    QUERY_DETECTION_PATTERNS,
    estimate_total_time
)

def detect_query_type(query: str) -> str:
    """
    Detects query type from user input.
    Returns one of:
      summary, factual, comparative, methodology,
      results, limitations, bibliography, explanation,
      novelty, timeline, verification, deep_research

    How it works:
    1. Lowercase the query
    2. Check each type's keyword patterns
    3. Count how many keywords match
    4. Return type with highest match count
    5. Default to factual if no match

    No LLM needed — pure keyword matching
    Runs in milliseconds
    """
    query_lower = query.lower()
    scores      = {}

    for qtype, patterns in QUERY_DETECTION_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if pattern.lower() in query_lower:
                score += 1
        if score > 0:
            scores[qtype] = score

    if not scores:
        return "factual"  # default

    # Return type with most keyword matches
    detected = max(scores, key=scores.get)

    # Special overrides:
    # Bibliography always wins if detected
    if scores.get("bibliography", 0) > 0:
        detected = "bibliography"

    # Summary wins over explanation for full-doc queries
    if (scores.get("summary", 0) > 0 and
            "this paper" in query_lower):
        detected = "summary"

    return detected


def get_agents_for_query(query: str,
                          query_type: str = None
                          ) -> dict:
    """
    Returns the exact agents to run for this query.
    Called by Agent 10 before pipeline starts.

    Returns:
    {
      "query_type":       "summary",
      "required_agents":  ["agent3_navigator",...],
      "optional_agents":  ["agent8_temporal"],
      "skip_agents":      ["agent1_router",...],
      "agent10_checks":   [...],
      "max_requery":      0,
      "is_bibliography":  False,
      "use_full_document": True,
      "estimated_secs":   45,
      "description":      "Full document overview"
    }
    """
    if query_type is None:
        query_type = detect_query_type(query)

    rules = ROUTING_RULES.get(
        query_type,
        ROUTING_RULES["factual"]  # safe default
    )

    estimate = estimate_total_time(query_type)

    print(
        f"[Agent1] Query type detected: "
        f"{query_type.upper()}"
    )
    print(
        f"[Agent1] Agents needed: "
        f"{len(rules['required_agents'])} required + "
        f"{len(rules['optional_agents'])} optional"
    )
    print(
        f"[Agent1] Skipping: "
        f"{len(rules['skip_agents'])} agents"
    )
    print(
        f"[Agent1] Est. time: "
        f"~{estimate['estimated_secs']}s"
    )

    return {
        "query_type":        query_type,
        "required_agents":   rules["required_agents"],
        "optional_agents":   rules["optional_agents"],
        "skip_agents":       rules["skip_agents"],
        "agent10_checks":    rules["agent10_checks"],
        "max_requery":       rules["max_requery"],
        "min_atoms_needed":  rules.get(
            "min_atoms_needed", 5
        ),
        "is_bibliography":   rules.get(
            "is_bibliography", False
        ),
        "use_full_document": rules.get(
            "use_full_document", False
        ),
        "estimated_secs":    estimate["estimated_secs"],
        "description":       rules["description"]
    }


# ============================================================
# FILE 2: agent10_super.py
# REPLACE your execute() function with this version
# ============================================================

"""
REPLACE your existing execute() function
with this smart routing version.

Key changes:
1. Calls get_agents_for_query() first
2. Only runs required + optional agents
3. Skips everything in skip_agents list
4. Agent 10 reviews each agent per rules
5. Shows estimated time before starting
"""

import time
from agent_routing_rules import (
    AGENT10_REVIEW_RULES
)

# Import all agents
from agents import agent1_router   as a1
from agents import agent2_decomposer as a2
from agents import agent3_navigator  as a3
from agents import agent4_retrieval  as a4
from agents import agent5_expansion  as a5
from agents import agent6_validation as a6
from agents import agent7_contradiction as a7
from agents import agent8_temporal   as a8
from agents import agent9_calibration as a9
from agents import agent11_synthesis  as a11
from agents import agent12_websearch  as a12
from agents.agent6_validation import RequerySignal


def _agent10_review(agent_name: str,
                     result,
                     routing: dict) -> dict:
    """
    Agent 10 reviews each agent output.
    Checks against AGENT10_REVIEW_RULES.
    Returns: {pass: bool, action: str, reason: str}
    """
    rules = AGENT10_REVIEW_RULES.get(agent_name, {})

    if not rules:
        return {
            "pass":   True,
            "action": "proceed",
            "reason": "No rules defined"
        }

    # Check each condition
    failed_checks = []

    for condition in rules.get("pass_if", []):

        passed = _evaluate_condition(
            condition, result, routing
        )
        if not passed:
            failed_checks.append(condition)

    if failed_checks:
        action = rules.get("on_fail", "skip")
        return {
            "pass":           False,
            "action":         action,
            "reason":         f"Failed: {failed_checks}",
            "failed_checks":  failed_checks
        }

    return {
        "pass":   True,
        "action": "proceed",
        "reason": "All checks passed"
    }


def _evaluate_condition(condition: str,
                          result,
                          routing: dict) -> bool:
    """
    Evaluates a condition string against agent result.
    Simple rule evaluator — no LLM needed.
    """
    try:
        c = condition.lower()

        # Agent 1 checks
        if "intent is not 'unknown'" in c:
            return (isinstance(result, dict) and
                    result.get("intent", "unknown")
                    != "unknown")

        if "rewritten_query length" in c:
            q = result.get("rewritten_query", "")
            return len(q.split()) > 5

        if "confidence > 0.6" in c:
            return result.get("confidence", 0) > 0.6

        if "confidence > 0.5" in c:
            return result.get("confidence", 0) > 0.5

        if "confidence > 0.3" in c:
            return (result.get(
                "confidence_score", 0
            ) > 0.3)

        # Agent 2 checks
        if "sub_queries is a non-empty list" in c:
            return (isinstance(result, list) and
                    len(result) > 0)

        if "each sub_query length" in c:
            if not isinstance(result, list):
                return False
            return all(
                len(q.split()) > 3 for q in result
            )

        # Agent 3 checks
        if "selected_nodes is not empty" in c:
            nodes = (result.get("selected_nodes", [])
                     if isinstance(result, dict)
                     else [])
            return len(nodes) > 0

        if "at least 1 valid node_id" in c:
            nodes = (result.get("selected_nodes", [])
                     if isinstance(result, dict)
                     else [])
            return len(nodes) > 0

        # Agent 4 checks
        if "anchors count >= 2" in c:
            return (isinstance(result, list) and
                    len(result) >= 2)

        if "at least 1 anchor has score > 0.1" in c:
            if not isinstance(result, list):
                return False
            return any(
                a.get("combined_score", 0) > 0.1
                for a in result
            )

        # Agent 5 checks
        if "atom_count >=" in c:
            min_needed = routing.get(
                "min_atoms_needed", 3
            )
            return (isinstance(result, dict) and
                    result.get("atom_count", 0)
                    >= min_needed)

        if "narrative length > 200" in c:
            return (isinstance(result, dict) and
                    len(result.get("narrative",""))
                    > 200)

        if "gap_count <= 5" in c:
            return (isinstance(result, dict) and
                    result.get("gap_count", 0) <= 5)

        # Agent 6 checks
        if "verdict in" in c:
            verdict = (result.get("verdict","")
                       if isinstance(result,dict)
                       else "")
            return verdict in [
                "grounded", "partially_grounded"
            ]

        # Default: pass
        return True

    except Exception:
        return True  # don't block on check errors


def execute_smart(question: str,
                   doc_id: str,
                   tree: list,
                   atom_store,
                   bm25_index,
                   triple_store,
                   causal_store,
                   feedback_index,
                   forced_query_type: str = None
                   ) -> dict:
    """
    Smart routing execute function.
    Replaces the original execute() in agent10_super.py

    Changes from original:
    1. Detects query type first
    2. Only runs needed agents
    3. Skips irrelevant agents
    4. Reviews each agent per rules
    5. Shows progress clearly

    Usage:
    result = execute_smart(
        question="Summarise this paper",
        doc_id="abc123",
        tree=tree,
        ...
    )
    """
    start_time = time.time()

    # ── STEP 1: DETECT QUERY TYPE ─────────────────────────
    query_type = (forced_query_type or
                  detect_query_type(question))
    routing    = get_agents_for_query(
        question, query_type
    )

    required = routing["required_agents"]
    optional = routing["optional_agents"]
    skipped  = routing["skip_agents"]
    is_bib   = routing["is_bibliography"]
    use_full = routing["use_full_document"]

    # Build display list: agent12 only shows when paper/guide query
    _a12_active = query_type in ("paper_writing", "implementation_guide")
    _display_running = [
        a for a in (required + optional)
        if a != "agent12_websearch"
    ]
    if _a12_active:
        _display_running.append("agent12_websearch")
    _display_skipping = list(skipped)
    if not _a12_active and "agent12_websearch" not in _display_skipping:
        _display_skipping.append("agent12_websearch (paper/guide only)")

    print(f"\n{'='*50}")
    print(f"[Agent10] Query type: {query_type.upper()}")
    print(f"[Agent10] Running: {_display_running}")
    print(f"[Agent10] Skipping: {_display_skipping}")
    print(f"[Agent10] Est. time: ~{routing['estimated_secs']}s")
    print(f"{'='*50}\n")

    # ── STEP 2: WARM START ────────────────────────────────
    warm        = feedback_index.get_warm_start(
        question, doc_id
    )
    warm_atoms  = warm["warm_atom_ids"] if warm else []
    warm_nodes  = warm["warm_node_ids"]  if warm else []

    # ── STEP 3: INITIALISE STATE ──────────────────────────
    agent_results  = []
    routed         = None
    sub_queries    = [question]
    navigation     = None
    selected_nodes = warm_nodes or []
    anchors        = []
    expansion      = None
    temporal       = None
    validation     = None
    contradiction  = None
    calibration    = None
    synthesis      = None

    def is_needed(agent_name: str) -> bool:
        """Check if agent should run."""
        if agent_name in skipped:
            return False
        if agent_name in required:
            return True
        if agent_name in optional:
            return True
        return False

    def record(name, result, grade, score,
               elapsed, skipped_flag=False,
               action="proceed"):
        agent_results.append({
            "agent":   name,
            "grade":   grade,
            "score":   score,
            "elapsed": round(elapsed, 2),
            "skipped": skipped_flag,
            "action":  action
        })
        status = "SKIP" if skipped_flag else f"grade={grade}"
        print(
            f"[Agent10] ✓ {name}: "
            f"{status} score={score:.2f} "
            f"({elapsed:.1f}s)"
        )

    # ── STEP 4: AGENT 1 — ROUTER ──────────────────────────
    t0 = time.time()
    if is_needed("agent1_router"):
        try:
            routed = a1.route(question)
            routed["query_type"]     = query_type
            routed["is_bibliography"] = is_bib

            review = _agent10_review(
                "agent1_router", routed, routing
            )
            if review["pass"]:
                record("agent1_router", routed,
                       "B", 0.85,
                       time.time()-t0)
            else:
                print(
                    f"[Agent10] agent1_router check "
                    f"failed: {review['reason']}"
                )
                record("agent1_router", routed,
                       "C", 0.50,
                       time.time()-t0,
                       action=review["action"])
        except Exception as e:
            print(f"[Agent10] agent1_router error: {e}")
            routed = {
                "intent":         query_type,
                "rewritten_query": question,
                "key_entities":   [],
                "is_complex":     False,
                "original_query": question,
                "query_type":     query_type,
                "is_bibliography": is_bib
            }
    else:
        # Router skipped — build minimal routed dict
        routed = {
            "intent":         query_type,
            "rewritten_query": question,
            "key_entities":   [],
            "is_complex":     False,
            "original_query": question,
            "query_type":     query_type,
            "is_bibliography": is_bib
        }
        record("agent1_router", None,
               "S", 1.0, 0.0,
               skipped_flag=True)

    # ── STEP 5: AGENT 2 — DECOMPOSER ─────────────────────
    t0 = time.time()
    if is_needed("agent2_decomposer"):
        try:
            sub_queries = a2.decompose(routed)
            review = _agent10_review(
                "agent2_decomposer",
                sub_queries, routing
            )
            if not review["pass"]:
                print(
                    "[Agent10] Decomposer failed "
                    "check — using original query"
                )
                sub_queries = [
                    routed["rewritten_query"]
                ]
            record("agent2_decomposer", sub_queries,
                   "B", 0.70, time.time()-t0)
        except Exception as e:
            print(f"[Agent10] agent2 error: {e}")
            sub_queries = [routed["rewritten_query"]]
    else:
        sub_queries = [routed["rewritten_query"]]
        record("agent2_decomposer", None,
               "S", 1.0, 0.0,
               skipped_flag=True)

    # ── STEP 6: AGENT 3 — NAVIGATOR ──────────────────────
    t0 = time.time()
    if is_needed("agent3_navigator") and not is_bib:
        try:
            # For summary — navigate all sections
            if query_type == "summary" or use_full:
                selected_nodes = [
                    n["node_id"] for n in tree
                ]
                navigation = {
                    "selected_nodes": selected_nodes,
                    "reasoning": "Full document summary",
                    "confidence": 1.0
                }
                print(
                    "[Agent3] Summary mode — "
                    "all sections selected"
                )
            else:
                q = routed["rewritten_query"]
                navigation = a3.navigate(q, tree)
                selected_nodes = navigation.get(
                    "selected_nodes",
                    warm_nodes or []
                )

            review = _agent10_review(
                "agent3_navigator",
                navigation, routing
            )
            if not review["pass"]:
                print(
                    "[Agent10] Agent3 failed — "
                    "expanding to all sections"
                )
                selected_nodes = [
                    n["node_id"] for n in tree
                ]

            record("agent3_navigator", navigation,
                   "B", navigation.get("confidence",0.7),
                   time.time()-t0)
        except Exception as e:
            print(f"[Agent10] agent3 error: {e}")
            selected_nodes = [
                n["node_id"] for n in tree
            ]
    elif is_bib:
        # Bibliography — force last section
        if tree:
            selected_nodes = [tree[-1]["node_id"]]
        print(
            "[Agent3] Bibliography mode — "
            "last section forced"
        )
        record("agent3_navigator", None,
               "S", 1.0, 0.0,
               skipped_flag=True)
    else:
        selected_nodes = [n["node_id"] for n in tree]
        record("agent3_navigator", None,
               "S", 1.0, 0.0,
               skipped_flag=True)

    # ── STEP 7: AGENT 4 — RETRIEVAL ──────────────────────
    t0 = time.time()
    all_narratives = []
    all_atom_ids   = []
    all_pages      = []

    for sq in sub_queries:
        time.time()
        try:
            anchors = a4.retrieve(
                sq, selected_nodes, tree,
                atom_store, bm25_index,
                triple_store, warm_atoms,
                is_bibliography=is_bib
            )
            review = _agent10_review(
                "agent4_retrieval",
                anchors, routing
            )
            if not review["pass"]:
                action = review["action"]
                print(
                    f"[Agent10] Agent4 failed: "
                    f"{review['reason']} → {action}"
                )
                if action == "expand_scope_to_full_doc":
                    anchors = a4.retrieve(
                        sq,
                        [n["node_id"] for n in tree],
                        tree, atom_store, bm25_index,
                        triple_store, warm_atoms,
                        is_bibliography=is_bib
                    )
        except Exception as e:
            print(f"[Agent10] agent4 error: {e}")
            anchors = []

        # ── STEP 8: AGENT 5 — EXPANSION ──────────────────
        time.time()
        try:
            expansion = a5.expand(
                anchor_atoms=anchors,
                atom_store=atom_store,
                query=sq,
                is_bibliography=is_bib
            )
            review5 = _agent10_review(
                "agent5_expansion",
                expansion, routing
            )
            if not review5["pass"]:
                print(
                    f"[Agent10] Agent5 narrative "
                    f"thin: {review5['reason']}"
                )

            all_narratives.append(expansion["narrative"])
            all_atom_ids.extend(
                expansion["atom_ids_used"]
            )
            all_pages.extend(
                expansion["pages_referenced"]
            )
        except Exception as e:
            print(f"[Agent10] agent5 error: {e}")

    # Merge all sub-query narratives
    merged_narrative = "\n\n---\n\n".join(
        all_narratives
    )
    unique_atom_ids  = list(set(all_atom_ids))
    unique_pages     = sorted(set(all_pages))

    record("agent4_retrieval", anchors,
           "C", 0.70, time.time()-t0)
    record("agent5_expansion", expansion,
           "A", 1.00, time.time()-t0)

    # ── STEP 9: AGENT 8 — TEMPORAL ───────────────────────
    t0 = time.time()
    temporal = {
        "time_ordered":      False,
        "ordered_narrative": merged_narrative
    }
    if is_needed("agent8_temporal"):
        try:
            temporal = a8.analyze(
                merged_narrative, []
            )
            record("agent8_temporal", temporal,
                   "B", 0.75, time.time()-t0)
        except Exception as e:
            print(f"[Agent10] agent8 error: {e}")
            record("agent8_temporal", None,
                   "C", 0.50, time.time()-t0)
    else:
        record("agent8_temporal", None,
               "S", 1.0, 0.0,
               skipped_flag=True)

    final_narrative = (
        temporal.get("ordered_narrative",
                     merged_narrative)
    )

    # ── STEP 10: GENERATE ANSWER ──────────────────────────
    print(
        "\n[Agent10] → Generating final "
        "context-grounded answer..."
    )
    from llm.router import generate as llm_generate

    answer = llm_generate(
        "answer_generation",
        f"Answer ONLY from context below.\n"
        f"Question: {question}\n"
        f"Context:\n{final_narrative}\nAnswer:",
        temperature=0.1
    )

    # ── STEP 11: AGENT 6 — VALIDATION ────────────────────
    t0 = time.time()
    validation = {
        "verdict":           "unknown",
        "confidence_score":  0.5,
        "grounded_claims":   [],
        "ungrounded_claims": [],
        "requery_count":     0
    }

    if is_needed("agent6_validation"):
        max_requery = routing["max_requery"]
        for attempt in range(max_requery + 1):
            try:
                validation = a6.validate(
                    answer, final_narrative, question
                )
                validation["requery_count"] = attempt
                break
            except RequerySignal as rs:
                if attempt < max_requery:
                    print(
                        f"[Agent10] Requery {attempt+1}"
                        f": {rs.reason}"
                    )
                    re_anchors = a4.retrieve(
                        rs.refined_query,
                        selected_nodes, tree,
                        atom_store, bm25_index,
                        triple_store
                    )
                    re_exp = a5.expand(
                        re_anchors, atom_store,
                        rs.refined_query
                    )
                    final_narrative = re_exp["narrative"]
                    answer = llm_generate(
                        "answer_generation",
                        f"Answer ONLY from context.\n"
                        f"Question: {question}\n"
                        f"Context:\n"
                        f"{final_narrative}\nAnswer:",
                        temperature=0.1
                    )
                else:
                    validation["requery_count"] = attempt

        record("agent6_validation", validation,
               "B", validation.get(
                   "confidence_score", 0.5
               ), time.time()-t0)
    else:
        record("agent6_validation", None,
               "S", 1.0, 0.0,
               skipped_flag=True)

    # ── STEP 12: AGENT 7 — CONTRADICTION ─────────────────
    t0 = time.time()
    contradiction = {
        "contradictions_found": False,
        "llm_contradictions":   [],
        "consistency_score":    1.0
    }

    if is_needed("agent7_contradiction"):
        try:
            contradiction = a7.detect(
                unique_atom_ids,
                triple_store,
                final_narrative
            )
            record(
                "agent7_contradiction",
                contradiction,
                "B",
                contradiction.get(
                    "consistency_score", 0.5
                ),
                time.time()-t0
            )
        except Exception as e:
            print(f"[Agent10] agent7 error: {e}")
            record("agent7_contradiction", None,
                   "C", 0.50, time.time()-t0)
    else:
        print(
            "[SuperAgent] Skipping Agent7 "
            f"(contradiction audit) for "
            f"{query_type} query."
        )
        record("agent7_contradiction", None,
               "S", 1.0, 0.0,
               skipped_flag=True)

    # ── STEP 13: AGENT 9 — CALIBRATION ───────────────────
    t0 = time.time()
    exp_stats = {
        "gap_count":   expansion.get("gap_count", 0)
                       if expansion else 0,
        "atom_count":  len(unique_atom_ids)
    }
    calibration = a9.calibrate(
        validation, contradiction,
        exp_stats, temporal,
        routed.get("intent", query_type)
    )
    record("agent9_calibration", calibration,
           "A", calibration["calibrated_score"],
           time.time()-t0)

    # ── STEP 14: AGENT 11 — SYNTHESIS ────────────────────
    t0 = time.time()
    synthesis = {
        "novel_connections":   [],
        "synthesis_performed": False
    }

    if is_needed("agent11_synthesis"):
        try:
            synthesis = a11.synthesize(
                routed.get("key_entities", []),
                causal_store,
                atom_store
            )
            record("agent11_synthesis", synthesis,
                   "B" if synthesis[
                       "synthesis_performed"
                   ] else "F",
                   0.75 if synthesis[
                       "synthesis_performed"
                   ] else 0.0,
                   time.time()-t0)
        except Exception as e:
            print(f"[Agent10] agent11 error: {e}")
            record("agent11_synthesis", None,
                   "F", 0.0, time.time()-t0)
    else:
        record("agent11_synthesis", None,
               "S", 1.0, 0.0,
               skipped_flag=True)

    # ── STEP 15: AGENT 12 — WEB SEARCH ───────────────────
    t0 = time.time()
    web_result = None

    if is_needed("agent12_websearch"):
        should_search = (
            calibration["calibrated_score"] <
            0.70 and
            synthesis.get("novel_connections")
        )
        if should_search:
            try:
                gap_desc = " ".join([
                    f"{n.get('from','')} "
                    f"to {n.get('to','')}"
                    for n in synthesis[
                        "novel_connections"
                    ][:1]
                ])
                web_result = a12.search_and_solve(
                    gap_desc,
                    synthesis["novel_connections"]
                )
                record("agent12_websearch",
                       web_result, "A", 0.85,
                       time.time()-t0)
            except Exception as e:
                print(f"[Agent10] agent12 error: {e}")
        else:
            record("agent12_websearch", None,
                   "S", 1.0, 0.0,
                   skipped_flag=True)
    else:
        record("agent12_websearch", None,
               "S", 1.0, 0.0,
               skipped_flag=True)

    # ── STEP 16: SAVE EXPERIENCE ──────────────────────────
    try:
        from learning.experience_store import (
            record_query_experience
        )
        record_query_experience(
            question=question,
            doc_id=doc_id,
            useful_atom_ids=unique_atom_ids,
            useful_node_ids=selected_nodes,
            confidence=calibration["calibrated_score"],
            trust_level=calibration["trust_level"],
            agent_grades={
                r["agent"]: r["grade"]
                for r in agent_results
            }
        )
    except Exception:
        pass

    # ── STEP 17: BUILD PIPELINE REPORT ───────────────────
    ran_agents     = [r for r in agent_results
                      if not r["skipped"]]
    skipped_agents = [r for r in agent_results
                      if r["skipped"]]

    avg_score = (
        sum(r["score"] for r in ran_agents) /
        len(ran_agents)
    ) if ran_agents else 0.5

    def grade(s):
        if s >= 0.85:
            return "A"
        if s >= 0.70:
            return "B"
        if s >= 0.50:
            return "C"
        return "F"

    elapsed_total = round(time.time()-start_time, 2)

    return {
        # Main answer
        "answer":             answer,
        "confidence":         calibration[
            "calibrated_score"
        ],
        "trust_level":        calibration["trust_level"],
        "pages_referenced":   unique_pages,

        # Query routing info
        "query_type":         query_type,
        "routing_description": routing["description"],
        "agents_run":         len(ran_agents),
        "agents_skipped":     len(skipped_agents),

        # Analysis
        "contradictions_found": contradiction.get(
            "contradictions_found", False
        ),
        "contradiction_details": contradiction.get(
            "llm_contradictions", []
        ),
        "temporal_ordered":   temporal.get(
            "time_ordered", False
        ),
        "novel_connections":  synthesis.get(
            "novel_connections", []
        ),

        # Pipeline
        "pipeline_grade":     grade(avg_score),
        "reasoning_trail":    agent_results,
        "elapsed_seconds":    elapsed_total,
        "requery_count":      validation.get(
            "requery_count", 0
        ),

        # Web search
        "web_search_run":     web_result is not None,
        "new_papers_added":   (
            web_result.get("new_papers_added", 0)
            if web_result else 0
        ),
    }
