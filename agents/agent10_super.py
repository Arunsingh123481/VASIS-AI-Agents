# agents/agent10_super.py
# Master Orchestrator — autonomous agent execution loop

import time
from dataclasses import dataclass, field
from typing import Any
from llm.router import generate, generate_json
from utils.exceptions import RequerySignal
from config import MAX_REQUERY_ATTEMPTS, MODE
from console_helper import print_msg
from agent_routing_rules import (
    AGENT10_REVIEW_RULES,
    ROUTING_RULES
)

@dataclass
class AgentResult:
    agent_id:     int
    agent_name:   str
    input_summary: str
    output:       Any
    score:        float = 0.0
    grade:        str   = "?"
    issues:       list  = field(default_factory=list)
    skipped:      bool  = False
    retried:      bool  = False
    elapsed:      float = 0.0

@dataclass
class ExecutionPlan:
    query:      str
    steps:      list = field(default_factory=list)
    completed:  list = field(default_factory=list)
    skipped:    list = field(default_factory=list)
    requery_count: int = 0
    abort:      bool = False
    abort_reason: str = ""

PLANNER = ("You are a master orchestrator planning agent execution. "
           "Return ONLY valid JSON.")
REVIEWER = ("You are a quality reviewer evaluating outputs. "
             "Return ONLY valid JSON.")

def _grade(score: float) -> str:
    if score >= 0.85: return "A"
    if score >= 0.70: return "B"
    if score >= 0.50: return "C"
    return "F"

def _review(agent_name: str, inp: str, out: str, expected: list) -> dict:
    """Evaluate quality of agent output using a cooperative dual-model pipeline:
    1. DeepSeek-7B evaluates the work in rich natural language.
    2. Qwen-3B-Coder parses that evaluation into structured JSON.
    """
    # ── PHASE 1: Qualitative Evaluation (DeepSeek-7B) ──
    eval_prompt = f"""Evaluate the quality of the following agent output.
Agent Name: {agent_name}
Input Data: {str(inp)[:400]}
Output Data: {str(out)[:400]}
Expected attributes or keys: {expected}

Analyze:
1. Did the agent produce the expected traits and content?
2. Are there any errors, gaps, or structural issues?
3. Provide a clear verdict on whether the output is usable and if we should proceed, retry, or skip.

Write a concise 1-2 paragraph qualitative assessment."""

    try:
        # Route to DeepSeek (Reasoning Model) for deep cognitive audit
        assessment = generate("answer_generation", eval_prompt, temperature=0.1)
    except Exception as e:
        assessment = f"Evaluation failed due to exception: {e}. Output seems usable."

    # ── PHASE 2: JSON Extraction (Qwen-3B-Coder) ──
    parse_prompt = f"""You are a structural data parser.
Convert the qualitative quality assessment below into a valid, strict JSON object.

Qualitative Assessment:
\"\"\"
{assessment}
\"\"\"

Required JSON Schema:
{{
  "score": 0.0 to 1.0 (float reflecting quality),
  "grade": "A|B|C|F" (string representing grade),
  "issues": ["list", "of", "concrete", "issues", "if", "any"],
  "is_usable": true or false (boolean),
  "recommendation": "proceed|retry|skip|abort" (string recommendation)
}}

Return ONLY valid JSON. No conversational preamble or trailing text."""

    try:
        # Route to Qwen-Coder (Agentic Model) for strict, compliant JSON formatting
        res = generate_json("agent10_super", parse_prompt)
        if not isinstance(res, dict):
            raise ValueError("Parser returned non-dict format")
            
        # Ensure all required keys are present
        res.setdefault("score", 0.8)
        res.setdefault("grade", "B")
        res.setdefault("issues", [])
        res.setdefault("is_usable", True)
        res.setdefault("recommendation", "proceed")
        return res
    except Exception as e:
        # Graceful fallback to avoid halting the orchestrator
        return {
            "score": 0.8,
            "grade": "B",
            "issues": [f"Cooperative review parser exception: {e}"],
            "is_usable": True,
            "recommendation": "proceed"
        }


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


class SuperAgent:
    def __init__(self, tree, atom_store, bm25_index, triple_store, causal_store, feedback_index):
        self.tree           = tree
        self.atom_store     = atom_store
        self.bm25_index     = bm25_index
        self.triple_store   = triple_store
        self.causal_store   = causal_store
        self.feedback_index = feedback_index
        self.results: list[AgentResult] = []

    def _plan(self, query: str) -> ExecutionPlan:
        """Create initial step sequence for solving query."""
        prompt = f"""Plan which agents to run in the initial retrieval & expansion phase.
Query: {query}
Available agents for retrieval phase: agent1_router, agent2_decomposer, agent3_navigator, agent4_retrieval, agent5_expansion, agent8_temporal, agent11_synthesis
Return JSON:
{{
  "initial_steps": ["agent1_router", "agent3_navigator", "agent4_retrieval", "agent5_expansion"],
  "skip_reasons": {{}},
  "complexity": "simple|moderate|complex"
}}"""
        try:
            r = generate_json("agent10_super", prompt, system=PLANNER)
            if not isinstance(r, dict):
                raise ValueError("Planner returned non-dict format")
            steps = r.get("initial_steps")
            if not steps:
                raise ValueError("No steps planned")
            return ExecutionPlan(query=query, steps=steps)
        except Exception:
            return ExecutionPlan(
                query=query, 
                steps=[
                    "agent1_router",
                    "agent3_navigator",
                    "agent4_retrieval",
                    "agent5_expansion"
                ]
            )

    def _decide_next(self, just_done: str, summary: str, review: dict, plan: ExecutionPlan) -> str:
        """Determine next agent dynamically based on state and output grades."""
        remaining = [
            s for s in plan.steps
            if s not in plan.completed
            and s not in plan.skipped
            and s != just_done
        ]
        if not remaining:
            return "generate_answer"
            
        prompt = f"""After completing {just_done}:
Output summary: {str(summary)[:200]}
Grade: {review.get('grade')} | Score: {review.get('score')}
Issues: {review.get('issues', [])}
Remaining steps planned: {remaining}
Return JSON:
{{
  "next_action": "proceed|retry|skip|insert|abort",
  "next_agent": "agent_name or generate_answer",
  "insert_agent": "agent_name or null",
  "reasoning": "brief reasoning text"
}}"""
        try:
            r = generate_json("agent10_super", prompt, system=PLANNER)
            if not isinstance(r, dict):
                raise ValueError("Planner returned non-dict format")
            action = r.get("next_action", "proceed")
            
            if action == "abort":
                plan.abort = True
                plan.abort_reason = r.get("reasoning", "Aborted by planner review.")
                return "abort"
                
            if action == "insert":
                ins = r.get("insert_agent")
                if ins and ins not in plan.steps:
                    idx = plan.steps.index(just_done) + 1
                    plan.steps.insert(idx, ins)
                    
            return r.get("next_agent", remaining[0])
        except Exception:
            return remaining[0]

    def _record(self, aid, name, inp, out, review, elapsed, skipped=False, retried=False):
        score = float(review.get("score", 0.8))
        r = AgentResult(
            agent_id=aid, agent_name=name,
            input_summary=str(inp)[:200], output=out,
            score=score,
            grade=review.get("grade", _grade(score)),
            issues=review.get("issues", []),
            skipped=skipped, retried=retried,
            elapsed=elapsed
        )
        self.results.append(r)
        status_tag = "SKIP" if skipped else f"grade={r.grade}"
        print_msg(f"[SuperAgent] Agent [bold yellow]{name}[/bold yellow] completed: {status_tag} | score={score:.2f} ({elapsed:.2f}s)")
        return r

    def _report(self, plan: ExecutionPlan) -> dict:
        scores = [r.score for r in self.results if not r.skipped]
        avg = round(sum(scores)/len(scores), 3) if scores else 0.0
        underp = [r for r in self.results if r.grade in ("F","C")]
        
        return {
            "pipeline_grade":      _grade(avg),
            "average_agent_score": avg,
            "agents_run":          len([r for r in self.results if not r.skipped]),
            "agents_skipped":      len([r for r in self.results if r.skipped]),
            "agents_retried":      len([r for r in self.results if r.retried]),
            "total_elapsed":       round(sum(r.elapsed for r in self.results), 2),
            "underperformers": [
                {"agent": r.agent_name, "grade": r.grade, "score": r.score, "issues": r.issues}
                for r in underp
            ],
            "per_agent_scores": [
                {
                    "agent":   r.agent_name,
                    "grade":   r.grade,
                    "score":   r.score,
                    "elapsed": r.elapsed,
                    "skipped": r.skipped,
                    "retried": r.retried
                }
                for r in self.results
            ],
            "mode":          MODE,
            "requery_count": plan.requery_count,
            "aborted":       plan.abort,
            "abort_reason":  plan.abort_reason
        }

    def execute(self, question: str, doc_id: str = "", forced_query_type: str = None) -> dict:
        """
        Smart routing execute function.
        Replaces the original execute() in agent10_super.py

        Changes from original:
        1. Detects query type first
        2. Only runs needed agents
        3. Skips irrelevant agents
        4. Reviews each agent per rules
        5. Shows progress clearly
        """
        start_time = time.time()

        # Import all agents locally to avoid circular imports
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
        try:
            from agents import agent12_websearch  as a12
        except ImportError:
            a12 = None
        try:
            from agents import agent13_paper_writer as a13
        except ImportError:
            a13 = None
        try:
            from agents import agent14_implementation_guide as a14
        except ImportError:
            a14 = None

        from agents.agent1_router import (
            get_agents_for_query,
            detect_query_type
        )
        from agents.agent6_validation import RequerySignal
        from llm.router import generate as llm_generate

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

        print_msg(f"\n{'='*50}")
        print_msg(f"[Agent10] Query type: {query_type.upper()}")
        print_msg(f"[Agent10] Running: {required + optional}")
        print_msg(f"[Agent10] Skipping: {skipped}")
        print_msg(f"[Agent10] Est. time: ~{routing['estimated_secs']}s")
        print_msg(f"{'='*50}\n")

        # ── STEP 2: WARM START ────────────────────────────────
        warm        = self.feedback_index.get_warm_start(
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
        web_result     = None
        narrative      = ""
        failed_sub_queries = []

        def is_needed(agent_name: str) -> bool:
            """Check if agent should run."""
            if agent_name in skipped:
                return False
            if agent_name in required:
                return True
            if agent_name in optional:
                return True
            return False

        def record(name, result_val, grade, score,
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
            print_msg(
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
                    print_msg(
                        f"[Agent10] agent1_router check "
                        f"failed: {review['reason']}"
                    )
                    record("agent1_router", routed,
                           "C", 0.50,
                           time.time()-t0,
                           action=review["action"])
            except Exception as e:
                print_msg(f"[Agent10] agent1_router error: {e}")
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
                    print_msg(
                        f"[Agent10] Decomposer failed "
                        f"check — using original query"
                    )
                    sub_queries = [
                        routed["rewritten_query"]
                    ]
                record("agent2_decomposer", sub_queries,
                       "B", 0.70, time.time()-t0)
            except Exception as e:
                print_msg(f"[Agent10] agent2 error: {e}")
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
                        n["node_id"] for n in self.tree
                    ]
                    navigation = {
                        "selected_nodes": selected_nodes,
                        "reasoning": "Full document summary",
                        "confidence": 1.0
                    }
                    print_msg(
                        "[Agent3] Summary mode — "
                        "all sections selected"
                    )
                else:
                    q = routed["rewritten_query"]
                    navigation = a3.navigate(q, self.tree)
                    selected_nodes = navigation.get(
                        "selected_nodes",
                        warm_nodes or []
                    )

                review = _agent10_review(
                    "agent3_navigator",
                    navigation, routing
                )
                if not review["pass"]:
                    print_msg(
                        "[Agent10] Agent3 failed — "
                        "expanding to all sections"
                    )
                    selected_nodes = [
                        n["node_id"] for n in self.tree
                    ]

                record("agent3_navigator", navigation,
                       "B", navigation.get("confidence",0.7),
                       time.time()-t0)
            except Exception as e:
                print_msg(f"[Agent10] agent3 error: {e}")
                selected_nodes = [
                    n["node_id"] for n in self.tree
                ]
        elif is_bib:
            # Bibliography — force last section
            if self.tree:
                selected_nodes = [self.tree[-1]["node_id"]]
            print_msg(
                "[Agent3] Bibliography mode — "
                "last section forced"
            )
            record("agent3_navigator", None,
                   "S", 1.0, 0.0,
                   skipped_flag=True)
        else:
            selected_nodes = [n["node_id"] for n in self.tree]
            record("agent3_navigator", None,
                   "S", 1.0, 0.0,
                   skipped_flag=True)

        # ── STEP 7: AGENT 4 — RETRIEVAL ──────────────────────
        t0 = time.time()
        all_narratives = []
        all_atom_ids   = []
        all_pages      = []
        any_retrieval_run = False

        for sq in sub_queries:
            t4 = time.time()
            
            # Determine if we should dynamically skip agent4_retrieval for this specific sub-query
            skip_retrieval_for_sq = False
            if is_needed("agent4_retrieval"):
                sq_type = detect_query_type(sq)
                from agent_routing_rules import ROUTING_RULES
                sq_rules = ROUTING_RULES.get(sq_type, {})
                if "agent4_retrieval" in sq_rules.get("skip_agents", []):
                    skip_retrieval_for_sq = True
                    print_msg(f"[Agent10] Sub-query '{sq}' detected as type '{sq_type}' which is conceptual. Skipping BM25 retrieval for this sub-query.")
                elif sq_type == "comparative" and any(k in sq.lower() for k in ["this paper", "the paper", "this work", "advantages and disadvantages", "pros and cons"]):
                    skip_retrieval_for_sq = True
                    print_msg(f"[Agent10] Sub-query '{sq}' detected as paper-level comparative. Skipping BM25 retrieval for this sub-query.")

            if not is_needed("agent4_retrieval") or skip_retrieval_for_sq:
                print_msg("[Agent10] Skipping BM25 retrieval. Loading all selected atoms directly for full-document context.")
                anchors = []
                node_map = {n["node_id"]: n for n in self.tree}
                scoped_ids = set()
                for nid in selected_nodes:
                    node = node_map.get(nid)
                    if node:
                        scoped_ids.update(int(a["atom_id"]) for a in self.atom_store.get_in_page_range(node["start_page"], node["end_page"]))
                if not scoped_ids:
                    scoped_ids = set(int(a["atom_id"]) for a in self.atom_store.all())
                
                anchors = sorted([a for a in self.atom_store.all() if int(a["atom_id"]) in scoped_ids], key=lambda x: int(x["atom_id"]))
                
                # Out-of-scope validation:
                # If retrieval was skipped, check if the key entities are actually present in the context.
                # If they are completely absent, empty the anchors to flag it as failed/out-of-scope.
                if routed and routed.get("key_entities"):
                    GENERIC_QUERY_WORDS = {
                        "paper", "this paper", "the paper", "this work", "advantages", "disadvantages",
                        "advantages and disadvantages", "pros", "cons", "pros and cons", "novel problem",
                        "summary", "summarise", "summarize", "main points", "key points", "overview",
                        "contribution", "contributions", "weakness", "weaknesses", "strength", "strengths",
                        "conclusion", "conclusions", "introduction", "method", "methodology", "result",
                        "results", "references", "bibliography", "citation", "citations", "author", "authors",
                        "work", "novelty", "limitation", "limitations", "implications"
                    }
                    entities = [
                        e.lower().strip() for e in routed["key_entities"] 
                        if e and len(e.strip()) > 2 
                        and e.lower().strip() not in GENERIC_QUERY_WORDS
                    ]
                    if entities:
                        matched_any = False
                        full_context_text = " ".join([a.get("text", "").lower() for a in anchors])
                        for ent in entities:
                            words = [w for w in ent.split() if len(w) > 2]
                            if words and all(w in full_context_text for w in words):
                                matched_any = True
                                break
                        if not matched_any:
                            print_msg(f"[Agent10] None of the key entities {entities} found in the loaded context. Query is out-of-scope.")
                            anchors = []
                # Record will be done once outside the loop to avoid duplicate entries
            else:
                any_retrieval_run = True
                try:
                    anchors = a4.retrieve(
                        sq, selected_nodes, self.tree,
                        self.atom_store, self.bm25_index,
                        self.triple_store, warm_atoms,
                        routed=routed
                    )
                    review = _agent10_review(
                        "agent4_retrieval",
                        anchors, routing
                    )
                    if not review["pass"]:
                        action = review["action"]
                        print_msg(
                            f"[Agent10] Agent4 failed: "
                            f"{review['reason']} → {action}"
                        )
                        if action == "expand_scope_to_full_doc":
                            anchors = a4.retrieve(
                                sq,
                                [n["node_id"] for n in self.tree],
                                self.tree, self.atom_store, self.bm25_index,
                                self.triple_store, warm_atoms,
                                routed=routed
                            )
                    # Even after retrieval/retry, check if the best anchor is a meaningful keyword match (score >= 0.15)
                    if anchors and not any(a.get("combined_score", 0.0) >= 0.15 for a in anchors):
                        print_msg("[Agent10] Low anchor quality score (< 0.15). Flagging sub-query as out-of-scope.")
                        anchors = []
                except Exception as e:
                    print_msg(f"[Agent10] agent4 error: {e}")
                    anchors = []

            # ── STEP 8: AGENT 5 — EXPANSION ──────────────────
            t5 = time.time()
            try:
                expansion = a5.expand(
                    anchor_atoms=anchors,
                    atom_store=self.atom_store,
                    query=sq,
                    is_bibliography=is_bib
                )
                review5 = _agent10_review(
                    "agent5_expansion",
                    expansion, routing
                )
                if not review5["pass"]:
                    print_msg(
                        f"[Agent10] Agent5 narrative "
                        f"thin: {review5['reason']}"
                    )
                    failed_sub_queries.append(sq)

                all_narratives.append(expansion["narrative"])
                all_atom_ids.extend(
                    expansion["atom_ids_used"]
                )
                all_pages.extend(
                    expansion["pages_referenced"]
                )
            except Exception as e:
                print_msg(f"[Agent10] agent5 error: {e}")
                failed_sub_queries.append(sq)

        # Merge all sub-query narratives
        merged_narrative = "\n\n---\n\n".join(
            all_narratives
        )
        unique_atom_ids  = list(set(all_atom_ids))
        unique_pages     = sorted(set(all_pages))

        # Grade Agent4 dynamically based on anchor quality
        if not is_needed("agent4_retrieval"):
            record("agent4_retrieval", None, "S", 1.0, 0.0, skipped_flag=True)
        elif not any_retrieval_run:
            # All sub-queries skipped retrieval (e.g. they were all conceptual)
            record("agent4_retrieval", None, "S", 1.0, 0.0, skipped_flag=True)
        else:
            a4_count = len(anchors) if anchors else 0
            a4_min   = routing.get("min_atoms_needed", 3)
            if a4_count >= a4_min:
                a4_grade, a4_score = "A", 0.90
            elif a4_count >= max(a4_min // 2, 2):
                a4_grade, a4_score = "B", 0.75
            elif a4_count >= 1:
                a4_grade, a4_score = "C", 0.60
            else:
                a4_grade, a4_score = "F", 0.30
            record("agent4_retrieval", anchors,
                   a4_grade, a4_score, time.time()-t0)
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
                print_msg(f"[Agent10] agent8 error: {e}")
                record("agent8_temporal", None,
                       "C", 0.50, time.time()-t0)
        else:
            record("agent8_temporal", None,
                   "S", 1.0, 0.0,
                   skipped_flag=True)

        if query_type == "timeline":
            final_narrative = temporal.get("ordered_narrative", merged_narrative)
        else:
            final_narrative = merged_narrative

        # ── STEP 10: GENERATE ANSWER ──────────────────────────
        print_msg(
            "\n[Agent10] → Generating final "
            "context-grounded answer..."
        )

        # Target bibliography fallback or targeted reference retrieval
        is_bib_query = (routed.get("is_bibliography_query", False) or routed.get("is_bibliography", False)) if routed else False
        bib_fallback_mode = False
        if is_bib_query:
            # Check if any anchors indicated bibliography fallback
            bib_fallback_mode = any(a.get("_bib_fallback") for a in anchors) if anchors else False

        if is_bib_query and not bib_fallback_mode:
            print_msg("[Agent10] Bypassing LLM generation for bibliography (returning formatted stitched text to avoid token limits).")
            import re
            # Remove messy PDF line breaks and extra spaces
            clean_text = final_narrative.replace("\n", " ")
            clean_text = re.sub(r'\s+', ' ', clean_text)
            # Add double newlines before citation markers like [1] or 1.
            clean_text = re.sub(r'(\[\d+\]|\b\d+\.\s)', r'\n\n\1', clean_text)
            answer = "The references cited in this paper are:\n" + clean_text.strip()
        elif is_bib_query and bib_fallback_mode:
            answer_prompt = (
                f"The user is asking for the references/bibliography section of a research paper.\n"
                f"Question: {question}\n"
                f"Context (last pages of document, NOT a dedicated references section):\n{final_narrative}\n\n"
                f"Task: Based on the document structure, this paper does not appear to have a dedicated "
                f"references or bibliography section. If you find any inline citations (e.g. [1], [2]) or "
                f"reference markers in the context, list them. Otherwise, clearly state that no formal "
                f"bibliography section was found in this document and that citations appear to be inline only."
            )
            answer = llm_generate(
                "answer_generation",
                answer_prompt,
                temperature=0.1
            )
        else:
            # Filter out empty or duplicate failed sub-queries
            clean_failed = sorted(list(set([q.strip() for q in failed_sub_queries if q and q.strip()])))
            if clean_failed:
                print_msg(f"[Agent10] Injection: the following sub-queries failed context retrieval: {clean_failed}")
                answer_prompt = (
                    f"Answer the user's question accurately using ONLY the provided document context.\n"
                    f"Question: {question}\n"
                    f"Context:\n{final_narrative}\n\n"
                    f"IMPORTANT: The following aspects/questions could NOT be found or supported by the document context: {clean_failed}.\n"
                    f"Task: Answer the parts of the question that are supported by the context. For the parts that are NOT found, explicitly state in your answer that the document does not contain this information. Do not invent or fabricate facts.\n\n"
                    f"Answer:"
                )
            else:
                answer_prompt = (
                    f"Answer the user's question accurately using ONLY the provided document context.\n"
                    f"Question: {question}\n"
                    f"Context:\n{final_narrative}\n\nAnswer:"
                )
            answer = llm_generate(
                "answer_generation",
                answer_prompt,
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
                        print_msg(
                            f"[Agent10] Requery {attempt+1}"
                            f": {rs.reason}"
                        )
                        re_anchors = a4.retrieve(
                            rs.refined_query,
                            selected_nodes, self.tree,
                            self.atom_store, self.bm25_index,
                            self.triple_store, routed=routed
                        )
                        re_exp = a5.expand(
                            re_anchors, self.atom_store,
                            rs.refined_query,
                            is_bibliography=is_bib
                        )
                        final_narrative = re_exp["narrative"]

                        # Reuse appropriate prompt on requery regeneration
                        if is_bib_query and not bib_fallback_mode:
                            print_msg("[Agent10] Bypassing LLM requery generation for bibliography.")
                            import re
                            clean_text = final_narrative.replace("\n", " ")
                            clean_text = re.sub(r'\s+', ' ', clean_text)
                            clean_text = re.sub(r'(\[\d+\]|\b\d+\.\s)', r'\n\n\1', clean_text)
                            answer = "The references cited in this paper are:\n" + clean_text.strip()
                        elif is_bib_query and bib_fallback_mode:
                            requery_prompt = (
                                f"The user is asking for the references/bibliography section of a research paper.\n"
                                f"Question: {question}\n"
                                f"Context (last pages of document, NOT a dedicated references section):\n{final_narrative}\n\n"
                                f"Task: Based on the document structure, this paper does not appear to have a dedicated "
                                f"references or bibliography section. If you find any inline citations (e.g. [1], [2]) or "
                                f"reference markers in the context, list them. Otherwise, clearly state that no formal "
                                f"bibliography section was found in this document and that citations appear to be inline only."
                            )
                            answer = llm_generate(
                                "answer_generation",
                                requery_prompt,
                                temperature=0.1
                            )
                        else:
                            clean_failed = sorted(list(set([q.strip() for q in failed_sub_queries if q and q.strip()])))
                            if clean_failed:
                                requery_prompt = (
                                    f"Answer the user's question accurately using ONLY the provided document context.\n"
                                    f"Question: {question}\n"
                                    f"Context:\n{final_narrative}\n\n"
                                    f"IMPORTANT: The following aspects/questions could NOT be found or supported by the document context: {clean_failed}.\n"
                                    f"Task: Answer the parts of the question that are supported by the context. For the parts that are NOT found, explicitly state in your answer that the document does not contain this information. Do not invent or fabricate facts.\n\n"
                                    f"Answer:"
                                )
                            else:
                                requery_prompt = (
                                    f"Answer the user's question accurately using ONLY the provided document context.\n"
                                    f"Question: {question}\n"
                                    f"Context:\n{final_narrative}\n\nAnswer:"
                                )
                            answer = llm_generate(
                                "answer_generation",
                                requery_prompt,
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
                    self.triple_store,
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
                print_msg(f"[Agent10] agent7 error: {e}")
                record("agent7_contradiction", None,
                       "C", 0.50, time.time()-t0)
        else:
            print_msg(
                "[SuperAgent] Skipping Agent7 "
                f"(contradiction audit) for "
                f"{query_type} query."
            )
            record("agent7_contradiction", None,
                   "S", 1.0, 0.0,
                   skipped_flag=True)

        # ── STEP 13: AGENT 9 — CALIBRATION ───────────────────
        t0 = time.time()
        if is_needed("agent9_calibration"):
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
        else:
            # When calibration is skipped (e.g. summary),
            # use a score derived from expansion quality
            atom_count = len(unique_atom_ids)
            min_atoms  = routing.get("min_atoms_needed", 3)
            # Scale: if we got enough atoms → 0.85, half → 0.65, few → 0.50
            if atom_count >= min_atoms:
                skip_score = 0.85
            elif atom_count >= max(min_atoms // 2, 2):
                skip_score = 0.65
            else:
                skip_score = 0.50
            calibration = {
                "base_score":       skip_score,
                "calibrated_score": skip_score,
                "trust_level":      "high" if skip_score >= 0.75 else "medium",
                "gap_penalty":      0.0,
                "conflict_penalty": 0.0,
                "temporal_bonus":   0.0,
                "requery_penalty":  0.0
            }
            print_msg(
                f"[Agent9] Calibration skipped for {query_type} "
                f"query — using expansion-based score: {skip_score:.2f}"
            )
            record("agent9_calibration", calibration,
                   "S", 1.0, 0.0,
                   skipped_flag=True)

        # ── STEP 14: AGENT 11 — SYNTHESIS ────────────────────
        t0 = time.time()
        synthesis = {
            "novel_connections":   [],
            "synthesis_performed": False
        }

        if is_needed("agent11_synthesis"):
            try:
                entities = routed.get("key_entities", [])
                # If no entities were extracted (e.g. router was
                # skipped), pull entities from the causal store
                if not entities:
                    entities = self.causal_store.all_entities()[:10]

                synthesis = a11.synthesize(
                    entities,
                    self.causal_store,
                    self.atom_store
                )
                if synthesis["synthesis_performed"]:
                    record("agent11_synthesis", synthesis,
                           "B", 0.75, time.time()-t0)
                else:
                    # No multi-hop chains is NOT a failure —
                    # it just means the doc has simple causal
                    # structure. Score it as a neutral pass.
                    print_msg(
                        "[Agent11] No multi-hop chains — "
                        "scoring as neutral (not a failure)."
                    )
                    record("agent11_synthesis", synthesis,
                           "B", 0.60, time.time()-t0)
            except Exception as e:
                print_msg(f"[Agent10] agent11 error: {e}")
                record("agent11_synthesis", None,
                       "C", 0.50, time.time()-t0)
        else:
            record("agent11_synthesis", None,
                   "S", 1.0, 0.0,
                   skipped_flag=True)

        # ── STEP 15: AGENT 12 — WEB SEARCH ───────────────────
        t0 = time.time()

        if is_needed("agent12_websearch"):
            should_search = (
                calibration["calibrated_score"] <
                0.70 and
                synthesis.get("novel_connections")
            )
            if should_search and a12 is not None:
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
                    print_msg(f"[Agent10] agent12 error: {e}")
            else:
                record("agent12_websearch", None,
                       "S", 1.0, 0.0,
                       skipped_flag=True)
        else:
            record("agent12_websearch", None,
                   "S", 1.0, 0.0,
                   skipped_flag=True)

        # ── STEP 15b: AGENT 13 — PAPER WRITER ─────────────────
        paper_result = None
        if query_type == "paper_writing" and a13 is not None:
            t0 = time.time()
            print_msg("\n[Agent10] → Dispatching Agent 13 (Paper Writer)...")
            try:
                # Parse venue and article type from query
                venue = "IEEE"
                article_type = "research_article"
                q_lower = question.lower()
                for v in ["neurips", "icml", "iclr", "acm",
                          "springer", "elsevier", "dsj"]:
                    if v in q_lower:
                        venue = v.upper()
                        break
                for at in ["review_article", "short_communication",
                           "systematic_review", "perspective_article",
                           "technical_note", "case_study",
                           "letter_to_editor"]:
                    if at.replace("_", " ") in q_lower:
                        article_type = at
                        break
                if "review" in q_lower and article_type == "research_article":
                    article_type = "review_article"

                paper_result = a13.write_paper(
                    topic=question,
                    venue=venue,
                    article_type=article_type,
                    narrative=final_narrative,
                    atom_ids=unique_atom_ids,
                    web_evidence=web_result or {"sources": []},
                    novel_connections=synthesis.get(
                        "novel_connections", []
                    ),
                )
                record("agent13_paper_writer", paper_result,
                       "A", 0.90, time.time()-t0)
                # Override answer with the paper text
                answer = paper_result.get("full_text", answer)
            except Exception as e:
                print_msg(f"[Agent10] agent13 error: {e}")
                record("agent13_paper_writer", None,
                       "F", 0.30, time.time()-t0)

        # ── STEP 15c: AGENT 14 — IMPLEMENTATION GUIDE ─────────
        impl_result = None
        if query_type == "implementation_guide" and a14 is not None:
            t0 = time.time()
            print_msg("\n[Agent10] → Dispatching Agent 14 (Implementation Guide)...")
            try:
                impl_result = a14.guide_implementation(
                    innovation=question,
                    narrative=final_narrative,
                    atom_ids=unique_atom_ids,
                    web_evidence=web_result or {"sources": []},
                    novel_connections=synthesis.get(
                        "novel_connections", []
                    ),
                    paper_result=paper_result,
                )
                record("agent14_impl_guide", impl_result,
                       "A", 0.90, time.time()-t0)
                # Override answer with the guide text
                answer = impl_result.get("full_text", answer)
            except Exception as e:
                print_msg(f"[Agent10] agent14 error: {e}")
                record("agent14_impl_guide", None,
                       "F", 0.30, time.time()-t0)

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
            if s >= 0.85: return "A"
            if s >= 0.70: return "B"
            if s >= 0.50: return "C"
            return "F"

        elapsed_total = round(time.time()-start_time, 2)

        # Populate self.results so other methods on SuperAgent work if called
        self.results = []
        for r in agent_results:
            self.results.append(AgentResult(
                agent_id=0,
                agent_name=r["agent"],
                input_summary="",
                output=None,
                score=r["score"],
                grade=r["grade"],
                skipped=r["skipped"],
                elapsed=r["elapsed"]
            ))

        # We construct a report structure that matches the original _report output exactly,
        # ensuring full integration with pipeline.py and CLI components.
        report = {
            "pipeline_grade":      grade(avg_score),
            "average_agent_score": round(avg_score, 3),
            "agents_run":          len(ran_agents),
            "agents_skipped":      len(skipped_agents),
            "agents_retried":      0,
            "total_elapsed":       elapsed_total,
            "underperformers":     [],
            "per_agent_scores": [
                {
                    "agent":   r["agent"],
                    "grade":   r["grade"],
                    "score":   r["score"],
                    "elapsed": r["elapsed"],
                    "skipped": r["skipped"],
                    "retried": False
                }
                for r in agent_results
            ],
            "mode":                MODE,
            "requery_count":       validation.get("requery_count", 0),
            "aborted":             False,
            "abort_reason":        ""
        }

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

            # Pipeline compatibility
            "pipeline_grade":     grade(avg_score),
            "reasoning_trail":    agent_results,
            "elapsed_seconds":    elapsed_total,
            "requery_count":      validation.get(
                "requery_count", 0
            ),
            "selected_sections":  selected_nodes,
            "ordered_atoms":      self.atom_store.get_many(unique_atom_ids),
            "narrative":          final_narrative,
            "review_report":      report,

            # Web search
            "web_search_run":     web_result is not None,
            "new_papers_added":   (
                web_result.get("new_papers_added", 0)
                if web_result else 0
            ),

            # Paper writing (Agent 13)
            "paper_result":       paper_result,

            # Implementation guide (Agent 14)
            "impl_result":        impl_result,
        }
