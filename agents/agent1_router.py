# agents/agent1_router.py
# Query router and optimizer agent — uses local Qwen Coder (via router)

from llm.router import generate_json
from console_helper import print_msg

SYSTEM = ("You are a query routing agent. "
          "Return ONLY valid JSON.")

VALID_INTENTS = {"factual", "comparative", "definitional", "procedural", "causal", "unknown"}

# Keywords that signal the user wants to find the references/bibliography section
BIBLIOGRAPHY_KEYWORDS = {
    "references", "bibliography", "citations", "cited", "cite",
    "authors cited", "papers cited", "works cited", "cited papers",
    "list of references", "reference list", "reference section"
}

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

    # Multi-intent detection:
    # If the user is asking for multiple distinct things, decide how to route.
    # We ignore bibliography here because asking for references alongside a fact is common and simple.
    intents_detected = [qtype for qtype in scores if qtype != "bibliography"]
    if len(intents_detected) >= 2:
        has_paper = scores.get("paper_writing", 0) > 0
        has_impl  = scores.get("implementation_guide", 0) > 0

        if has_paper and has_impl:
            # Paper + guide combo: route as paper_writing.
            # agent10_super's is_impl_guide_requested keyword check will
            # also fire Agent 14 when impl keywords are present under paper_writing.
            detected = "paper_writing"
        elif has_paper:
            # Paper writing requested alongside other intents
            detected = "paper_writing"
        elif has_impl:
            # Implementation guide requested alongside other intents
            detected = "implementation_guide"
        else:
            # Generic multi-intent: deep_research handles decomposition
            detected = "deep_research"
    else:
        # Single-intent special overrides:
        # Bibliography always wins if detected
        if scores.get("bibliography", 0) > 0:
            detected = "bibliography"

        # Summary wins over explanation for full-doc queries
        elif (scores.get("summary", 0) > 0 and
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

    required_agents = list(rules["required_agents"])
    optional_agents = list(rules["optional_agents"])
    skip_agents = list(rules["skip_agents"])

    query_lower = query.lower()
    if query_type == "comparative" and any(k in query_lower for k in ["this paper", "the paper", "this work", "advantages and disadvantages", "pros and cons"]):
        if "agent4_retrieval" in required_agents:
            required_agents.remove("agent4_retrieval")
        if "agent4_retrieval" not in skip_agents:
            skip_agents.append("agent4_retrieval")
            
    estimate = estimate_total_time(query_type)

    print_msg(
        f"[Agent1] Query type detected: "
        f"{query_type.upper()}"
    )
    print_msg(
        f"[Agent1] Agents needed: "
        f"{len(required_agents)} required + "
        f"{len(optional_agents)} optional"
    )
    print_msg(
        f"[Agent1] Skipping: "
        f"{len(skip_agents)} agents"
    )
    print_msg(
        f"[Agent1] Est. time: "
        f"~{estimate['estimated_secs']}s"
    )

    return {
        "query_type":        query_type,
        "required_agents":   required_agents,
        "optional_agents":   optional_agents,
        "skip_agents":       skip_agents,
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


def route(query: str) -> dict:
    """Analyze query and rewrite it for indexing/retrieval, specifying intent."""
    prompt = f"""Analyze query intent. Choose EXACTLY ONE intent from this list:
- factual      : asks for a specific value, number, or attribute (What is, How many, Which)
- comparative  : contrasts two concepts (How does X differ from Y, X vs Y, better than)
- definitional : asks what something is or means (Define, Explain what, What is meant by)
- procedural   : asks for steps or a process (How to, What steps, Describe the procedure)
- causal       : asks why something happens or what causes an effect (Why does, What causes,
                 What leads to, How does X improve/affect Y, Why is X better)
- unknown      : none of the above apply

Return JSON:
{{
  "intent": "causal",
  "rewritten_query": "optimized for retrieval",
  "key_entities": ["entity1", "entity2"],
  "requires_multi_section": true,
  "is_complex": false,
  "confidence": 0.9
}}
intent must be exactly one of: factual, comparative, definitional, procedural, causal, unknown
Query: {query}"""

    try:
        result = generate_json("agent1_router", prompt, system=SYSTEM)
        if not isinstance(result, dict):
            result = {}
    except Exception:
        result = {}

    # Sanitize intent — must be one of the 6 valid options
    intent_raw = result.get("intent", "")
    if not isinstance(intent_raw, str) or intent_raw.split("|")[0].strip().lower() not in VALID_INTENTS:
        result["intent"] = "factual"
    else:
        result["intent"] = intent_raw.split("|")[0].strip().lower()
    if result.get("rewritten_query") is None:
        result["rewritten_query"] = query
    if result.get("key_entities") is None:
        result["key_entities"] = []
    if result.get("requires_multi_section") is None:
        result["requires_multi_section"] = False
    if result.get("is_complex") is None:
        result["is_complex"] = False
    if result.get("confidence") is None:
        result["confidence"] = 1.0
    result["original_query"] = query

    # Deep Research Overrides: Ensure decomposition and comprehensive section navigation
    query_type = detect_query_type(query)
    if query_type == "deep_research":
        result["is_complex"] = True
        result["rewritten_query"] = query
        print_msg("[Agent1] Deep research query detected — enforcing complexity and preserving full query intent.")

    # Bibliography detection: if the query is asking for references/citations,
    # flag it so Agent4 can target the last pages directly instead of using BM25
    query_lower = query.lower()
    result["is_bibliography_query"] = any(kw in query_lower for kw in BIBLIOGRAPHY_KEYWORDS)
    if result["is_bibliography_query"]:
        result["intent"] = "factual"  # force factual — we want precise targeted retrieval
        # CRITICAL: Override the LLM rewritten_query with a stable bibliography-scoped
        # retrieval term. The LLM often rewrites "What are the references?" into semantically
        # wrong queries (e.g. "compared to other papers") which sabotages Agent5's
        # Jaccard relevance scoring and causes it to reject the correct citation atoms.
        result["rewritten_query"] = "references bibliography citations authors cited works"
        result["key_entities"] = ["references", "bibliography", "citations"]
        print_msg("[Agent1] Bibliography query detected — enabling last-page targeted retrieval.")
        print_msg("[Agent1] Locked rewritten_query to bibliography retrieval terms (bypassing LLM rewrite).")

    print_msg(f"[Agent1] Query categorized: intent={result['intent']} | rewritten={result['rewritten_query']}")
    return result
