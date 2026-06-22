# agents/agent7_contradiction.py
# Factual contradiction and logical consistency checker — uses local DeepSeek (via router)

from llm.router import generate_json
from db.triple_store import TripleStore
from collections import defaultdict
from console_helper import print_msg

SYSTEM = ("You are a logical consistency checker. "
          "Return ONLY valid JSON.")

def detect(atom_ids: list[int], triple_store: TripleStore, narrative: str) -> dict:
    """Scan accumulated atoms and stitched context to locate factual, cross-doc, or numerical contradictions."""
    # Step 1: Query structural database for triple-level collisions
    triple_conflicts = triple_store.find_conflicts(atom_ids)

    # Step 2: Query for cross-document subject-relation divergence
    scoped = [t for t in triple_store.triples if t.get("atom_id") is not None and int(t["atom_id"]) in set(atom_ids)]
    doc_ids = set(t.get("doc_id") for t in scoped)
    cross_doc = []
    
    if len(doc_ids) > 1:
        sr = defaultdict(lambda: defaultdict(list))
        for t in scoped:
            key = (t.get("subject", "").lower().strip(), t.get("relation", "").lower().strip())
            sr[key][t.get("doc_id")].append(t.get("object", "").strip())
            
        for key, dm in sr.items():
            if len(dm) > 1:
                # Map doc_id -> first object representation
                objs = {d: v[0] for d, v in dm.items()}
                if len(set(objs.values())) > 1:
                    cross_doc.append({
                        "subject":  key[0],
                        "relation": key[1],
                        "per_document": objs
                    })

    # Step 3: LLM reasoning audit for logical/numerical discrepancies
    if narrative:
        prompt = f"""Find contradictions in this text.
Return JSON:
{{
  "contradictions_found": true or false,
  "contradiction_details": [
    {{"claim_a":"...","claim_b":"...",
      "type":"numerical|factual|logical",
      "severity":"high|medium|low"}}
  ],
  "consistency_score": 0.0 to 1.0
}}
Text: {narrative[:2000]}"""
        try:
            llm = generate_json("agent7_contradiction", prompt, system=SYSTEM)
            if not isinstance(llm, dict):
                llm = {}
        except Exception:
            llm = {}
    else:
        llm = {}

    llm.setdefault("contradictions_found", False)
    llm.setdefault("contradiction_details", [])
    llm.setdefault("consistency_score", 1.0)

    # Sanitize contradiction details to guarantee dict format
    details = llm.get("contradiction_details", [])
    if not isinstance(details, list):
        details = []
    valid_details = []
    for c in details:
        if isinstance(c, dict):
            c.setdefault("claim_a", "")
            c.setdefault("claim_b", "")
            c.setdefault("type", "logical")
            c.setdefault("severity", "low")
            valid_details.append(c)
    llm["contradiction_details"] = valid_details

    score = float(llm.get("consistency_score", 1.0))
    structural_conflicts = len(triple_conflicts) > 0 or len(cross_doc) > 0
    llm_found = llm.get("contradictions_found", False)
    
    print_msg(f"[Agent7] Audit complete. Triple collisions={len(triple_conflicts)} | Cross-doc conflicts={len(cross_doc)} | Consistency score={score}")
    if llm_found and not structural_conflicts:
        print_msg("[Agent7] LLM detected soft logical inconsistencies (no structural triple conflicts).")
    
    return {
        "triple_conflicts":     triple_conflicts,
        "cross_doc_conflicts":  cross_doc,
        "llm_contradictions":   llm.get("contradiction_details", []),
        # Only raise the WARNING banner for real structural conflicts (triple/cross-doc)
        # LLM soft inconsistencies affect calibration score but not the alert level
        "contradictions_found": structural_conflicts,
        "llm_contradictions_found": llm_found,
        "consistency_score": score
    }
