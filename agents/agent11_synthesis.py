# agents/agent11_synthesis.py
# Knowledge synthesis agent — uses local DeepSeek (via router)

from llm.router import generate_json
from db.causal_store import CausalStore
from config import MIN_SYNTHESIS_CONFIDENCE, MAX_SYNTHESIS_HOPS
from console_helper import print_msg

SYSTEM = ("You are a knowledge synthesis agent. "
          "Return ONLY valid JSON.")

def synthesize(entities: list[str], causal_store: CausalStore, atom_store) -> dict:
    """Traverse causal relation chains and synthesise indirect connections to uncover novel inferences."""
    all_chains = []
    # Scope traversal to top 5 entities to prevent context blowup
    for entity in entities[:5]:
        chains = causal_store.traverse_chain(entity, max_hops=MAX_SYNTHESIS_HOPS)
        all_chains.extend(chains)

    # Filter for indirect connections (>= 2 hops)
    indirect = [c for c in all_chains if c["hops"] >= 2]
    if not indirect:
        print_msg("[Agent11] No multi-hop indirect causal chains found in graph.")
        return {
            "novel_connections": [],
            "synthesis_performed": False,
            "chains_explored": len(all_chains),
            "synthesis_quality": 0.0
        }

    # Format summaries for LLM prompt
    summaries = [
        {
            "path": " → ".join(c["path"]),
            "hops": c["hops"],
            "atom_ids": c["atom_ids"]
        }
        for c in indirect[:10]
    ]

    prompt = f"""Analyze causal chains from documents.
Find novel insights not stated directly anywhere.
Chains: {summaries}
Return JSON:
{{
  "novel_connections": [
    {{
      "from": "start entity",
      "to": "end entity",
      "via": ["intermediates"],
      "inference": "what this novel connection means",
      "confidence": 0.0 to 1.0,
      "type": "inferred",
      "supporting_atom_ids": [1,2,3]
    }}
  ],
  "synthesis_quality": 0.0 to 1.0
}}
Only include confidence >= {MIN_SYNTHESIS_CONFIDENCE}"""

    try:
        result = generate_json("agent11_synthesis", prompt, system=SYSTEM)
        if not isinstance(result, dict):
            result = {}
            
        novel = result.get("novel_connections", [])
        if not isinstance(novel, list):
            novel = []
            
        valid_novel = []
        for n in novel:
            if isinstance(n, dict):
                try:
                    conf = float(n.get("confidence", 0.0))
                    if conf >= MIN_SYNTHESIS_CONFIDENCE:
                        n["confidence"] = conf
                        n.setdefault("from", "")
                        n.setdefault("to", "")
                        n.setdefault("via", [])
                        n.setdefault("inference", "")
                        valid_novel.append(n)
                except (ValueError, TypeError):
                    pass
        novel = valid_novel
        
        print_msg(f"[Agent11] Synthesis complete: novel_connections={len(novel)} | chains_explored={len(all_chains)}")
        
        return {
            "novel_connections":   novel,
            "synthesis_performed": True,
            "chains_explored":     len(all_chains),
            "synthesis_quality":   float(result.get("synthesis_quality", 0.0))
        }
    except Exception as e:
        print_msg(f"[yellow][Agent11] Synthesis failed: {e}[/yellow]")
        return {
            "novel_connections": [],
            "synthesis_performed": False,
            "chains_explored": len(all_chains),
            "synthesis_quality": 0.0
        }
