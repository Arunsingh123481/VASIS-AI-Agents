# agents/agent2_decomposer.py
# Query decomposer agent — uses local Qwen Coder (via router)

from llm.router import generate_json
from config import MAX_SUB_QUERIES
from console_helper import print_msg

SYSTEM = ("You are a query decomposition agent. "
          "Return ONLY valid JSON.")

def decompose(routed: dict) -> list[str]:
    """Partition a complex query into simpler independent sub-queries."""
    if not routed.get("is_complex", False):
        return [routed.get("rewritten_query", routed.get("original_query"))]

    prompt = f"""Split into {MAX_SUB_QUERIES} or fewer
independent sub-queries. Return JSON array of strings:
["sub-query 1", "sub-query 2"]
Original: {routed.get("original_query")}"""

    try:
        result = generate_json("agent2_decomposer", prompt, system=SYSTEM)
        print_msg(f"[Agent10] Raw generate_json output: {type(result)} - {result}")
        
        if isinstance(result, dict):
            # Extract from common LLM wrapper keys
            if "sub_queries" in result:
                result = result["sub_queries"]
            elif "sub-queries" in result:
                result = result["sub-queries"]
            else:
                # take first list found
                found_list = False
                for v in result.values():
                    if isinstance(v, list):
                        result = v
                        found_list = True
                        break
                # if no list found, maybe the LLM returned {"sub-query 1": "text1", ...}
                if not found_list:
                    extracted = [v for v in result.values() if isinstance(v, str)]
                    if extracted:
                        result = extracted
        
        if isinstance(result, list) and result:
            print_msg(f"[Agent2] Decomposed into {len(result)} sub-queries: {result}")
            return result[:MAX_SUB_QUERIES]
    except Exception as e:
        print_msg(f"[yellow][Agent2] Decomposition failed, falling back: {e}[/yellow]")
        
    return [routed.get("rewritten_query", routed.get("original_query"))]
