# agents/agent3_navigator.py
# Section navigator agent — uses local Qwen (via router)

from llm.router import generate_json
from console_helper import print_msg

SYSTEM = ("You are a document tree navigation agent. "
          "Return ONLY valid JSON.")

def navigate(query: str, tree: list[dict], is_bibliography: bool = False) -> dict:
    """Identify which PageIndex section nodes are most relevant for a given query."""
    if is_bibliography and tree:
        last_node = tree[-1]
        result = {
            "selected_nodes": [last_node.get("node_id")],
            "reasoning": "Forced last-section selection for bibliography query.",
            "confidence": 1.0
        }
        print_msg(f"[Agent3] Bibliography query - forcing last-section selection: {result['selected_nodes']}")
        return result

    compact = [
        {
            "node_id": n.get("node_id"),
            "title":   n.get("title"),
            "summary": n.get("summary", "")[:200],
            "topics":  n.get("key_topics", []),
            "pages":   f"{n.get('start_page')}-{n.get('end_page')}"
        }
        for n in tree
    ]
    
    prompt = f"""Find 1-3 most relevant sections.
Query: {query}
Sections: {compact}
Return JSON:
{{
  "selected_nodes": ["<insert exact node_id strings from the Sections list>"],
  "reasoning": "brief explanation",
  "confidence": 0.0 to 1.0
}}"""

    try:
        result = generate_json("agent3_navigator", prompt, system=SYSTEM)
        if not isinstance(result, dict):
            result = {}
    except Exception:
        result = {}

    # Set default values if keys are missing or None
    if result.get("selected_nodes") is None:
        result["selected_nodes"] = []
        
    # Sanitize node IDs to ensure they actually exist in the tree
    valid_ids = {str(n.get("node_id")) for n in tree}
    
    sanitized_nodes = []
    for nid in result["selected_nodes"]:
        nid_str = str(nid)
        if nid_str in valid_ids:
            sanitized_nodes.append(nid_str)
        else:
            # Sometimes the model appends 'node_id' to the integer e.g., 'node_id3' instead of '0003'
            # Try to fix it by matching the last digit(s)
            import re
            match = re.search(r'\d+', nid_str)
            if match:
                num_str = match.group()
                # Pad to 4 digits which is the typical format '0000'
                padded = num_str.zfill(4)
                if padded in valid_ids:
                    sanitized_nodes.append(padded)
                    continue
            
    result["selected_nodes"] = sanitized_nodes

    if result.get("reasoning") is None:
        result["reasoning"] = "No navigation path found."
    if result.get("confidence") is None:
        result["confidence"] = 0.5

    print_msg(f"[Agent3] Sections selected: {result['selected_nodes']} | reasoning={result['reasoning']}")
    return result
