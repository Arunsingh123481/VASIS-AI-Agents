"""
Tree Navigator — LLM-based reasoning to navigate the PageIndex tree.
Identifies the most relevant section nodes for a given query.
This replaces vector similarity search for section-level retrieval.
"""
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from console_helper import print_msg


import json
from typing import List, Dict, Tuple
from llm.ollama_client import ask_llm


MAX_NODES_TO_RETURN = 3
MAX_TREE_CHARS_IN_PROMPT = 6000


def navigate_tree(query: str, tree_nodes: List[Dict]) -> Tuple[List[Dict], str]:
    """
    Use LLM reasoning to navigate the PageIndex tree and identify
    the most relevant section nodes for the given query.

    Returns:
        - List of selected tree nodes
        - Reasoning explanation from the LLM
    """
    if not tree_nodes:
        return [], "No tree nodes available."

    # Build a compact tree representation for the LLM
    tree_text = _format_tree_for_llm(tree_nodes)

    prompt = f"""You are an expert document navigator with deep reasoning ability.

A user has asked: "{query}"

Below is the document's Table of Contents with section summaries.
Your task is to identify which sections are MOST LIKELY to contain the answer.

Think carefully:
1. What specific information does the query require?
2. Which sections by their title and summary are most relevant?
3. Consider that answers may span multiple sections.

Document Sections:
{tree_text}

Return a JSON object with:
- "selected_node_ids": array of the top {MAX_NODES_TO_RETURN} most relevant node_ids (e.g. ["0002", "0005"])
- "reasoning": 2-3 sentences explaining WHY you selected these sections

Return ONLY valid JSON. No markdown fences."""

    response = ask_llm(prompt, expect_json=True)

    try:
        result = json.loads(response)
        selected_ids = result.get("selected_node_ids", [])
        reasoning = result.get("reasoning", "No reasoning provided.")
    except (json.JSONDecodeError, ValueError):
        # Fallback: return first 2 nodes
        selected_ids = [tree_nodes[0]["node_id"]]
        if len(tree_nodes) > 1:
            selected_ids.append(tree_nodes[1]["node_id"])
        reasoning = "Fallback selection due to parsing error."

    # Filter to valid nodes only
    selected_nodes = [n for n in tree_nodes if n["node_id"] in selected_ids]

    # Deduplicate while preserving order
    seen = set()
    unique_nodes = []
    for node in selected_nodes:
        if node["node_id"] not in seen:
            seen.add(node["node_id"])
            unique_nodes.append(node)

    # Always return at least one node
    if not unique_nodes and tree_nodes:
        unique_nodes = [tree_nodes[0]]

    print_msg("\n[bold green]Tree Navigation Result:[/bold green]")
    print_msg(f"[dim]Reasoning: {reasoning}[/dim]")
    for node in unique_nodes:
        print_msg(f"  → [yellow][{node['node_id']}] {node['title']}[/yellow] (pages {node['start_page']}–{node['end_page']})")

    return unique_nodes, reasoning


def _format_tree_for_llm(tree_nodes: List[Dict]) -> str:
    """Format the tree nodes into a compact, LLM-readable representation."""
    lines = []
    total_chars = 0

    for node in tree_nodes:
        line = (
            f"[Node {node['node_id']}] {node['title']} "
            f"(pages {node['start_page']}–{node['end_page']})\n"
            f"  Summary: {node['summary'][:200]}"
        )
        if node.get("key_topics"):
            line += f"\n  Topics: {', '.join(node['key_topics'][:4])}"

        total_chars += len(line)
        if total_chars > MAX_TREE_CHARS_IN_PROMPT:
            lines.append("... [tree truncated for length]")
            break

        lines.append(line)

    return "\n\n".join(lines)
