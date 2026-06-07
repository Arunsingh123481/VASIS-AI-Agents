"""
Tree Builder — Builds the PageIndex hierarchical Table-of-Contents tree.
Uses LLM reasoning to generate section titles and summaries.
This is the macro-layer of the dual-layer indexing system.
"""
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from console_helper import print_msg, tqdm, print_panel


import json
from typing import List, Dict
from llm.router import generate_json


# Number of pages per section group (adaptive based on doc length)
MIN_SECTION_SIZE = 3
MAX_SECTION_SIZE = 8
DEFAULT_SECTION_SIZE = 5

# Max characters sent to LLM per section summary request
MAX_CHARS_FOR_SUMMARY = 3000


def build_tree(pages: List[Dict], doc_id: str) -> List[Dict]:
    """
    Build a hierarchical PageIndex tree from document pages.
    Returns a list of tree nodes, each representing a document section.
    """
    section_size = _adaptive_section_size(len(pages))
    print_msg(f"[cyan]Building PageIndex tree (section size: {section_size} pages)...[/cyan]")

    # Group pages into sections
    page_groups = []
    for i in range(0, len(pages), section_size):
        page_groups.append(pages[i:i+section_size])

    tree_nodes = []
    for idx, group in enumerate(tqdm(page_groups, desc="Building tree nodes")):
        node = _build_node(idx, group, doc_id)
        tree_nodes.append(node)

    # Add parent-child relationships for adjacent nodes
    for i, node in enumerate(tree_nodes):
        node["prev_node_id"] = tree_nodes[i-1]["node_id"] if i > 0 else None
        node["next_node_id"] = tree_nodes[i+1]["node_id"] if i < len(tree_nodes)-1 else None

    print_msg(f"[green]Built tree with {len(tree_nodes)} section nodes.[/green]")
    return tree_nodes


def _build_node(idx: int, page_group: List[Dict], doc_id: str) -> Dict:
    """Build a single tree node from a group of pages."""
    combined_text = " ".join([p["text"] for p in page_group])
    truncated_text = combined_text[:MAX_CHARS_FOR_SUMMARY]
    start_page = page_group[0]["page_num"]
    end_page = page_group[-1]["page_num"]

    prompt = f"""You are an expert document analyst building a semantic index.

Read this document passage (pages {start_page}-{end_page}) and return a JSON object with:
- "title": a concise section title (5-10 words, captures the main topic)
- "summary": a 2-3 sentence summary of what key information this section contains
- "key_topics": a list of 3-5 key topics or entities mentioned

Passage:
{truncated_text}

Return ONLY valid JSON. No explanation, no markdown fences. Just the JSON object."""

    meta = generate_json("tree_builder", prompt)
    if not isinstance(meta, dict):
        meta = {
            "title": f"Section {idx + 1} (Pages {start_page}-{end_page})",
            "summary": truncated_text[:300] + "...",
            "key_topics": []
        }

    return {
        "node_id": f"{idx:04d}",
        "doc_id": doc_id,
        "title": meta.get("title", f"Section {idx + 1}"),
        "summary": meta.get("summary", ""),
        "key_topics": meta.get("key_topics", []),
        "start_page": start_page,
        "end_page": end_page,
        "page_count": len(page_group),
        "prev_node_id": None,  # filled after all nodes built
        "next_node_id": None
    }


def _adaptive_section_size(total_pages: int) -> int:
    """
    Determine section size based on document length.
    Short docs get smaller sections for finer granularity.
    Long docs get larger sections to keep tree manageable.
    """
    if total_pages <= 20:
        return MIN_SECTION_SIZE
    elif total_pages <= 60:
        return DEFAULT_SECTION_SIZE
    elif total_pages <= 150:
        return 7
    else:
        return MAX_SECTION_SIZE


def print_tree(tree_nodes: List[Dict]) -> None:
    """Pretty print the tree structure."""
    print_msg("\n[bold cyan]═══ PageIndex Tree Structure ═══[/bold cyan]")
    for node in tree_nodes:
        print_msg(f"\n[bold yellow][{node['node_id']}] {node['title']}[/bold yellow]")
        print_msg(f"  Pages: {node['start_page']}–{node['end_page']}")
        print_msg(f"  Summary: [dim]{node['summary'][:150]}...[/dim]")
        if node["key_topics"]:
            print_msg(f"  Topics: {', '.join(node['key_topics'])}")
    print_msg("\n[bold cyan]═══════════════════════════════[/bold cyan]\n")
