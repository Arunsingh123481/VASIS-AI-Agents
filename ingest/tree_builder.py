"""
Tree Builder — Builds the PageIndex hierarchical Table-of-Contents tree.
Uses LLM reasoning to generate section titles and summaries.
If Markdown heading markers (#, ##, ###) are detected in the document text,
sections are built around those natural boundaries instead of arbitrary page chunks.
This is the macro-layer of the dual-layer indexing system.
"""
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from console_helper import print_msg, tqdm, print_panel


import json
import re
from typing import List, Dict, Tuple
from llm.router import generate_json


# Number of pages per section group (adaptive based on doc length)
MIN_SECTION_SIZE = 3
MAX_SECTION_SIZE = 8
DEFAULT_SECTION_SIZE = 5

# Max characters sent to LLM per section summary request
MAX_CHARS_FOR_SUMMARY = 3000

# Regex to detect Markdown headings at the start of a line
_HEADING_RE = re.compile(r'^(#{1,4})\s+(.+)', re.MULTILINE)


def _detect_headings(pages: List[Dict]) -> List[Dict]:
    """Scan all pages for Markdown-style headings.
    Returns a list of dicts: {level, title, page_num, char_offset}
    """
    headings: List[Dict] = []
    for page in pages:
        text = page["text"]
        for m in _HEADING_RE.finditer(text):
            headings.append({
                "level": len(m.group(1)),  # number of # chars
                "title": m.group(2).strip(),
                "page_num": page["page_num"],
                "char_offset": m.start()
            })
    return headings


def _group_pages_by_headings(
    pages: List[Dict],
    headings: List[Dict],
    max_heading_level: int = 2
) -> List[Tuple[str, List[Dict]]]:
    """Group pages into sections based on detected headings.
    
    Only headings at level <= max_heading_level are used as section boundaries
    (i.e. # and ## by default). Deeper headings (### etc.) are kept inside
    their parent section.
    
    Returns a list of (section_title, [page_dicts]) tuples.
    """
    # Filter to section-breaking headings only
    breaks = [h for h in headings if h["level"] <= max_heading_level]

    if not breaks:
        return []

    page_lookup = {p["page_num"]: p for p in pages}
    all_page_nums = sorted(page_lookup.keys())

    sections: List[Tuple[str, List[Dict]]] = []

    # Pages before the first heading become a preamble section
    first_break_page = breaks[0]["page_num"]
    preamble_pages = [page_lookup[pn] for pn in all_page_nums if pn < first_break_page]
    if preamble_pages:
        sections.append(("Preamble", preamble_pages))

    # Each heading starts a new section that runs until the next heading
    for i, brk in enumerate(breaks):
        start_page = brk["page_num"]
        if i + 1 < len(breaks):
            end_page = breaks[i + 1]["page_num"]
            # If the next heading is on the same page, the section is just that page
            section_pages = [page_lookup[pn] for pn in all_page_nums
                            if start_page <= pn < end_page]
        else:
            # Last heading: include all remaining pages
            section_pages = [page_lookup[pn] for pn in all_page_nums
                            if pn >= start_page]

        if section_pages:
            sections.append((brk["title"], section_pages))

    return sections


def build_tree(pages: List[Dict], doc_id: str) -> List[Dict]:
    """
    Build a hierarchical PageIndex tree from document pages.
    
    Strategy:
      1. Scan for Markdown headings in the extracted text.
      2. If sufficient headings are found (>= 3 at level 1-2), use them
         as natural section boundaries.
      3. Otherwise, fall back to adaptive fixed-size page grouping.
    
    Returns a list of tree nodes, each representing a document section.
    """
    headings = _detect_headings(pages)
    top_headings = [h for h in headings if h["level"] <= 2]

    if len(top_headings) >= 3:
        # ── Header-aware path ─────────────────────────────────────────────
        print_msg(f"[cyan]Detected {len(top_headings)} Markdown headings — building header-aware tree...[/cyan]")
        heading_groups = _group_pages_by_headings(pages, headings, max_heading_level=2)

        tree_nodes = []
        for idx, (title, group) in enumerate(tqdm(heading_groups, desc="Building header-aware nodes")):
            node = _build_node(idx, group, doc_id, preset_title=title)
            tree_nodes.append(node)
    else:
        # ── Fallback: adaptive page-chunk grouping ────────────────────────
        section_size = _adaptive_section_size(len(pages))
        print_msg(f"[cyan]No Markdown headings detected — building tree by page groups (section size: {section_size})...[/cyan]")

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


def _build_node(idx: int, page_group: List[Dict], doc_id: str,
                preset_title: str = None) -> Dict:
    """Build a single tree node from a group of pages.
    
    If preset_title is provided (from a detected Markdown heading), the LLM
    is still used for the summary and key_topics but the title is locked to
    the detected heading text.
    """
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

    # Use the detected Markdown heading as the title if available
    title = preset_title if preset_title else meta.get("title", f"Section {idx + 1}")

    return {
        "node_id": f"{idx:04d}",
        "doc_id": doc_id,
        "title": title,
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
