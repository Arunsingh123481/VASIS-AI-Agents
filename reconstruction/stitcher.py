"""
Stitcher — Deduplication, ordering, and Narrative Reconstruction.
Consolidates all atoms from the State Cache into a single coherent
Reconstructed Narrative Block ready for LLM consumption.
"""
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from console_helper import print_msg, tqdm, print_panel


from typing import List, Dict, Tuple


MAX_NARRATIVE_CHARS = 30000  # Safe context limit for Llama 3 8B

def stitch(expanded_atoms: List[Dict], max_chars: int = MAX_NARRATIVE_CHARS) -> Tuple[str, List[Dict]]:
    """
    Deduplicate, sort, and stitch atoms into a Reconstructed Narrative Block.
    """
    if not expanded_atoms:
        return "", []

    # Step 1: Deduplicate by atom_id
    seen = {}
    for atom in expanded_atoms:
        seen[atom["atom_id"]] = atom

    # Step 2: Sort by atom_id (restores original document order)
    ordered = sorted(seen.values(), key=lambda x: x["atom_id"])

    # Step 3: Detect gaps and insert gap markers
    stitched_parts = []
    prev_id = None

    for atom in ordered:
        current_id = atom["atom_id"]

        if prev_id is not None:
            gap = current_id - prev_id - 1
            if gap > 3:
                # Large gap — insert separator to prevent false continuity
                stitched_parts.append(f"\n[...document section omitted ({gap} atoms)...]\n")
            elif gap > 0:
                # Small gap — still mark it but keep reading flow
                stitched_parts.append(" [...] ")

        stitched_parts.append(atom["text"])
        prev_id = current_id

    # Step 4: Join into narrative
    narrative = " ".join(stitched_parts)

    # Step 5: Truncate if needed (preserve from start — most relevant atoms first)
    if len(narrative) > max_chars:
        narrative = narrative[:max_chars]

    _print_stitch_summary(ordered, narrative)

    return narrative, ordered


def _print_stitch_summary(ordered_atoms: List[Dict], narrative: str) -> None:
    """Print summary of the stitching result."""
    if not ordered_atoms:
        return

    pages = sorted(set(a["page_num"] for a in ordered_atoms))
    page_range = f"{pages[0]}–{pages[-1]}" if len(pages) > 1 else str(pages[0])

    print_msg(f"[bold green]Reconstructed Narrative Block:[/bold green]")
    print_msg(f"  Atoms used: {len(ordered_atoms)}")
    print_msg(f"  Pages covered: {page_range} ({len(pages)} unique pages)")
    print_msg(f"  Narrative length: {len(narrative)} chars")
    print_msg(f"  Preview: [dim]{narrative[:200]}...[/dim]\n")


def get_provenance(ordered_atoms: List[Dict], tree_nodes: List[Dict], reasoning: str) -> Dict:
    """
    Build a full provenance record for audit-ready output.
    Includes both macro (tree) and micro (atomic) provenance chains.
    """
    atom_refs = [
        {
            "atom_id": a["atom_id"],
            "page": a["page_num"],
            "section_node_id": a.get("section_node_id"),
            "text_preview": a["text"][:80]
        }
        for a in ordered_atoms
    ]

    section_refs = []
    used_node_ids = set(a.get("section_node_id") for a in ordered_atoms if a.get("section_node_id"))
    for node in tree_nodes:
        if node["node_id"] in used_node_ids:
            section_refs.append({
                "node_id": node["node_id"],
                "title": node["title"],
                "pages": f"{node['start_page']}–{node['end_page']}"
            })

    return {
        "reasoning_path": reasoning,
        "sections_used": section_refs,
        "atoms_used": atom_refs,
        "total_atoms": len(ordered_atoms),
        "pages_referenced": sorted(set(a["page_num"] for a in ordered_atoms))
    }
