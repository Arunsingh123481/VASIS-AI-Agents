"""
Stateful Expander — The Core RE-MSE Engine.
Performs Multi-Segment Sequential Expansion from Anchor Atoms.

Each anchor spawns independent expansion passes with progressively
wider windows. A shared State Cache accumulates all retrieved atoms
across passes, so every subsequent pass benefits from prior results.

This eliminates boundary blindness without a fixed, arbitrary segment count.
"""
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from console_helper import print_msg, tqdm, print_panel


from typing import List, Dict, Optional


# Default expansion configuration
DEFAULT_EXPANSION_PASSES = 4
DEFAULT_WINDOW_SIZES = [1, 2, 4, 6]  # tokens of neighbor atoms per pass


def expand_anchors(
    anchors: List[Dict],
    all_atoms: List[Dict],
    expansion_passes: int = DEFAULT_EXPANSION_PASSES,
    window_sizes: Optional[List[int]] = None
) -> Dict:
    """
    Run Multi-Segment Stateful Expansion from all anchor atoms.

    Args:
        anchors: List of Anchor Atoms (seed points)
        all_atoms: Complete list of all atoms in the document
        expansion_passes: Number of expansion passes per anchor
        window_sizes: Neighbor window radius for each pass

    Returns:
        dict with:
          - 'atoms': all expanded atoms from State Cache
          - 'state_cache': full State Cache (atom_id -> atom)
          - 'expansion_log': log of each pass for traceability
    """
    if window_sizes is None:
        window_sizes = DEFAULT_WINDOW_SIZES[:expansion_passes]

    # Ensure window_sizes matches expansion_passes
    while len(window_sizes) < expansion_passes:
        window_sizes.append(window_sizes[-1] + 2)

    # Build fast lookup map
    atom_map: Dict[int, Dict] = {a["atom_id"]: a for a in all_atoms}

    # State Cache — persists across ALL anchors and ALL passes
    state_cache: Dict[int, Dict] = {}
    expansion_log = []

    print_msg(f"[cyan]Running {expansion_passes}-pass stateful expansion from {len(anchors)} anchors...[/cyan]")

    for anchor_idx, anchor in enumerate(anchors):
        anchor_id = anchor["atom_id"]

        # Seed the State Cache with this anchor
        state_cache[anchor_id] = anchor
        anchor_log = {
            "anchor_atom_id": anchor_id,
            "anchor_page": anchor["page_num"],
            "passes": []
        }

        for pass_num in range(expansion_passes):
            window = window_sizes[pass_num]

            # Only expand FROM the anchor atom, not all previously added neighbors
            # to prevent exponential explosion that retrieves the entire document
            snapshot_ids = [anchor_id]
            new_this_pass = 0

            for atom_id in snapshot_ids:
                # Expand neighbors within window radius
                for offset in range(-window, window + 1):
                    if offset == 0:
                        continue
                    neighbor_id = atom_id + offset

                    if neighbor_id >= 0 and neighbor_id in atom_map and neighbor_id not in state_cache:
                        state_cache[neighbor_id] = atom_map[neighbor_id]
                        new_this_pass += 1

            pass_info = {
                "pass_num": pass_num + 1,
                "window_radius": window,
                "new_atoms_added": new_this_pass,
                "cache_size_after": len(state_cache)
            }
            anchor_log["passes"].append(pass_info)

        expansion_log.append(anchor_log)

    expanded_atoms = list(state_cache.values())
    print_msg(f"[green]Stateful expansion complete: {len(expanded_atoms)} atoms in State Cache.[/green]")

    # Print expansion summary
    _print_expansion_summary(expansion_log, len(state_cache))

    return {
        "atoms": expanded_atoms,
        "state_cache": state_cache,
        "expansion_log": expansion_log
    }


def _print_expansion_summary(expansion_log: List[Dict], final_cache_size: int) -> None:
    """Print a readable summary of the expansion process."""
    print_msg("\n[bold cyan]Expansion Summary:[/bold cyan]")
    for anchor_log in expansion_log:
        print_msg(f"  Anchor atom [{anchor_log['anchor_atom_id']}] (page {anchor_log['anchor_page']}):")
        for p in anchor_log["passes"]:
            print_msg(
                f"    Pass {p['pass_num']} (window ±{p['window_radius']}): "
                f"+{p['new_atoms_added']} atoms → cache size: {p['cache_size_after']}"
            )
    print_msg(f"  [bold]Final State Cache: {final_cache_size} atoms[/bold]\n")
