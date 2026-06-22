"""
Storage Layer — Persistent storage for document indexes.
Saves and loads PageIndex trees and Atom collections to/from disk.
Enables reuse without re-indexing on every run.
"""
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from console_helper import print_msg


import json
import os
import hashlib
from typing import List, Dict, Optional


STORAGE_DIR = ".pageindex_cache"


def _ensure_dir(doc_id: str) -> str:
    """Create storage directory for a document."""
    path = os.path.join(STORAGE_DIR, doc_id)
    os.makedirs(path, exist_ok=True)
    return path


def get_doc_id(pdf_path: str) -> str:
    """Generate a stable doc_id from the PDF file path and size."""
    stat = os.stat(pdf_path)
    raw = f"{os.path.basename(pdf_path)}_{stat.st_size}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def save_index(doc_id: str, tree_nodes: List[Dict], atoms: List[Dict], triples: List[Dict] = None) -> None:
    """Save tree nodes, atoms, and triples to disk."""
    path = _ensure_dir(doc_id)

    tree_path = os.path.join(path, "tree.json")
    atoms_path = os.path.join(path, "atoms.json")
    triples_path = os.path.join(path, "triples.json")

    with open(tree_path, "w", encoding="utf-8") as f:
        json.dump(tree_nodes, f, indent=2, ensure_ascii=False)

    with open(atoms_path, "w", encoding="utf-8") as f:
        json.dump(atoms, f, indent=2, ensure_ascii=False)

    if triples is not None:
        with open(triples_path, "w", encoding="utf-8") as f:
            json.dump(triples, f, indent=2, ensure_ascii=False)

    print_msg(f"[green]Index saved to {path}[/green]")


def load_index(doc_id: str) -> Optional[Dict]:
    """
    Load tree, atoms, and triples from disk.
    Returns None if no cached index exists.
    """
    path = os.path.join(STORAGE_DIR, doc_id)
    tree_path = os.path.join(path, "tree.json")
    atoms_path = os.path.join(path, "atoms.json")
    triples_path = os.path.join(path, "triples.json")

    if not os.path.exists(tree_path) or not os.path.exists(atoms_path):
        return None

    with open(tree_path, "r", encoding="utf-8") as f:
        tree_nodes = json.load(f)

    with open(atoms_path, "r", encoding="utf-8") as f:
        atoms = json.load(f)

    triples = []
    if os.path.exists(triples_path):
        with open(triples_path, "r", encoding="utf-8") as f:
            triples = json.load(f)

    # Convert atom_id keys back to int if needed
    for a in atoms:
        a["atom_id"] = int(a["atom_id"])

    print_msg(f"[green]Loaded cached index: {len(tree_nodes)} nodes, {len(atoms)} atoms, {len(triples)} triples.[/green]")
    return {"tree": tree_nodes, "atoms": atoms, "triples": triples}


def index_exists(doc_id: str) -> bool:
    """Check if a cached index exists for this doc_id."""
    path = os.path.join(STORAGE_DIR, doc_id)
    return (
        os.path.exists(os.path.join(path, "tree.json")) and
        os.path.exists(os.path.join(path, "atoms.json"))
    )


def save_query_result(doc_id: str, query: str, answer: str, provenance: Dict) -> None:
    """Save a query result for audit trail."""
    path = _ensure_dir(doc_id)
    results_path = os.path.join(path, "query_log.jsonl")

    import time
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "query": query,
        "answer": answer,
        "provenance": provenance
    }

    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_query_log(doc_id: str) -> List[Dict]:
    """Load all past query results for a document."""
    path = os.path.join(STORAGE_DIR, doc_id)
    results_path = os.path.join(path, "query_log.jsonl")

    if not os.path.exists(results_path):
        return []

    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


def save_note_card(doc_id: str, card_data: Dict) -> None:
    """Save a note card with its anchor atoms and citation."""
    path = _ensure_dir(doc_id)
    cards_path = os.path.join(path, "note_cards.jsonl")

    import time
    if "timestamp" not in card_data:
        card_data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    with open(cards_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(card_data, ensure_ascii=False) + "\n")


def load_note_cards(doc_id: str) -> List[Dict]:
    """Load all saved note cards for a document."""
    path = os.path.join(STORAGE_DIR, doc_id)
    cards_path = os.path.join(path, "note_cards.jsonl")

    if not os.path.exists(cards_path):
        return []

    cards = []
    with open(cards_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    cards.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return cards


def delete_index(doc_id: str) -> bool:
    """
    Delete the entire cached index directory for a document.
    Removes tree.json, atoms.json, triples.json, query_log.jsonl,
    note_cards.jsonl, and any other files under .pageindex_cache/<doc_id>/.
    
    Returns True if the directory was found and deleted, False otherwise.
    """
    import shutil

    path = os.path.join(STORAGE_DIR, doc_id)
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print_msg(f"[yellow]Deleted cached index for doc_id={doc_id}[/yellow]")
            return True
        except Exception as e:
            print_msg(f"[red]Failed to delete index for doc_id={doc_id}: {e}[/red]")
            return False
    return False


