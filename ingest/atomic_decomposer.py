"""
Atomic Decomposer — Breaks document pages into Atoms (50-100 token units).
Each Atom carries Sequential Metadata Tags for bidirectional navigation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import List, Dict
from console_helper import print_msg, tqdm

try:
    import tiktoken as _tiktoken
    ENC = _tiktoken.get_encoding("cl100k_base")
except (ImportError, Exception):
    ENC = None


def count_tokens(text: str) -> int:
    if ENC:
        return len(ENC.encode(text))
    return len(text.split())


def decompose_to_atoms(pages: List[Dict], doc_id: str, target_tokens: int = 75) -> List[Dict]:
    atoms = []
    atom_id = 0
    print_msg(f"[cyan]Decomposing {len(pages)} pages into atoms (target: ~{target_tokens} tokens each)...[/cyan]")

    for page in tqdm(pages, desc="Atomizing pages"):
        text = page["text"]
        # Split by paragraphs instead of words to preserve Markdown tables and structure!
        paragraphs = text.split('\n\n')
        buffer = []
        token_count = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_tokens = count_tokens(para)
            
            # If a single paragraph/table is huge, split it adaptively by line breaks to preserve granularity
            if para_tokens > 1.5 * target_tokens:
                # Flush the current paragraph buffer first
                if buffer:
                    atoms.append(_make_atom(atom_id, doc_id, page["page_num"], "\n\n".join(buffer)))
                    atom_id += 1
                    buffer = []
                    token_count = 0
                
                # Split huge paragraph/table into lines
                lines = para.split('\n')
                line_buffer = []
                line_token_count = 0
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    line_tokens = count_tokens(line)
                    if line_token_count + line_tokens > target_tokens and line_buffer:
                        atoms.append(_make_atom(atom_id, doc_id, page["page_num"], "\n".join(line_buffer)))
                        atom_id += 1
                        line_buffer = [line]
                        line_token_count = line_tokens
                    else:
                        line_buffer.append(line)
                        line_token_count += line_tokens
                        
                if line_buffer:
                    atoms.append(_make_atom(atom_id, doc_id, page["page_num"], "\n".join(line_buffer)))
                    atom_id += 1
                    
            elif token_count + para_tokens > target_tokens and buffer:
                atoms.append(_make_atom(atom_id, doc_id, page["page_num"], "\n\n".join(buffer)))
                atom_id += 1
                buffer = [para]
                token_count = para_tokens
            else:
                buffer.append(para)
                token_count += para_tokens

        if buffer:
            atom_text = "\n\n".join(buffer)
            if atoms and count_tokens(atom_text) < 20:
                atoms[-1]["text"] += "\n\n" + atom_text
                atoms[-1]["token_count"] = count_tokens(atoms[-1]["text"])
            else:
                atoms.append(_make_atom(atom_id, doc_id, page["page_num"], atom_text))
                atom_id += 1

    for i, atom in enumerate(atoms):
        atom["prev_atom_id"] = atoms[i-1]["atom_id"] if i > 0 else None
        atom["next_atom_id"] = atoms[i+1]["atom_id"] if i < len(atoms) - 1 else None
        atom["section_node_id"] = None

    print_msg(f"[green]Created {len(atoms)} atoms from {len(pages)} pages.[/green]")
    return atoms


def _make_atom(atom_id: int, doc_id: str, page_num: int, text: str) -> Dict:
    return {
        "atom_id": atom_id,
        "doc_id": doc_id,
        "page_num": page_num,
        "text": text,
        "token_count": count_tokens(text),
        "prev_atom_id": None,
        "next_atom_id": None,
        "section_node_id": None
    }


def cross_reference_atoms_to_tree(atoms: List[Dict], tree_nodes: List[Dict]) -> List[Dict]:
    for atom in atoms:
        for node in tree_nodes:
            if node["start_page"] <= atom["page_num"] <= node["end_page"]:
                atom["section_node_id"] = node["node_id"]
                break
    return atoms
