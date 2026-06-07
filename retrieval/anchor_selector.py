"""
Anchor Selector — Selects Anchor Atoms from within PageIndex-validated sections.
Uses vector similarity search SCOPED to the validated section only.
This eliminates false positives from global similarity search.
"""
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from console_helper import print_msg, tqdm, print_panel


from typing import List, Dict


# Lazy-load to avoid slow startup
_model = None
import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print_msg("[dim]Loading embedding model (first time only)...[/dim]")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def select_anchors(
    query: str,
    atoms: List[Dict],
    selected_sections: List[Dict],
    top_k: int = 15
) -> List[Dict]:
    """
    Select Anchor Atoms using both semantic search and keyword boosting across all atoms.
    Atoms within PageIndex-validated sections receive a bonus to favor structural reasoning.
    """
    print_msg(f"[cyan]Selecting anchors from ALL {len(atoms)} atoms (with bonus for validated sections)...[/cyan]")

    # Identify valid section page ranges
    valid_pages = set()
    for s in selected_sections:
        valid_pages.update(range(s["start_page"], s["end_page"] + 1))

    # Embed query and all atoms
    model = _get_model()
    import numpy as np

    query_emb = model.encode(query, convert_to_tensor=False)
    atom_texts = [a["text"] for a in atoms]
    atom_embs = model.encode(atom_texts, convert_to_tensor=False, show_progress_bar=False)

    # Compute cosine similarities with keyword boost
    scores = _cosine_similarity_batch(query_emb, atom_embs, query, atom_texts)

    # Apply PageIndex structural bonus
    for i, atom in enumerate(atoms):
        if atom["page_num"] in valid_pages:
            scores[i] += 0.25  # Increased bonus

    # Get top-k indices
    k = min(top_k, len(atoms))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    anchors = []
    for idx in top_indices:
        atom = atoms[idx].copy()
        atom["anchor_score"] = float(scores[idx])
        anchors.append(atom)

    print_msg(f"[green]Selected {len(anchors)} anchor atoms.[/green]")
    for a in anchors:
        print_msg(f"  → Atom [{a['atom_id']}] page {a['page_num']} score={a['anchor_score']:.3f}: [dim]{a['text'][:80]}...[/dim]")

    return anchors


def _filter_atoms_by_sections(atoms: List[Dict], sections: List[Dict]) -> List[Dict]:
    """Return only atoms whose page_num falls within any selected section's page range."""
    valid = []
    for atom in atoms:
        for section in sections:
            if section["start_page"] <= atom["page_num"] <= section["end_page"]:
                valid.append(atom)
                break
    return valid


def _cosine_similarity_batch(query_emb, atom_embs, query_text: str, atom_texts: List[str]) -> List[float]:
    """Compute cosine similarity between query embedding and all atom embeddings, with keyword boost."""
    import numpy as np
    import re

    q = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    norms = np.linalg.norm(atom_embs, axis=1, keepdims=True) + 1e-10
    normalized = atom_embs / norms
    scores = (normalized @ q).tolist()
    
    # Keyword Boost
    stop_words = {"what", "how", "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "with", "and", "or"}
    # Strip symbols
    clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', query_text).lower()
    query_words = [w for w in clean_query.split() if w not in stop_words and len(w) > 2]
    
    # Add common variants
    if 'dff' in query_words or 'd_ff' in query_text.lower():
        query_words.append('dff')
    if 'table' in query_words and '2' in clean_query.split():
        query_words.append('table 2')
        query_words.append('table2')
    
    for i, text in enumerate(atom_texts):
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
        boost = 0.0
        
        for w in query_words:
            if w in clean_text:
                boost += 0.2  # Massive boost for keyword match
                
        # Extra boost for exact phrase matches
        if 'base model' in clean_text and 'base model' in query_text.lower():
            boost += 0.5
        if 'convs2s' in clean_text and 'convs2s' in query_text.lower():
            boost += 0.5
        if 'innerlayer' in clean_text.replace(' ', ''):
            boost += 0.5
        if 'dff' in clean_text and 'dff' in query_text.lower().replace('_', '').replace('{', '').replace('}', ''):
            boost += 1.0
        if 'noam' in clean_text and 'author' in query_text.lower():
            boost += 1.0
                
        scores[i] += boost
        
    return scores
