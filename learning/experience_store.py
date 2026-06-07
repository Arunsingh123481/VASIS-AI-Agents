# learning/experience_store.py
# Persistent store for query execution experiences

import json
import os
from pathlib import Path
from datetime import datetime
from config import EXPERIENCE_STORE_PATH, MAX_EXPERIENCE_ENTRIES
from console_helper import print_msg

def save_experience(entry: dict):
    """Append a new experience entry to the local logs."""
    try:
        os.makedirs(os.path.dirname(EXPERIENCE_STORE_PATH), exist_ok=True)
        with open(EXPERIENCE_STORE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print_msg(f"[yellow]Failed to append query experience: {e}[/yellow]")


def load_experiences() -> list[dict]:
    """Load cached query experiences up to the MAX entries limit."""
    path = Path(EXPERIENCE_STORE_PATH)
    if not path.exists():
        return []
        
    entries = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
    except Exception as e:
        print_msg(f"[yellow]Failed to read query experiences: {e}[/yellow]")
        
    return entries[-MAX_EXPERIENCE_ENTRIES:]


def record_query_experience(
    question: str,
    doc_id: str,
    useful_atom_ids: list[int],
    useful_node_ids: list[str],
    confidence: float,
    trust_level: str,
    agent_grades: dict
):
    """Assemble and write query experience to storage."""
    entry = {
        "timestamp":       datetime.now().isoformat(),
        "doc_id":          doc_id,
        "question":        question,
        "question_words":  [w for w in question.lower().split() if len(w) > 2],
        "useful_atom_ids": [int(x) for x in useful_atom_ids],
        "useful_node_ids": [str(x) for x in useful_node_ids],
        "confidence":      float(confidence),
        "trust_level":     str(trust_level),
        "agent_grades":    agent_grades
    }
    save_experience(entry)
