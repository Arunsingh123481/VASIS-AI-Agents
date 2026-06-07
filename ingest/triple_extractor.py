# ingest/triple_extractor.py
# Uses local Qwen Coder (via router) for causal triple extraction from atomic segments

from llm.router import generate_json
from console_helper import print_msg

SYSTEM = ("You are a knowledge extraction assistant. "
          "Return ONLY valid JSON array.")

import re

def extract_triples(atom: dict) -> list[dict]:
    """Extract factual and causal triples from an atomic text segment using heuristic regex."""
    atom_id = int(atom.get("atom_id", 0))
    doc_id = atom.get("doc_id", "")
    page_number = int(atom.get("page_number", atom.get("page_num", 1)))
    text = atom.get("text", "")

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    triples = []

    # Ordering verbs from longest to shortest to avoid partial matches
    relation_verbs = [
        "results in", "result in", "resulting in",
        "leads to", "lead to", "leading to",
        "consists of", "consist of", "consisting of",
        "utilizes", "utilize", "utilizing",
        "implements", "implement", "implementing",
        "proposes", "propose", "proposing",
        "achieves", "achieve", "achieving",
        "improves", "improve", "improving",
        "reduces", "reduce", "reducing",
        "increases", "increase", "increasing",
        "defines", "define", "defining",
        "contains", "contain", "containing",
        "represents", "represent", "representing",
        "enables", "enable", "enabling",
        "allows", "allow", "allowing",
        "causes", "cause", "causing",
        "is", "are", "was", "were", "has", "have", "had", "uses", "use", "using"
    ]

    verb_pattern = re.compile(r'\b(' + '|'.join(re.escape(v) for v in relation_verbs) + r')\b', re.IGNORECASE)

    for sentence in sentences:
        sentence_clean = re.sub(r'[.!?]+$', '', sentence).strip()
        if not sentence_clean:
            continue

        match = verb_pattern.search(sentence_clean)
        if match:
            verb = match.group(1)
            start_idx = match.start()
            end_idx = match.end()

            subject = sentence_clean[:start_idx].strip()
            obj = sentence_clean[end_idx:].strip()

            # Strip leading common determiners or punctuation
            subject = re.sub(r'^(the|a|an|this|that|these|those)\s+', '', subject, flags=re.IGNORECASE)
            subject = subject.strip(",; ")

            obj = re.sub(r'^(the|a|an|this|that|these|those)\s+', '', obj, flags=re.IGNORECASE)
            obj = obj.strip(",; ")

            # Ensure the subject and object are reasonable lengths
            if 2 <= len(subject) <= 100 and 2 <= len(obj) <= 150:
                verb_lower = verb.lower()
                causal_type = "null"
                if verb_lower in ("causes", "cause", "causing"):
                    causal_type = "causes"
                elif verb_lower in ("leads to", "lead to", "leading to"):
                    causal_type = "leads_to"
                elif verb_lower in ("results in", "result in", "resulting in"):
                    causal_type = "results_in"
                elif verb_lower in ("enables", "enable", "enabling", "allows", "allow", "allowing"):
                    causal_type = "enables"

                causal_chain = None
                if causal_type != "null":
                    causal_chain = f"{subject} {verb_lower} {obj}"

                triples.append({
                    "subject": subject,
                    "relation": verb,
                    "object": obj,
                    "causal_type": causal_type,
                    "causal_chain": causal_chain,
                    "atom_id": atom_id,
                    "doc_id": doc_id,
                    "page_number": page_number
                })

                # Keep at most 4 triples per atom
                if len(triples) >= 4:
                    break

    return triples


def extract_all_triples(atoms: list[dict]) -> list[dict]:
    """Iterate over all atoms and extract knowledge triples."""
    all_triples = []
    total = len(atoms)
    
    print_msg(f"[cyan]Extracting triples from {total} atoms...[/cyan]")
    for i, atom in enumerate(atoms):
        triples = extract_triples(atom)
        all_triples.extend(triples)
        
        if (i+1) % 10 == 0 or (i+1) == total:
            print_msg(f"  -> Triple Ingestion progress: {i+1}/{total} atoms processed | {len(all_triples)} triples extracted")
            
    return all_triples
