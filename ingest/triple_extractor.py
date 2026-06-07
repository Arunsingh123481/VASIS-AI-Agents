# ingest/triple_extractor.py
# Deterministic rule-based triple extraction with lightweight coreference resolution.
# Resolves pronouns (it, they, this, etc.) to the last concrete subject before
# building [Subject]-[Predicate]-[Object] triples, eliminating meaningless graph nodes.

from console_helper import print_msg

import re

# ── Coreference Resolution ───────────────────────────────────────────────────

# Pronouns that should be replaced by the last concrete subject
_SUBJECT_PRONOUNS = {"it", "they", "he", "she", "this", "that", "these", "those", "we"}
_POSSESSIVE_PRONOUNS = {"its", "their", "his", "her", "our"}

# Determiners to strip from extracted spans
_DETERMINERS = re.compile(r'^(the|a|an)\s+', re.IGNORECASE)

# Minimum length for a candidate to be considered a "concrete" subject
_MIN_CONCRETE_LEN = 3


def _is_pronoun(text: str) -> bool:
    """Check if the text (after stripping determiners) is a bare pronoun."""
    cleaned = _DETERMINERS.sub('', text).strip(",; ").lower()
    return cleaned in _SUBJECT_PRONOUNS


def _starts_with_possessive(text: str) -> tuple[bool, str, str]:
    """Check if text starts with a possessive pronoun.
    Returns (is_possessive, pronoun, remainder).
    Example: 'its design' -> (True, 'its', 'design')
    """
    lower = text.lower().strip()
    for pron in _POSSESSIVE_PRONOUNS:
        if lower.startswith(pron + " "):
            remainder = text[len(pron):].strip()
            return True, pron, remainder
    return False, "", text


def _resolve_coreference(text: str, last_subject: str) -> str:
    """Replace a pronoun subject/object with the last known concrete subject.
    Handles:
      - Bare pronouns: 'It' -> 'Software'
      - Possessives:   'Its design' -> 'Software design'
    """
    if not last_subject:
        return text

    stripped = _DETERMINERS.sub('', text).strip(",; ")

    # Case 1: entire span is a bare pronoun
    if stripped.lower() in _SUBJECT_PRONOUNS:
        return last_subject

    # Case 2: span starts with a possessive pronoun
    is_poss, _, remainder = _starts_with_possessive(stripped)
    if is_poss and remainder:
        return f"{last_subject} {remainder}"

    return text


# ── Triple Extraction ────────────────────────────────────────────────────────

def extract_triples(atom: dict) -> list[dict]:
    """Extract factual and causal triples from an atomic text segment.
    
    Uses a two-pass approach:
      1. Rule-based coreference: resolve pronouns to last concrete noun.
      2. Regex verb matching: extract [Subject]-[Relation]-[Object] triples.
    """
    atom_id = int(atom.get("atom_id", 0))
    doc_id = atom.get("doc_id", "")
    page_number = int(atom.get("page_number", atom.get("page_num", 1)))
    text = atom.get("text", "")

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    triples = []

    # Track the last concrete (non-pronoun) subject across sentences in this atom
    last_concrete_subject = ""

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

            subject_raw = sentence_clean[:start_idx].strip()
            obj_raw = sentence_clean[end_idx:].strip()

            # Strip leading determiners
            subject = re.sub(r'^(the|a|an|this|that|these|those)\s+', '', subject_raw, flags=re.IGNORECASE)
            subject = subject.strip(",; ")

            obj = re.sub(r'^(the|a|an|this|that|these|those)\s+', '', obj_raw, flags=re.IGNORECASE)
            obj = obj.strip(",; ")

            # ── Coreference Resolution Pass ──
            # Resolve pronoun subjects
            if _is_pronoun(subject):
                subject = _resolve_coreference(subject, last_concrete_subject)
            else:
                # Check possessive start
                is_poss, _, _ = _starts_with_possessive(subject)
                if is_poss:
                    subject = _resolve_coreference(subject, last_concrete_subject)

            # Resolve pronoun objects
            if _is_pronoun(obj):
                obj = _resolve_coreference(obj, last_concrete_subject)
            else:
                is_poss, _, _ = _starts_with_possessive(obj)
                if is_poss:
                    obj = _resolve_coreference(obj, last_concrete_subject)

            # Update the last concrete subject tracker if this subject is concrete
            if len(subject) >= _MIN_CONCRETE_LEN and not _is_pronoun(subject):
                last_concrete_subject = subject

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
    
    print_msg(f"[cyan]Extracting triples from {total} atoms (with coreference resolution)...[/cyan]")
    for i, atom in enumerate(atoms):
        triples = extract_triples(atom)
        all_triples.extend(triples)
        
        if (i+1) % 10 == 0 or (i+1) == total:
            print_msg(f"  -> Triple Ingestion progress: {i+1}/{total} atoms processed | {len(all_triples)} triples extracted")
            
    return all_triples
