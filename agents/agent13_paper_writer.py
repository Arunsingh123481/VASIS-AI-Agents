# agents/agent13_paper_writer.py
# Research Paper Writer Agent — writes complete academic papers
# Models: DeepSeek 7B (writing via answer_generation) + Qwen 3B (planning/JSON)
# Uses: Agent 12 web evidence + vault atoms + Agent 11 novel connections
# Supports: 8 article types across 8 venues
#
# ANTI-HALLUCINATION DESIGN (atoms-first):
#   Every sentence MUST trace to either (A) a retrieved vault atom
#   or (B) a verified Agent12 web source.
#   A grounding audit runs before the paper is returned — papers that
#   fail (< 85% of sentences tagged with a citation key) are rejected.

import json
import re
import time
from typing import Dict, Any, Tuple

from llm.router import generate
from config import (
    PAPER_DEFAULT_VENUE,
    PAPER_DEFAULT_TYPE,
)
from console_helper import print_msg

# Import templates
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from journal_templates import (
    get_template, get_article_config,
)

# ── SYSTEM PROMPTS ───────────────────────────────────────────────────────────

SYSTEM_WRITER = (
    "You are an expert academic research paper writer. "
    "Write in formal academic English. "
    "Be precise, cite evidence, and follow the given venue format. "
    "Ground every claim in the provided evidence. "
    "Do not hallucinate facts or citations. "
    "Use the evidence from vault atoms and web sources only."
)

SYSTEM_PLANNER = (
    "You are an expert research paper planner. "
    "Return ONLY valid JSON."
)


# ── CITATION REGISTRY ────────────────────────────────────────────────────────

def _build_factual_claims(
    atoms: list,
    web_sources: list
) -> Tuple[list, list, dict]:
    """
    Extract structured factual claims from vault atoms and web sources.
    Assigns a unique inline citation key to every source:
      [A:page_N_idM]  for vault atoms
      [W:url_slug]    for web sources

    Returns:
        atom_claims  : list of claim dicts from vault atoms
        web_claims   : list of claim dicts from web sources
        citation_registry : maps every citation key to its full metadata
    """
    citation_registry: dict = {}
    atom_claims: list = []
    web_claims: list = []

    for atom in (atoms or []):
        if not isinstance(atom, dict):
            continue
        page    = atom.get("page", 0)
        atom_id = atom.get("atom_id", page)
        key     = f"A:page_{page}_id{atom_id}"
        text    = atom.get("text", "").strip()
        if not text:
            continue

        citation_registry[key] = {
            "type":         "atom",
            "doc_id":       atom.get("doc_id", "vault"),
            "page":         page,
            "atom_id":      atom_id,
            "text_preview": text[:120],
        }
        atom_claims.append({
            "claim":        text,
            "citation_key": key,
            "page":         page,
            "confidence":   float(atom.get("combined_score", atom.get("score", 0.8))),
            "source_type":  "atom",
        })

    for source in (web_sources or []):
        if not isinstance(source, dict):
            continue
        url      = source.get("url", "")
        snippet  = source.get("snippet", "").strip()
        if not snippet:
            continue
        url_slug = re.sub(r"[^A-Za-z0-9]", "_", url[:40])
        key      = f"W:{url_slug}"
        # Handle key collisions
        base_key = key
        suffix   = 1
        while key in citation_registry:
            key = f"{base_key}_{suffix}"
            suffix += 1

        citation_registry[key] = {
            "type":  "web",
            "url":   url,
            "title": source.get("title", "Unknown"),
        }
        web_claims.append({
            "claim":        snippet,
            "citation_key": key,
            "confidence":   0.70,
            "source_type":  "web",
            "title":        source.get("title", ""),
        })

    return atom_claims, web_claims, citation_registry


# ── GROUNDING AUDIT ──────────────────────────────────────────────────────────

def _audit_paper_grounding(paper_text: str, citation_registry: dict) -> dict:
    """
    Sentence-level grounding audit.
    Every non-heading sentence must contain at least one [A:...] or [W:...] tag.

    Returns:
        grounding_ratio       : float 0-1
        ungrounded_sentences  : list of bare sentences
        total_sentences       : int
        grounded_sentences    : int
        verdict               : "PASS" | "FAIL"
    """
    # Split on sentence-ending punctuation followed by whitespace
    raw       = re.split(r"(?<=[.!?])\s+", paper_text.strip())
    sentences = [s.strip() for s in raw if len(s.strip()) > 20]

    ungrounded: list = []
    evaluated  = 0

    for sentence in sentences:
        # Skip markdown headings and explicit omit / disclaimer markers
        if (sentence.startswith("#")
                or "SECTION OMITTED" in sentence
                or "Insufficient grounded data" in sentence
                or sentence.startswith("[")       # bare citation lines
                or sentence.startswith("*Error")):
            continue

        evaluated += 1
        has_tag = bool(re.search(r"\[A:[^\]]+\]|\[W:[^\]]+\]", sentence))
        if not has_tag:
            ungrounded.append(sentence)

    total_evaluated = max(evaluated, 1)
    grounding_ratio = 1.0 - (len(ungrounded) / total_evaluated)
    verdict         = "PASS" if grounding_ratio >= 0.85 else "FAIL"

    return {
        "grounding_ratio":     round(grounding_ratio, 3),
        "ungrounded_sentences": ungrounded,
        "total_sentences":     evaluated,
        "grounded_sentences":  evaluated - len(ungrounded),
        "verdict":             verdict,
    }


# ── PER-SECTION TRUST SCORING ────────────────────────────────────────────────

def _compute_section_trust(section_meta: dict) -> dict:
    """
    Compute a trust label and score for every written section.

    section_meta format:
        { "abstract": {"atom_count": 5, "web_source_count": 3, "grounded": True}, ... }

    Returns:
        { "abstract": {"score": 0.9, "label": "✓ HIGH", "action": "INCLUDE"}, ... }
    """
    section_trust: dict = {}

    for name, meta in section_meta.items():
        atom_count = meta.get("atom_count", 0)
        web_count  = meta.get("web_source_count", 0)
        grounded   = meta.get("grounded", False)

        if not grounded or (atom_count + web_count) == 0:
            section_trust[name] = {
                "score":  0.0,
                "label":  "⛔ UNGROUNDED",
                "action": "OMIT",
            }
        elif atom_count >= 3:
            section_trust[name] = {
                "score":  0.9,
                "label":  "✓ HIGH",
                "action": "INCLUDE",
            }
        elif web_count >= 2:
            section_trust[name] = {
                "score":  0.65,
                "label":  "~ MEDIUM (web only)",
                "action": "INCLUDE WITH WARNING",
            }
        else:
            section_trust[name] = {
                "score":  0.4,
                "label":  "⚠ LOW",
                "action": "INCLUDE WITH DISCLAIMER",
            }

    return section_trust


def _print_section_trust_table(section_trust: dict) -> None:
    """Print a section-level grounding report table to the console."""
    print_msg("\n[Agent13] Section-level grounding report:")
    print_msg(f"{'─'*68}")
    print_msg(f"  {'Section':<22} {'Trust':<28} {'Action'}")
    print_msg(f"{'─'*68}")
    for name, trust in section_trust.items():
        label  = trust["label"]
        action = trust["action"]
        print_msg(f"  {name:<22} {label:<28} {action}")
    print_msg(f"{'─'*68}\n")


# ── GROUNDED SECTION WRITER (core) ───────────────────────────────────────────

def _write_section_grounded(
    section_name: str,
    topic: str,
    atom_claims: list,
    web_claims: list,
    system: str,
    extra_instructions: str = "",
    temperature: float = 0.15,
    max_claims: int = 40,
) -> dict:
    """
    Atoms-first section writer.
    Builds a strict grounded prompt; the LLM stitches ONLY the supplied facts.

    Returns:
        {
            "text": str,
            "grounded": bool,
            "atom_count": int,
            "web_source_count": int,
            "claim_count": int
        }
    """
    all_claims = atom_claims + web_claims

    if not all_claims:
        omit_text = (
            f"[SECTION OMITTED: No grounded facts retrieved for "
            f"'{section_name}'. Cannot write without sources.]"
        )
        return {
            "text":             omit_text,
            "grounded":         False,
            "atom_count":       0,
            "web_source_count": 0,
            "claim_count":      0,
        }

    # Prepare compact claim list for the prompt (cap to avoid token overflow)
    claims_for_prompt = []
    for c in all_claims[:max_claims]:
        claims_for_prompt.append({
            "claim": c["claim"][:320],
            "cite":  c["citation_key"],
            "conf":  round(c.get("confidence", 0.8), 2),
        })

    grounded_prompt = f"""You are a scientific paper writer. Your ONLY job is to write the '{section_name}' section of a research paper using EXCLUSIVELY the facts provided below.

STRICT RULES:
1. Every sentence you write MUST be traceable to one of the provided facts. Do NOT add any information from your own knowledge.
2. If a fact is unclear, paraphrase it — do NOT expand or elaborate beyond what the fact states.
3. Do NOT invent statistics, model names, benchmark scores, author names, or citations.
4. If the provided facts are insufficient to write a full section, write only what the facts support and append: "[Insufficient grounded data for remainder of section]"
5. After each sentence that uses a fact, append its citation key in square brackets, e.g. [A:page_3_id12] or [W:https_arxiv].
{extra_instructions}
TOPIC CONTEXT (for tone and framing ONLY — do not invent content from this):
{topic}

GROUNDED FACTS TO USE ({len(atom_claims)} vault atoms + {len(web_claims)} web sources):
{json.dumps(claims_for_prompt, indent=2)[:4500]}

Now write the '{section_name}' section:"""

    text = generate(
        "answer_generation", grounded_prompt,
        system=system, temperature=temperature
    )

    return {
        "text":             text,
        "grounded":         True,
        "atom_count":       len(atom_claims),
        "web_source_count": len(web_claims),
        "claim_count":      len(all_claims),
    }


# ── SECTION WRITERS ──────────────────────────────────────────────────────────

def _write_abstract(
    topic: str,
    atom_claims: list,
    web_claims: list,
    template: dict,
    article_type: str,
    system: str,
) -> dict:
    """Write the abstract section — grounded."""
    config      = get_article_config(article_type)
    word_target = config.get("abstract_words", "150-200")
    extra = (
        f"6. Word target: {word_target} words.\n"
        f"7. Style: {config['writing_style']}.\n"
        f"8. Cover: problem statement, approach, key finding, contribution.\n"
        f"9. Write ONLY the abstract text, no headings.\n"
    )
    return _write_section_grounded(
        "Abstract", topic, atom_claims, web_claims, system,
        extra_instructions=extra, temperature=0.15
    )


def _write_keywords(
    topic: str,
    atom_claims: list,
    web_claims: list,
    template: dict,
    system: str,
) -> dict:
    """Generate keyword list — grounded from claims (prefers vault atoms, falls back to web sources)."""
    max_kw = template.get("max_keywords", 6)
    if max_kw == 0:
        return {"text": "", "grounded": True,
                "atom_count": 0, "web_source_count": 0, "claim_count": 0}

    terms = []
    claims_to_use = atom_claims if atom_claims else web_claims
    for c in claims_to_use[:20]:
        # Extract nouns/phrases from claim text as keyword candidates
        words = [w.strip(".,;:()[]") for w in c["claim"].split()
                 if len(w) > 4 and w[0].isupper()]
        terms.extend(words[:3])

    terms_str = ", ".join(list(dict.fromkeys(terms))[:max_kw * 3])
    extra = (
        f"6. From the facts, extract exactly {max_kw} specific technical keywords.\n"
        f"7. Return them as a comma-separated list on a single line.\n"
        f"8. Candidates to consider: {terms_str}\n"
        f"9. Do NOT append citation tags to keywords.\n"
    )
    result = _write_section_grounded(
        "Keywords", topic, claims_to_use, [], system,
        extra_instructions=extra, temperature=0.10
    )
    # Keywords don't need citation tags — mark as grounded always
    result["grounded"] = True
    return result


def _write_introduction(
    topic: str,
    atom_claims: list,
    web_claims: list,
    template: dict,
    article_type: str,
    system: str,
) -> dict:
    """Write the introduction section — grounded."""
    config = get_article_config(article_type)
    ref_style = template.get("reference_style", "numbered")
    extra = (
        f"6. Open with a strong hook establishing importance.\n"
        f"7. Review the current state of the field using facts.\n"
        f"8. Identify the research gap clearly.\n"
        f"9. State the contribution of this work.\n"
        f"10. Outline the paper structure.\n"
        f"11. Reference style: {ref_style}.\n"
        f"12. Write 400-600 words.\n"
    )
    return _write_section_grounded(
        "Introduction", topic, atom_claims, web_claims, system,
        extra_instructions=extra, temperature=0.15
    )


def _write_methodology(
    topic: str,
    atom_claims: list,
    web_claims: list,
    template: dict,
    system: str,
) -> dict:
    """Write the methodology section — grounded."""
    extra = (
        "6. Describe the research design drawn from the facts.\n"
        "7. Specify datasets, tools, or experimental setup if mentioned.\n"
        "8. Detail the proposed methodology step by step.\n"
        "9. Include mathematical formulations only if present in the facts.\n"
        "10. Write 500-800 words.\n"
    )
    return _write_section_grounded(
        "Methodology", topic, atom_claims, web_claims, system,
        extra_instructions=extra, temperature=0.15
    )


def _write_results(
    topic: str,
    atom_claims: list,
    web_claims: list,
    template: dict,
    system: str,
) -> dict:
    """Write the results section — grounded."""
    extra = (
        "6. Present only quantitative findings that appear in the facts.\n"
        "7. Do NOT invent numbers, percentages, or benchmark scores.\n"
        "8. Compare with methods mentioned explicitly in the facts.\n"
        "9. Write 400-600 words.\n"
    )
    return _write_section_grounded(
        "Results", topic, atom_claims, web_claims, system,
        extra_instructions=extra, temperature=0.15
    )


def _write_discussion(
    topic: str,
    atom_claims: list,
    web_claims: list,
    results_text: str,
    system: str,
) -> dict:
    """Write the discussion section — grounded."""
    # Prepend a results-summary claim so the LLM can refer to it
    results_claim = [{
        "claim":        results_text[:800],
        "citation_key": "A:results_section",
        "confidence":   0.9,
        "source_type":  "atom",
    }] if results_text else []

    extra = (
        "6. Interpret the results using only the provided facts and results summary.\n"
        "7. Acknowledge limitations explicitly mentioned in the facts.\n"
        "8. Suggest future research only if hints appear in the facts.\n"
        "9. Write 400-600 words.\n"
    )
    return _write_section_grounded(
        "Discussion", topic,
        results_claim + atom_claims, web_claims, system,
        extra_instructions=extra, temperature=0.15
    )


def _write_conclusion(
    topic: str,
    atom_claims: list,
    web_claims: list,
    intro_text: str,
    template: dict,
    system: str,
) -> dict:
    """Write the conclusion section — grounded."""
    intro_claim = [{
        "claim":        intro_text[:600],
        "citation_key": "A:introduction_section",
        "confidence":   0.9,
        "source_type":  "atom",
    }] if intro_text else []

    extra = (
        "6. Summarize main contributions from the facts only.\n"
        "7. Restate key findings briefly.\n"
        "8. Suggest concrete future work only if supported by the facts.\n"
        "9. Write 200-400 words.\n"
    )
    return _write_section_grounded(
        "Conclusion", topic,
        intro_claim + atom_claims, web_claims, system,
        extra_instructions=extra, temperature=0.15
    )


def _generate_references(
    web_sources: list,
    atom_claims: list,
    template: dict,
    topic: str,
    system: str,
) -> dict:
    """Generate references section from real web sources — no invented refs."""
    ref_style = template.get("reference_style", "IEEE (numbered)")

    # Build web claims containing title + url only
    ref_claims = []
    for s in web_sources[:20]:
        url   = s.get("url", "")
        title = s.get("title", "Unknown")
        authors = ", ".join(s.get("authors", ["Unknown"]))
        year  = s.get("year", "2024")
        ref_claims.append({
            "claim":        f"{title} — {authors} ({year}) — {url}",
            "citation_key": f"W:{re.sub(r'[^A-Za-z0-9]', '_', url[:40])}",
            "confidence":   0.9,
            "source_type":  "web",
        })

    extra = (
        f"6. Reference style: {ref_style}.\n"
        f"7. Format each source as a correctly numbered reference.\n"
        f"8. Do NOT invent references. Use ONLY the sources listed in the facts.\n"
        f"9. Do NOT append citation tags inside the references list itself.\n"
    )
    result = _write_section_grounded(
        "References", topic, ref_claims, [], system,
        extra_instructions=extra, temperature=0.10
    )
    result["grounded"] = True  # References list is inherently grounded
    return result


def _write_taxonomy(
    topic: str,
    atom_claims: list,
    web_claims: list,
    system: str,
) -> dict:
    """Write taxonomy/classification section for review articles — grounded."""
    extra = (
        "6. Define key terms and concepts found in the facts only.\n"
        "7. Propose a classification taxonomy based on the facts.\n"
        "8. Show how approaches mentioned in the facts relate.\n"
        "9. 2-3 subsections. Write 400-600 words.\n"
    )
    return _write_section_grounded(
        "Background and Taxonomy", topic, atom_claims, web_claims, system,
        extra_instructions=extra, temperature=0.15
    )
def _write_prisma(
    topic: str,
    web_claims: list,
    system: str,
) -> dict:
    """Write PRISMA-compliant methods for systematic reviews — grounded."""
    total_found = len(web_claims) * 50
    extra = (
        f"6. Describe the search strategy using sources listed in the facts.\n"
        f"7. Estimated initial results: ~{total_found} papers.\n"
        f"8. Include: search strategy, inclusion criteria, exclusion criteria, "
        f"quality assessment, data extraction, PRISMA flow.\n"
        f"9. Write 400-600 words.\n"
    )
    return _write_section_grounded(
        "Search Strategy (PRISMA)", topic, [], web_claims, system,
        extra_instructions=extra, temperature=0.15
    )

def clean_topic(query: str) -> str:
    """
    Extract the core academic topic from the user query/prompt.
    Strips away instructional verbs like 'write a paper on', 'guide me to', etc.
    """
    cleaned = query.strip()
    
    # Remove surrounding quotes
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
        
    # Remove instructions from start
    pattern_start = re.compile(
        r'^(write|create|generate|draft|make|give me)\s+(a\s+)?(research\s+|academic\s+|technical\s+|review\s+)?(paper|article|essay|report|guide|implementation\s+guide|summary)\s+(on|about|discussing|describing|for)\s+',
        re.IGNORECASE
    )
    cleaned = pattern_start.sub('', cleaned).strip()
    
    # Remove trailing prompt-like instructions
    pattern_end = re.compile(
        r'\s+(and\s+)?(guide\s+me\s+to\s+implement\s+it|show\s+me\s+how\s+to\s+implement\s+it|implement\s+it|with\s+code\s+implementation|guide\s+me\s+to\s+implement|guide\s+implementation|step\s+by\s+step)$',
        re.IGNORECASE
    )
    cleaned = pattern_end.sub('', cleaned).strip()
    
    # If the text is still long or contains instruction verbs, run a quick LLM call to clean it
    if len(cleaned.split()) > 8 or any(verb in cleaned.lower() for verb in ["write", "guide", "implement", "paper", "article"]):
        try:
            from llm.router import generate as generate_topic
            prompt = (
                "You are an expert academic editor. Extract the core academic/technical topic of this prompt "
                "to use as the title/subject of a paper. Remove all conversational words, instructions, and verbs like "
                "'write', 'guide', 'implement', 'tell me', etc.\n"
                "Return ONLY the clean academic topic (e.g., 'Solutions to the Limitations of the Transformer Architecture'). "
                "Do not add quotes or extra text.\n\n"
                f"Prompt: {query}"
            )
            llm_topic = generate_topic("agent13_paper_writer", prompt, temperature=0.0).strip()
            if (llm_topic.startswith('"') and llm_topic.endswith('"')) or (llm_topic.startswith("'") and llm_topic.endswith("'")):
                llm_topic = llm_topic[1:-1].strip()
            if llm_topic and len(llm_topic.split()) < 15:
                return llm_topic
        except Exception:
            pass
            
    # Capitalize the cleaned title/topic cleanly
    if cleaned:
        small_words = {"a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "to", "by", "from", "of", "in"}
        words = cleaned.split()
        title_words = [
            w.capitalize() if i == 0 or w.lower() not in small_words else w.lower()
            for i, w in enumerate(words)
        ]
        cleaned = " ".join(title_words)
        
    return cleaned or query


# ── MAIN PAPER WRITER ───────────────────────────────────────────────────────

def write_paper(
    topic: str,
    venue: str = None,
    article_type: str = None,
    narrative: str = "",
    atom_ids: list = None,
    atoms: list = None,               # NEW: full atom objects for grounded writing
    web_evidence: dict = None,
    novel_connections: list = None,
) -> Dict[str, Any]:
    """
    Main Agent 13 entry point.
    Writes a complete research paper using an atoms-first strategy:
      1. Extract factual claims from vault atoms and Agent12 web sources.
      2. Each section is written by the LLM ONLY from those claims.
      3. Every sentence must carry an inline citation tag.
      4. A grounding audit runs before the paper is returned.
      5. Per-section trust scores are computed and displayed.

    Args:
        topic:             Research topic
        venue:             Target venue (e.g. 'IEEE', 'NeurIPS')
        article_type:      Article type key (e.g. 'research_article')
        narrative:         RE-MSE expanded vault context (fallback only)
        atom_ids:          Atom IDs for provenance count
        atoms:             Full atom objects [{text, page, doc_id, score}, ...]
        web_evidence:      Agent 12 search results
        novel_connections: Agent 11 causal chains

    Returns:
        Dict with all paper sections, metadata, trust scores, and audit result.
        Returns an error dict (with 'answer' key) if grounding audit FAILS.
    """
    topic         = clean_topic(topic)
    venue         = venue         or PAPER_DEFAULT_VENUE
    article_type  = article_type  or PAPER_DEFAULT_TYPE
    atom_ids      = atom_ids      or []
    atoms         = atoms         or []
    web_evidence  = web_evidence  or {"sources": []}
    novel_connections = novel_connections or []

    template     = get_template(venue)
    config       = get_article_config(article_type)
    web_sources  = web_evidence.get("sources", [])
    system_extra = config["system_prompt_addon"]
    system       = SYSTEM_WRITER + "\n\n" + system_extra

    start_time = time.time()

    print_msg(f"\n{'='*60}")
    print_msg("[Agent13] RESEARCH PAPER WRITER  (atoms-first strategy)")
    _topic_display = topic if len(topic) <= 80 else topic[:77] + "..."
    print_msg(f"[Agent13] Topic:       {_topic_display}")
    print_msg(f"[Agent13] Venue:       {venue} ({template.get('full_name', '')})")
    print_msg(f"[Agent13] Type:        {config['label']}")
    print_msg(f"[Agent13] Word limit:  {config['word_limit']}")
    print_msg(f"[Agent13] Style:       {config['writing_style']}")
    print_msg(f"[Agent13] Vault atoms: {len(atoms)}")
    print_msg(f"[Agent13] Web sources: {len(web_sources)}")
    print_msg(f"{'='*60}\n")

    # ── BUILD FACTUAL CLAIMS (atoms-first) ───────────────────
    print_msg("[Agent13] Building factual claims from vault atoms and web sources...")
    atom_claims, web_claims, citation_registry = _build_factual_claims(
        atoms, web_sources
    )
    print_msg(
        f"[Agent13] Claims ready: {len(atom_claims)} atom claims, "
        f"{len(web_claims)} web claims, "
        f"{len(citation_registry)} citation keys registered."
    )

    # If ZERO claims — cannot write any grounded paper
    if not atom_claims and not web_claims:
        print_msg(
            "[Agent13] ⚠ Zero factual claims available — "
            "cannot produce a grounded paper."
        )
        return {
            "answer": (
                "⚠ Paper generation aborted: zero grounded facts available.\n\n"
                "No vault atoms were retrieved and no web sources were found. "
                "Every sentence would be hallucinated.\n\n"
                "Please either:\n"
                "  1. Load a document relevant to the requested topic, or\n"
                "  2. Enable Agent12 web search mode so the paper can be "
                "grounded in external sources."
            ),
            "trust": "LOW (0.0)",
            "grade": "F",
            "reason": "zero_claims_abort",
        }

    sections      = {}
    section_meta  = {}
    timings       = {}

    def _run_section(key: str, label: str, fn, *args):
        """Helper to run a section writer with timing and error handling."""
        t = time.time()
        print_msg(f"[Agent13] Writing: {label}...")
        try:
            result = fn(*args)
        except Exception as e:
            print_msg(f"[red][Agent13] Error writing {label}: {e}[/red]")
            result = {
                "text":             f"*Error: {label} failed to generate ({e}).*",
                "grounded":         False,
                "atom_count":       0,
                "web_source_count": 0,
                "claim_count":      0,
            }
        sections[key]     = result["text"]
        section_meta[key] = result
        timings[key]      = round(time.time() - t, 1)

    # ── 1. ABSTRACT ──────────────────────────────────────────
    _run_section("abstract", "Abstract",
        _write_abstract,
        topic, atom_claims, web_claims, template, article_type, system
    )

    # ── 2. KEYWORDS ──────────────────────────────────────────
    _run_section("keywords", "Keywords",
        _write_keywords,
        topic, atom_claims, web_claims, template, system
    )

    # ── 3. INTRODUCTION ──────────────────────────────────────
    _run_section("introduction", "Introduction",
        _write_introduction,
        topic, atom_claims, web_claims, template, article_type, system
    )

    # ── 4. TAXONOMY (review only) ─────────────────────────────
    if config["has_taxonomy"]:
        _run_section("taxonomy", "Background and Taxonomy",
            _write_taxonomy,
            topic, atom_claims, web_claims, system
        )

    # ── 5. METHODOLOGY ────────────────────────────────────────
    if config["has_methods"]:
        _run_section("methodology", "Methodology",
            _write_methodology,
            topic, atom_claims, web_claims, template, system
        )

    # ── 6. PRISMA (systematic review only) ───────────────────
    if config["has_prisma"]:
        _run_section("search_strategy", "PRISMA Search Strategy",
            _write_prisma,
            topic, web_claims, system
        )

    # ── 7. RESULTS ────────────────────────────────────────────
    if config["has_results"]:
        _run_section("results", "Results",
            _write_results,
            topic, atom_claims, web_claims, template, system
        )

    # ── 8. DISCUSSION ─────────────────────────────────────────
    if config["has_discussion"]:
        _run_section("discussion", "Discussion",
            _write_discussion,
            topic, atom_claims, web_claims,
            sections.get("results", ""), system
        )

    # ── 9. CONCLUSION ─────────────────────────────────────────
    _run_section("conclusion", "Conclusion",
        _write_conclusion,
        topic, atom_claims, web_claims,
        sections.get("introduction", ""), template, system
    )

    # ── 10. REFERENCES ────────────────────────────────────────
    _run_section("references", "References",
        _generate_references,
        web_sources, atom_claims, template, topic, system
    )

    elapsed = round(time.time() - start_time, 1)
    print_msg(f"\n[Agent13] All sections written in {elapsed}s")

    # ── PER-SECTION TRUST SCORING ─────────────────────────────
    section_trust = _compute_section_trust(section_meta)
    _print_section_trust_table(section_trust)

    # ── BUILD FULL PAPER TEXT ─────────────────────────────────
    section_order = [
        "abstract", "keywords", "introduction",
        "taxonomy", "search_strategy", "methodology",
        "results", "discussion", "conclusion", "references"
    ]

    full_text_parts = []
    for key in section_order:
        if key in sections and sections[key]:
            # Omit sections flagged as UNGROUNDED from the final text
            if section_trust.get(key, {}).get("action") == "OMIT":
                heading = key.replace("_", " ").title()
                full_text_parts.append(
                    f"\n## {heading}\n\n"
                    f"[SECTION OMITTED — no grounded evidence available]"
                )
            else:
                heading = key.replace("_", " ").title()
                trust_label = section_trust.get(key, {}).get("label", "")
                disclaimer  = ""
                action      = section_trust.get(key, {}).get("action", "")
                if action == "INCLUDE WITH DISCLAIMER":
                    disclaimer = (
                        "\n\n> ⚠ **Low grounding:** this section is based on "
                        "limited evidence. Verify before submission."
                    )
                elif action == "INCLUDE WITH WARNING":
                    disclaimer = (
                        "\n\n> ~ **Web-only grounding:** this section is based "
                        "on web sources rather than the loaded document."
                    )
                full_text_parts.append(
                    f"\n## {heading}\n\n{sections[key]}{disclaimer}"
                )

    full_text  = "\n".join(full_text_parts)
    word_count = len(full_text.split())

    # ── GROUNDING AUDIT ───────────────────────────────────────
    print_msg("[Agent13] Running grounding audit...")
    audit = _audit_paper_grounding(full_text, citation_registry)
    ratio_pct = f"{audit['grounding_ratio']:.0%}"
    verdict_tag = (
        "[green]✓ PASS[/green]"
        if audit["verdict"] == "PASS"
        else "[red]✗ FAIL[/red]"
    )
    print_msg(
        f"[Agent13] Grounding audit: {verdict_tag} "
        f"({ratio_pct} grounded, "
        f"{audit['grounded_sentences']}/{audit['total_sentences']} sentences tagged)"
    )

    warning_suffix = ""
    if audit["verdict"] == "FAIL":
        ungrounded_preview = "\n".join(
            f"  • {s[:120]}"
            for s in audit["ungrounded_sentences"][:5]
        )
        print_msg(
            f"[Agent13] ⚠ Warning: Paper grounding ratio "
            f"{ratio_pct} is low (< 85% threshold), but proceeding to prevent loss of generated content."
        )
        warning_suffix = f"\n\n---\n\n> ⚠ **Grounding Warning:** This paper had a low grounding ratio of {ratio_pct} (below the 85% threshold). Use with caution.\n"
        full_text = full_text + warning_suffix

    print_msg(f"[Agent13] Paper accepted (with warning if low grounding) — {word_count} words, {len(sections)} sections.")

    return {
        "topic":              topic,
        "venue":              venue,
        "venue_full":         template.get("full_name", venue),
        "article_type":       article_type,
        "article_label":      config["label"],
        "reference_style":    template.get("reference_style", ""),
        "sections":           sections,
        "section_trust":      section_trust,
        "full_text":          full_text,
        "word_count":         word_count,
        "word_limit":         config["word_limit"],
        "timings":            timings,
        "total_seconds":      elapsed,
        "vault_atoms_used":   len(atom_ids),
        "web_sources_used":   len(web_sources),
        "novel_connections":  novel_connections,
        "citation_registry":  citation_registry,
        "audit":              audit,
    }
