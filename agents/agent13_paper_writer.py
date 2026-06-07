# agents/agent13_paper_writer.py
# Research Paper Writer Agent — writes complete academic papers
# Models: DeepSeek 7B (writing via answer_generation) + Qwen 3B (planning/JSON)
# Uses: Agent 12 web evidence + vault atoms + Agent 11 novel connections
# Supports: 8 article types across 8 venues

import time
from typing import Dict, List, Optional, Any

from llm.router import generate, generate_json
from config import (
    PAPER_DEFAULT_WORD_LIMIT,
    PAPER_DEFAULT_VENUE,
    PAPER_DEFAULT_TYPE,
)
from console_helper import print_msg

# Import templates
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from journal_templates import (
    JOURNAL_TEMPLATES, ARTICLE_TYPE_CONFIGS,
    get_template, get_article_config,
    list_journal_names, list_conferences, list_article_types,
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


# ── SECTION WRITERS ──────────────────────────────────────────────────────────

def _write_abstract(
    topic: str, narrative: str, web_sources: list,
    template: dict, article_type: str
) -> str:
    """Write the abstract section."""
    config = get_article_config(article_type)
    word_target = config.get("abstract_words", "150-200")

    source_refs = "\n".join([
        f"- {s.get('title', '')}: {s.get('snippet', '')[:100]}"
        for s in web_sources[:5]
    ])

    prompt = (
        f"Write an abstract for a {config['label']} on: {topic}\n\n"
        f"Evidence from vault:\n{narrative[:2000]}\n\n"
        f"Web sources:\n{source_refs}\n\n"
        f"Word target: {word_target} words\n"
        f"Style: {config['writing_style']}\n"
        f"Requirements:\n"
        f"- State the problem clearly\n"
        f"- Describe the approach briefly\n"
        f"- Summarize key findings\n"
        f"- State the main contribution\n\n"
        f"Write ONLY the abstract text, no headings."
    )
    system = SYSTEM_WRITER + "\n\n" + config["system_prompt_addon"]
    return generate("answer_generation", prompt, system=system, temperature=0.15)


def _write_keywords(
    topic: str, narrative: str, template: dict
) -> str:
    """Generate keywords for the paper."""
    max_kw = template.get("max_keywords", 6)
    if max_kw == 0:
        return ""

    prompt = (
        f"Generate {max_kw} academic keywords for a paper on: {topic}\n"
        f"Context: {narrative[:500]}\n\n"
        f"Return keywords as a comma-separated list.\n"
        f"Use specific technical terms, not generic words."
    )
    return generate("answer_generation", prompt, system=SYSTEM_WRITER, temperature=0.1)


def _write_introduction(
    topic: str, narrative: str, web_sources: list,
    template: dict, article_type: str
) -> str:
    """Write the introduction section."""
    config = get_article_config(article_type)
    source_refs = "\n".join([
        f"- {s.get('title', '')}: {s.get('snippet', '')[:120]}"
        for s in web_sources[:6]
    ])

    prompt = (
        f"Write the Introduction for a {config['label']} on: {topic}\n\n"
        f"Evidence from vault:\n{narrative[:2500]}\n\n"
        f"Related work from web:\n{source_refs}\n\n"
        f"Requirements:\n"
        f"- Open with a strong hook establishing importance\n"
        f"- Review the current state of the field\n"
        f"- Identify the research gap clearly\n"
        f"- State the contribution of this work\n"
        f"- Outline the paper structure\n"
        f"- Reference style: {template.get('reference_style', 'numbered')}\n\n"
        f"Write 400-600 words."
    )
    system = SYSTEM_WRITER + "\n\n" + config["system_prompt_addon"]
    return generate("answer_generation", prompt, system=system, temperature=0.15)


def _write_methodology(
    topic: str, narrative: str, template: dict
) -> str:
    """Write the methodology/methods section."""
    prompt = (
        f"Write the Methodology section for a research paper on: {topic}\n\n"
        f"Evidence from vault:\n{narrative[:2500]}\n\n"
        f"Requirements:\n"
        f"- Describe the research design\n"
        f"- Specify datasets, tools, or experimental setup\n"
        f"- Detail the proposed methodology step by step\n"
        f"- Include any mathematical formulations\n"
        f"- Explain evaluation criteria\n\n"
        f"Write 500-800 words."
    )
    return generate("answer_generation", prompt, system=SYSTEM_WRITER, temperature=0.15)


def _write_results(
    topic: str, narrative: str, web_sources: list,
    template: dict
) -> str:
    """Write the results section."""
    source_refs = "\n".join([
        f"- {s.get('title', '')}: {s.get('snippet', '')[:100]}"
        for s in web_sources[:5]
    ])

    prompt = (
        f"Write the Results section for a research paper on: {topic}\n\n"
        f"Evidence from vault:\n{narrative[:2500]}\n\n"
        f"Comparison sources:\n{source_refs}\n\n"
        f"Requirements:\n"
        f"- Present quantitative findings\n"
        f"- Reference tables and figures (describe them textually)\n"
        f"- Compare with baseline methods\n"
        f"- Use specific numbers and percentages\n"
        f"- Report statistical significance where applicable\n\n"
        f"Write 400-600 words."
    )
    return generate("answer_generation", prompt, system=SYSTEM_WRITER, temperature=0.15)


def _write_discussion(
    topic: str, narrative: str, results_text: str,
    web_sources: list
) -> str:
    """Write the discussion section."""
    source_refs = "\n".join([
        f"- {s.get('title', '')}: {s.get('snippet', '')[:100]}"
        for s in web_sources[:5]
    ])

    prompt = (
        f"Write the Discussion section for a research paper on: {topic}\n\n"
        f"Results:\n{results_text[:1500]}\n\n"
        f"Related work:\n{source_refs}\n\n"
        f"Requirements:\n"
        f"- Interpret the results in context\n"
        f"- Compare with existing literature\n"
        f"- Discuss implications of findings\n"
        f"- Acknowledge limitations honestly\n"
        f"- Suggest future research directions\n\n"
        f"Write 400-600 words."
    )
    return generate("answer_generation", prompt, system=SYSTEM_WRITER, temperature=0.15)


def _write_conclusion(
    topic: str, narrative: str, intro_text: str,
    template: dict
) -> str:
    """Write the conclusion section."""
    prompt = (
        f"Write the Conclusion for a research paper on: {topic}\n\n"
        f"Introduction (for consistency):\n{intro_text[:1000]}\n\n"
        f"Requirements:\n"
        f"- Summarize the main contributions\n"
        f"- Restate key findings briefly\n"
        f"- Discuss broader impact\n"
        f"- Suggest concrete future work\n"
        f"- End with a strong closing statement\n\n"
        f"Write 200-400 words."
    )
    return generate("answer_generation", prompt, system=SYSTEM_WRITER, temperature=0.15)


def _generate_references(
    web_sources: list, narrative: str,
    template: dict, topic: str
) -> str:
    """Generate references section based on web sources found."""
    ref_style = template.get("reference_style", "IEEE (numbered)")

    sources_list = "\n".join([
        f"- Title: {s.get('title', 'Unknown')}, "
        f"Authors: {', '.join(s.get('authors', ['Unknown']))}, "
        f"Year: {s.get('year', '2024')}, "
        f"URL: {s.get('url', '')}, "
        f"Source: {s.get('source', 'Web')}"
        for s in web_sources[:20]
    ])

    prompt = (
        f"Generate a References section for a paper on: {topic}\n\n"
        f"Available sources:\n{sources_list}\n\n"
        f"Reference style: {ref_style}\n\n"
        f"Requirements:\n"
        f"- Format each reference correctly\n"
        f"- Number references sequentially\n"
        f"- Include all available metadata\n"
        f"- Add 3-5 classic/foundational references for the field\n\n"
        f"Write the complete references section."
    )
    return generate("answer_generation", prompt, system=SYSTEM_WRITER, temperature=0.1)


def _write_taxonomy(
    topic: str, narrative: str, web_sources: list, system: str
) -> str:
    """Write taxonomy/classification section for review articles."""
    prompt = (
        f"Write a Background and Taxonomy section "
        f"for a review article on: {topic}\n\n"
        f"Evidence:\n{narrative[:2000]}\n\n"
        f"Requirements:\n"
        f"- Define key terms and concepts\n"
        f"- Propose a classification taxonomy\n"
        f"- Show how different approaches relate\n"
        f"- 2-3 subsections\n"
        f"- Include comparison table description\n"
    )
    return generate("answer_generation", prompt, system=system, temperature=0.15)


def _write_prisma(
    topic: str, web_sources: list, system: str
) -> str:
    """Write PRISMA-compliant methods section for systematic reviews."""
    total_found = len(web_sources) * 50

    prompt = (
        f"Write the Methods section for a systematic "
        f"review on: {topic}\n\n"
        f"Include:\n"
        f"2.1 Search Strategy:\n"
        f"  Databases: Google Scholar, ArXiv, "
        f"IEEE Xplore, PubMed, ACM DL\n"
        f"  Search terms: {topic} AND related terms\n"
        f"  Date range: 2018-2025\n"
        f"  Initial results: ~{total_found} papers\n"
        f"2.2 Inclusion criteria (3-4 criteria)\n"
        f"2.3 Exclusion criteria (3-4 criteria)\n"
        f"2.4 Quality assessment (10-point checklist)\n"
        f"2.5 Data extraction template\n"
        f"2.6 PRISMA flow: "
        f"{total_found} -> screened -> included\n"
    )
    return generate("answer_generation", prompt, system=system, temperature=0.15)


# ── MAIN PAPER WRITER ───────────────────────────────────────────────────────

def write_paper(
    topic: str,
    venue: str = None,
    article_type: str = None,
    narrative: str = "",
    atom_ids: list = None,
    web_evidence: dict = None,
    novel_connections: list = None,
) -> Dict[str, Any]:
    """
    Main Agent 13 entry point.
    Writes a complete research paper section by section.

    Args:
        topic:             Research topic
        venue:             Target venue (e.g. 'IEEE', 'NeurIPS')
        article_type:      Article type key (e.g. 'research_article')
        narrative:         RE-MSE expanded vault context
        atom_ids:          Atom IDs for provenance
        web_evidence:      Agent 12 search results
        novel_connections: Agent 11 causal chains

    Returns:
        Dict with all paper sections, metadata, and timings
    """
    venue = venue or PAPER_DEFAULT_VENUE
    article_type = article_type or PAPER_DEFAULT_TYPE
    atom_ids = atom_ids or []
    web_evidence = web_evidence or {"sources": []}
    novel_connections = novel_connections or []

    template = get_template(venue)
    config = get_article_config(article_type)
    web_sources = web_evidence.get("sources", [])
    system_extra = config["system_prompt_addon"]
    system = SYSTEM_WRITER + "\n\n" + system_extra

    start_time = time.time()

    print_msg(f"\n{'='*60}")
    print_msg(f"[Agent13] RESEARCH PAPER WRITER")
    _topic_display = topic if len(topic) <= 80 else topic[:77] + "..."
    print_msg(f"[Agent13] Topic: {_topic_display}")
    print_msg(f"[Agent13] Venue: {venue} ({template.get('full_name', '')})")
    print_msg(f"[Agent13] Type: {config['label']}")
    print_msg(f"[Agent13] Word limit: {config['word_limit']}")
    print_msg(f"[Agent13] Style: {config['writing_style']}")
    print_msg(f"[Agent13] Web sources: {len(web_sources)}")
    print_msg(f"{'='*60}\n")

    sections = {}
    timings = {}

    # ── 1. ABSTRACT ──────────────────────────────────────
    t = time.time()
    print_msg("[Agent13] Writing: Abstract...")
    sections["abstract"] = _write_abstract(
        topic, narrative, web_sources, template, article_type
    )
    timings["abstract"] = round(time.time() - t, 1)

    # ── 2. KEYWORDS ──────────────────────────────────────
    t = time.time()
    print_msg("[Agent13] Writing: Keywords...")
    sections["keywords"] = _write_keywords(
        topic, narrative, template
    )
    timings["keywords"] = round(time.time() - t, 1)

    # ── 3. INTRODUCTION ──────────────────────────────────
    t = time.time()
    print_msg("[Agent13] Writing: Introduction...")
    sections["introduction"] = _write_introduction(
        topic, narrative, web_sources, template, article_type
    )
    timings["introduction"] = round(time.time() - t, 1)

    # ── 4. TAXONOMY (review only) ────────────────────────
    if config["has_taxonomy"]:
        t = time.time()
        print_msg("[Agent13] Writing: Background and Taxonomy...")
        sections["taxonomy"] = _write_taxonomy(
            topic, narrative, web_sources, system
        )
        timings["taxonomy"] = round(time.time() - t, 1)

    # ── 5. METHODOLOGY ───────────────────────────────────
    if config["has_methods"]:
        t = time.time()
        print_msg("[Agent13] Writing: Methodology...")
        sections["methodology"] = _write_methodology(
            topic, narrative, template
        )
        timings["methodology"] = round(time.time() - t, 1)

    # ── 6. PRISMA (systematic review only) ───────────────
    if config["has_prisma"]:
        t = time.time()
        print_msg("[Agent13] Writing: PRISMA Search Strategy...")
        sections["search_strategy"] = _write_prisma(
            topic, web_sources, system
        )
        timings["search_strategy"] = round(time.time() - t, 1)

    # ── 7. RESULTS ───────────────────────────────────────
    if config["has_results"]:
        t = time.time()
        print_msg("[Agent13] Writing: Results...")
        sections["results"] = _write_results(
            topic, narrative, web_sources, template
        )
        timings["results"] = round(time.time() - t, 1)

    # ── 8. DISCUSSION ────────────────────────────────────
    if config["has_discussion"]:
        t = time.time()
        print_msg("[Agent13] Writing: Discussion...")
        sections["discussion"] = _write_discussion(
            topic, narrative,
            sections.get("results", ""),
            web_sources
        )
        timings["discussion"] = round(time.time() - t, 1)

    # ── 9. CONCLUSION ────────────────────────────────────
    t = time.time()
    print_msg("[Agent13] Writing: Conclusion...")
    sections["conclusion"] = _write_conclusion(
        topic, narrative,
        sections["introduction"],
        template
    )
    timings["conclusion"] = round(time.time() - t, 1)

    # ── 10. REFERENCES ───────────────────────────────────
    t = time.time()
    print_msg("[Agent13] Writing: References...")
    sections["references"] = _generate_references(
        web_sources, narrative, template, topic
    )
    timings["references"] = round(time.time() - t, 1)

    elapsed = round(time.time() - start_time, 1)
    print_msg(f"\n[Agent13] Paper complete in {elapsed}s")
    print_msg(f"[Agent13] Sections written: {len(sections)}")

    # ── BUILD FULL PAPER TEXT ─────────────────────────────
    section_order = [
        "abstract", "keywords", "introduction",
        "taxonomy", "search_strategy", "methodology",
        "results", "discussion", "conclusion", "references"
    ]

    full_text_parts = []
    for key in section_order:
        if key in sections and sections[key]:
            heading = key.replace("_", " ").title()
            full_text_parts.append(f"\n## {heading}\n\n{sections[key]}")

    full_text = "\n".join(full_text_parts)

    # Estimate word count
    word_count = len(full_text.split())

    return {
        "topic":              topic,
        "venue":              venue,
        "venue_full":         template.get("full_name", venue),
        "article_type":       article_type,
        "article_label":      config["label"],
        "reference_style":    template.get("reference_style", ""),
        "sections":           sections,
        "full_text":          full_text,
        "word_count":         word_count,
        "word_limit":         config["word_limit"],
        "timings":            timings,
        "total_seconds":      elapsed,
        "vault_atoms_used":   len(atom_ids),
        "web_sources_used":   len(web_sources),
        "novel_connections":  novel_connections,
    }
