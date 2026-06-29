#!/usr/bin/env python3
"""
VASIS AI — Custom Agent Studio
================================
Lets researchers create, connect, run, and delete their own agents
without touching any code.  The 14 fixed agents are always protected.

COMMANDS
--------
  /agent build references                    7-step wizard → new agent
  /my-agents                                 list fixed + custom agents
  /connect abstract introduction references  wire agents into a loop
     --name paper-sections                   optional loop name
     --quality                               add quality gate
  /my-loops                                  list custom loops
  /delete agent references                   delete a custom agent
  /delete loop paper-sections                delete a custom loop
  /references "topic"                        run a custom agent directly
  /loop paper-sections "topic"               run a custom loop

INTEGRATION
-----------
    from agent_builder import AgentStudio

    studio = AgentStudio(llm_fn=your_llm_call)

    # in VasisCLI.__init__:
    self.studio = studio

    # add to dispatch table:
    "/agent":      lambda: self.studio.cmd_agent(args, session=self.session),
    "/my-agents":  lambda: self.studio.cmd_list_agents(),
    "/my-loops":   lambda: self.studio.cmd_list_loops(),
    "/connect":    lambda: self.studio.cmd_connect(args),
    "/delete":     lambda: self.studio.cmd_delete(args),

    # dynamic dispatch for /agentname commands:
    # in your command dispatcher, after checking fixed commands:
    if cmd in self.studio.custom_command_names():
        self.studio.cmd_run_agent(cmd, args)
        return
"""

import json
import time
import re
import hashlib
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable


# =============================================================================
# CONSTANTS
# =============================================================================

STORE_PATH   = Path(".vasis_custom_agents.json")
FIXED_AGENTS = {
    "router", "decomposer", "navigator", "retrieval", "expansion",
    "validation", "contradiction", "temporal", "calibration", "supervisor",
    "synthesis", "websearch", "paperwriter", "implguide",
}

INPUT_OPTIONS = {
    "1": "paper_text",
    "2": "web_search",
    "3": "vault_atoms",
    "4": "all",
    "paper text":    "paper_text",
    "web":           "web_search",
    "vault":         "vault_atoms",
    "all":           "all",
    "all sources":   "all",
}

INPUT_LABELS = {
    "paper_text":  "Paper text",
    "web_search":  "Web search",
    "vault_atoms": "Vault atoms",
    "all":         "All sources",
}

# =============================================================================
# AGENT BLUEPRINT LIBRARY
# =============================================================================
# Each blueprint is a pre-built template for a recognised agent type.
# When the user types /agent build references, the system matches "references"
# to the blueprint below, pre-fills smart defaults, and asks only 3 questions
# instead of 7.  Unknown names fall through to the full 7-question wizard.
#
# Structure of each blueprint:
#   keywords      list of names that trigger this blueprint (fuzzy matched)
#   category      paper_section | analysis | extraction | specialized
#   description   pre-filled Q1 answer
#   input_type    pre-filled Q3 answer
#   output_desc   pre-filled Q4 answer
#   quality_bar   pre-filled Q6 answer
#   citation      default citation style (overridden by domain selection)
#   tip           one-line hint shown to the user when the blueprint fires

AGENT_BLUEPRINTS = {

    # ── Paper section agents ─────────────────────────────────────────────────
    "abstract": {
        "keywords":    ["abstract", "summary", "synopsis"],
        "category":    "paper_section",
        "description": "Write a concise academic abstract summarising the paper's background, methods, key results, and conclusions",
        "input_type":  "paper_text",
        "output_desc": "150–250 word abstract with four parts: background, methods, results, conclusion",
        "quality_bar": "Must cover all four parts (background/methods/results/conclusion), 150–250 words, no citations",
        "citation":    "None",
        "tip":         "Tip: abstracts don't cite sources. The quality bar enforces this automatically.",
    },
    "introduction": {
        "keywords":    ["introduction", "intro", "background"],
        "category":    "paper_section",
        "description": "Write an introduction section that establishes context, identifies the research gap, and states the paper's objective",
        "input_type":  "all",
        "output_desc": "3–4 paragraph introduction ending with a clear research gap statement and objective",
        "quality_bar": "Must end with an explicit research gap and objective statement; every factual claim cited",
        "citation":    "APA 7th",
        "tip":         "Tip: introduction agents work best with vault atoms — they ground claims in real literature.",
    },
    "literaturereview": {
        "keywords":    ["literature", "litreview", "lit_review", "related_work", "relatedwork", "survey"],
        "category":    "paper_section",
        "description": "Write a literature review section that surveys existing work, groups it by theme, and identifies gaps",
        "input_type":  "all",
        "output_desc": "Thematically organised literature review, 400–600 words, grouped under 3–4 sub-headings",
        "quality_bar": "Minimum 8 citations, grouped by theme not chronology, ends with a gap identification paragraph",
        "citation":    "APA 7th",
        "tip":         "Tip: set input to 'all' so it draws on both vault atoms and live web sources.",
    },
    "methodology": {
        "keywords":    ["methodology", "methods", "method", "approach", "procedure"],
        "category":    "paper_section",
        "description": "Write a methodology section describing the research design, data collection process, and analysis approach",
        "input_type":  "paper_text",
        "output_desc": "Methodology section structured as: research design, data/materials, procedure, analysis method",
        "quality_bar": "Reproducibility standard: another researcher could replicate the study from this section alone",
        "citation":    "APA 7th",
        "tip":         "Tip: methodology is usually written from the paper draft — paper text input is sufficient.",
    },
    "results": {
        "keywords":    ["results", "findings", "outcomes", "experiments"],
        "category":    "paper_section",
        "description": "Write a results section presenting findings objectively without interpretation",
        "input_type":  "paper_text",
        "output_desc": "Results section with data presented in logical order; tables and figures referenced by number",
        "quality_bar": "No interpretation or discussion in this section; every number must reference a table or figure",
        "citation":    "None",
        "tip":         "Tip: results sections should be purely descriptive. The quality bar blocks interpretation.",
    },
    "discussion": {
        "keywords":    ["discussion", "interpretation", "analysis", "implications"],
        "category":    "paper_section",
        "description": "Write a discussion section that interprets results, compares with prior work, and discusses implications",
        "input_type":  "all",
        "output_desc": "Discussion section covering: interpretation of results, comparison with literature, limitations, implications",
        "quality_bar": "Must include a limitations sub-section; every comparison with prior work must be cited",
        "citation":    "APA 7th",
        "tip":         "Tip: discussion benefits most from vault atoms — they provide the prior work to compare against.",
    },
    "conclusion": {
        "keywords":    ["conclusion", "conclusions", "summary", "closing"],
        "category":    "paper_section",
        "description": "Write a conclusion section that summarises findings, restates contributions, and suggests future work",
        "input_type":  "paper_text",
        "output_desc": "2–3 paragraph conclusion: key findings summary, research contributions, future directions",
        "quality_bar": "Must not introduce new information; must include at least one concrete future work suggestion",
        "citation":    "None",
        "tip":         "Tip: conclusions should never cite — they synthesise what is already in the paper.",
    },
    "references": {
        "keywords":    ["references", "bibliography", "refs", "citations", "works_cited", "workscited"],
        "category":    "paper_section",
        "description": "Collect and format all references cited in the paper into a complete, properly formatted bibliography",
        "input_type":  "paper_text",
        "output_desc": "Numbered reference list, alphabetically ordered, each entry with complete bibliographic details",
        "quality_bar": "Every in-text citation must appear in the reference list; DOI included where available",
        "citation":    "APA 7th",
        "tip":         "Tip: the citation style you choose here is enforced as a hard rule in the agent's prompt.",
    },

    # ── Analysis agents ──────────────────────────────────────────────────────
    "gapanalysis": {
        "keywords":    ["gap", "gap_analysis", "gapanalysis", "research_gap", "researchgap"],
        "category":    "analysis",
        "description": "Identify and articulate research gaps in the existing literature for a given topic",
        "input_type":  "all",
        "output_desc": "Structured list of 3–5 research gaps, each with: gap description, evidence from literature, suggested direction",
        "quality_bar": "Each gap must be supported by at least 2 citations; gaps must be specific not generic",
        "citation":    "APA 7th",
        "tip":         "Tip: gap analysis is most powerful when vault atoms contain recent literature on the topic.",
    },
    "comparison": {
        "keywords":    ["comparison", "compare", "comparative", "versus", "vs"],
        "category":    "analysis",
        "description": "Compare two or more approaches, methods, or papers across defined criteria",
        "input_type":  "all",
        "output_desc": "Comparison table followed by a 200-word narrative summary of key differences",
        "quality_bar": "Comparison table must cover at least 5 criteria; every cell must be evidence-backed",
        "citation":    "APA 7th",
        "tip":         "Tip: specify the items to compare in your topic when running this agent.",
    },
    "criticalreview": {
        "keywords":    ["critical", "critique", "peer_review", "peerreview", "review"],
        "category":    "analysis",
        "description": "Critically evaluate a paper or approach, identifying strengths, weaknesses, and areas for improvement",
        "input_type":  "paper_text",
        "output_desc": "Structured critical review: strengths, weaknesses, methodological issues, suggestions for improvement",
        "quality_bar": "Each weakness must include a specific suggestion for improvement; no vague criticisms",
        "citation":    "APA 7th",
        "tip":         "Tip: the quality bar forces constructive criticism — every weakness needs a fix.",
    },

    # ── Extraction agents ────────────────────────────────────────────────────
    "keywords": {
        "keywords":    ["keywords", "keyword", "tags", "index_terms", "indexterms"],
        "category":    "extraction",
        "description": "Extract and rank the most relevant keywords and index terms from a paper",
        "input_type":  "paper_text",
        "output_desc": "List of 8–12 keywords ordered by relevance, grouped into: core concepts, methods, applications",
        "quality_bar": "Keywords must appear in the paper text; no generic terms like 'research' or 'study'",
        "citation":    "None",
        "tip":         "Tip: keyword extraction only needs the paper text — no other sources required.",
    },
    "factcheck": {
        "keywords":    ["factcheck", "fact_check", "verify", "verification", "validate"],
        "category":    "extraction",
        "description": "Verify factual claims in the paper against vault sources and web evidence",
        "input_type":  "all",
        "output_desc": "Fact-check report: list of claims with status (verified / unverified / contradicted) and evidence",
        "quality_bar": "Every claim marked 'contradicted' must include the conflicting source; no claims left without status",
        "citation":    "APA 7th",
        "tip":         "Tip: this agent works best with high-quality vault atoms from authoritative sources.",
    },
}


# ── Comprehensive alias map — all venue/journal/conference naming variants ─────
#
# Different publication venues use different section names for the same thing.
# This table maps every known variant to the correct blueprint key so the
# researcher can type the name they know from their own field.
#
# Format:  slugified_alias → blueprint_key

BLUEPRINT_ALIASES = {

    # ── References section ────────────────────────────────────────────────────
    # IEEE, APA, Vancouver, most science journals
    "ref":                      "references",
    "refs":                     "references",
    "referencelist":            "references",
    "reference list":           "references",
    # MLA style (humanities, literature)
    "workscited":               "references",
    "works cited":              "references",
    "workcited":                "references",
    # Chicago / Harvard / humanities
    "bibliography":             "references",
    "bib":                      "references",
    "bibliographies":           "references",
    "selectedbibliography":     "references",
    # Some reports and books
    "sources":                  "references",
    "furtherreading":           "references",
    "further reading":          "references",
    "suggestedreading":         "references",
    "suggested reading":        "references",
    "worksconfulted":           "references",
    "works consulted":          "references",
    "consultedworks":           "references",
    # Footnote/endnote styles (Chicago, legal)
    "notes":                    "references",
    "endnotes":                 "references",
    "footnotes":                "references",
    "notesandbibliography":     "references",
    # Medical / clinical
    "literaturedcited":         "references",
    "literature cited":         "references",
    "litcited":                 "references",
    "citedliterature":          "references",
    "cited literature":         "references",
    # Generic
    "citations":                "references",
    "cites":                    "references",
    "cited":                    "references",

    # ── Methodology section ────────────────────────────────────────────────────
    # Natural sciences (Nature, Science, Cell)
    "methods":                  "methodology",
    "materialsandmethods":      "methodology",
    "materials and methods":    "methodology",
    "materialsmethod":          "methodology",
    "experimentalsection":      "methodology",
    "experimental section":     "methodology",
    "experimental":             "methodology",
    "experimentaldesign":       "methodology",
    # Medical / clinical trials
    "patientsandmethods":       "methodology",
    "patients and methods":     "methodology",
    "studydesign":              "methodology",
    "study design":             "methodology",
    "clinicalmethods":          "methodology",
    # Social sciences
    "researchdesign":           "methodology",
    "research design":          "methodology",
    "researchmethod":           "methodology",
    "research method":          "methodology",
    "procedure":                "methodology",
    "procedures":               "methodology",
    "datacollection":           "methodology",
    # CS / AI / Engineering
    "approach":                 "methodology",
    "systemoverview":           "methodology",
    "system overview":          "methodology",
    "technicalapproach":        "methodology",
    "technical approach":       "methodology",
    "meth":                     "methodology",
    "method":                   "methodology",

    # ── Literature Review ──────────────────────────────────────────────────────
    # CS / AI (most common in ACM, IEEE CS papers)
    "relatedwork":              "literaturereview",
    "related work":             "literaturereview",
    "relatedworks":             "literaturereview",
    "priorwork":                "literaturereview",
    "prior work":               "literaturereview",
    "previouswork":             "literaturereview",
    "previous work":            "literaturereview",
    "priorart":                 "literaturereview",
    "prior art":                "literaturereview",
    # Engineering / general science
    "stateoftheart":            "literaturereview",
    "state of the art":         "literaturereview",
    "sota":                     "literaturereview",
    "literaturesurvey":         "literaturereview",
    "literature survey":        "literaturereview",
    "reviewofliterature":       "literaturereview",
    "review of literature":     "literaturereview",
    # Social sciences / humanities
    "theoreticalframework":     "literaturereview",
    "theoretical framework":    "literaturereview",
    "conceptualframework":      "literaturereview",
    "conceptual framework":     "literaturereview",
    # Short forms
    "lit":                      "literaturereview",
    "rw":                       "literaturereview",
    "litreview":                "literaturereview",

    # ── Results section ────────────────────────────────────────────────────────
    # Social sciences / qualitative research
    "findings":                 "results",
    "finding":                  "results",
    # Medical / clinical
    "outcomes":                 "results",
    "clinicaloutcomes":         "results",
    "clinical outcomes":        "results",
    # Natural sciences
    "observations":             "results",
    "data":                     "results",
    "experimentalresults":      "results",
    "experimental results":     "results",
    "dataanalysis":             "results",
    "data analysis":            "results",
    # Short forms
    "res":                      "results",
    "result":                   "results",

    # ── Discussion section ─────────────────────────────────────────────────────
    "analysis":                 "discussion",
    "interpretation":           "discussion",
    "implications":             "discussion",
    "discussionandimplications":"discussion",
    "clinicalimplications":     "discussion",
    "disc":                     "discussion",

    # ── Conclusion section ─────────────────────────────────────────────────────
    # CS / engineering (common to combine with future work)
    "conclusionandfuturework":  "conclusion",
    "conclusion and future work":"conclusion",
    "conclusions":              "conclusion",
    "concludingremarks":        "conclusion",
    "concluding remarks":       "conclusion",
    "closingremarks":           "conclusion",
    "closing remarks":          "conclusion",
    "finalremarks":             "conclusion",
    "final remarks":            "conclusion",
    "closingstatement":         "conclusion",
    # Reports
    "recommendations":          "conclusion",
    "summaryandrecommendations":"conclusion",
    # Short forms
    "conc":                     "conclusion",
    "concl":                    "conclusion",
    "closing":                  "conclusion",

    # ── Abstract section ───────────────────────────────────────────────────────
    # Reports / business / government
    "executivesummary":         "abstract",
    "executive summary":        "abstract",
    # Books / book chapters
    "synopsis":                 "abstract",
    "precis":                   "abstract",
    "précis":                   "abstract",
    "overview":                 "abstract",
    # General
    "abs":                      "abstract",
    "summ":                     "abstract",

    # ── Introduction section ───────────────────────────────────────────────────
    # Medical / some engineering
    "background":               "introduction",
    "motivation":               "introduction",
    "context":                  "introduction",
    "preliminaries":            "introduction",
    "preamble":                 "introduction",
    "scope":                    "introduction",
    # Short forms
    "intro":                    "introduction",
    "bg":                       "introduction",

    # ── Keywords section ───────────────────────────────────────────────────────
    # IEEE standard
    "indexterms":               "keywords",
    "index terms":              "keywords",
    "indexterm":                "keywords",
    # ACM
    "ccsconcepts":              "keywords",
    "ccs concepts":             "keywords",
    # Medical / NLM / PubMed
    "meshterms":                "keywords",
    "mesh terms":               "keywords",
    "meshterminals":            "keywords",
    # General academic
    "keyterms":                 "keywords",
    "key terms":                "keywords",
    "subjectterms":             "keywords",
    "subject terms":            "keywords",
    "subjectheadings":          "keywords",
    "subject headings":         "keywords",
    "descriptors":              "keywords",
    "tags":                     "keywords",
    # Short forms
    "kw":                       "keywords",
    "kws":                      "keywords",

    # ── Gap analysis ───────────────────────────────────────────────────────────
    "researchgap":              "gapanalysis",
    "research gap":             "gapanalysis",
    "knowledgegap":             "gapanalysis",
    "knowledge gap":            "gapanalysis",
    "gaps":                     "gapanalysis",
    "gapidentification":        "gapanalysis",
    "gap":                      "gapanalysis",

    # ── Comparison ────────────────────────────────────────────────────────────
    "compare":                  "comparison",
    "comparative":              "comparison",
    "comparativeanalysis":      "comparison",
    "comparative analysis":     "comparison",
    "comparativestudy":         "comparison",
    "vs":                       "comparison",
    "versus":                   "comparison",

    # ── Critical review ───────────────────────────────────────────────────────
    "critique":                 "criticalreview",
    "criticalanalysis":         "criticalreview",
    "critical analysis":        "criticalreview",
    "qualityassessment":        "criticalreview",
    "quality assessment":       "criticalreview",
    "peerreviewreport":         "criticalreview",
    "peer review report":       "criticalreview",
    "peerreview":               "criticalreview",
    "crit":                     "criticalreview",
}


# ── Category → which 3 questions to ask in the blueprint fast path ────────────
#
# Each category has three question slots: Q1 (domain), Q2 (varies by type),
# Q3 (extra / specific preferences).  This replaces the old one-size-fits-all
# "domain + citation style + extra rules" for all blueprints.

CATEGORY_QUESTIONS = {

    # Sections with NO citations (asking for citation style would confuse things)
    "no_citation": {
        "q2_label": "Target length or format?",
        "q2_hint": (
            "How long or how many items should the output be?\n"
            "  e.g.  '150–250 words' (abstract)\n"
            "  e.g.  '2–3 paragraphs, no citations allowed' (conclusion)\n"
            "  e.g.  '8–12 keywords grouped by concept' (keywords)"
        ),
        "q2_key":   "output_format",
        "q2_type":  "free",
        "q3_label": "Any extra rules? (press Enter to skip)",
        "q3_hint": (
            "Domain-specific instructions for this section.\n"
            "  e.g.  'must not introduce any new information'\n"
            "  e.g.  'write in passive voice throughout'\n"
            "  e.g.  'no acronyms without spelling them out first'"
        ),
    },

    # Sections that cite heavily (introduction, methodology, discussion, lit review)
    "citation_section": {
        "q2_label": "Citation style?",
        "q2_hint":  "",     # filled dynamically from domain selection
        "q2_key":   "citation_style",
        "q2_type":  "citation_map",
        "q3_label": "Any extra rules? (press Enter to skip)",
        "q3_hint": (
            "Domain-specific instructions.\n"
            "  e.g.  'must end with a clear research gap statement' (introduction)\n"
            "  e.g.  'include a limitations sub-section' (discussion)\n"
            "  e.g.  'minimum 8 citations grouped by theme' (literature review)"
        ),
    },

    # References section — citation style + formatting preferences
    "references_section": {
        "q2_label": "Citation style?",
        "q2_hint":  "",     # filled dynamically from domain selection
        "q2_key":   "citation_style",
        "q2_type":  "citation_map",
        "q3_label": "Formatting preferences?",
        "q3_hint": (
            "How should individual entries be ordered and formatted?\n"
            "  e.g.  'alphabetical order by first author surname'\n"
            "  e.g.  'numbered in order of appearance in the paper'\n"
            "  e.g.  'include DOI for every entry where available'\n"
            "  e.g.  'include URL and date accessed for web sources'"
        ),
    },

    # Analysis agents — citation style + coverage requirements
    "analysis": {
        "q2_label": "Citation style?",
        "q2_hint":  "",     # filled dynamically
        "q2_key":   "citation_style",
        "q2_type":  "citation_map",
        "q3_label": "Minimum coverage or scope requirement?",
        "q3_hint": (
            "What must the agent cover to be considered complete?\n"
            "  e.g.  'minimum 5 distinct sources per gap identified'\n"
            "  e.g.  'must cover literature from the last 5 years only'\n"
            "  e.g.  'comparison must cover at least 6 criteria per approach'\n"
            "  e.g.  'every weakness must include a specific suggestion to fix it'"
        ),
    },

    # Extraction agents — output format + extra rules
    "extraction": {
        "q2_label": "Output format?",
        "q2_hint": (
            "How should the output be structured or presented?\n"
            "  e.g.  'bulleted list of 8–12 terms, grouped by concept type'\n"
            "  e.g.  'table with columns: term | definition | source'\n"
            "  e.g.  'ranked list with relevance score for each term'\n"
            "  e.g.  'fact-check report as: claim | status | evidence'"
        ),
        "q2_key":  "output_format",
        "q2_type": "free",
        "q3_label": "Any extra rules? (press Enter to skip)",
        "q3_hint": (
            "Domain-specific restrictions.\n"
            "  e.g.  'no generic terms like research or study'\n"
            "  e.g.  'only include terms that appear verbatim in the paper'\n"
            "  e.g.  'flag all unverified claims separately from contradicted ones'"
        ),
    },
}

# Maps each blueprint key to its category
BLUEPRINT_CATEGORY_MAP = {
    "abstract":         "no_citation",
    "results":          "no_citation",
    "conclusion":       "no_citation",
    "keywords":         "extraction",
    "factcheck":        "extraction",
    "introduction":     "citation_section",
    "literaturereview": "citation_section",
    "methodology":      "citation_section",
    "discussion":       "citation_section",
    "references":       "references_section",
    "gapanalysis":      "analysis",
    "comparison":       "analysis",
    "criticalreview":   "analysis",
}


# ── Domain → preferred citation styles (ordered by relevance) ────────────────
DOMAIN_CITATION_PREFERENCE = {
    "CS / AI":               ["IEEE", "APA 7th", "ACM", "MLA",     "Chicago", "None"],
    "Medicine / Biology":    ["Vancouver", "APA 7th", "MLA",       "Chicago", "IEEE", "None"],
    "Physics / Chemistry":   ["APA 7th",  "Chicago", "MLA",        "Vancouver","IEEE","None"],
    "Social Sciences":       ["APA 7th",  "Chicago", "MLA",        "Vancouver","IEEE","None"],
    "Engineering":           ["IEEE",     "APA 7th", "Chicago",    "MLA",  "Vancouver","None"],
    "General / Other":       ["APA 7th",  "IEEE",    "MLA",        "Chicago","Vancouver","None"],
}


# =============================================================================
# BLUEPRINT DETECTION
# =============================================================================

def detect_blueprint(agent_name: str, llm_fn=None) -> tuple[str, dict | None]:
    """
    Given an agent name, return (blueprint_key, blueprint_dict) or
    (None, None) if no match found.

    Matching order:
      1. Exact slug match             e.g. "references" → references blueprint
      2. Alias match                  e.g. "ref"        → references blueprint
      3. Keyword fuzzy match          e.g. "refs2024"   → references blueprint
      4. LLM inference (if llm_fn)    e.g. "drug_interaction_checker"
      5. No match                     → full 7-question wizard
    """
    slug = _slugify(agent_name)

    # 1. Exact match
    if slug in AGENT_BLUEPRINTS:
        return slug, AGENT_BLUEPRINTS[slug]

    # 2. Alias match
    if slug in BLUEPRINT_ALIASES:
        key = BLUEPRINT_ALIASES[slug]
        return key, AGENT_BLUEPRINTS[key]

    # 3. Fuzzy keyword match — any blueprint keyword contained in slug or vice versa
    for key, bp in AGENT_BLUEPRINTS.items():
        for kw in bp["keywords"]:
            kw_slug = _slugify(kw)
            if kw_slug in slug or slug in kw_slug:
                return key, bp

    # 4. LLM inference for unknown names
    if llm_fn:
        known = list(AGENT_BLUEPRINTS.keys())
        prompt = (
            f"A researcher wants to build a research agent called '{agent_name}'.\n"
            f"Which of these known agent types is it most similar to?\n"
            f"Known types: {', '.join(known)}\n"
            f"Reply with ONLY the matching type name, or 'none' if nothing fits."
        )
        try:
            result = llm_fn(prompt).strip().lower().replace("'", "").replace('"', "")
            inferred = _slugify(result)
            if inferred in AGENT_BLUEPRINTS:
                return inferred, AGENT_BLUEPRINTS[inferred]
        except Exception:
            pass

    # 5. No match
    return None, None


def domain_ordered_citations(domain: str) -> list[str]:
    """
    Return citation styles ordered by relevance for the given domain.
    The first item is the suggested default.
    """
    return DOMAIN_CITATION_PREFERENCE.get(domain, DOMAIN_CITATION_PREFERENCE["General / Other"])


def format_citation_options(domain: str) -> str:
    """
    Build the citation Q5 hint string with domain-relevant ordering
    and a '← suggested' marker on the top pick.
    """
    ordered = domain_ordered_citations(domain)
    parts = []
    for i, style in enumerate(ordered[:6], start=1):
        marker = " ← suggested" if i == 1 else ""
        parts.append(f"[{i}] {style}{marker}")
    return "  ".join(parts)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class CustomAgent:
    """A researcher-defined agent stored persistently."""
    agent_id:       str     # slug, e.g. "references"
    command:        str     # "/references"
    created:        float
    description:    str     # Q1 — what it does
    research_domain:str     # Q2 — e.g. "CS / AI", "Medicine"
    input_type:     str     # Q3 — "paper_text" | "web_search" | "vault_atoms" | "all"
    output_desc:    str     # Q4 — what it produces
    citation_style: str     # Q5 — "APA 7th", "IEEE", "None", etc.
    quality_bar:    str     # Q6 — what makes a good output
    extra:          str     # Q7 — any special rules (optional)
    system_prompt:  str     # generated from all the above
    model:          str = "deepseek-llm:7b"
    run_count:      int = 0
    last_run:       float = 0.0


@dataclass
class CustomLoop:
    """A sequence of custom (or fixed) agents wired into a loop."""
    loop_id:      str           # slug, e.g. "paper-sections"
    command:      str           # "paper-sections" (no slash — loops use /loop)
    created:      float
    agent_ids:    list          # ["abstract", "introduction", "references"]
    quality_gate: bool = False  # retry each agent if output too short
    run_count:    int = 0
    last_run:     float = 0.0


# =============================================================================
# STORE
# =============================================================================

class CustomAgentStore:
    """Atomic JSON persistence for custom agents and loops."""

    SCHEMA = {
        "version": "1.0",
        "agents":  {},    # agent_id → CustomAgent dict
        "loops":   {},    # loop_id  → CustomLoop dict
    }

    def __init__(self, path: Path = STORE_PATH):
        self.path = path
        self._data: dict = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text("utf-8"))
                return
            except (json.JSONDecodeError, OSError):
                pass
        self._data = {"version": "1.0", "agents": {}, "loops": {}}
        self._save()

    def _save(self):
        dir_ = self.path.parent
        try:
            fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError as e:
            print(f"[AgentStore] Write error: {e}")

    # ── agents ────────────────────────────────────────────────────────────────

    def save_agent(self, agent: CustomAgent):
        self._data["agents"][agent.agent_id] = asdict(agent)
        self._save()

    def get_agent(self, agent_id: str) -> Optional[CustomAgent]:
        d = self._data["agents"].get(agent_id)
        return CustomAgent(**d) if d else None

    def all_agents(self) -> list[CustomAgent]:
        return [CustomAgent(**d) for d in self._data["agents"].values()]

    def delete_agent(self, agent_id: str) -> bool:
        if agent_id in self._data["agents"]:
            del self._data["agents"][agent_id]
            self._save()
            return True
        return False

    def agent_exists(self, agent_id: str) -> bool:
        return agent_id in self._data["agents"]

    # ── loops ─────────────────────────────────────────────────────────────────

    def save_loop(self, loop: CustomLoop):
        self._data["loops"][loop.loop_id] = asdict(loop)
        self._save()

    def get_loop(self, loop_id: str) -> Optional[CustomLoop]:
        d = self._data["loops"].get(loop_id)
        return CustomLoop(**d) if d else None

    def all_loops(self) -> list[CustomLoop]:
        return [CustomLoop(**d) for d in self._data["loops"].values()]

    def delete_loop(self, loop_id: str) -> bool:
        if loop_id in self._data["loops"]:
            del self._data["loops"][loop_id]
            self._save()
            return True
        return False

    def loops_using_agent(self, agent_id: str) -> list[str]:
        """Return loop IDs that include this agent."""
        return [
            lid for lid, ldata in self._data["loops"].items()
            if agent_id in ldata.get("agent_ids", [])
        ]


# =============================================================================
# PROMPT GENERATOR
# =============================================================================

def generate_system_prompt(
    agent_id:       str,
    description:    str,
    research_domain:str,
    input_type:     str,
    output_desc:    str,
    citation_style: str,
    quality_bar:    str,
    extra:          str,
    llm_fn:         Optional[Callable],
) -> str:
    """
    Ask the LLM to write an expert system prompt using all 7 wizard answers.
    Falls back to a rich template if llm_fn is None or fails.
    """
    citation_line = (
        f"Citation style: {citation_style}. Apply this style to every reference."
        if citation_style.lower() not in ("none", "n/a", "")
        else "No specific citation style required."
    )

    quality_line = (
        f"Quality standard: {quality_bar}. Reject your own output if it does not meet this."
        if quality_bar
        else ""
    )

    extra_line = (
        f"Additional rule: {extra}."
        if extra
        else ""
    )

    meta_prompt = (
        f"You are a prompt engineer. Write a concise, expert system prompt for "
        f"a research AI agent. Use all details below.\n\n"
        f"Agent name: /{agent_id}\n"
        f"Research domain: {research_domain}\n"
        f"Specialisation: {description}\n"
        f"Input: {INPUT_LABELS.get(input_type, input_type)}\n"
        f"Output: {output_desc}\n"
        f"Quality requirement: {quality_bar}\n"
        f"{citation_line}\n"
        f"{quality_line}\n"
        f"{extra_line}\n\n"
        f"Rules for writing the system prompt:\n"
        f"- 4 to 6 sentences only\n"
        f"- Be specific to the domain ({research_domain})\n"
        f"- Include the quality bar as a hard requirement\n"
        f"- Include the citation style as a hard requirement if given\n"
        f"- End with: 'Output must be well-formatted markdown.'\n"
        f"Write ONLY the system prompt — no preamble."
    )

    if llm_fn:
        try:
            result = llm_fn(meta_prompt)
            if result and len(result) > 40:
                return result.strip()
        except Exception:
            pass

    # ── rich template fallback ────────────────────────────────────────────────
    parts = [
        f"You are an expert research assistant specialising in {research_domain}.",
        f"Your task is to {description}.",
        f"You receive {INPUT_LABELS.get(input_type, input_type)} as your context "
        f"and must produce {output_desc}.",
    ]
    if citation_style.lower() not in ("none", "n/a", ""):
        parts.append(f"Apply {citation_style} citation style to every reference.")
    if quality_bar:
        parts.append(f"Your output must satisfy: {quality_bar}.")
    if extra:
        parts.append(f"{extra}.")
    parts.append("Output must be well-formatted markdown.")

    return " ".join(parts)


# =============================================================================
# DOMAIN & CITATION MAPS
# =============================================================================

DOMAIN_OPTIONS = {
    "1": "CS / AI",         "2": "Medicine / Biology",
    "3": "Physics / Chemistry", "4": "Social Sciences",
    "5": "Engineering",     "6": "General / Other",
    "cs": "CS / AI",        "ai": "CS / AI",
    "medicine": "Medicine / Biology", "bio": "Medicine / Biology",
    "physics": "Physics / Chemistry", "chemistry": "Physics / Chemistry",
    "social": "Social Sciences",      "engineering": "Engineering",
    "general": "General / Other",     "other": "General / Other",
}

CITATION_OPTIONS = {
    "1": "APA 7th",   "2": "IEEE",     "3": "MLA",
    "4": "Chicago",   "5": "Vancouver","6": "None",
    "apa": "APA 7th", "ieee": "IEEE",  "mla": "MLA",
    "chicago": "Chicago", "vancouver": "Vancouver",
    "none": "None",   "n/a": "None",   "": "None",
}

WIZARD_QUESTIONS = [
    {
        "step":     1,
        "label":    "What is the main job of this agent?",
        "hint":     "Be specific. e.g. 'collect and format all references cited in the paper into APA'\n"
                    "  or 'write a 200-word abstract summarising methodology and results'",
        "key":      "description",
        "optional": False,
    },
    {
        "step":     2,
        "label":    "Which research field is this for?",
        "hint":     "[1] CS / AI   [2] Medicine / Biology   [3] Physics / Chemistry\n"
                    "  [4] Social Sciences   [5] Engineering   [6] General / Other",
        "key":      "research_domain",
        "map":      DOMAIN_OPTIONS,
        "optional": False,
    },
    {
        "step":     3,
        "label":    "What should it read as input?",
        "hint":     "[1] Paper text   [2] Web / academic sources   [3] Vault documents   [4] All of these",
        "key":      "input_type",
        "map":      INPUT_OPTIONS,
        "optional": False,
    },
    {
        "step":     4,
        "label":    "What exactly should it produce?",
        "hint":     "Describe the output precisely. e.g. 'numbered APA reference list with DOI links'\n"
                    "  or '3-paragraph introduction with a clear research gap statement'",
        "key":      "output_desc",
        "optional": False,
    },
    {
        "step":     5,
        "label":    "Which citation style should it use?",
        "hint":     "[1] APA 7th   [2] IEEE   [3] MLA   [4] Chicago   [5] Vancouver   [6] None",
        "key":      "citation_style",
        "map":      CITATION_OPTIONS,
        "optional": False,
    },
    {
        "step":     6,
        "label":    "What is your quality bar for this agent?",
        "hint":     "What must be true for the output to be acceptable?\n"
                    "  e.g. 'every reference must have a DOI', 'minimum 250 words',\n"
                    "  'must cite at least 5 sources', 'alphabetical order required'",
        "key":      "quality_bar",
        "optional": False,
    },
    {
        "step":     7,
        "label":    "Any extra rules or restrictions? (press Enter to skip)",
        "hint":     "Domain-specific instructions. e.g. 'only peer-reviewed sources',\n"
                    "  'ignore self-citations', 'use passive voice throughout'",
        "key":      "extra",
        "optional": True,
    },
]


# =============================================================================
# UNKNOWN AGENT — OPEN-ENDED QUESTIONS (no options, free-form text)
# =============================================================================
# These 7 questions are shown only for agents whose names don't match
# any of the 14 blueprints.  No numbered choices — the user writes
# everything in their own words.  The answers are then validated for
# research relevance before the agent is built.

UNKNOWN_AGENT_QUESTIONS = [
    {
        "key":   "specialisation",
        "label": "What is this agent specialised in?",
        "hint": (
            "Describe its expertise domain in detail.\n"
            "  Write as much as you need — more detail = better agent.\n"
            "  e.g.  'biomedical named entity recognition in clinical trial abstracts'\n"
            "  e.g.  'automated quality scoring of systematic literature reviews'\n"
            "  e.g.  'extracting causal relationships between variables in social science papers'"
        ),
    },
    {
        "key":   "input_desc",
        "label": "What does this agent receive as input?",
        "hint": (
            "Describe the raw data or content it works with.\n"
            "  e.g.  'raw clinical trial PDF reports from PubMed'\n"
            "  e.g.  'a set of research paper abstracts on a given topic'\n"
            "  e.g.  'survey response data from a qualitative study'"
        ),
    },
    {
        "key":   "output_desc",
        "label": "What exactly does this agent produce as output?",
        "hint": (
            "Be precise about format AND content.\n"
            "  e.g.  'a structured JSON list of adverse drug events with confidence scores'\n"
            "  e.g.  'a ranked comparison table of methodological approaches'\n"
            "  e.g.  'annotated dataset with entity labels and relationship types'"
        ),
    },
    {
        "key":   "tasks",
        "label": "What specific tasks does this agent perform, step by step?",
        "hint": (
            "List the steps in order — this becomes the agent's internal logic.\n"
            "  e.g.  '1. reads the abstract  2. identifies drug names  "
            "3. maps them to targets  4. classifies interaction type  5. outputs structured data'\n"
            "  e.g.  '1. receives paper text  2. segments by section  "
            "3. scores each section for methodological quality  4. flags weak sections'"
        ),
    },
    {
        "key":   "research_link",
        "label": "How does this agent connect to your research workflow?",
        "hint": (
            "Explain where it fits in your pipeline — before, during, or after writing?\n"
            "  e.g.  'its output feeds into my gap_analysis agent as structured evidence'\n"
            "  e.g.  'I run it before /paper to pre-process my corpus into labelled entities'\n"
            "  e.g.  'it runs after /references to verify every citation is peer-reviewed'"
        ),
    },
    {
        "key":   "quality_signal",
        "label": "How will you know if this agent did a good job?",
        "hint": (
            "Describe exactly what a correct, high-quality output looks like.\n"
            "  This becomes the agent's quality bar — it checks its own output against this.\n"
            "  e.g.  'every identified entity has a confidence score above 0.7 and a source sentence'\n"
            "  e.g.  'all claims are backed by a DOI-linked citation from the input corpus'\n"
            "  e.g.  'the output covers all 5 methodological dimensions: design, sample, "
            "instrument, analysis, validity'"
        ),
    },
    {
        "key":   "domain_knowledge",
        "label": "What specific domain knowledge does this agent need?",
        "hint": (
            "Mention any terminology, standards, ontologies, or conventions it must know.\n"
            "  e.g.  'UMLS medical terminology, ICD-10 codes, MeSH headings'\n"
            "  e.g.  'IEEE citation format, transformer architecture concepts, BLEU scoring'\n"
            "  e.g.  'APA style, grounded theory methodology, thematic analysis conventions'"
        ),
    },
]


# =============================================================================
# RESEARCH RELEVANCE VALIDATOR
# =============================================================================

# Vocabulary of concepts found in genuine research workflows
RESEARCH_VOCABULARY = {
    # core research actions
    "research", "study", "analysis", "review", "literature", "paper",
    "publication", "abstract", "methodology", "experiment", "hypothesis",
    "dataset", "corpus", "citation", "reference", "bibliography", "findings",
    "results", "conclusion", "survey", "systematic", "meta", "evaluation",
    "benchmark", "baseline", "annotation", "labelling", "labeling",
    # academic domain words
    "biomedical", "clinical", "medical", "pharmaceutical", "biology",
    "chemistry", "physics", "engineering", "machine learning", "nlp",
    "natural language", "social science", "psychology", "economics",
    "linguistics", "mathematics", "neuroscience", "genomics", "proteomics",
    "epidemiology", "immunology", "cognitive", "computational", "statistical",
    # research artifact words
    "model", "algorithm", "framework", "architecture", "pipeline", "workflow",
    "classification", "extraction", "detection", "prediction", "generation",
    "summarization", "parsing", "segmentation", "clustering", "ranking",
    "verification", "validation", "grounding", "indexing", "retrieval",
    # academic publishing signals
    "journal", "conference", "thesis", "dissertation", "preprint", "arxiv",
    "pubmed", "semantic scholar", "ieee", "acm", "springer", "elsevier",
    "peer-reviewed", "peer review", "doi", "orcid", "scopus", "citation",
    # agent task verbs relevant to research
    "extract", "classify", "identify", "detect", "generate", "summarize",
    "compare", "validate", "verify", "annotate", "analyse", "analyze",
    "evaluate", "rank", "score", "segment", "parse", "map", "align",
    "synthesise", "synthesize", "integrate", "aggregate", "transform",
    # research pipeline connections
    "feed", "input", "output", "pipeline", "workflow", "stage", "pre-process",
    "post-process", "upstream", "downstream", "agent", "module", "component",
}

# Signals that suggest the agent has nothing to do with research
NON_RESEARCH_SIGNALS = {
    "pizza", "food", "delivery", "restaurant", "recipe", "cooking", "menu",
    "shopping", "buy", "sell", "ecommerce", "retail", "fashion", "clothing",
    "game", "gaming", "entertainment", "movie", "music", "playlist", "stream",
    "social media", "instagram", "twitter", "tiktok", "likes", "followers",
    "travel", "booking", "flight", "hotel", "tourism", "vacation",
    "weather", "forecast", "sports", "betting", "lottery",
    "customer service", "support ticket", "chatbot", "faq",
}

# Which answer keys count most toward research relevance
ANSWER_WEIGHTS = {
    "specialisation":  3.0,   # most important
    "research_link":   2.5,   # second most important
    "tasks":           2.0,
    "quality_signal":  1.5,
    "output_desc":     1.0,
    "input_desc":      1.0,
    "domain_knowledge":1.0,
}

RESEARCH_SCORE_THRESHOLD   = 4.0   # weighted score must exceed this
REJECTION_SCORE_THRESHOLD  = 2.0   # non-research signals above this = reject


def validate_research_relevance(
    answers: dict,
    llm_fn: Optional[Callable] = None,
) -> dict:
    """
    Score the 7 free-form answers for research relevance.

    Returns a dict:
        valid        bool   — True if the agent should be built
        score        float  — weighted research relevance score
        reasons      list   — what matched (positives)
        rejections   list   — what didn't match (negatives)
        explanation  str    — human-readable verdict
        llm_verdict  str    — "yes"/"no"/"skip" from LLM double-check
    """
    reasons:    list[str] = []
    rejections: list[str] = []
    weighted_score  = 0.0
    non_res_score   = 0.0

    for key, weight in ANSWER_WEIGHTS.items():
        text   = answers.get(key, "").lower()
        tokens = set(re.findall(r"[a-z][a-z0-9\-]+", text))

        # positive matches
        matched = tokens & RESEARCH_VOCABULARY
        # also match multi-word phrases
        for phrase in RESEARCH_VOCABULARY:
            if " " in phrase and phrase in text:
                matched.add(phrase)

        if matched:
            weighted_score += weight * min(len(matched), 5) / 5
            reasons.append(
                f"{key}: matched '{', '.join(list(matched)[:3])}'"
            )

        # negative matches
        non_matched = tokens & NON_RESEARCH_SIGNALS
        for phrase in NON_RESEARCH_SIGNALS:
            if " " in phrase and phrase in text:
                non_matched.add(phrase)
        if non_matched:
            non_res_score += weight
            rejections.append(
                f"{key}: non-research terms '{', '.join(list(non_matched)[:3])}'"
            )

    # ── special check: research_link must mention pipeline/workflow concepts ──
    link_text = answers.get("research_link", "").lower()
    pipeline_words = {
        "agent", "pipeline", "workflow", "feed", "run", "before", "after",
        "loop", "connect", "output", "input", "process", "paper", "research",
        "vault", "query", "index", "loop", "learn",
    }
    if not any(w in link_text for w in pipeline_words):
        rejections.append(
            "research_link: does not describe a connection to a research pipeline"
        )
        weighted_score *= 0.5   # penalise heavily

    # ── LLM double-check on borderline cases ─────────────────────────────────
    llm_verdict = "skip"
    if llm_fn and REJECTION_SCORE_THRESHOLD <= weighted_score <= RESEARCH_SCORE_THRESHOLD + 1:
        prompt = (
            "You are a research system validator. A user wants to build a custom "
            "research agent. Read their answers and decide if this agent belongs "
            "in an academic research pipeline.\n\n"
            + "\n".join(f"{k}: {v}" for k, v in answers.items())
            + "\n\nReply with ONLY 'yes' or 'no'."
        )
        try:
            verdict = llm_fn(prompt).strip().lower()
            llm_verdict = "yes" if "yes" in verdict else "no"
            if llm_verdict == "yes":
                weighted_score = max(weighted_score, RESEARCH_SCORE_THRESHOLD + 0.1)
            else:
                weighted_score = min(weighted_score, RESEARCH_SCORE_THRESHOLD - 0.1)
        except Exception:
            pass

    # ── final decision ────────────────────────────────────────────────────────
    valid = (
        weighted_score >= RESEARCH_SCORE_THRESHOLD
        and non_res_score < REJECTION_SCORE_THRESHOLD
    )

    if valid:
        explanation = (
            f"Research relevance confirmed (score {weighted_score:.1f}). "
            f"Agent matches academic research patterns."
        )
    elif non_res_score >= REJECTION_SCORE_THRESHOLD:
        explanation = (
            f"Agent does not match research work.\n"
            f"  Non-research terms detected in: "
            f"{'; '.join(rejections[:3])}.\n"
            f"  VASIS is a research system — agents must connect to an academic "
            f"research pipeline (reading papers, analysing literature, generating "
            f"academic content, processing research data)."
        )
    else:
        explanation = (
            f"Agent description is too vague or generic to confirm research relevance "
            f"(score {weighted_score:.1f}, threshold {RESEARCH_SCORE_THRESHOLD}).\n"
            f"  Try being more specific about:\n"
            f"  — Which research domain it serves\n"
            f"  — How it connects to your research pipeline\n"
            f"  — What academic output it produces"
        )

    return {
        "valid":       valid,
        "score":       weighted_score,
        "reasons":     reasons,
        "rejections":  rejections,
        "explanation": explanation,
        "llm_verdict": llm_verdict,
    }


class AgentStudio:
    """
    The Custom Agent Studio.
    Handles /agent build, /my-agents, /my-loops, /connect, /delete.

    llm_fn: Callable(prompt: str) -> str
        Your existing LLM call (deepseek-llm:7b etc.)
        Used to generate system prompts for custom agents.
        If None, falls back to a rich template.

    print_fn: Callable(text: str, style: str)
        Your terminal print function.
        If None, uses plain print().
    """

    def __init__(
        self,
        llm_fn:   Optional[Callable] = None,
        print_fn: Optional[Callable] = None,
        store_path: Path = STORE_PATH,
    ):
        self.store    = CustomAgentStore(store_path)
        self.llm_fn   = llm_fn
        self._print   = print_fn or (lambda text, style="": print(text))

    # =========================================================================
    # /agent build <name>   (entry point for the /agent meta-command)
    # =========================================================================

    def cmd_agent(self, args: str, session=None):
        """
        Handles the /build agent meta-command (legacy: /agent build).
          /build agent references   → 7-step wizard
          /build agent              → asks for agent name interactively
        """
        parts = args.strip().split(maxsplit=1)
        sub   = parts[0].lower() if parts else ""
        rest  = parts[1].strip() if len(parts) > 1 else ""

        if sub == "build":
            return self.cmd_build(rest, session)
        else:
            self._p(
                "  Usage: /build agent <agentname>\n"
                "  Example: /build agent references",
                "error",
            )

    def cmd_build(self, args: str, session=None) -> Optional[CustomAgent]:
        """
        Smart wizard that adapts based on the agent name.

        Known agent names  → blueprint fires → 3 focused questions
        Unknown names      → LLM infers type → offer blueprint or skip to 7 questions
        """
        # ── get agent name ────────────────────────────────────────────────────
        command = args.strip().lstrip("/").strip()

        if not command:
            self._p(
                "\n  /agent build — Custom Agent Wizard\n"
                "  ─────────────────────────────────────\n"
                "  What do you want to call this agent?\n"
                "  Examples: references  abstract  introduction  methodology\n"
                "            gap_analysis  comparison  keywords  fact_check",
                "header",
            )
            command = self._ask("  > /", session).strip().lstrip("/")
            if not command:
                self._p("  Cancelled — no name given.", "muted")
                return None

        agent_id = _slugify(command)

        # ── protect fixed agents ──────────────────────────────────────────────
        if agent_id in FIXED_AGENTS:
            self._p(
                f"\n  ✗ '{command}' is one of the 14 fixed system agents.\n"
                f"  Fixed agents cannot be overridden. Choose a different name.",
                "error",
            )
            return None

        # ── check overwrite ───────────────────────────────────────────────────
        if self.store.agent_exists(agent_id):
            self._p(
                f"\n  ⚠  Agent /{agent_id} already exists. Overwrite it? (yes/no)",
                "warning",
            )
            if self._ask("  > ", session).strip().lower() not in ("yes", "y"):
                self._p("  Cancelled.", "muted")
                return None

        # ── blueprint detection ───────────────────────────────────────────────
        bp_key, blueprint = detect_blueprint(command, self.llm_fn)
        answers: dict = {}

        if blueprint:
            answers = self._run_blueprint_wizard(
                agent_id, command, bp_key, blueprint, session
            )
        else:
            answers = self._run_full_wizard(agent_id, command, session)

        if answers is None:
            return None

        # ── generate system prompt ────────────────────────────────────────────
        self._p("\n  Generating system prompt from your answers…", "muted")

        system_prompt = generate_system_prompt(
            agent_id        = agent_id,
            description     = answers["description"],
            research_domain = answers.get("research_domain", "General / Other"),
            input_type      = answers["input_type"],
            output_desc     = answers["output_desc"],
            citation_style  = answers.get("citation_style", "None"),
            quality_bar     = answers.get("quality_bar", ""),
            extra           = answers.get("extra", ""),
            llm_fn          = self.llm_fn,
        )

        # ── save ──────────────────────────────────────────────────────────────
        agent = CustomAgent(
            agent_id        = agent_id,
            command         = f"/{agent_id}",
            created         = time.time(),
            description     = answers["description"],
            research_domain = answers.get("research_domain", "General / Other"),
            input_type      = answers["input_type"],
            output_desc     = answers["output_desc"],
            citation_style  = answers.get("citation_style", "None"),
            quality_bar     = answers.get("quality_bar", ""),
            extra           = answers.get("extra", ""),
            system_prompt   = system_prompt,
        )
        self.store.save_agent(agent)

        self._p(
            f"\n  ✓  Agent /{agent_id} is ready.\n"
            f"  ─────────────────────────────────────\n"
            f"  Run it:          /{agent_id} \"your topic\"\n"
            f"  Add to a loop:   /connect … {agent_id}\n"
            f"  See all agents:  /my-agents\n"
            f"  Delete it:       /delete agent {agent_id}",
            "success",
        )
        return agent

    # =========================================================================
    # BLUEPRINT WIZARD  — 3 focused questions when a blueprint fires
    # =========================================================================

    def _run_blueprint_wizard(
        self, agent_id, command, bp_key, blueprint, session
    ) -> Optional[dict]:
        """
        Category-aware fast path: 3 focused questions, different per category.

        no_citation     → domain + target length/format + extra rules
        citation_section→ domain + citation style + extra rules
        references_section→ domain + citation style + formatting preferences
        analysis        → domain + citation style + coverage requirement
        extraction      → domain + output format + extra rules
        """
        category = BLUEPRINT_CATEGORY_MAP.get(bp_key, "citation_section")
        cq       = CATEGORY_QUESTIONS[category]

        self._p(
            f"\n  Building agent: /{agent_id}\n"
            f"  ─────────────────────────────────────\n"
            f"  Recognised as: {bp_key.upper()} agent  "
            f"[{blueprint['category']}]\n"
            f"  {blueprint['tip']}",
            "header",
        )
        self._p("\n  Smart defaults ready:", "section")

        rows = [
            ("Task",        blueprint["description"]),
            ("Input",       INPUT_LABELS.get(
                                blueprint["input_type"],
                                blueprint["input_type"])),
            ("Output",      blueprint["output_desc"]),
            ("Quality bar", blueprint["quality_bar"]),
        ]
        for label, value in rows:
            self._p(f"  {label:<14} {value[:70]}", "muted")

        self._p(
            "\n  [1] Use these defaults  "
            "(you set 3 final details below)\n"
            "  [2] Edit the defaults   "
            "(run open-ended wizard instead)",
            "hint",
        )
        choice = self._ask("  > ", session).strip()
        if choice == "2":
            return self._run_full_wizard(agent_id, command, session)

        # ── accepted defaults — start with pre-fills ─────────────────────────
        answers = {
            "description":   blueprint["description"],
            "input_type":    blueprint["input_type"],
            "output_desc":   blueprint["output_desc"],
            "quality_bar":   blueprint["quality_bar"],
            "citation_style":"None",
            "output_format": "",
            "extra":         "",
        }

        # ── Q1 of 3 — research domain (always asked) ─────────────────────────
        self._p(
            "\n  Step 1/3 — Which research field is this for?",
            "question",
        )
        self._p(
            "  [1] CS / AI   [2] Medicine / Biology   "
            "[3] Physics / Chemistry\n"
            "  [4] Social Sciences   [5] Engineering   "
            "[6] General / Other",
            "hint",
        )
        while True:
            raw    = self._ask("  > ", session).strip()
            domain = DOMAIN_OPTIONS.get(raw.lower(), DOMAIN_OPTIONS.get(raw))
            if domain:
                answers["research_domain"] = domain
                self._p(f"  ✓  {domain}\n", "success")
                break
            self._p("  Please enter 1–6.", "warning")

        # ── Q2 of 3 — varies by category ─────────────────────────────────────
        self._p(f"  Step 2/3 — {cq['q2_label']}", "question")

        if cq["q2_type"] == "citation_map":
            # Show domain-ordered citation styles
            ordered    = domain_ordered_citations(answers["research_domain"])
            style_map  = {str(i + 1): s for i, s in enumerate(ordered[:6])}
            for k, v in style_map.items():
                marker = " ← suggested" if k == "1" else ""
                self._p(f"  [{k}] {v}{marker}", "hint")
            self._p(
                "  (Ordered by relevance for "
                f"{answers['research_domain']})",
                "hint",
            )
            while True:
                raw   = self._ask("  > ", session).strip()
                style = style_map.get(raw) or \
                        CITATION_OPTIONS.get(raw.lower())
                if style:
                    answers["citation_style"] = style
                    self._p(f"  ✓  {style}\n", "success")
                    break
                self._p("  Please enter 1–6.", "warning")

        else:
            # Free-text answer (no_citation / extraction categories)
            self._p(cq["q2_hint"], "hint")
            while True:
                raw = self._ask("  > ", session).strip()
                if raw:
                    answers["output_format"] = raw
                    # fold into quality_bar so it reaches the system prompt
                    answers["quality_bar"] = (
                        answers["quality_bar"] + f"; {raw}"
                    ).strip("; ")
                    self._p("  ✓  Saved.\n", "success")
                    break
                self._p("  Please describe the format.", "warning")

        # ── Q3 of 3 — category-specific final question ────────────────────────
        self._p(f"  Step 3/3 — {cq['q3_label']}", "question")
        self._p(cq["q3_hint"], "hint")
        raw = self._ask("  > ", session).strip()
        if raw:
            answers["extra"] = raw
            self._p("  ✓  Saved.\n", "success")
        else:
            self._p("  Skipped.\n", "muted")

        return self._review_and_confirm(agent_id, answers, session)

    # =========================================================================
    # OPEN-ENDED WIZARD  — for unknown agents, no options, validation required
    # =========================================================================

    def _run_full_wizard(self, agent_id, command, session) -> Optional[dict]:
        """
        Open-ended 7-question wizard for agents not matching any blueprint.
        No numbered options — the user writes everything in their own words.
        Answers are validated for research relevance before building.
        """
        self._p(
            f"\n  Building agent: /{agent_id}\n"
            f"  ─────────────────────────────────────\n"
            f"  No recognised pattern for '{command}'.\n"
            f"  Answer 7 questions in your own words — no options, no shortcuts.\n"
            f"  The system will validate whether this agent belongs in a\n"
            f"  research pipeline before building it.\n",
            "header",
        )

        total   = len(UNKNOWN_AGENT_QUESTIONS)
        answers = {}

        for i, q in enumerate(UNKNOWN_AGENT_QUESTIONS):
            self._p(
                f"  ── Step {i+1}/{total}  —  {q['label']}",
                "question",
            )
            self._p(q["hint"], "hint")
            self._p("", "")

            while True:
                raw = self._ask("  > ", session).strip()
                if not raw:
                    self._p(
                        "  This question is required — please describe in your own words.",
                        "warning",
                    )
                    continue
                if len(raw.split()) < 4:
                    self._p(
                        "  Too short. Please write at least a sentence so the "
                        "system understands the agent clearly.",
                        "warning",
                    )
                    continue
                answers[q["key"]] = raw
                self._p(f"  ✓  Recorded.\n", "success")
                break

        # ── map to the CustomAgent field names expected by cmd_build ──────────
        # The open-ended answers need to be translated into the standard
        # agent fields (description, input_type, output_desc, etc.)
        # We derive them from the free-form answers.
        structured = self._structure_open_answers(answers, agent_id, session)
        if not structured:
            return None

        return structured

    def _structure_open_answers(
        self, raw_answers: dict, agent_id: str, session
    ) -> Optional[dict]:
        """
        Step 1: Validate research relevance.
        Step 2: Show the user a concise summary of what was understood.
        Step 3: Ask for domain + citation style (the two things free-form can't reliably capture).
        Step 4: Confirm and return structured answers.
        """
        # ── validation ────────────────────────────────────────────────────────
        self._p("\n  Validating research relevance…", "muted")
        result = validate_research_relevance(raw_answers, self.llm_fn)

        if not result["valid"]:
            self._p(
                f"\n  ✗  Agent not built.\n"
                f"  ─────────────────────────────────────\n"
                f"  {result['explanation']}",
                "error",
            )
            self._p(
                "\n  Examples of valid custom agents:\n"
                "  • drug_interaction_checker   — extracts drug-target pairs from biomedical papers\n"
                "  • causal_chain_mapper        — maps cause-effect relationships in social science data\n"
                "  • replication_checker        — verifies whether study results can be reproduced\n"
                "  • protocol_extractor         — pulls experimental protocols from methods sections\n"
                "  • hypothesis_ranker          — ranks competing hypotheses by evidence strength\n"
                "\n  Type /agent build <name> to try again with a different description.",
                "muted",
            )
            return None

        self._p(
            f"  ✓  Research relevance confirmed  "
            f"(score {result['score']:.1f})\n",
            "success",
        )

        # ── show what was understood ──────────────────────────────────────────
        self._p(
            "  What the system understood about your agent:\n"
            "  ─────────────────────────────────────",
            "section",
        )
        understand_rows = [
            ("Specialisation", raw_answers.get("specialisation", "")[:72]),
            ("Input",          raw_answers.get("input_desc",     "")[:72]),
            ("Output",         raw_answers.get("output_desc",    "")[:72]),
            ("Tasks",          raw_answers.get("tasks",          "")[:72]),
            ("Research link",  raw_answers.get("research_link",  "")[:72]),
            ("Quality signal", raw_answers.get("quality_signal", "")[:72]),
            ("Domain knowl.",  raw_answers.get("domain_knowledge","")[:72]),
        ]
        for label, value in understand_rows:
            self._p(f"  {label:<18} {value}", "muted")

        # ── two follow-up questions the open wizard can't infer ───────────────
        self._p(
            "\n  Two final questions to complete the agent.\n",
            "section",
        )

        # Research domain (needed for citation ordering)
        self._p(
            "  Which research domain best describes this agent?\n"
            "  [1] CS / AI   [2] Medicine / Biology   [3] Physics / Chemistry\n"
            "  [4] Social Sciences   [5] Engineering   [6] General / Other",
            "question",
        )
        domain = "General / Other"
        while True:
            raw = self._ask("  > ", session).strip()
            mapped = DOMAIN_OPTIONS.get(raw.lower(), DOMAIN_OPTIONS.get(raw))
            if mapped:
                domain = mapped
                self._p(f"  ✓  {domain}\n", "success")
                break
            self._p("  Please enter 1–6.", "warning")

        # Citation style (domain-ordered)
        self._p("  Citation style?", "question")
        ordered = domain_ordered_citations(domain)
        style_map = {str(i+1): s for i, s in enumerate(ordered[:6])}
        for k, v in style_map.items():
            marker = " ← suggested" if k == "1" else ""
            self._p(f"  [{k}] {v}{marker}", "hint")

        citation = "APA 7th"
        while True:
            raw = self._ask("  > ", session).strip()
            style = style_map.get(raw) or CITATION_OPTIONS.get(raw.lower())
            if style:
                citation = style
                self._p(f"  ✓  {citation}\n", "success")
                break
            self._p("  Please enter 1–6.", "warning")

        # ── build the structured answers dict expected by cmd_build ───────────
        structured = {
            # core fields
            "description":     raw_answers["specialisation"],
            "input_type":      "all",       # open answers don't constrain this
            "output_desc":     raw_answers["output_desc"],
            "quality_bar":     raw_answers["quality_signal"],
            "research_domain": domain,
            "citation_style":  citation,
            "extra":           raw_answers.get("domain_knowledge", ""),
            # store the full open answers in metadata for the system prompt
            "_open_specialisation": raw_answers["specialisation"],
            "_open_tasks":          raw_answers["tasks"],
            "_open_research_link":  raw_answers["research_link"],
            "_open_input_desc":     raw_answers["input_desc"],
        }

        # confirm
        return self._review_and_confirm(agent_id, structured, session)


    # =========================================================================
    # REVIEW + CONFIRM
    # =========================================================================

    def _review_and_confirm(self, agent_id: str, answers: dict, session) -> Optional[dict]:
        """Show all answers, let user confirm, edit, or cancel."""
        self._p(
            f"\n  Review — /{agent_id}\n"
            f"  ─────────────────────────────────────",
            "section",
        )
        rows = [
            ("Task",           answers.get("description", "")),
            ("Domain",         answers.get("research_domain", "")),
            ("Input",          INPUT_LABELS.get(answers.get("input_type", ""), answers.get("input_type", ""))),
            ("Output",         answers.get("output_desc", "")),
            ("Citation style", answers.get("citation_style", "None")),
            ("Quality bar",    answers.get("quality_bar", "")),
            ("Extra rules",    answers.get("extra", "") or "—"),
        ]
        for label, value in rows:
            self._p(f"  {label:<18} {value[:72]}", "muted")

        self._p(
            "\n  Save? (yes / no / edit)",
            "question",
        )
        confirm = self._ask("  > ", session).strip().lower()

        if confirm in ("edit", "e"):
            self._p("\n  Restarting full wizard…\n", "muted")
            return self._run_full_wizard(agent_id, "", session)

        if confirm not in ("yes", "y"):
            self._p("  Cancelled — agent not saved.", "muted")
            return None

        return answers

    # =========================================================================
    # /my-agents
    # =========================================================================

    def cmd_list_agents(self):
        """Show all fixed agents + all custom agents."""
        fixed = [
            ("01", "Router"),       ("02", "Decomposer"),  ("03", "Navigator"),
            ("04", "Retrieval"),    ("05", "Expansion"),   ("06", "Validation"),
            ("07", "Contradiction"),("08", "Temporal"),    ("09", "Calibration"),
            ("10", "Supervisor"),   ("11", "Synthesis"),   ("12", "Web Search"),
            ("13", "Paper Writer"), ("14", "Impl. Guide"),
        ]

        self._p("\n  FIXED AGENTS  (coded — cannot be deleted)", "section")
        for num, name in fixed:
            self._p(f"  {num}  {name}", "fixed")

        custom = self.store.all_agents()
        self._p(f"\n  YOUR AGENTS  ({len(custom)} custom)", "section")

        if not custom:
            self._p("  None yet.  Use /agent build <name> to create one.", "muted")
        else:
            for a in sorted(custom, key=lambda x: x.created):
                self._p(
                    f"  ✦ /{a.agent_id:<18} "
                    f"{a.description[:50]}  ·  {INPUT_LABELS.get(a.input_type, a.input_type)}  "
                    f"·  used {a.run_count}×",
                    "custom",
                )
        self._p("", "")

    # =========================================================================
    # /connect
    # =========================================================================

    def cmd_connect(self, args: str) -> Optional[CustomLoop]:
        """
        /connect abstract introduction references --name paper-sections --quality

        Wires a list of custom agents into a named loop.
        Agents can be custom or fixed (by name).
        """
        parts     = args.strip().split()
        loop_name = None
        quality   = False
        agent_names: list[str] = []

        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "--name" and i + 1 < len(parts):
                loop_name = _slugify(parts[i + 1])
                i += 2
            elif p in ("--quality", "--q"):
                quality = True
                i += 1
            else:
                agent_names.append(p.lstrip("/"))
                i += 1

        if len(agent_names) < 2:
            self._p("  Usage: /connect agent1 agent2 agent3 --name myloop", "error")
            return None

        # validate all agents exist (custom or fixed by name)
        for name in agent_names:
            slug = _slugify(name)
            if slug not in FIXED_AGENTS and not self.store.agent_exists(slug):
                self._p(f"  ✗ Agent '{name}' not found. Build it first: /agent build {name}", "error")
                return None

        # auto-name if not given
        if not loop_name:
            loop_name = "-".join(a[:6] for a in agent_names[:3])

        # check if loop name already taken
        if self.store.get_loop(loop_name):
            self._p(f"  ✗ Loop '{loop_name}' already exists. Choose another --name.", "error")
            return None

        # show suggested order
        self._p(
            f"\n  Loop: {loop_name}\n"
            f"  Order: {' → '.join('/' + a for a in agent_names)}\n"
            f"  Quality gate: {'yes' if quality else 'no'}",
            "info",
        )

        loop = CustomLoop(
            loop_id      = loop_name,
            command      = loop_name,
            created      = time.time(),
            agent_ids    = [_slugify(a) for a in agent_names],
            quality_gate = quality,
        )
        self.store.save_loop(loop)

        self._p(f"\n  ✓ Loop '{loop_name}' created.", "success")
        self._p(
            f"  Run:    /loop {loop_name} \"your topic\"\n"
            f"  Delete: /delete loop {loop_name}",
            "muted",
        )
        return loop

    # =========================================================================
    # /my-loops
    # =========================================================================

    def cmd_list_loops(self):
        """List all custom loops."""
        loops = self.store.all_loops()

        self._p(f"\n  YOUR LOOPS  ({len(loops)} custom)", "section")
        if not loops:
            self._p("  None yet.  Use /connect to create one.", "muted")
        else:
            for lp in sorted(loops, key=lambda x: x.created):
                agents_str = " → ".join(f"/{a}" for a in lp.agent_ids)
                self._p(
                    f"  ◈ {lp.loop_id:<20} "
                    f"{agents_str}  "
                    f"{'· quality gate' if lp.quality_gate else ''}  "
                    f"· used {lp.run_count}×",
                    "custom",
                )
        self._p("", "")

    # =========================================================================
    # /delete agent <name>  or  /delete loop <name>
    # =========================================================================

    def cmd_delete(self, args: str, session=None) -> bool:
        """
        /delete agent references   — delete a custom agent
        /delete loop paper-sections — delete a custom loop
        """
        parts = args.strip().split(maxsplit=1)
        if len(parts) < 2:
            self._p("  Usage: /delete agent <name>  or  /delete loop <name>", "error")
            return False

        kind = parts[0].lower()
        name = _slugify(parts[1].lstrip("/"))

        if kind == "agent":
            return self._delete_agent(name, session)
        elif kind in ("loop", "loops"):
            return self._delete_loop(name, session)
        else:
            self._p(f"  ✗ Unknown type '{kind}'. Use 'agent' or 'loop'.", "error")
            return False

    def _delete_agent(self, agent_id: str, session) -> bool:
        # block fixed agents
        if agent_id in FIXED_AGENTS:
            self._p(
                f"  ✗ /{agent_id} is a fixed system agent and cannot be deleted.",
                "error",
            )
            return False

        if not self.store.agent_exists(agent_id):
            self._p(f"  ✗ Agent /{agent_id} not found.", "error")
            return False

        # check if used in any loops
        loops_using = self.store.loops_using_agent(agent_id)
        if loops_using:
            self._p(
                f"  ⚠  /{agent_id} is used in loop(s): {', '.join(loops_using)}.\n"
                f"  Deleting it will also remove those loops.",
                "warning",
            )
            confirm = self._ask("  Confirm? (yes/no): ", session).strip().lower()
            if confirm not in ("yes", "y"):
                self._p("  Cancelled.", "muted")
                return False

            # cascade delete loops
            for lid in loops_using:
                self.store.delete_loop(lid)
                self._p(f"  ✓ Loop '{lid}' deleted.", "success")

        self.store.delete_agent(agent_id)
        self._p(f"  ✓ Agent /{agent_id} deleted.", "success")
        self._p("  Fixed agents 01–14 are untouched.", "muted")
        return True

    def _delete_loop(self, loop_id: str, session) -> bool:
        if not self.store.get_loop(loop_id):
            self._p(f"  ✗ Loop '{loop_id}' not found.", "error")
            return False

        self.store.delete_loop(loop_id)
        self._p(f"  ✓ Loop '{loop_id}' deleted.", "success")
        self._p("  Agents in this loop are not deleted.", "muted")
        return True

    # =========================================================================
    # Run a custom agent
    # =========================================================================

    def cmd_run_agent(
        self,
        command:    str,
        topic:      str,
        paper_text: str = "",
        atoms:      list = None,
        web_results:list = None,
    ) -> dict:
        """
        Run a single custom agent by its command name.

        Returns: { output_text, agent_id, elapsed_s }
        """
        agent_id = _slugify(command.lstrip("/"))
        agent = self.store.get_agent(agent_id)

        if not agent:
            self._p(f"  ✗ Agent /{agent_id} not found.", "error")
            return {}

        self._p(f"  ◆ Running /{agent_id}…", "running")
        t0 = time.time()

        # assemble context based on input_type
        context = _build_context(
            agent.input_type,
            topic       = topic,
            paper_text  = paper_text,
            atoms       = atoms or [],
            web_results = web_results or [],
        )

        user_prompt = (
            f"Topic: {topic}\n\n"
            f"Context:\n{context}\n\n"
            f"Your task: {agent.description}\n"
            f"Output: {agent.output_desc}"
        )
        if agent_id == "references":
            user_prompt += "\nNote: Be exhaustive. Extract and list all unique references from the context. Do not truncate, summarize, or omit items."

        output_text = ""
        if self.llm_fn:
            try:
                output_text = self.llm_fn(
                    agent.system_prompt + "\n\n" + user_prompt
                )
            except Exception as e:
                self._p(f"  ✗ LLM error: {e}", "error")
                return {}
        else:
            output_text = f"[/{agent_id} stub — wire llm_fn to get real output]"

        elapsed = time.time() - t0

        # update stats
        agent.run_count += 1
        agent.last_run   = time.time()
        self.store.save_agent(agent)

        self._p(
            f"  ✓ /{agent_id}  {elapsed:.1f}s  "
            f"{len(output_text.split())} words",
            "success",
        )
        return {
            "output_text": output_text,
            "agent_id":    agent_id,
            "elapsed_s":   elapsed,
        }

    # =========================================================================
    # Run a custom loop
    # =========================================================================

    def cmd_run_loop(
        self,
        loop_id:    str,
        topic:      str,
        paper_text: str = "",
        atoms:      list = None,
        web_results:list = None,
        fixed_agent_runner: Optional[Callable] = None,
    ) -> dict:
        """
        Run a custom loop sequentially.
        Output of each agent is appended to paper_text for the next.

        fixed_agent_runner(agent_id, topic, paper_text) -> str
            Optional: provide this to run fixed agents (01–14) within
            a custom loop.
        """
        loop = self.store.get_loop(loop_id)
        if not loop:
            self._p(f"  ✗ Loop '{loop_id}' not found.", "error")
            return {}

        self._p(
            f"\n  Running custom loop: {loop_id}\n"
            f"  {len(loop.agent_ids)} agent(s)  "
            f"{'· quality gate on' if loop.quality_gate else ''}",
            "info",
        )

        accumulated_text = paper_text
        all_results = {}
        t_total = time.time()

        for agent_id in loop.agent_ids:
            if agent_id in FIXED_AGENTS and fixed_agent_runner:
                # run a fixed agent via the provided runner
                try:
                    output = fixed_agent_runner(agent_id, topic, accumulated_text)
                    accumulated_text += "\n\n" + output
                    all_results[agent_id] = output
                    self._p(f"  ✓ {agent_id} (fixed)", "success")
                except Exception as e:
                    self._p(f"  ✗ Fixed agent {agent_id} failed: {e}", "error")
            else:
                # run a custom agent
                result = self.cmd_run_agent(
                    command     = agent_id,
                    topic       = topic,
                    paper_text  = accumulated_text,
                    atoms       = atoms,
                    web_results = web_results,
                )
                if result.get("output_text"):
                    accumulated_text += "\n\n" + result["output_text"]
                    all_results[agent_id] = result["output_text"]

                    # quality gate: retry once if output is very short
                    if loop.quality_gate and len(result["output_text"].split()) < 50:
                        self._p(
                            f"  ⚠ /{agent_id} output short — retrying once",
                            "warning",
                        )
                        retry = self.cmd_run_agent(
                            command     = agent_id,
                            topic       = topic,
                            paper_text  = accumulated_text,
                            atoms       = atoms,
                            web_results = web_results,
                        )
                        if retry.get("output_text"):
                            accumulated_text = accumulated_text.rsplit("\n\n", 1)[0]
                            accumulated_text += "\n\n" + retry["output_text"]
                            all_results[agent_id] = retry["output_text"]

        # update loop stats
        loop.run_count += 1
        loop.last_run   = time.time()
        self.store.save_loop(loop)

        elapsed = time.time() - t_total
        self._p(f"\n  ✓ Loop complete  ·  {elapsed:.1f}s total", "success")

        return {
            "output_text":  accumulated_text,
            "agent_results":all_results,
            "loop_id":      loop_id,
            "elapsed_s":    elapsed,
        }

    # =========================================================================
    # Helpers for the CLI dispatcher
    # =========================================================================

    def custom_command_names(self) -> set[str]:
        """Return the set of custom agent command names (without /)."""
        return {a.agent_id for a in self.store.all_agents()}

    def custom_loop_names(self) -> set[str]:
        """Return the set of custom loop IDs."""
        return {lp.loop_id for lp in self.store.all_loops()}

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _p(self, text: str, style: str = ""):
        self._print(text, style)

    def _ask(self, prompt: str, session) -> str:
        if session:
            try:
                return session.prompt(prompt)
            except (KeyboardInterrupt, EOFError):
                return ""
        return input(prompt)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _slugify(text: str) -> str:
    """'Paper Writer' → 'paperwriter', '/references' → 'references'"""
    return re.sub(r"[^a-z0-9]+", "", text.lower().strip())


def _build_context(
    input_type:  str,
    topic:       str,
    paper_text:  str,
    atoms:       list,
    web_results: list,
) -> str:
    """
    Assemble context string from all available sources.

    The input_type hint is used for ordering priority, but any context
    that is actually populated is always included so standalone custom
    agents receive maximum grounding.
    """
    parts = []

    # Paper text (first 4000 chars)
    if paper_text:
        parts.append(f"PAPER TEXT:\n{paper_text[:4000]}")

    # Vault atoms — up to 60, ranked by BM25 relevance if caller sorted them
    if atoms:
        atom_text = "\n\n".join(
            a.get('text', '').strip()
            for a in atoms[:60]
        )
        parts.append(f"VAULT ATOMS:\n{atom_text}")

    # Web search results — up to 8
    if web_results:
        web_text = "\n".join(
            f"[W{i+1}] {r.get('title','')} — {r.get('snippet','')[:200]}"
            for i, r in enumerate(web_results[:8])
        )
        parts.append(f"WEB SOURCES:\n{web_text}")

    return "\n\n".join(parts) if parts else f"Topic: {topic}"


# =============================================================================
# CLI PRINT FUNCTION  — wire to Rich in VasisCLI
# =============================================================================

def rich_print_fn(console):
    """
    Returns a print function compatible with AgentStudio,
    using Rich colours.  Pass to AgentStudio(print_fn=...).

    Usage:
        from rich.console import Console
        console = Console()
        studio = AgentStudio(print_fn=rich_print_fn(console), llm_fn=your_llm)
    """
    from rich.text import Text

    STYLES = {
        "header":   "bold #A78BFA",
        "section":  "#6B7280",
        "success":  "#10B981",
        "error":    "bold #EF4444",
        "warning":  "#F59E0B",
        "muted":    "#6B7280",
        "info":     "#60A5FA",
        "fixed":    "#374151",
        "custom":   "#A78BFA",
        "running":  "bold #7C3AED",
        "question": "bold #E5E7EB",
        "hint":     "#4b5563",
        "":         "#E5E7EB",
    }

    def _print(text: str, style: str = ""):
        console.print(Text(text, style=STYLES.get(style, "#E5E7EB")))

    return _print


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    import tempfile, shutil

    tmp = Path(tempfile.mkdtemp()) / "test_studio.json"
    logs = []

    studio = AgentStudio(
        print_fn   = lambda t, s="": logs.append(t),
        store_path = tmp,
    )

    # ── test detect_blueprint ─────────────────────────────────────────────
    key,  bp  = detect_blueprint("references")
    key2, bp2 = detect_blueprint("ref")
    key3, bp3 = detect_blueprint("bibliography")
    key4, bp4 = detect_blueprint("drug_interaction_checker")
    assert key  == "references", f"exact match failed: {key}"
    assert key2 == "references", f"alias match failed: {key2}"
    assert key3 == "references", f"keyword fuzzy match failed: {key3}"
    assert key4 is None,         f"unknown should return None: {key4}"
    print("[test] detect_blueprint  PASS")

    # ── test blueprint fast path — all 5 categories ──────────────────────

    # references_section: domain + citation + formatting pref
    bp_ref = iter(["1","1","1","alphabetical order with DOI","yes"])
    studio._ask = lambda p, s: next(bp_ref)
    a_ref = studio.cmd_agent("build references")
    assert a_ref is not None,               "references failed"
    assert a_ref.citation_style == "IEEE",  f"cite: {a_ref.citation_style}"
    assert "alphabetical" in a_ref.extra,   "formatting pref not saved"
    print("[test] blueprint references_section  PASS")

    # no_citation: domain + length/format + extra
    bp_abs = iter(["1","1","150-250 words","no citations allowed","yes"])
    studio._ask = lambda p, s: next(bp_abs)
    a_abs = studio.cmd_agent("build abstract")
    assert a_abs is not None,                   "abstract failed"
    assert "150-250" in a_abs.quality_bar,      "length not in quality_bar"
    assert a_abs.citation_style == "None",      "abstract should have None citation"
    print("[test] blueprint no_citation (abstract)  PASS")

    # citation_section: domain + citation + extra
    bp_intro = iter(["1","1","1","must end with research gap","yes"])
    studio._ask = lambda p, s: next(bp_intro)
    a_intro = studio.cmd_agent("build introduction")
    assert a_intro is not None,                 "introduction failed"
    assert a_intro.citation_style == "IEEE",    f"cite: {a_intro.citation_style}"
    print("[test] blueprint citation_section (introduction)  PASS")

    # analysis: domain + citation + coverage req
    bp_gap = iter(["1","1","1","minimum 5 sources per gap","yes"])
    studio._ask = lambda p, s: next(bp_gap)
    a_gap = studio.cmd_agent("build gapanalysis")
    assert a_gap is not None,               "gapanalysis failed"
    assert "minimum 5" in a_gap.extra,      "coverage req not saved"
    print("[test] blueprint analysis (gapanalysis)  PASS")

    # extraction: domain + output format + extra
    bp_kw = iter(["1","1","bulleted list of 8-12 terms","only verbatim terms","yes"])
    studio._ask = lambda p, s: next(bp_kw)
    a_kw = studio.cmd_agent("build keywords")
    assert a_kw is not None,               "keywords failed"
    assert "8-12" in a_kw.quality_bar,     "format not in quality_bar"
    print("[test] blueprint extraction (keywords)  PASS")


    # ── test research validator — valid ───────────────────────────────────
    valid_answers = {
        "specialisation":   "extract drug-target interaction pairs from biomedical research abstracts",
        "input_desc":       "raw clinical trial PDF reports and PubMed abstracts",
        "output_desc":      "structured list of drug-target pairs with confidence scores and evidence sentences",
        "tasks":            "1. reads abstract  2. identifies drug names  3. maps to biological targets  4. classifies interaction  5. outputs structured data",
        "research_link":    "its output feeds into my gap_analysis agent as structured evidence for the literature review pipeline",
        "quality_signal":   "every pair must include confidence score above 0.7 and a source sentence from the corpus",
        "domain_knowledge": "UMLS medical terminology ICD-10 codes MeSH headings biomedical NER",
    }
    rv = validate_research_relevance(valid_answers)
    assert rv["valid"], f"valid agent rejected: {rv['explanation']}"
    print(f"[test] validator PASS (valid, score={rv['score']:.1f})")

    # ── test research validator — invalid ─────────────────────────────────
    invalid_answers = {
        "specialisation":   "order pizza from nearby restaurants based on user food preferences",
        "input_desc":       "user food preferences and delivery address",
        "output_desc":      "pizza order confirmation with estimated delivery time",
        "tasks":            "1. gets location  2. finds restaurant  3. shows menu  4. places food order",
        "research_link":    "I use it when I am hungry during work",
        "quality_signal":   "pizza arrives hot and in under 30 minutes",
        "domain_knowledge": "food delivery apps and restaurant menus",
    }
    ri = validate_research_relevance(invalid_answers)
    assert not ri["valid"], f"invalid agent accepted: {ri['explanation']}"
    print(f"[test] validator PASS (invalid rejected, score={ri['score']:.1f})")

    # ── test open-ended wizard — valid agent ──────────────────────────────
    open_valid = iter([
        "extract causal relationships between variables in social science research papers",
        "a corpus of social science research papers in PDF format from JSTOR and SSRN",
        "a structured table of causal claims with source sentence confidence score and variable names",
        "1. reads paper  2. segments sections  3. identifies causal language  4. extracts variable pairs  5. scores confidence  6. outputs structured table",
        "its output feeds into my comparison agent as pre-processed causal evidence for the literature review pipeline",
        "every causal claim includes a confidence score source sentence and both cause and effect variable names clearly identified",
        "APA citation style grounded theory methodology social science causal inference terminology",
        "4",    # Social Sciences
        "1",    # APA 7th
        "yes",  # confirm
    ])
    studio._ask = lambda p, s: next(open_valid)
    agent3 = studio.cmd_agent("build causal_extractor")
    assert agent3 is not None,                          "open valid build failed"
    assert agent3.research_domain == "Social Sciences", f"domain: {agent3.research_domain}"
    print("[test] open wizard (valid)  PASS")

    # ── test open-ended wizard — invalid agent rejected ───────────────────
    open_invalid = iter([
        "help users find and order pizza from nearby restaurants",
        "user location and food preferences from their phone",
        "pizza order with delivery time estimate and restaurant menu",
        "1. gets location  2. finds restaurants  3. shows food menu  4. places pizza order  5. tracks delivery",
        "I use it when I am hungry during my lunch break at work",
        "pizza arrives hot and on time within 30 minutes",
        "knowledge of local restaurant menus and food delivery app integrations",
    ])
    studio._ask = lambda p, s: next(open_invalid)
    agent4 = studio.cmd_agent("build pizza_finder")
    assert agent4 is None, "non-research agent should have been rejected"
    print("[test] open wizard (invalid → rejected)  PASS")

    # ── test domain_ordered_citations ─────────────────────────────────────
    assert domain_ordered_citations("CS / AI")[0]           == "IEEE"
    assert domain_ordered_citations("Medicine / Biology")[0] == "Vancouver"
    assert domain_ordered_citations("Social Sciences")[0]    == "APA 7th"
    print("[test] domain_ordered_citations  PASS")

    # ── test /my-agents ────────────────────────────────────────────────────
    studio.cmd_list_agents()
    print("[test] /my-agents  PASS")

    # ── test /connect ──────────────────────────────────────────────────────
    # build methodology to have 3 agents ready
    bp_meth = iter(["1","1","1","","yes"])
    studio._ask = lambda p, s: next(bp_meth)
    studio.cmd_agent("build methodology")

    loop = studio.cmd_connect(
        "abstract introduction methodology --name paper-core"
    )
    assert loop is not None, "connect failed"
    print("[test] /connect  PASS")

    # ── test alias recognition across venues ───────────────────────────────
    test_aliases = [
        ("bibliography",     "references"),
        ("workscited",       "references"),
        ("relatedwork",      "literaturereview"),
        ("materialsandmethods", "methodology"),
        ("findings",         "results"),
        ("indexterms",       "keywords"),
        ("executivesummary", "abstract"),
        ("concludingremarks","conclusion"),
        ("stateoftheart",    "literaturereview"),
        ("patientsandmethods","methodology"),
    ]
    for alias, expected_key in test_aliases:
        key, _ = detect_blueprint(alias)
        assert key == expected_key, \
            f"alias '{alias}' → got '{key}', expected '{expected_key}'"
    print(f"[test] alias recognition ({len(test_aliases)} venue variants)  PASS")

    studio._ask = lambda p, s: "yes"
    ok = studio.cmd_delete("agent abstract")
    assert ok
    assert not studio.store.agent_exists("abstract")
    assert not studio.store.get_loop("paper-core")
    print("[test] /delete + cascade  PASS")

    shutil.rmtree(tmp.parent)
    print("\nAll tests PASS")
