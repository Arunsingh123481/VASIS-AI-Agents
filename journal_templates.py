# journal_templates.py
# Journal/Conference venue configs and article type definitions for Agent 13
# Used by agents/agent13_paper_writer.py

# ── JOURNAL / CONFERENCE TEMPLATES ────────────────────────────────────────────

JOURNAL_TEMPLATES = {

    "DSJ": {
        "full_name": "Decision Sciences Journal",
        "reference_style": "APA",
        "sections_required": [
            "Abstract", "Keywords", "Introduction",
            "Literature Review", "Methodology",
            "Results", "Discussion", "Conclusion",
            "References"
        ],
        "abstract_limit": 200,
        "max_keywords": 6,
    },

    "IEEE": {
        "full_name": "IEEE Transactions",
        "reference_style": "IEEE (numbered)",
        "sections_required": [
            "Abstract", "Index Terms", "Introduction",
            "Related Work", "Proposed Method",
            "Experiments", "Results and Discussion",
            "Conclusion", "References"
        ],
        "abstract_limit": 200,
        "max_keywords": 8,
    },

    "Elsevier": {
        "full_name": "Elsevier Journal",
        "reference_style": "Vancouver (numbered)",
        "sections_required": [
            "Abstract", "Keywords", "Nomenclature",
            "Introduction", "Material and Methods",
            "Results", "Discussion", "Conclusion",
            "Acknowledgement", "References"
        ],
        "abstract_limit": 200,
        "max_keywords": 6,
    },

    "Springer": {
        "full_name": "Springer Nature",
        "reference_style": "Vancouver (numbered)",
        "sections_required": [
            "Abstract", "Keywords", "Introduction",
            "Methods", "Results", "Discussion",
            "Conclusion", "References"
        ],
        "abstract_limit": 250,
        "max_keywords": 6,
    },

    "ACM": {
        "full_name": "ACM Computing Surveys / SIGCHI",
        "reference_style": "ACM",
        "sections_required": [
            "Abstract", "CCS Concepts", "Keywords",
            "Introduction", "Background",
            "Methodology", "Evaluation",
            "Discussion", "Conclusion", "References"
        ],
        "abstract_limit": 200,
        "max_keywords": 5,
    },

    "NeurIPS": {
        "full_name": "NeurIPS Conference",
        "reference_style": "NeurIPS (author-year)",
        "sections_required": [
            "Abstract", "Introduction",
            "Related Work", "Preliminaries",
            "Method", "Experiments",
            "Analysis and Discussion",
            "Conclusion", "References"
        ],
        "abstract_limit": 200,
        "max_keywords": 0,
        "is_conference": True,
        "page_limit": 8,
    },

    "ICML": {
        "full_name": "ICML Conference",
        "reference_style": "ICML (author-year)",
        "sections_required": [
            "Abstract", "Introduction",
            "Related Work", "Method",
            "Experiments", "Conclusion",
            "References"
        ],
        "abstract_limit": 200,
        "max_keywords": 0,
        "is_conference": True,
        "page_limit": 8,
    },

    "ICLR": {
        "full_name": "ICLR Conference",
        "reference_style": "ICLR (author-year)",
        "sections_required": [
            "Abstract", "Introduction",
            "Background", "Method",
            "Experiments", "Conclusion",
            "References"
        ],
        "abstract_limit": 200,
        "max_keywords": 0,
        "is_conference": True,
        "page_limit": 8,
    },
}


# ── ARTICLE TYPE CONFIGURATIONS ───────────────────────────────────────────────

ARTICLE_TYPE_CONFIGS = {

    "research_article": {
        "label":         "Research Article",
        "word_limit":    4000,
        "abstract_words": "150-200",
        "sections": [
            "Abstract", "Keywords", "Nomenclature",
            "1. Introduction",
            "2. Material and Methods",
            "3. Results",
            "4. Discussion",
            "5. Conclusion",
            "References", "Acknowledgement"
        ],
        "has_methods":      True,
        "has_results":      True,
        "has_discussion":   True,
        "has_taxonomy":     False,
        "has_prisma":       False,
        "writing_style":    "objective experimental",
        "citation_density": "high",
        "system_prompt_addon": (
            "This is an original research paper. "
            "Report new experimental findings. "
            "Include specific quantitative results. "
            "Be precise about methodology. "
        )
    },

    "review_article": {
        "label":         "Review Article",
        "word_limit":    6000,
        "abstract_words": "200-250",
        "sections": [
            "Abstract", "Keywords",
            "1. Introduction",
            "2. Background and Taxonomy",
            "3. [Theme 1]",
            "4. [Theme 2]",
            "5. [Theme 3]",
            "6. Open Challenges",
            "7. Future Directions",
            "8. Conclusion",
            "References"
        ],
        "has_methods":      False,
        "has_results":      False,
        "has_discussion":   True,
        "has_taxonomy":     True,
        "has_prisma":       False,
        "writing_style":    "analytical synthesising",
        "citation_density": "very high (30-60 refs)",
        "system_prompt_addon": (
            "This is a review article synthesising "
            "existing literature. Do not present new "
            "experimental results. Instead: critically "
            "analyse, compare, and synthesise findings "
            "from multiple papers. Organise thematically. "
            "Identify trends, consensus, and conflicts. "
        )
    },

    "short_communication": {
        "label":         "Short Communication",
        "word_limit":    2000,
        "abstract_words": "80-100",
        "sections": [
            "Abstract", "Keywords",
            "1. Introduction",
            "2. Methods and Results",
            "3. Discussion and Conclusion",
            "References"
        ],
        "has_methods":      True,
        "has_results":      True,
        "has_discussion":   True,
        "has_taxonomy":     False,
        "has_prisma":       False,
        "writing_style":    "concise focused",
        "citation_density": "low (5-10 refs)",
        "system_prompt_addon": (
            "This is a short communication — "
            "report ONE key finding concisely. "
            "Combine methods and results. "
            "No lengthy introduction. Maximum brevity. "
            "Every sentence must add value. "
        )
    },

    "letter_to_editor": {
        "label":         "Letter to Editor",
        "word_limit":    1200,
        "abstract_words": "none",
        "sections": [
            "Opening statement",
            "Key argument",
            "Supporting evidence",
            "Conclusion",
            "References (3-5 max)"
        ],
        "has_methods":      False,
        "has_results":      False,
        "has_discussion":   False,
        "has_taxonomy":     False,
        "has_prisma":       False,
        "writing_style":    "direct persuasive",
        "citation_density": "very low (3-5 refs)",
        "system_prompt_addon": (
            "This is a letter to the editor. "
            "State one clear position. "
            "Support with brief evidence. "
            "Be direct and persuasive. "
            "No sections or headers. "
            "Addressed to 'Dear Editor'. "
        )
    },

    "systematic_review": {
        "label":         "Systematic Review",
        "word_limit":    10000,
        "abstract_words": "250-300",
        "sections": [
            "Structured Abstract",
            "1. Introduction",
            "2. Methods",
            "  2.1 Search Strategy",
            "  2.2 Inclusion/Exclusion Criteria",
            "  2.3 Quality Assessment",
            "  2.4 Data Extraction",
            "3. Results",
            "  3.1 Search Results (PRISMA flow)",
            "  3.2 Study Characteristics",
            "  3.3 Quality Assessment",
            "  3.4 Synthesis",
            "4. Discussion",
            "5. Conclusion",
            "References (40-80)"
        ],
        "has_methods":      True,
        "has_results":      True,
        "has_discussion":   True,
        "has_taxonomy":     False,
        "has_prisma":       True,
        "writing_style":    "rigorous systematic",
        "citation_density": "very high (40-80 refs)",
        "system_prompt_addon": (
            "This is a PRISMA-compliant systematic review. "
            "Include structured abstract with 5 headings. "
            "Describe search strategy explicitly. "
            "Report inclusion/exclusion criteria. "
            "Describe PRISMA flow: initial -> screened -> "
            "eligible -> included. "
            "Be rigorous and transparent. "
        )
    },

    "perspective_article": {
        "label":         "Perspective Article",
        "word_limit":    3000,
        "abstract_words": "150",
        "sections": [
            "Abstract",
            "1. Introduction — The Position",
            "2. Current State and Its Limitations",
            "3. A New Way of Thinking",
            "4. Evidence and Implications",
            "5. Conclusion — Call to Action",
            "References (10-20)"
        ],
        "has_methods":      False,
        "has_results":      False,
        "has_discussion":   True,
        "has_taxonomy":     False,
        "has_prisma":       False,
        "writing_style":    "opinionated persuasive",
        "citation_density": "medium (10-20 refs)",
        "system_prompt_addon": (
            "This is a perspective article — "
            "express a clear opinion or viewpoint. "
            "First person (we/our) is acceptable. "
            "Make provocative but supported claims. "
            "Challenge existing assumptions. "
            "End with a clear call to action. "
        )
    },

    "technical_note": {
        "label":         "Technical Note",
        "word_limit":    2500,
        "abstract_words": "100-150",
        "sections": [
            "Abstract", "Keywords",
            "1. Introduction",
            "2. Technical Description",
            "3. Validation/Testing",
            "4. Conclusions",
            "References (10-15)"
        ],
        "has_methods":      True,
        "has_results":      True,
        "has_discussion":   False,
        "has_taxonomy":     False,
        "has_prisma":       False,
        "writing_style":    "technical precise",
        "citation_density": "medium",
        "system_prompt_addon": (
            "This is a technical note describing "
            "a specific technical contribution — "
            "an algorithm, tool, dataset, or method. "
            "Be precise and reproducible. "
            "Focus on technical details. "
        )
    },

    "case_study": {
        "label":         "Case Study",
        "word_limit":    4000,
        "abstract_words": "150-200",
        "sections": [
            "Abstract", "Keywords",
            "1. Introduction",
            "2. Case Description",
            "  2.1 Context",
            "  2.2 System Description",
            "3. Implementation",
            "4. Findings",
            "  4.1 Quantitative Outcomes",
            "  4.2 Qualitative Outcomes",
            "5. Discussion — Lessons Learned",
            "6. Conclusion",
            "References (15-25)"
        ],
        "has_methods":      True,
        "has_results":      True,
        "has_discussion":   True,
        "has_taxonomy":     False,
        "has_prisma":       False,
        "writing_style":    "narrative descriptive",
        "citation_density": "medium",
        "system_prompt_addon": (
            "This is a case study analysing "
            "a specific real deployment or scenario. "
            "Be narrative and descriptive. "
            "Include both positive and negative findings. "
            "Generalise lessons learned carefully. "
        )
    },
}


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def list_journal_names() -> list:
    """Return list of journal venue short names."""
    return [k for k, v in JOURNAL_TEMPLATES.items()
            if not v.get("is_conference")]


def list_conferences() -> list:
    """Return list of conference venue short names."""
    return [k for k, v in JOURNAL_TEMPLATES.items()
            if v.get("is_conference")]


def get_template(venue: str) -> dict:
    """Get template config for a venue (journal or conference)."""
    return JOURNAL_TEMPLATES.get(
        venue, JOURNAL_TEMPLATES["IEEE"]
    )


def list_article_types() -> list:
    """Return list of all article type keys."""
    return list(ARTICLE_TYPE_CONFIGS.keys())


def get_article_config(article_type: str) -> dict:
    """Get config for a specific article type."""
    return ARTICLE_TYPE_CONFIGS.get(
        article_type,
        ARTICLE_TYPE_CONFIGS["research_article"]
    )
