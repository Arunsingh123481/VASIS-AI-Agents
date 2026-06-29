"""
Test Suite — Custom Agent Studio (agent_builder.py)
Run with: python -m pytest tests/test_agent_builder.py -v
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agent_builder import (
    AgentStudio,
    CustomAgent,
    CustomLoop,
    CustomAgentStore,
    detect_blueprint,
    validate_research_relevance,
    domain_ordered_citations,
    format_citation_options,
    generate_system_prompt,
    AGENT_BLUEPRINTS,
    BLUEPRINT_ALIASES,
    FIXED_AGENTS,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def tmp_store(tmp_path):
    """Return a path for a temporary store file."""
    return tmp_path / "test_agents.json"


@pytest.fixture
def studio(tmp_store):
    """Return an AgentStudio wired to a temp store with no LLM."""
    logs = []
    return AgentStudio(
        print_fn=lambda t, s="": logs.append(t),
        store_path=tmp_store,
    )


# =============================================================================
# BLUEPRINT DETECTION
# =============================================================================

class TestBlueprintDetection:
    def test_exact_match(self):
        key, bp = detect_blueprint("references")
        assert key == "references"
        assert bp is not None

    def test_alias_match_ref(self):
        key, _ = detect_blueprint("ref")
        assert key == "references"

    def test_alias_match_bibliography(self):
        key, _ = detect_blueprint("bibliography")
        assert key == "references"

    def test_alias_match_workscited(self):
        key, _ = detect_blueprint("workscited")
        assert key == "references"

    def test_keyword_fuzzy_match(self):
        key, _ = detect_blueprint("refs")
        assert key == "references"

    def test_no_match_returns_none(self):
        key, bp = detect_blueprint("drug_interaction_checker")
        assert key is None
        assert bp is None

    @pytest.mark.parametrize("alias,expected", [
        ("relatedwork",       "literaturereview"),
        ("materialsandmethods", "methodology"),
        ("findings",          "results"),
        ("indexterms",        "keywords"),
        ("executivesummary",  "abstract"),
        ("concludingremarks", "conclusion"),
        ("stateoftheart",     "literaturereview"),
        ("patientsandmethods","methodology"),
        ("critique",          "criticalreview"),
        ("compare",           "comparison"),
    ])
    def test_alias_recognition(self, alias, expected):
        key, _ = detect_blueprint(alias)
        assert key == expected, f"alias '{alias}' → got '{key}', expected '{expected}'"


# =============================================================================
# RESEARCH RELEVANCE VALIDATION
# =============================================================================

class TestResearchRelevance:
    def test_valid_research_agent(self):
        answers = {
            "specialisation": "extract drug-target interaction pairs from biomedical research abstracts",
            "input_desc":     "raw clinical trial PDF reports and PubMed abstracts",
            "output_desc":    "structured list of drug-target pairs with confidence scores and evidence sentences",
            "tasks":          "1. reads abstract  2. identifies drug names  3. maps to biological targets  4. classifies interaction  5. outputs structured data",
            "research_link":  "its output feeds into my gap_analysis agent as structured evidence for the literature review pipeline",
            "quality_signal": "every pair must include confidence score above 0.7 and a source sentence from the corpus",
            "domain_knowledge": "UMLS medical terminology ICD-10 codes MeSH headings biomedical NER",
        }
        result = validate_research_relevance(answers)
        assert result["valid"], f"Valid agent rejected: {result['explanation']}"
        assert result["score"] > 0

    def test_invalid_nonresearch_agent(self):
        answers = {
            "specialisation": "order pizza from nearby restaurants based on user food preferences",
            "input_desc":     "user food preferences and delivery address",
            "output_desc":    "pizza order confirmation with estimated delivery time",
            "tasks":          "1. gets location  2. finds restaurant  3. shows menu  4. places food order",
            "research_link":  "I use it when I am hungry during work",
            "quality_signal": "pizza arrives hot and in under 30 minutes",
            "domain_knowledge": "food delivery apps and restaurant menus",
        }
        result = validate_research_relevance(answers)
        assert not result["valid"], f"Invalid agent accepted: {result['explanation']}"


# =============================================================================
# DOMAIN CITATION ORDERING
# =============================================================================

class TestDomainCitations:
    def test_cs_ai_default_is_ieee(self):
        ordered = domain_ordered_citations("CS / AI")
        assert ordered[0] == "IEEE"

    def test_medicine_default_is_vancouver(self):
        ordered = domain_ordered_citations("Medicine / Biology")
        assert ordered[0] == "Vancouver"

    def test_social_sciences_default_is_apa(self):
        ordered = domain_ordered_citations("Social Sciences")
        assert ordered[0] == "APA 7th"

    def test_unknown_domain_falls_back(self):
        ordered = domain_ordered_citations("Unknown Field")
        assert len(ordered) > 0


# =============================================================================
# CUSTOM AGENT STORE
# =============================================================================

class TestCustomAgentStore:
    def test_save_and_get_agent(self, tmp_store):
        store = CustomAgentStore(tmp_store)
        agent = CustomAgent(
            agent_id="testref",
            command="/testref",
            created=1000.0,
            description="test agent",
            research_domain="CS / AI",
            input_type="paper_text",
            output_desc="reference list",
            citation_style="IEEE",
            quality_bar="must have DOI",
            extra="",
            system_prompt="You are a test agent.",
        )
        store.save_agent(agent)

        retrieved = store.get_agent("testref")
        assert retrieved is not None
        assert retrieved.agent_id == "testref"
        assert retrieved.description == "test agent"

    def test_delete_agent(self, tmp_store):
        store = CustomAgentStore(tmp_store)
        agent = CustomAgent(
            agent_id="to_delete",
            command="/to_delete",
            created=1000.0,
            description="delete me",
            research_domain="General / Other",
            input_type="all",
            output_desc="nothing",
            citation_style="None",
            quality_bar="",
            extra="",
            system_prompt="test",
        )
        store.save_agent(agent)
        assert store.agent_exists("to_delete")

        ok = store.delete_agent("to_delete")
        assert ok
        assert not store.agent_exists("to_delete")

    def test_delete_nonexistent_returns_false(self, tmp_store):
        store = CustomAgentStore(tmp_store)
        assert not store.delete_agent("nonexistent")

    def test_save_and_get_loop(self, tmp_store):
        store = CustomAgentStore(tmp_store)
        loop = CustomLoop(
            loop_id="test-loop",
            command="test-loop",
            created=1000.0,
            agent_ids=["abstract", "introduction"],
        )
        store.save_loop(loop)

        retrieved = store.get_loop("test-loop")
        assert retrieved is not None
        assert retrieved.agent_ids == ["abstract", "introduction"]

    def test_loops_using_agent(self, tmp_store):
        store = CustomAgentStore(tmp_store)
        loop = CustomLoop(
            loop_id="myloop",
            command="myloop",
            created=1000.0,
            agent_ids=["abstract", "references"],
        )
        store.save_loop(loop)

        assert "myloop" in store.loops_using_agent("abstract")
        assert "myloop" not in store.loops_using_agent("methodology")

    def test_all_agents(self, tmp_store):
        store = CustomAgentStore(tmp_store)
        for name in ("a1", "a2", "a3"):
            agent = CustomAgent(
                agent_id=name,
                command=f"/{name}",
                created=1000.0,
                description=f"agent {name}",
                research_domain="General / Other",
                input_type="all",
                output_desc="test",
                citation_style="None",
                quality_bar="",
                extra="",
                system_prompt="test",
            )
            store.save_agent(agent)

        all_a = store.all_agents()
        assert len(all_a) == 3


# =============================================================================
# AGENT STUDIO — BLUEPRINT WIZARD
# =============================================================================

class TestAgentStudioBlueprintWizard:
    def test_build_references_agent(self, studio):
        # Simulate answers: [1]=use defaults, [1]=CS/AI, [1]=IEEE, formatting pref, [yes]=confirm
        inputs = iter(["1", "1", "1", "alphabetical order with DOI", "yes"])
        studio._ask = lambda p, s: next(inputs)

        agent = studio.cmd_agent("build references")
        assert agent is not None
        assert agent.agent_id == "references"
        assert agent.citation_style == "IEEE"
        assert "alphabetical" in agent.extra

    def test_build_abstract_no_citation(self, studio):
        inputs = iter(["1", "1", "150-250 words", "no citations allowed", "yes"])
        studio._ask = lambda p, s: next(inputs)

        agent = studio.cmd_agent("build abstract")
        assert agent is not None
        assert agent.citation_style == "None"
        assert "150-250" in agent.quality_bar

    def test_build_introduction_citation_section(self, studio):
        inputs = iter(["1", "1", "1", "must end with research gap", "yes"])
        studio._ask = lambda p, s: next(inputs)

        agent = studio.cmd_agent("build introduction")
        assert agent is not None
        assert agent.citation_style == "IEEE"

    def test_reject_fixed_agent_name(self, studio):
        agent = studio.cmd_agent("build router")
        assert agent is None

    def test_overwrite_refused(self, studio):
        # First build
        inputs1 = iter(["1", "1", "1", "with DOI", "yes"])
        studio._ask = lambda p, s: next(inputs1)
        studio.cmd_agent("build keywords")

        # Attempt overwrite, say no
        inputs2 = iter(["no"])
        studio._ask = lambda p, s: next(inputs2)
        result = studio.cmd_agent("build keywords")
        assert result is None


# =============================================================================
# AGENT STUDIO — CONNECT AND LOOPS
# =============================================================================

class TestAgentStudioLoops:
    def test_connect_agents(self, studio):
        # Build two agents first
        for name in ("abstract", "introduction"):
            inputs = iter(["1", "1", "1", "", "yes"])
            studio._ask = lambda p, s: next(inputs)
            studio.cmd_agent(f"build {name}")

        loop = studio.cmd_connect("abstract introduction --name test-loop")
        assert loop is not None
        assert loop.loop_id == "testloop"  # slugified: "test-loop" → "testloop"

    def test_connect_too_few_agents(self, studio):
        loop = studio.cmd_connect("abstract")
        assert loop is None


# =============================================================================
# AGENT STUDIO — DELETE WITH CASCADE
# =============================================================================

class TestAgentStudioDelete:
    def test_delete_agent_cascades_loops(self, studio):
        # Build agents
        for name in ("abstract", "introduction", "methodology"):
            inputs = iter(["1", "1", "1", "", "yes"])
            studio._ask = lambda p, s: next(inputs)
            studio.cmd_agent(f"build {name}")

        # Create a loop
        studio.cmd_connect("abstract introduction methodology --name paper-core")

        # Delete abstract (confirm cascade)
        studio._ask = lambda p, s: "yes"
        ok = studio.cmd_delete("agent abstract")
        assert ok
        assert not studio.store.agent_exists("abstract")
        # Loop should be cascade-deleted
        assert studio.store.get_loop("papercore") is None  # slugified


# =============================================================================
# SYSTEM PROMPT GENERATION
# =============================================================================

class TestSystemPromptGeneration:
    def test_template_fallback_no_llm(self):
        prompt = generate_system_prompt(
            agent_id="testref",
            description="Collect and format all references",
            research_domain="CS / AI",
            input_type="paper_text",
            output_desc="Numbered APA reference list",
            citation_style="APA 7th",
            quality_bar="Every citation must have DOI",
            extra="Only peer-reviewed sources",
            llm_fn=None,
        )
        assert len(prompt) > 50
        assert "APA 7th" in prompt
        assert "DOI" in prompt
        assert "peer-reviewed" in prompt
        assert prompt.endswith("Output must be well-formatted markdown.")


# =============================================================================
# FIXED AGENTS PROTECTION
# =============================================================================

class TestFixedAgentsProtection:
    @pytest.mark.parametrize("name", list(FIXED_AGENTS))
    def test_cannot_build_fixed_agent(self, studio, name):
        result = studio.cmd_agent(f"build {name}")
        assert result is None

    def test_cannot_delete_fixed_agent(self, studio):
        studio._ask = lambda p, s: "yes"
        result = studio.cmd_delete("agent router")
        assert not result
