import os
import json
from typing import List, Dict, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm.ollama_client import ask_llm

class ResearchIntelligence:
    def __init__(self):
        pass

    def extract_limitations(self, document_title: str, rag_instance) -> List[Dict[str, Any]]:
        """Stage 1: Extracts structured limitations and problem context."""
        query = "What are the limitations, unresolved issues, benchmark failures, reproducibility issues, or future work mentioned by the authors?"
        
        result = rag_instance.query(query, top_k_anchors=6, expansion_passes=3, show_provenance=False, save_result=False)
        narrative = result.get("narrative", "")
        
        prompt = f"""You are a strict research evolution analyst.
Extract explicit limitations, bottlenecks, or failures from the following paper excerpt: "{document_title}".
DO NOT paraphrase generic gaps. Be specific about WHAT failed and WHY.

Excerpt:
---
{narrative}
---

Return the output strictly as a JSON array of objects. Each object must contain:
- "limitation": A detailed description of the limitation
- "limitation_type": One of [scaling, memory, reasoning, generalization, benchmark_failure, reproducibility, multimodal_alignment, efficiency, interpretability, safety, temporal_reasoning, other]
- "affected_component": What part of the system or method failed
- "affected_domain": The task or domain where it failed
- "failure_condition": Under what exact conditions it failed
- "evidence_strength": [strong, moderate, weak] based on experimental backing
- "experimental_context": Context of the experiment
- "publication_year": Year if mentioned, else "Unknown"
- "quote": Exact text snippet supporting this
"""
        response = ask_llm(prompt, expect_json=True)
        try:
            limitations = json.loads(response)
            if isinstance(limitations, dict):
                limitations = [limitations]
            for lim in limitations:
                lim["source_paper"] = document_title
                lim["doc_id"] = rag_instance.doc_id
            return limitations
        except Exception as e:
            print(f"Error parsing limitations: {e}")
            return []

    def analyze_novelty_gaps(self, rag_instances: Dict[str, Any], all_limitations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 2-6: Analyzes limitations across all papers to find real gaps, contradictions, and blind spots."""
        if not all_limitations:
            return []
            
        limitations_json = json.dumps(all_limitations, indent=2)
        
        # We need a summary of what the papers solved to detect if limitations were addressed
        paper_summaries = {}
        for title, rag in rag_instances.items():
            query = "What did this paper solve, and what does it explicitly claim to fix?"
            res = rag.query(query, show_provenance=False, save_result=False)
            paper_summaries[title] = res.get("narrative", "")
            
        summaries_json = json.dumps(paper_summaries, indent=2)

        prompt = f"""You are a world-class Scientific Claim Graph Engine.
Your objective is to map the true scientific evolution of problems by grounding EVERY finding in a traceable, quantitative claim graph rather than generating isolated reasoning text.

MANDATORY CLAIM PROVENANCE & GRAPH GROUNDING:
You must NEVER generate an opportunity without tracing its exact origin. Every finding must extract explicit claim nodes containing the paper, benchmark, metric, experimental conditions, and architecture era (e.g., RNN era, Transformer era, State-Space era, RAG era). 

QUANTITATIVE STATE & BENCHMARK LINEAGE:
Replace vague language ("significant challenge remains") with quantitative modeling (failure distributions, benchmark variance, scaling thresholds). Track the lineage of benchmarks to detect evaluation drift and metric incompatibilities.

DEPLOYMENT-AWARE REASONING:
All opportunities must consider hardware constraints, memory thresholds, latency, and real-world feasibility.

CLASSIFICATION SYSTEM:
Every finding MUST be classified into EXACTLY ONE of these strings:
- "verified_unresolved_gap" (Confidence: 90-100)
- "partially_solved_bottleneck" (Confidence: 70-89)
- "emerging_contradiction" (Confidence: 50-69)
- "weak_signal" (Confidence: 30-49)
- "historical_resolved_limitation" (Confidence: 0-29)

Paper Claims & Achievements:
{summaries_json}

Extracted Limitations to Analyze:
{limitations_json}

Analyze the limitations against the claims. Return strictly a JSON array of objects. Each object MUST strictly adhere to this schema:
{{
  "id": "A unique slug string",
  "classification": "<one of the 5 allowed strings above>",
  "title": "Specific, nuanced title of the active tension",
  "domain": "Cross-Domain Research Area",
  "summary": "Nuanced synthesis of the bottleneck with conditional boundaries.",
  "confidence_score": <integer 0-100>,
  "unresolvedness_score": <integer 0-100>,
  
  "claim_graph_nodes": [
    {{
      "paper": "Exact paper title or ID",
      "claim": "Measurable claim (e.g. Method X improves Y by Z%)",
      "benchmark": "Dataset or benchmark used",
      "metric": "Evaluation metric",
      "experimental_conditions": "Key constraints",
      "architecture_family": "e.g., Transformer era, State-space model era",
      "publication_year": "Contextual year if known",
      "supporting_evidence": "List of supporting sources",
      "contradictory_evidence": "List of conflicting sources",
      "reproducibility_status": "Status of reproduction",
      "deployment_constraints": "Implementation bottlenecks"
    }}
  ],

  "benchmark_lineage": {{
    "benchmark_evolution": "How evaluation evolved",
    "dataset_limitations": "Flaws in data",
    "metric_incompatibilities": "Issues with metric comparison",
    "evaluation_drift": "Drift in measuring true capability",
    "cross_domain_reliability": "Transfer reliability"
  }},

  "quantitative_state": {{
    "benchmark_variance": "Variance in outcomes",
    "failure_distributions": "Where failures cluster",
    "reproduction_instability": "Instability rate",
    "deployment_scaling_thresholds": "Exact threshold boundaries",
    "edge_case_failure_probabilities": "Probability distributions"
  }},

  "scientific_consensus": {{
    "current_consensus": "Summary",
    "solved_status": "e.g., Partially resolved",
    "supporting_study_count": <integer>,
    "contradictory_study_count": <integer>,
    "benchmark_agreement_rate": "e.g. High, Low, or percentage",
    "replication_stability": "Stability metric",
    "temporal_persistence": "How long it has survived",
    "architecture_era_relevance": "Which eras this applies to",
    "attribution": "Derived from explicit sources in the graph"
  }},

  "deployment_reasoning": {{
    "hardware_constraints": "Specific hardware limits",
    "memory_thresholds": "Memory bounds",
    "latency_constraints": "Latency bottlenecks",
    "scalability_boundaries": "Where scaling breaks",
    "deployment_tradeoffs": "Tradeoffs required",
    "real_world_feasibility": "Practical viability"
  }},

  "score_explanations": {{
    "confidence_derivation": ["+ explicit evidence", "- contradiction"],
    "unresolvedness_derivation": ["+ persistent failure", "- modern solution"]
  }},

  "failure_boundaries": {{
    "solved_conditions": "Where systems successfully operate",
    "unresolved_conditions": "Where systems currently fail",
    "threshold_conditions": "Specific limits",
    "deployment_constraints": "Practical constraints",
    "benchmark_limitations": "Instability in evaluation",
    "edge_case_failures": "Critical failure modes"
  }},

  "temporal_evolution": {{
    "original_problem": "Original issue",
    "proposed_solutions": "Modern architectures attempted",
    "partial_improvements": "Actual improvements",
    "unresolved_failures": "Specific persistent failures",
    "current_research_tension": "Active debate"
  }},

  "suggested_research_directions": "Next steps",
  "contradiction_details": {{
     "claim_a": "Measurable claim A",
     "claim_b": "Measurable claim B",
     "disagreement_strength": "Strength",
     "benchmark_reasoning": "Reasoning",
     "reproducibility_status": "Status"
  }} // ONLY INCLUDE IF classification IS "emerging_contradiction", otherwise null
}}
"""
        response = ask_llm(prompt, expect_json=True)
        try:
            gaps = json.loads(response)
            if isinstance(gaps, dict):
                gaps = [gaps]
            return gaps
        except Exception as e:
            print(f"Error mapping gaps: {e}")
            return []

    def extract_themes(self, rag_instances: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extracts the main research themes and clusters across all uploaded papers."""
        paper_contributions = {}
        for title, rag in rag_instances.items():
            query = "What is the main contribution, method, and key result of this paper?"
            result = rag.query(query, top_k_anchors=4, expansion_passes=3, show_provenance=False, save_result=False)
            paper_contributions[title] = result.get("narrative", "")

        contributions_json = json.dumps(paper_contributions, indent=2)

        prompt = f"""You are a research synthesis expert. I am giving you the key contributions of multiple papers.
Identify 3-6 major research themes or clusters that group these papers together.
For each theme, list which papers belong to it and provide a short description.

Paper Contributions:
{contributions_json}

Return strictly a JSON array of objects with:
- "theme": Short name for the research theme (e.g., "Adversarial Robustness")
- "description": 1-2 sentence description of what this theme covers
- "papers": List of paper titles belonging to this theme
- "keywords": List of 3-5 key technical terms for this theme
"""
        response = ask_llm(prompt, expect_json=True)
        try:
            themes = json.loads(response)
            if isinstance(themes, dict):
                themes = [themes]
            return themes
        except Exception as e:
            print(f"Error extracting themes: {e}")
            return []

    def detect_debates(self, rag_instances: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detects debates and conflicting results across multiple papers."""
        paper_summaries = {}
        for title, rag in rag_instances.items():
            query = "What are the main experimental results, methodology choices, and core conclusions of this paper?"
            result = rag.query(query, show_provenance=False, save_result=False)
            paper_summaries[title] = result.get("narrative", "")
            
        summaries_json = json.dumps(paper_summaries, indent=2)
        
        prompt = f"""You are a peer reviewer. I am providing the core methodologies and conclusions of several papers.
Analyze them together to find Disagreements, Conflicting Results, or Unresolved Debates.
For example, Paper A says method X is fast, but Paper B says method X scales poorly.

Paper Summaries:
{summaries_json}

Return strictly a JSON array of objects with:
- "debate_topic": A short, specific, descriptive title for the debate (e.g. "PVA hydrogels in wound healing — efficacy vs degradation risk"). NEVER use a generic question like "What is the debate about?". Make it descriptive and informative.
- "side_a": Position and paper name
- "side_b": Conflicting position and paper name
- "open_issue": What remains unresolved
"""
        response = ask_llm(prompt, expect_json=True)
        try:
            debates = json.loads(response)
            if isinstance(debates, dict):
                debates = [debates]
            return debates
        except Exception:
            return []

intelligence_engine = ResearchIntelligence()
