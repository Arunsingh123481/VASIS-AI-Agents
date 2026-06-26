# agents/agent14_implementation_guide.py
# Implementation Guide Agent — generates step-by-step research implementation roadmaps
# Models: DeepSeek 7B (reasoning + code) + Qwen 3B (planning + structuring)
#
# OUTPUT:
#   1. Innovation breakdown (what exactly to build)
#   2. Technical architecture diagram (text)
#   3. Pseudocode for core algorithm
#   4. Python code skeleton
#   5. Dataset recommendations
#   6. Baseline comparisons
#   7. Evaluation metrics
#   8. Week-by-week implementation plan
#   9. Common pitfalls to avoid
#  10. Hardware requirements

import time
from typing import Dict, Any

from llm.router import generate, generate_json
from config import DEFAULT_RESEARCHER_LEVEL
from console_helper import print_msg
from agents.agent13_paper_writer import clean_topic

SYSTEM_GUIDE = (
    "You are an expert AI researcher and engineer "
    "guiding a researcher through implementing a "
    "novel innovation. "
    "Be specific, practical, and encouraging. "
    "Provide working code examples. "
    "Explain WHY each step matters. "
    "Flag common mistakes clearly. "
    "Assume the researcher has Python skills "
    "and basic ML knowledge."
)

SYSTEM_CODE = (
    "You are an expert Python/PyTorch developer. "
    "Write clean, well-commented, runnable code. "
    "Include type hints. "
    "Add TODO comments for researcher to fill in. "
    "Use modern PyTorch 2.x conventions. "
    "Every function must have a docstring."
)


# ── SECTION 1: INNOVATION BREAKDOWN ─────────────────────────────────────────

def _breakdown_innovation(
    innovation: str, narrative: str, web_sources: list
) -> dict:
    """Break down the innovation into clear technical components."""
    source_context = "\n".join([
        f"- {s.get('title', '')}: {s.get('snippet', '')[:150]}"
        for s in web_sources[:8]
    ])

    prompt = (
        f"Break down this research innovation "
        f"into clear technical components.\n\n"
        f"INNOVATION: {innovation}\n\n"
        f"Evidence from papers:\n{narrative[:2000]}\n\n"
        f"Related work found:\n{source_context}\n\n"
        f"Return JSON:\n"
        f"{{\n"
        f'  "innovation_name": "Short name",\n'
        f'  "core_idea": "1-2 sentence description",\n'
        f'  "problem_solved": "What gap it fills",\n'
        f'  "novelty_claim": "What is truly new",\n'
        f'  "technical_components": [\n'
        f'    {{"name": "Component 1",\n'
        f'     "description": "What it does",\n'
        f'     "complexity": "Low/Medium/High"}}\n'
        f"  ],\n"
        f'  "prior_work_gap": "What existing work does NOT do",\n'
        f'  "expected_improvements": [\n'
        f'    "Improvement 1 with % estimate"\n'
        f"  ],\n"
        f'  "risks": ["Risk 1", "Risk 2"]\n'
        f"}}"
    )
    try:
        result = generate_json(
            "agent12_websearch", prompt,
            "Return ONLY valid JSON."
        )
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    return {
        "innovation_name":       innovation[:80],
        "core_idea":             innovation,
        "problem_solved":        "See vault papers",
        "novelty_claim":         "Novel combination",
        "technical_components":  [],
        "prior_work_gap":        "See Agent 11 output",
        "expected_improvements": [],
        "risks":                 []
    }


# ── SECTION 2: ARCHITECTURE DESIGN ──────────────────────────────────────────

def _design_architecture(
    breakdown: dict, narrative: str
) -> str:
    """Create a detailed text-based architecture diagram."""
    components = breakdown.get("technical_components", [])
    comp_list = "\n".join([
        f"  - {c.get('name', '')}: {c.get('description', '')}" if isinstance(c, dict) else f"  - {c}"
        for c in components
    ])

    prompt = (
        f"Design the complete technical architecture "
        f"for: {breakdown.get('innovation_name', 'Innovation')}\n\n"
        f"Core idea: {breakdown.get('core_idea', '')}\n\n"
        f"Components:\n{comp_list}\n\n"
        f"Evidence from papers:\n{narrative[:2000]}\n\n"
        f"Requirements:\n"
        f"1. Draw ASCII architecture diagram\n"
        f"2. Show data flow (Input -> Layer -> Output)\n"
        f"3. Describe each component in detail\n"
        f"4. Specify input/output shapes\n"
        f"5. Describe forward pass step by step\n"
        f"6. Note where this differs from baseline\n\n"
        f"Write complete architecture specification."
    )
    return generate(
        "answer_generation", prompt,
        system=SYSTEM_GUIDE, temperature=0.1
    )


# ── SECTION 3: PSEUDOCODE ───────────────────────────────────────────────────

def _write_pseudocode(breakdown: dict, narrative: str) -> str:
    """Write clear pseudocode for the core algorithm."""
    prompt = (
        f"Write clear pseudocode for the core "
        f"algorithm of:\n"
        f"{breakdown.get('innovation_name', 'Innovation')}\n\n"
        f"Core idea: {breakdown.get('core_idea', '')}\n\n"
        f"Requirements:\n"
        f"- Use ALGORITHM ... END ALGORITHM format\n"
        f"- Number each step\n"
        f"- Include INPUT and OUTPUT specification\n"
        f"- Comment every non-obvious step\n"
        f"- Show the key innovation clearly\n"
        f"- Include time complexity analysis\n"
        f"- Include space complexity analysis\n\n"
        f"Write complete pseudocode."
    )
    return generate(
        "answer_generation", prompt,
        system=SYSTEM_GUIDE, temperature=0.1
    )


# ── SECTION 4: CODE SKELETON ────────────────────────────────────────────────

def _write_code_skeleton(breakdown: dict, narrative: str) -> str:
    """Write complete Python/PyTorch code skeleton."""
    components = [
        c.get("name", "") if isinstance(c, dict) else c for c in
        breakdown.get("technical_components", [])
    ]

    prompt = (
        f"Write a complete Python/PyTorch code "
        f"skeleton for:\n"
        f"{breakdown.get('innovation_name', 'Innovation')}\n\n"
        f"Core idea: {breakdown.get('core_idea', '')}\n\n"
        f"Components to implement:\n"
        f"{chr(10).join(f'  - {c}' for c in components)}\n\n"
        f"Requirements:\n"
        f"- Full Python 3.10+ with type hints\n"
        f"- PyTorch 2.x conventions\n"
        f"- Classes for each component\n"
        f"- Working __init__ and forward() methods\n"
        f"- TODO comments where researcher must fill in\n"
        f"- Docstrings for every class and method\n"
        f"- Example usage at bottom\n"
        f"- Import statements at top\n\n"
        f"Write complete working code skeleton."
    )
    return generate(
        "answer_generation", prompt,
        system=SYSTEM_CODE, temperature=0.1
    )


# ── SECTION 5: DATASET RECOMMENDATIONS ──────────────────────────────────────

def _recommend_datasets(
    breakdown: dict, web_sources: list, narrative: str
) -> str:
    """Recommend specific datasets for evaluation."""
    source_info = "\n".join([
        f"- {s.get('title', '')}: {s.get('snippet', '')[:100]}"
        for s in web_sources[:8]
    ])

    prompt = (
        f"Recommend datasets for evaluating:\n"
        f"{breakdown.get('innovation_name', 'Innovation')}\n\n"
        f"Innovation: {breakdown.get('core_idea', '')}\n\n"
        f"Related work datasets found:\n{source_info}\n\n"
        f"For each recommended dataset provide:\n"
        f"1. Dataset name and citation\n"
        f"2. Why it suits this innovation\n"
        f"3. Size (train/val/test splits)\n"
        f"4. Download URL or HuggingFace path\n"
        f"5. Preprocessing required\n"
        f"6. Evaluation metric used on it\n"
        f"7. State-of-the-art score to beat\n\n"
        f"Recommend 4-6 datasets in order of priority."
    )
    return generate(
        "answer_generation", prompt,
        system=SYSTEM_GUIDE, temperature=0.15
    )


# ── SECTION 6: BASELINE COMPARISONS ─────────────────────────────────────────

def _design_baselines(
    breakdown: dict, web_sources: list, narrative: str
) -> str:
    """Define what baselines to compare against."""
    source_methods = "\n".join([
        f"- {s.get('title', '')}: {s.get('snippet', '')[:120]}"
        for s in web_sources[:10]
        if any(
            w in s.get("snippet", "").lower()
            for w in ["baseline", "compare", "outperform",
                       "accuracy", "benchmark", "results"]
        )
    ])

    prompt = (
        f"Define baseline comparisons for:\n"
        f"{breakdown.get('innovation_name', 'Innovation')}\n\n"
        f"Related methods from literature:\n"
        f"{source_methods or 'Not available'}\n\n"
        f"For each baseline provide:\n"
        f"1. Method name and citation\n"
        f"2. Why it is a fair comparison\n"
        f"3. How to reproduce it\n"
        f"4. Its known score on standard benchmarks\n"
        f"5. What your method should improve over it\n\n"
        f"Also specify:\n"
        f"- Evaluation protocol (5-fold CV? held-out?)\n"
        f"- Statistical significance test to use\n"
        f"- What constitutes a meaningful improvement\n\n"
        f"List 5-7 baselines in order of importance."
    )
    return generate(
        "answer_generation", prompt,
        system=SYSTEM_GUIDE, temperature=0.15
    )


# ── SECTION 7: EVALUATION METRICS ───────────────────────────────────────────

def _define_metrics(breakdown: dict, narrative: str) -> str:
    """Define all evaluation metrics with code snippets."""
    prompt = (
        f"Define evaluation metrics for:\n"
        f"{breakdown.get('innovation_name', 'Innovation')}\n\n"
        f"Core idea: {breakdown.get('core_idea', '')}\n\n"
        f"For each metric provide:\n"
        f"1. Metric name\n"
        f"2. Formula or definition\n"
        f"3. Python implementation snippet\n"
        f"4. What score is considered good\n"
        f"5. What existing work scores on it\n\n"
        f"Also specify:\n"
        f"- Primary metric (most important)\n"
        f"- Secondary metrics\n"
        f"- Efficiency metrics (speed, memory)\n"
        f"- Ablation metrics\n\n"
        f"Include code snippets for each metric."
    )
    return generate(
        "answer_generation", prompt,
        system=SYSTEM_CODE, temperature=0.1
    )


# ── SECTION 8: IMPLEMENTATION PLAN ──────────────────────────────────────────

def _create_plan(
    breakdown: dict, researcher_level: str = "masters"
) -> str:
    """Create week-by-week implementation roadmap."""
    if researcher_level == "phd":
        weeks = 16
        depth = "deep theoretical analysis included"
    elif researcher_level == "masters":
        weeks = 12
        depth = "practical implementation focused"
    else:
        weeks = 8
        depth = "simplified version, core ideas only"

    components = [
        c.get("name", "") if isinstance(c, dict) else c for c in
        breakdown.get("technical_components", [])
    ]
    comp_str = "\n".join(
        f"  {i + 1}. {c}" for i, c in enumerate(components)
    )

    prompt = (
        f"Create a {weeks}-week implementation plan "
        f"for a {researcher_level} student building:\n"
        f"{breakdown.get('innovation_name', 'Innovation')}\n\n"
        f"Components to build:\n{comp_str}\n\n"
        f"Plan style: {depth}\n\n"
        f"For each week provide:\n"
        f"Week N:\n"
        f"  Goal: What to achieve this week\n"
        f"  Tasks: Specific actionable tasks\n"
        f"  Deliverable: What to have working\n"
        f"  Resources: Papers to read / code to study\n"
        f"  Pitfall: Common mistake to avoid\n\n"
        f"Be very specific — no vague tasks."
    )
    return generate(
        "answer_generation", prompt,
        system=SYSTEM_GUIDE, temperature=0.15
    )


# ── SECTION 9: PITFALLS AND TIPS ────────────────────────────────────────────

def _list_pitfalls(
    breakdown: dict, narrative: str, web_sources: list
) -> str:
    """List common pitfalls specific to this innovation."""
    failures = "\n".join([
        f"- {s.get('title', '')}: {s.get('snippet', '')[:100]}"
        for s in web_sources[:6]
        if any(
            w in s.get("snippet", "").lower()
            for w in ["fail", "challenge", "difficult",
                       "problem", "limitation", "issue"]
        )
    ])

    prompt = (
        f"List critical pitfalls for implementing:\n"
        f"{breakdown.get('innovation_name', 'Innovation')}\n\n"
        f"Known challenges from literature:\n"
        f"{failures if failures else 'Not available'}\n\n"
        f"Vault paper context:\n{narrative[:1000]}\n\n"
        f"Provide:\n"
        f"SECTION A: Top 5 Implementation Pitfalls\n"
        f"  For each: What goes wrong + How to avoid\n\n"
        f"SECTION B: Top 5 Experiment Pitfalls\n"
        f"  For each: What biases results + How to avoid\n\n"
        f"SECTION C: Top 5 Writing Pitfalls\n"
        f"  For each: What reviewers flag + How to avoid\n\n"
        f"SECTION D: 10 Pro Tips\n"
        f"  Specific actionable advice from literature"
    )
    return generate(
        "answer_generation", prompt,
        system=SYSTEM_GUIDE, temperature=0.2
    )


# ── SECTION 10: HARDWARE REQUIREMENTS ───────────────────────────────────────

def _estimate_hardware(breakdown: dict, narrative: str) -> str:
    """Estimate hardware requirements and suggest free alternatives."""
    prompt = (
        f"Estimate hardware requirements for:\n"
        f"{breakdown.get('innovation_name', 'Innovation')}\n\n"
        f"Core idea: {breakdown.get('core_idea', '')}\n\n"
        f"Provide:\n"
        f"1. Minimum hardware to get started (CPU-only)\n"
        f"2. Recommended hardware for full experiments\n"
        f"3. Ideal hardware for state-of-the-art results\n"
        f"4. Estimated training time per configuration\n"
        f"5. Free cloud options:\n"
        f"   - Google Colab (free tier)\n"
        f"   - Kaggle Notebooks (30hr/week GPU)\n"
        f"   - HuggingFace Spaces\n"
        f"6. Tips to reduce compute needs:\n"
        f"   - Which datasets to use for quick testing\n"
        f"   - How to run ablations cheaply\n"
        f"   - Mixed precision training\n"
        f"   - Gradient checkpointing\n\n"
        f"Be realistic — many researchers have limited hardware."
    )
    return generate(
        "answer_generation", prompt,
        system=SYSTEM_GUIDE, temperature=0.15
    )


# ── MAIN AGENT 14 FUNCTION ──────────────────────────────────────────────────

def guide_implementation(
    innovation: str,
    narrative: str = "",
    atom_ids: list = None,
    web_evidence: dict = None,
    novel_connections: list = None,
    researcher_level: str = None,
    paper_result: dict = None
) -> Dict[str, Any]:
    """
    Main Agent 14 function.
    Generates complete implementation guide.

    Args:
        innovation:        The novel idea to implement
        narrative:         RE-MSE expanded vault context
        atom_ids:          Atom IDs for provenance
        web_evidence:      Agent 12 search results
        novel_connections: Agent 11 causal chains
        researcher_level:  masters / phd / beginner
        paper_result:      Agent 13 paper if available

    Returns:
        Complete implementation guide as dict
    """
    innovation = clean_topic(innovation)
    atom_ids = atom_ids or []
    web_evidence = web_evidence or {"sources": []}
    novel_connections = novel_connections or []
    researcher_level = researcher_level or DEFAULT_RESEARCHER_LEVEL

    web_sources = web_evidence.get("sources", [])
    start_time = time.time()

    _innov_display = innovation if len(innovation) <= 80 else innovation[:77] + "..."
    panel_content = (
        f"[bold]Innovation:[/bold]  {_innov_display}\n"
        f"[bold]Level:[/bold]       {researcher_level}\n"
        f"[bold]Web sources:[/bold] {len(web_sources)}"
    )
    from console_helper import print_panel
    print_panel(panel_content, title="[Agent14] IMPLEMENTATION GUIDE")

    guide = {}
    timings = {}

    # ── 1. BREAKDOWN ─────────────────────────────────────
    t = time.time()
    print_msg("[Agent14] Step 1/10: Breaking down innovation...")
    try:
        guide["breakdown"] = _breakdown_innovation(
            innovation, narrative, web_sources
        )
    except Exception as e:
        print_msg(f"[red]Error breaking down innovation: {e}[/red]")
        guide["breakdown"] = {
            "innovation_name":       innovation[:60],
            "core_idea":             innovation,
            "problem_solved":        "Failed to generate innovation breakdown details due to error.",
            "novelty_claim":         "Novel combination",
            "technical_components":  [],
            "prior_work_gap":        "See Agent 11 output",
            "expected_improvements": [],
            "risks":                 []
        }
    timings["breakdown"] = round(time.time() - t, 1)
    name = guide["breakdown"].get("innovation_name", innovation[:60])
    print_msg(f"[Agent14] Innovation: {name}")

    # ── 2. ARCHITECTURE ──────────────────────────────────
    t = time.time()
    print_msg("[Agent14] Step 2/10: Designing architecture...")
    try:
        guide["architecture"] = _design_architecture(
            guide["breakdown"], narrative
        )
    except Exception as e:
        print_msg(f"[red]Error designing architecture: {e}[/red]")
        guide["architecture"] = "*Error: Architecture design failed to generate due to local LLM timeout or error.*"
    timings["architecture"] = round(time.time() - t, 1)

    # ── 3. PSEUDOCODE ────────────────────────────────────
    t = time.time()
    print_msg("[Agent14] Step 3/10: Writing pseudocode...")
    try:
        guide["pseudocode"] = _write_pseudocode(
            guide["breakdown"], narrative
        )
    except Exception as e:
        print_msg(f"[red]Error writing pseudocode: {e}[/red]")
        guide["pseudocode"] = "*Error: Pseudocode failed to generate due to local LLM timeout or error.*"
    timings["pseudocode"] = round(time.time() - t, 1)

    # ── 4. CODE SKELETON ─────────────────────────────────
    t = time.time()
    print_msg("[Agent14] Step 4/10: Writing code skeleton...")
    try:
        guide["code_skeleton"] = _write_code_skeleton(
            guide["breakdown"], narrative
        )
    except Exception as e:
        print_msg(f"[red]Error writing code skeleton: {e}[/red]")
        guide["code_skeleton"] = "*Error: Code skeleton failed to generate due to local LLM timeout or error.*"
    timings["code_skeleton"] = round(time.time() - t, 1)

    # ── 5. DATASETS ──────────────────────────────────────
    t = time.time()
    print_msg("[Agent14] Step 5/10: Recommending datasets...")
    try:
        guide["datasets"] = _recommend_datasets(
            guide["breakdown"], web_sources, narrative
        )
    except Exception as e:
        print_msg(f"[red]Error recommending datasets: {e}[/red]")
        guide["datasets"] = "*Error: Dataset recommendations failed to generate due to local LLM timeout or error.*"
    timings["datasets"] = round(time.time() - t, 1)

    # ── 6. BASELINES ─────────────────────────────────────
    t = time.time()
    print_msg("[Agent14] Step 6/10: Defining baselines...")
    try:
        guide["baselines"] = _design_baselines(
            guide["breakdown"], web_sources, narrative
        )
    except Exception as e:
        print_msg(f"[red]Error defining baselines: {e}[/red]")
        guide["baselines"] = "*Error: Baseline comparisons failed to generate due to local LLM timeout or error.*"
    timings["baselines"] = round(time.time() - t, 1)

    # ── 7. METRICS ───────────────────────────────────────
    t = time.time()
    print_msg("[Agent14] Step 7/10: Defining metrics...")
    try:
        guide["metrics"] = _define_metrics(
            guide["breakdown"], narrative
        )
    except Exception as e:
        print_msg(f"[red]Error defining metrics: {e}[/red]")
        guide["metrics"] = "*Error: Evaluation metrics failed to generate due to local LLM timeout or error.*"
    timings["metrics"] = round(time.time() - t, 1)

    # ── 8. IMPLEMENTATION PLAN ───────────────────────────
    t = time.time()
    print_msg("[Agent14] Step 8/10: Creating plan...")
    try:
        guide["implementation_plan"] = _create_plan(
            guide["breakdown"], researcher_level
        )
    except Exception as e:
        print_msg(f"[red]Error creating plan: {e}[/red]")
        guide["implementation_plan"] = "*Error: Implementation plan failed to generate due to local LLM timeout or error.*"
    timings["plan"] = round(time.time() - t, 1)

    # ── 9. PITFALLS ──────────────────────────────────────
    t = time.time()
    print_msg("[Agent14] Step 9/10: Listing pitfalls...")
    try:
        guide["pitfalls_and_tips"] = _list_pitfalls(
            guide["breakdown"], narrative, web_sources
        )
    except Exception as e:
        print_msg(f"[red]Error listing pitfalls: {e}[/red]")
        guide["pitfalls_and_tips"] = "*Error: Pitfalls and tips failed to generate due to local LLM timeout or error.*"
    timings["pitfalls"] = round(time.time() - t, 1)

    # ── 10. HARDWARE ─────────────────────────────────────
    t = time.time()
    print_msg("[Agent14] Step 10/10: Hardware requirements...")
    try:
        guide["hardware"] = _estimate_hardware(
            guide["breakdown"], narrative
        )
    except Exception as e:
        print_msg(f"[red]Error estimating hardware: {e}[/red]")
        guide["hardware"] = "*Error: Hardware requirements failed to generate due to local LLM timeout or error.*"
    timings["hardware"] = round(time.time() - t, 1)

    elapsed = round(time.time() - start_time, 1)
    print_msg(f"\n[Agent14] Guide complete in {elapsed}s")

    # ── BUILD FULL GUIDE TEXT ────────────────────────────
    section_order = [
        ("Innovation Breakdown", "breakdown"),
        ("Architecture", "architecture"),
        ("Pseudocode", "pseudocode"),
        ("Code Skeleton", "code_skeleton"),
        ("Recommended Datasets", "datasets"),
        ("Baseline Comparisons", "baselines"),
        ("Evaluation Metrics", "metrics"),
        ("Implementation Plan", "implementation_plan"),
        ("Pitfalls and Tips", "pitfalls_and_tips"),
        ("Hardware Requirements", "hardware"),
    ]

    full_text_parts = []
    for heading, key in section_order:
        content = guide.get(key, "")
        if isinstance(content, dict):
            import json
            content = json.dumps(content, indent=2)
        if content:
            full_text_parts.append(f"\n## {heading}\n\n{content}")

    full_text = "\n".join(full_text_parts)

    return {
        "innovation":         innovation,
        "researcher_level":   researcher_level,
        "guide":              guide,
        "full_text":          full_text,
        "timings":            timings,
        "total_seconds":      elapsed,
        "vault_atoms_used":   len(atom_ids),
        "web_sources_used":   len(web_sources),
        "novel_connections":  novel_connections,
    }
