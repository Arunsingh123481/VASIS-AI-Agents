"""
Agent13 + Agent14 Research Paper & Implementation Guide Generator
Novel Topic: "Adaptive Reciprocal Rank Fusion with Query-Type-Aware Weighting
             for Multi-Modal Retrieval in Long-Document RAG Systems"

Uses the full 14-agent pipeline:
  Agent1  → Route as PAPER_WRITE query
  Agent2  → Decompose topic into sub-questions
  Agent3  → Navigate to relevant sections of the Attention paper
  Agent4  → RRF retrieval of relevant atoms
  Agent5  → RE-MSE narrative expansion
  Agent6  → Validate groundedness of evidence
  Agent7  → Audit contradictions in RRF literature
  Agent8  → Temporal analysis of attention/retrieval timeline
  Agent9  → Calibrate trust score
  Agent11 → Synthesize novel connections
  Agent12 → Web search for prior RRF/RAG literature
  Agent13 → Write full academic paper
  Agent14 → Generate implementation guide with code

Run with: python tests/generate_paper.py
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from pipeline import PageIndexREMSE

console = Console(highlight=False)

# ─── NOVEL RESEARCH TOPIC ─────────────────────────────────────────────────────
NOVEL_TOPIC = (
    "Adaptive Reciprocal Rank Fusion with Query-Type-Aware Weighting "
    "for Multi-Modal Retrieval in Long-Document RAG Systems"
)

PAPER_QUERY = (
    "Write a complete research paper on the following novel problem: "
    "Standard Reciprocal Rank Fusion (RRF) in multi-modal RAG systems uses "
    "fixed weights across all query types (vector, BM25, graph). "
    "We propose Adaptive-RRF: a dynamic weight scheme where factual queries "
    "boost BM25, causal multi-hop queries boost graph-walk, and comparative "
    "queries boost vector similarity. "
    "Derive the mathematical formulation, design the architecture, "
    "show experimental results from the Attention paper benchmark, "
    "and propose a 12-week implementation roadmap."
)

INNOVATION_FOR_GUIDE = (
    "Adaptive-RRF: Dynamic Reciprocal Rank Fusion with query-type-aware weights. "
    "Implement a classifier that detects query type (factual/causal/comparative) "
    "and adjusts RRF fusion weights (vector, BM25, graph) in real-time before "
    "retrieval, improving recall by 15-25% over fixed-weight RRF baselines."
)


def main():
    console.print(Panel(
        "[bold cyan]Agent13 + Agent14 -- Full Research Paper Generation[/bold cyan]\n"
        f"Novel Problem: [bold]{NOVEL_TOPIC[:80]}...[/bold]\n"
        "Agents: 1 Router -> 2 Decomposer -> 3 Navigator -> 4 Retrieval ->\n"
        "        5 Expansion -> 6 Validation -> 7 Audit -> 8 Temporal ->\n"
        "        9 Calibration -> 11 Synthesis -> 12 Web -> 13 Paper -> 14 Guide",
        title="[*] 14-Agent Research Paper Pipeline",
        expand=False
    ))

    default_pdf = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "uploads",
        "1cf6b031-c99_NIPS-2017-attention-is-all-you-need-Paper.pdf"
    )
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else default_pdf

    if not os.path.exists(pdf_path):
        console.print(f"[bold red]PDF not found: {pdf_path}[/bold red]")
        sys.exit(1)

    # ── INIT + INGEST ─────────────────────────────────────────────────────────
    rag = PageIndexREMSE()
    with console.status("[yellow]Loading document index...[/yellow]"):
        rag.ingest(pdf_path)

    console.print("\n[bold green]Document ready. Starting 14-agent paper pipeline...[/bold green]")
    console.print(
        "[dim]This uses ALL agents: routing, retrieval, expansion, validation, "
        "contradiction audit, temporal analysis, synthesis, web search, "
        "paper writing, and implementation guide generation.[/dim]\n"
    )

    # ── RUN PAPER QUERY (triggers agents 1-13) ────────────────────────────────
    start = time.time()
    console.print(Panel(
        f"[bold white]PAPER QUERY:[/bold white]\n{PAPER_QUERY}",
        title="Agent Pipeline Input",
        expand=False
    ))

    paper_res = rag.query(
        question=PAPER_QUERY,
        show_provenance=False,
        save_result=True,
        forced_query_type="PAPER_WRITE",
        venue="NeurIPS",
        article_type="research_article",
        researcher_level="masters"
    )

    paper_elapsed = time.time() - start
    console.print(f"\n[bold green]Paper query complete in {paper_elapsed:.1f}s[/bold green]")

    # ── EXTRACT PAPER + NARRATIVE FOR AGENT14 ────────────────────────────────
    paper_result = paper_res.get("paper_result")
    narrative    = paper_res.get("narrative", "")
    novel_conns  = paper_res.get("novel_connections", [])
    ordered_atoms = paper_res.get("ordered_atoms", [])

    # ── RUN IMPLEMENTATION GUIDE (agent14) ───────────────────────────────────
    console.print("\n" + "="*60)
    console.print("[bold magenta]Starting Agent14 -- Implementation Guide...[/bold magenta]")

    from agents.agent14_implementation_guide import guide_implementation

    impl_start = time.time()
    impl_result = guide_implementation(
        innovation=INNOVATION_FOR_GUIDE,
        narrative=narrative,
        atom_ids=[a.get("atom_id", "") for a in ordered_atoms],
        web_evidence=paper_res.get("paper_result", {}) or {},
        novel_connections=novel_conns,
        researcher_level="masters",
        paper_result=paper_result,
    )
    impl_elapsed = time.time() - impl_start
    console.print(f"[bold green]Implementation guide complete in {impl_elapsed:.1f}s[/bold green]")

    # ── SAVE OUTPUTS ─────────────────────────────────────────────────────────
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Save full paper markdown
    paper_md_path = os.path.join(out_dir, "novel_paper.md")
    paper_text = ""
    if paper_result and isinstance(paper_result, dict):
        paper_text = paper_result.get("full_text", "")
        topic      = paper_result.get("topic", NOVEL_TOPIC)
        venue      = paper_result.get("venue_full", "NeurIPS")
        wc         = paper_result.get("word_count", 0)
    else:
        paper_text = paper_res.get("answer", "")
        topic      = NOVEL_TOPIC
        venue      = "NeurIPS"
        wc         = len(paper_text.split())

    with open(paper_md_path, "w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n")
        f.write(f"> **Venue:** {venue}  |  **Word count:** {wc:,}  |  **Generated by:** PageIndex-RE-MSE 14-Agent System\n\n")
        f.write("---\n\n")
        f.write(paper_text)
    console.print(f"\n[bold green]Paper saved:[/bold green] {paper_md_path}")

    # Save implementation guide markdown
    guide_md_path = os.path.join(out_dir, "novel_implementation_guide.md")
    guide_text = impl_result.get("full_text", "")
    with open(guide_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Implementation Guide: {INNOVATION_FOR_GUIDE[:80]}\n\n")
        f.write("> **Researcher Level:** Masters  |  **Generated by:** Agent14\n\n")
        f.write("---\n\n")
        f.write(guide_text)
    console.print(f"[bold green]Guide saved:[/bold green] {guide_md_path}")

    # Save combined JSON
    combined_json = {
        "topic": NOVEL_TOPIC,
        "paper_elapsed_s": round(paper_elapsed, 1),
        "impl_elapsed_s": round(impl_elapsed, 1),
        "paper_word_count": wc,
        "paper_path": paper_md_path,
        "guide_path": guide_md_path,
        "pipeline_grade": paper_res.get("pipeline_grade", "?"),
        "confidence": paper_res.get("confidence", 0.0),
        "trust_level": paper_res.get("trust_level", "?"),
        "atoms_used": paper_res.get("atoms_used", 0),
        "novel_connections": novel_conns,
    }
    json_path = os.path.join(out_dir, "paper_generation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined_json, f, indent=2, default=str)
    console.print(f"[bold green]Summary JSON:[/bold green] {json_path}")

    console.print(Panel(
        f"[bold white]Paper Generation Complete![/bold white]\n\n"
        f"  Paper:            {paper_md_path}\n"
        f"  Guide:            {guide_md_path}\n"
        f"  Word Count:       {wc:,}\n"
        f"  Pipeline Grade:   {paper_res.get('pipeline_grade', '?')}\n"
        f"  Total Time:       {paper_elapsed + impl_elapsed:.1f}s",
        title="Agent13+14 Output",
        expand=False
    ))

    return combined_json


if __name__ == "__main__":
    main()
