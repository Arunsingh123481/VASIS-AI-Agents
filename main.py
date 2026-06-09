"""
CLI Interface — Command-line interface for the PageIndex-RE-MSE CRDB system.
Supports interactive chat mode, single queries, index inspection, and detailed multi-agent audit logs.
"""

import click
import json
import os
import re
import sys
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.markdown import Markdown
from rich import box
from config import DEFAULT_MODEL

console = Console()

# ── OUTPUT DIRECTORY (for Agent 13 papers + Agent 14 guides) ──────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def _print_banner():
    banner_text = """
 [bold violet]██████╗  █████╗  ██████╗ ███████╗██╗███╗   ██╗██████╗ ███████╗██╗  ██╗[/bold violet]
 [bold violet]██╔══██╗██╔══██╗██╔════╝ ██╔════╝██║████╗  ██║██╔══██╗██╔════╝╚██╗██╔╝[/bold violet]
 [bold violet]██████╔╝███████║██║  ███╗█████╗  ██║██╔██╗ ██║██║  ██║█████╗   ╚███╔╝ [/bold violet]
 [bold violet]██╔═══╝ ██╔══██║██║   ██║██╔══╝  ██║██║╚██╗██║██║  ██║██╔══╝   ██╔██╗ [/bold violet]
 [bold violet]██║     ██║  ██║╚██████╔╝███████╗██║██║ ╚████║██████╔╝███████╗██╔╝ ██╗[/bold violet]
 [bold violet]╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝[/bold violet]

          [bold cyan]⚡ PageIndex-RE-MSE AI — 14-Agent Consensus Engine ⚡[/bold cyan]
    """
    console.print(Panel(banner_text, border_style="violet", box=box.ROUNDED, expand=False))


def _save_agent_output(
    content: str,
    agent_name: str,
    topic: str,
    venue: str = None,
    article_type: str = None,
    researcher_level: str = None,
) -> str:
    """
    Save Agent 13 (paper) or Agent 14 (guide) output to outputs/ as a .md file.
    Returns the full file path.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sanitise topic for filename — use up to 60 chars so long queries aren't cut off
    safe_topic = re.sub(r'[^\w\s-]', '', topic[:60]).strip()
    safe_topic = re.sub(r'[\s]+', '_', safe_topic).lower()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if agent_name == "paper":
        venue_tag   = f"_{venue.lower()}" if venue else ""
        type_tag    = f"_{article_type.replace('_', '')}" if article_type else ""
        filename    = f"paper{venue_tag}{type_tag}_{safe_topic}_{timestamp}.md"
        title_line  = f"# Research Paper — {topic}\n"
        meta_block  = (
            f"**Venue:** {venue or 'Unknown'}  \n"
            f"**Article Type:** {(article_type or 'research_article').replace('_', ' ').title()}  \n"
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n\n"
            f"---\n\n"
        )
    else:
        level_tag  = f"_{researcher_level}" if researcher_level else ""
        filename   = f"guide{level_tag}_{safe_topic}_{timestamp}.md"
        title_line = f"# Implementation Guide — {topic}\n"
        meta_block = (
            f"**Level:** {(researcher_level or 'masters').title()}  \n"
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n\n"
            f"---\n\n"
        )

    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(title_line)
        f.write(meta_block)
        f.write(content)

    return filepath


@click.group()
def cli():
    """PageIndex-RE-MSE CRDB Hybrid RAG System — Vectorless + Multi-Agent RAG"""
    pass


@cli.command()
@click.argument("pdf_path")
@click.option("--model", default=DEFAULT_MODEL, help=f"Ollama model to pull (default: {DEFAULT_MODEL})")
@click.option("--force-reindex", is_flag=True, help="Force re-indexing even if cache exists")
@click.option("--show-tree", is_flag=True, help="Show the PageIndex tree after indexing")
def index(pdf_path, model, force_reindex, show_tree):
    """Index a PDF document (build tree + atoms + causal knowledge triples)."""
    _check_pdf(pdf_path)
    _print_banner()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pipeline import PageIndexREMSE

    rag = PageIndexREMSE(model=model)
    with console.status("[bold cyan]Swarm active — Ingesting PDF and building causal graph...[/bold cyan]", spinner="dots"):
        rag.ingest(pdf_path, force_reindex=force_reindex)

    stats = rag.get_stats()
    _print_stats(stats)

    if show_tree:
        rag.show_tree()

    console.print("\n[bold green]✓ Document indexed and causal graph built successfully. Ready to query.[/bold green]")
    console.print(f"[dim]Run: python main.py chat {pdf_path}[/dim]")


@cli.command()
@click.argument("pdf_path")
@click.argument("question")
@click.option("--model", default=DEFAULT_MODEL, help="Ollama model to use")
@click.option("--top-k", default=5, help="Number of anchor atoms to select")
@click.option("--passes", default=4, help="Number of expansion passes")
@click.option("--no-provenance", is_flag=True, help="Hide provenance output")
@click.option("--json-output", is_flag=True, help="Output result as JSON")
def ask(pdf_path, question, model, top_k, passes, no_provenance, json_output):
    """Ask a single question about a PDF document using the 11-Agent CRDB engine."""
    _check_pdf(pdf_path)
    _print_banner()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pipeline import PageIndexREMSE

    rag = PageIndexREMSE(model=model)
    with console.status("[bold cyan]Swarm active — Ingesting PDF and executing agent pipeline...[/bold cyan]", spinner="arc"):
        rag.ingest(pdf_path)
        result = rag.query(
            question,
            top_k_anchors=top_k,
            expansion_passes=passes,
            show_provenance=not no_provenance
        )

    if json_output:
        output = {
            "question": question,
            "answer": result["answer"],
            "sections_used": result["selected_sections"],
            "atoms_used": result["atoms_used"],
            "provenance": result["provenance"],
            "confidence": result.get("confidence", 0.0),
            "trust_level": result.get("trust_level", "low"),
            "pipeline_grade": result.get("pipeline_grade", "F")
        }
        console.print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        # Display response inside a gorgeous Panel with Markdown rendering
        console.print(Panel(
            Markdown(result["answer"]),
            title="[bold green]✓ PageIndex AI Response[/bold green]",
            subtitle=f"[bold green]Trust: {result.get('trust_level', 'low').upper()} ({result.get('confidence', 0.0)}) | Grade: {result.get('pipeline_grade', 'F')}[/bold green]",
            border_style="green",
            box=box.ROUNDED,
            expand=False
        ))

        # Display advanced CRDB metrics
        console.print(f"\n[bold cyan]--- Swarm Analysis ({result.get('elapsed_seconds', 0.0)}s) ---[/bold cyan]")
        
        if result.get("contradictions_found"):
            console.print("\n[bold red][WARNING] CONTRADICTIONS DETECTED:[/bold red]")
            for c in result.get("contradiction_details", []):
                console.print(f"   [[bold]{c.get('severity', '?').upper()}[/bold]] {c.get('claim_a', '')} <-> {c.get('claim_b', '')}")
                
        if result.get("novel_connections"):
            console.print("\n[bold green][NOVEL INSIGHT] NOVEL SYNTHESISED CAUSAL CONNECTIONS:[/bold green]")
            for n in result["novel_connections"]:
                via = " -> ".join(n.get("via", []))
                console.print(f"   [bold]{n.get('from', '')}[/bold] -> {via} -> [bold]{n.get('to', '')}[/bold] (conf={n.get('confidence', 0.0):.2f})")
                console.print(f"   [dim]Inference: {n.get('inference', '')}[/dim]")


@cli.command()
@click.argument("pdf_path")
@click.option("--model", default=DEFAULT_MODEL, help="Ollama model to use")
@click.option("--top-k", default=5, help="Number of anchor atoms to select")
@click.option("--passes", default=4, help="Number of expansion passes")
def chat(pdf_path, model, top_k, passes):
    """Interactive chat mode — ask multiple questions utilizing the 11-Agent CRDB engine."""
    _check_pdf(pdf_path)
    _print_banner()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pipeline import PageIndexREMSE
    from agent_routing_rules import QUERY_DETECTION_PATTERNS

    rag = PageIndexREMSE(model=model)
    with console.status("[bold cyan]Swarm active — Loading and analyzing document...[/bold cyan]", spinner="dots"):
        rag.ingest(pdf_path)

    console.print(Panel(
        f"[bold violet]Document:[/bold violet] [cyan]{os.path.basename(pdf_path)}[/cyan]\n\n"
        "Ask questions about the paper. The 14-Agent Swarm will synthesize answers.\n"
        "Commands: [bold cyan]tree[/bold cyan] (show index) | [bold cyan]stats[/bold cyan] (show stats) | [bold cyan]quit[/bold cyan] (exit)",
        title="[bold green]● Swarm Chat Active[/bold green]",
        border_style="green",
        box=box.ROUNDED
    ))

    # Available options for interactive prompting
    VENUES = ["IEEE", "NeurIPS", "ICML", "ICLR", "ACM", "Springer", "Elsevier"]
    ARTICLE_TYPES = [
        "research_article", "review_article", "systematic_review",
        "short_communication", "perspective_article", "technical_note",
        "case_study", "letter_to_editor"
    ]
    RESEARCHER_LEVELS = ["beginner", "masters", "phd"]

    while True:
        try:
            # Claude Code style user prompt
            question = Prompt.ask("\n[bold violet]╭── user[/bold violet]\n[bold violet]╰──>[/bold violet]")

            if not question.strip():
                continue

            if question.strip().lower() in ("quit", "exit", "q"):
                console.print("[yellow]Goodbye![/yellow]")
                break

            if question.strip().lower() == "tree":
                rag.show_tree()
                continue

            if question.strip().lower() == "stats":
                _print_stats(rag.get_stats())
                continue

            # ── Detect if paper writing or implementation guide is requested ──
            q_lower = question.lower()
            is_paper = any(pat in q_lower for pat in QUERY_DETECTION_PATTERNS.get("paper_writing", []))
            is_guide = any(pat in q_lower for pat in QUERY_DETECTION_PATTERNS.get("implementation_guide", []))

            venue = None
            article_type = None
            researcher_level = None

            if is_paper:
                console.print(Panel(
                    "[bold yellow]📄 Research Paper Writing Mode detected![/bold yellow]\n"
                    "[dim]Please answer a few quick questions to customise your paper:[/dim]",
                    border_style="yellow",
                    box=box.ROUNDED
                ))

                # Ask for venue
                console.print("[bold]Target Venue / Journal:[/bold]")
                for i, v in enumerate(VENUES, 1):
                    console.print(f"  [{i}] {v}")
                v_choice = Prompt.ask("Select venue number (or press Enter for IEEE)", default="1")
                try:
                    venue = VENUES[int(v_choice) - 1]
                except (ValueError, IndexError):
                    venue = "IEEE"

                # Ask for article type
                console.print("\n[bold]Article Type:[/bold]")
                for i, at in enumerate(ARTICLE_TYPES, 1):
                    console.print(f"  [{i}] {at.replace('_', ' ').title()}")
                at_choice = Prompt.ask("Select article type number (or press Enter for Research Article)", default="1")
                try:
                    article_type = ARTICLE_TYPES[int(at_choice) - 1]
                except (ValueError, IndexError):
                    article_type = "research_article"

                console.print(f"\n[green]✓ Writing a [bold]{article_type.replace('_', ' ').title()}[/bold] for [bold]{venue}[/bold][/green]\n")

            if is_guide:
                console.print(Panel(
                    "[bold yellow]🔧 Implementation Guide Mode detected![/bold yellow]\n"
                    "[dim]Please answer a quick question to personalise your guide:[/dim]",
                    border_style="yellow",
                    box=box.ROUNDED
                ))

                # Ask for researcher level
                console.print("[bold]Your Researcher Level:[/bold]")
                for i, lv in enumerate(RESEARCHER_LEVELS, 1):
                    console.print(f"  [{i}] {lv.title()}")
                lv_choice = Prompt.ask("Select level number (or press Enter for Masters)", default="2")
                try:
                    researcher_level = RESEARCHER_LEVELS[int(lv_choice) - 1]
                except (ValueError, IndexError):
                    researcher_level = "masters"

                console.print(f"\n[green]✓ Generating guide for [bold]{researcher_level.title()}[/bold] level researcher[/green]\n")

            # Execute query inside styled status spinner
            with console.status("[bold cyan]Swarm active — Analyzing query & synthesizing consensus...[/bold cyan]", spinner="arc"):
                result = rag.query(
                    question,
                    top_k_anchors=top_k,
                    expansion_passes=passes,
                    show_provenance=True,
                    venue=venue,
                    article_type=article_type,
                    researcher_level=researcher_level
                )

            # Display response inside a gorgeous Panel with Markdown rendering
            console.print(Panel(
                Markdown(result["answer"]),
                title="[bold green]✓ PageIndex Response[/bold green]",
                subtitle=f"[bold green]Trust: {result.get('trust_level', 'low').upper()} ({result.get('confidence', 0.0)}) | Grade: {result.get('pipeline_grade', 'F')}[/bold green]",
                border_style="green",
                box=box.ROUNDED,
                expand=False
            ))

            # ── Auto-save Agent 13 / 14 outputs ─────────────────────────────
            paper_result = result.get("paper_result")
            impl_result  = result.get("impl_result")

            if paper_result and paper_result.get("full_text"):
                saved_path = _save_agent_output(
                    content=paper_result["full_text"],
                    agent_name="paper",
                    topic=question,
                    venue=venue,
                    article_type=article_type,
                )
                console.print(f"\n[bold green]📄 Paper saved →[/bold green] [cyan]{saved_path}[/cyan]")

            if impl_result and impl_result.get("full_text"):
                saved_path = _save_agent_output(
                    content=impl_result["full_text"],
                    agent_name="guide",
                    topic=question,
                    researcher_level=researcher_level,
                )
                console.print(f"\n[bold green]🔧 Guide saved →[/bold green] [cyan]{saved_path}[/cyan]")

            # Display advanced CRDB metrics in chat mode too
            console.print("\n[bold cyan]--- CRDB Engine Analysis ---[/bold cyan]")
            console.print(f"  [bold]TRUST LEVEL[/bold]: [bold green]{result.get('trust_level', 'low').upper()}[/bold green] ({result.get('confidence', 0.0)}) | [bold]GRADE[/bold]: [bold yellow]{result.get('pipeline_grade', 'F')}[/bold yellow]")

            if result.get("contradictions_found"):
                console.print("  [bold red][WARNING] CONTRADICTIONS DETECTED![/bold red] Severity details shown above.")
            if result.get("novel_connections"):
                console.print(f"  [bold green][NOVEL INSIGHT] {len(result['novel_connections'])} NOVEL INFERENCES SYNTHESISED![/bold green] Run single query to view full graph paths.")

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")



@cli.command()
@click.argument("pdf_path")
def history(pdf_path):
    """Show past query history for a document."""
    _check_pdf(pdf_path)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from storage.store import get_doc_id, load_query_log

    doc_id = get_doc_id(pdf_path)
    log = load_query_log(doc_id)

    if not log:
        console.print("[yellow]No query history found for this document.[/yellow]")
        return

    console.print(f"\n[bold cyan]Query History — {os.path.basename(pdf_path)}[/bold cyan]")
    for i, entry in enumerate(log, 1):
        console.print(f"\n[bold]{i}. [{entry['timestamp']}][/bold]")
        console.print(f"  Q: {entry['query']}")
        console.print(f"  A: [dim]{entry['answer'][:300]}...[/dim]")
        pages = entry.get("provenance", {}).get("pages_referenced", [])
        if pages:
            console.print(f"  Pages: {pages}")


@cli.command()
def list():
    """List all indexed documents in the local cache vault."""
    import os
    from storage.store import STORAGE_DIR
    if not os.path.exists(STORAGE_DIR) or not os.listdir(STORAGE_DIR):
        console.print("[yellow]No indexed documents found in the local cache vault.[/yellow]")
        return

    table = Table(title="Indexed Documents in Local Cache Vault")
    table.add_column("Doc ID", style="cyan")
    table.add_column("Index Files", style="green")
    table.add_column("Cache Size", style="magenta")

    for doc_id in sorted(os.listdir(STORAGE_DIR)):
        doc_dir = os.path.join(STORAGE_DIR, doc_id)
        if os.path.isdir(doc_dir):
            files = os.listdir(doc_dir)
            files_str = ", ".join(files)
            total_size = sum(os.path.getsize(os.path.join(doc_dir, f)) for f in files if os.path.isfile(os.path.join(doc_dir, f)))
            table.add_row(doc_id, files_str, f"{total_size / 1024:.1f} KB")
            
    console.print(table)


def _check_pdf(pdf_path: str) -> None:
    """Verify PDF file exists."""
    if not os.path.exists(pdf_path):
        console.print(f"[red]File not found: {pdf_path}[/red]")
        sys.exit(1)


def _print_stats(stats: dict) -> None:
    """Print system stats as a table."""
    table = Table(title="System Stats")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    for k, v in stats.items():
        table.add_row(str(k), str(v))
    console.print(table)


if __name__ == "__main__":
    cli()
