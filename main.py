"""
CLI Interface — Command-line interface for the PageIndex-RE-MSE CRDB system.
Supports interactive chat mode, single queries, index inspection, and detailed multi-agent audit logs.
"""

import click
import json
import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from config import DEFAULT_MODEL

console = Console()


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

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pipeline import PageIndexREMSE

    rag = PageIndexREMSE(model=model)
    rag.ingest(pdf_path, force_reindex=force_reindex)

    stats = rag.get_stats()
    _print_stats(stats)

    if show_tree:
        rag.show_tree()

    console.print("\n[bold green]Document indexed and causal graph built successfully. Ready to query.[/bold green]")
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

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pipeline import PageIndexREMSE

    rag = PageIndexREMSE(model=model)
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
        # Display advanced CRDB metrics
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold white]CRDB MULTI-AGENT PIPELINE ANALYSIS[/bold white]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"  [bold]TRUST LEVEL[/bold]  : [bold green]{result.get('trust_level', 'low').upper()}[/bold green] ({result.get('confidence', 0.0)})")
        console.print(f"  [bold]PIPELINE GRADE[/bold]: [bold yellow]{result.get('pipeline_grade', 'F')}[/bold yellow]")
        console.print(f"  [bold]ELAPSED TIME[/bold]  : {result.get('elapsed_seconds', 0.0)}s")
        
        if result.get("contradictions_found"):
            console.print(f"\n[bold red][WARNING] CONTRADICTIONS DETECTED:[/bold red]")
            for c in result.get("contradiction_details", []):
                console.print(f"   [[bold]{c.get('severity', '?').upper()}[/bold]] {c.get('claim_a', '')} <-> {c.get('claim_b', '')}")
                
        if result.get("novel_connections"):
            console.print(f"\n[bold green][NOVEL INSIGHT] NOVEL SYNTHESISED CAUSAL CONNECTIONS:[/bold green]")
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

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pipeline import PageIndexREMSE

    rag = PageIndexREMSE(model=model)
    rag.ingest(pdf_path)

    console.print(Panel(
        "[bold cyan]PageIndex-RE-MSE Interactive Multi-Agent Chat (CRDB Engine)[/bold cyan]\n"
        f"Document: {os.path.basename(pdf_path)}\n"
        "Type your question and press Enter.\n"
        "Commands: [bold]tree[/bold] = show index | [bold]stats[/bold] = show stats | [bold]quit[/bold] = exit",
        title="Chat Mode"
    ))

    while True:
        try:
            question = Prompt.ask("\n[bold cyan]Your question[/bold cyan]")

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

            result = rag.query(
                question,
                top_k_anchors=top_k,
                expansion_passes=passes,
                show_provenance=True
            )

            # Display advanced CRDB metrics in chat mode too
            console.print(f"\n[bold cyan]--- CRDB Engine Analysis ---[/bold cyan]")
            console.print(f"  [bold]TRUST LEVEL[/bold]: [bold green]{result.get('trust_level', 'low').upper()}[/bold green] ({result.get('confidence', 0.0)}) | [bold]GRADE[/bold]: [bold yellow]{result.get('pipeline_grade', 'F')}[/bold yellow]")
            
            if result.get("contradictions_found"):
                console.print(f"  [bold red][WARNING] CONTRADICTIONS DETECTED![/bold red] Severity details shown above.")
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
