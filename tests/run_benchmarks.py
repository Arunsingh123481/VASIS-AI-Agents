"""
PageIndex-RE-MSE CRDB Multi-Agent RAG Benchmark Suite
Evaluates Page Retrieval Recall, Factual Accuracy, and Hallucination Rejection.
Run with: python tests/run_benchmarks.py
"""

import sys
import os
import time

# Add workspace directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pipeline import PageIndexREMSE

console = Console()

# ─── BENCHMARK DATASET ────────────────────────────────────────────────────────
BENCHMARK_CASES = [
    {
        "id": "CASE-01",
        "category": "Factual Retrieval & Accuracy",
        "question": "What is the dimension of the keys d_k in Scaled Dot-Product Attention?",
        "ground_truth_pages": {4, 5},
        "expected_keywords": ["64", "d_k", "dimension"],
        "description": "Tests precise page retrieval and keyword accuracy (d_k = 64)."
    },
    {
        "id": "CASE-02",
        "category": "Hallucination Rejection (Negative Control)",
        "question": "What are the quantum computing experiments and quantum hardware constraints mentioned in this paper?",
        "ground_truth_pages": set(),  # Should not be in the paper
        "expected_keywords": [],       # Rejection verification
        "is_negative": True,
        "description": "Tests if the 11-agent pipeline rejects out-of-scope topics and flags them as ungrounded."
    },
    {
        "id": "CASE-03",
        "category": "Causal Multi-Hop Reasoning",
        "question": "Why does the Transformer utilize masking in the decoder self-attention?",
        "ground_truth_pages": {2, 3, 5},
        "expected_keywords": ["subsequent", "prevent", "attend", "future", "mask"],
        "description": "Tests multi-page traversal and reasoning for causal mechanisms."
    }
]

def run_benchmarks(pdf_path: str):
    console.print(Panel(
        "[bold cyan]PageIndex-RE-MSE CRDB RAG System Benchmark Suite[/bold cyan]\n"
        f"Target PDF: [bold]{os.path.basename(pdf_path)}[/bold]\n"
        "Evaluating: [bold green]Page Retrieval Recall, Factual Accuracy, Hallucination Rejection[/bold green]",
        title="RAG Evaluation Console",
        expand=False
    ))

    if not os.path.exists(pdf_path):
        console.print(f"[bold red]Error: Target PDF not found at {pdf_path}[/bold red]")
        sys.exit(1)

    # Initialize RAG Pipeline (uses config.py settings by default, which is now qwen2.5-coder:3b)
    rag = PageIndexREMSE()
    
    # Ingest document (loads from cache if already processed)
    with console.status("[bold yellow]Ingesting and indexing document...[/bold yellow]"):
        rag.ingest(pdf_path)

    results = []
    
    for case in BENCHMARK_CASES:
        console.print("\n[bold magenta]====================================================================[/bold magenta]")
        console.print(f"[bold blue][{case['id']}] Category: {case['category']}[/bold blue]")
        console.print(f"[bold white]Question:[/bold white] \"{case['question']}\"")
        console.print(f"[dim]Description: {case['description']}[/dim]\n")

        start_time = time.time()
        # Execute query without verbose standard console print inside loop
        res = rag.query(
            question=case["question"],
            show_provenance=False,
            save_result=False
        )
        elapsed = time.time() - start_time

        # Gather metrics
        answer = res.get("answer", "")
        provenance = res.get("provenance", {})
        pages_referenced = set(provenance.get("pages_referenced", []))
        trust_level = res.get("trust_level", "low")
        pipeline_grade = res.get("pipeline_grade", "F")
        confidence = res.get("confidence", 0.0)

        # 1. Page Retrieval Check (Recall + Precision)
        retrieval_pass = False
        recall_score = 0.0
        precision_score = None
        f1_score = None
        if case.get("is_negative"):
            # For negative controls, referencing few or no pages with low confidence is a pass
            retrieval_pass = len(pages_referenced) <= 2 or trust_level in ("low", "medium")
            recall_score = 1.0 if retrieval_pass else 0.0
        else:
            gt = case["ground_truth_pages"]
            intersection = pages_referenced.intersection(gt)
            if intersection:
                retrieval_pass = True
                recall_score = len(intersection) / len(gt)
            else:
                recall_score = 0.0
            # Precision: recall alone is gameable by retrieving most/all pages.
            # This catches that — e.g. retrieving 11 of 11 pages to "find" a
            # 2-page answer scores 100% recall but only ~18% precision.
            if len(pages_referenced) == 0:
                precision_score = 1.0 if not gt else 0.0
            else:
                precision_score = len(intersection) / len(pages_referenced)
            f1_score = (
                2 * precision_score * recall_score / (precision_score + recall_score)
                if (precision_score + recall_score) > 0 else 0.0
            )

        # 2. Factual Accuracy Check (Keyword matching / Rejection matching)
        accuracy_pass = False
        accuracy_score = 0.0
        if case.get("is_negative"):
            # Check if answer correctly handles rejection (should say not mentioned/found)
            rejection_words = ["not mention", "not discuss", "not explicitly", "no mention", "not found", "does not contain"]
            accuracy_pass = any(word in answer.lower() for word in rejection_words) or trust_level == "low"
            accuracy_score = 1.0 if accuracy_pass else 0.0
        else:
            matches = [kw for kw in case["expected_keywords"] if kw.lower() in answer.lower()]
            accuracy_score = len(matches) / len(case["expected_keywords"])
            accuracy_pass = accuracy_score >= 0.4  # Matches at least 40% of key facts

        # 3. Grounding Audit Rejection (Hallucination protection test)
        hallucination_pass = True
        if case.get("is_negative"):
            # If trust score is low/medium and system flagged it correctly
            hallucination_pass = trust_level in ("low", "medium")

        results.append({
            "id": case["id"],
            "category": case["category"],
            "recall": recall_score,
            "precision": precision_score,
            "f1": f1_score,
            "accuracy": accuracy_score,
            "hallucination_pass": hallucination_pass,
            "elapsed": elapsed,
            "grade": pipeline_grade,
            "trust": trust_level,
            "confidence": confidence,
            "pages": list(pages_referenced)
        })

        # Display query metrics
        console.print("[bold green]> Result Answer Summary:[/bold green]")
        console.print(f"  {answer[:180]}...")
        console.print("\n[bold yellow]> Local Execution Metrics:[/bold yellow]")
        console.print(f"  Pages Retrieved : {list(pages_referenced)} (Target: {list(case['ground_truth_pages']) if not case.get('is_negative') else 'None'})")
        console.print(f"  Recall Score    : [bold]{recall_score * 100:.1f}%[/bold]")
        console.print(f"  Precision Score : [bold]{f'{precision_score * 100:.1f}%' if precision_score is not None else 'N/A'}[/bold]")
        console.print(f"  Accuracy Score  : [bold]{accuracy_score * 100:.1f}%[/bold]")
        console.print(f"  Safety Audited  : [bold]{'PASSED (Safe rejection/rating)' if hallucination_pass else 'FAILED (Vulnerable to Hallucination)'}[/bold]")
        console.print(f"  Trust Level     : [bold cyan]{trust_level.upper()}[/bold cyan] (Confidence: {confidence:.2f})")
        console.print(f"  Orchestration   : Grade [bold white]{pipeline_grade}[/bold white] | Time: {elapsed:.2f}s")

    # ─── OVERALL SCORECARD DASHBOARD ──────────────────────────────────────────
    console.print("\n[bold magenta]====================================================================[/bold magenta]")
    console.print(Panel("[bold green]OVERALL SYSTEM BENCHMARK SCORECARD[/bold green]", expand=False))

    summary_table = Table(title="PageIndex-RE-MSE CRDB 11-Agent Benchmark Metrics", show_header=True, header_style="bold magenta")
    summary_table.add_column("Case ID", style="dim")
    summary_table.add_column("Category", width=30)
    summary_table.add_column("Page Recall", justify="right")
    summary_table.add_column("Page Precision", justify="right")
    summary_table.add_column("Fact Accuracy", justify="right")
    summary_table.add_column("Hallucination Protection", justify="center")
    summary_table.add_column("Trust / Conf", justify="center")
    summary_table.add_column("Pipeline Grade", justify="center")
    summary_table.add_column("Time", justify="right")

    total_recall = 0.0
    total_accuracy = 0.0
    total_safety_passes = 0
    total_time = 0.0
    precision_vals = []

    for r in results:
        recall_pct = f"{r['recall'] * 100:.0f}%"
        prec_pct = f"{r['precision'] * 100:.0f}%" if r.get("precision") is not None else "N/A"
        acc_pct = f"{r['accuracy'] * 100:.0f}%"
        safety_status = "[green]SAFE[/green]" if r["hallucination_pass"] else "[red]RISKY[/red]"
        conf_str = f"{r['trust'].upper()} ({r['confidence']:.2f})"
        
        summary_table.add_row(
            r["id"],
            r["category"],
            recall_pct,
            prec_pct,
            acc_pct,
            safety_status,
            conf_str,
            f"[bold]{r['grade']}[/bold]",
            f"{r['elapsed']:.1f}s"
        )
        total_recall += r["recall"]
        total_accuracy += r["accuracy"]
        if r.get("precision") is not None:
            precision_vals.append(r["precision"])
        if r["hallucination_pass"]:
            total_safety_passes += 1
        total_time += r["elapsed"]

    num_cases = len(BENCHMARK_CASES)
    avg_recall = total_recall / num_cases
    avg_accuracy = total_accuracy / num_cases
    safety_score = total_safety_passes / num_cases
    avg_precision_str = f"{(sum(precision_vals)/len(precision_vals))*100:.1f}%" if precision_vals else "N/A"

    console.print(summary_table)

    console.print(Panel(
        f"[bold white]Summary Scores:[/bold white]\n"
        f"  - [bold cyan]Average Page Retrieval Recall[/bold cyan] : [bold yellow]{avg_recall * 100:.1f}%[/bold yellow]\n"
        f"  - [bold cyan]Average Page Retrieval Precision[/bold cyan] : [bold yellow]{avg_precision_str}[/bold yellow] "
        f"(catches recall inflated by over-retrieval; computed over {len(precision_vals)} non-negative-control cases)\n"
        f"  - [bold cyan]Average Factual Accuracy Match[/bold cyan] : [bold yellow]{avg_accuracy * 100:.1f}%[/bold yellow]\n"
        f"  - [bold cyan]Hallucination Protection Rate[/bold cyan] : [bold green]{safety_score * 100:.1f}% (Full Audit Rejection)[/bold green]\n"
        f"  - [bold cyan]Total Benchmark Execution Time[/bold cyan]: [bold white]{total_time:.1f}s[/bold white]\n\n"
        f"[dim]Note: with only {num_cases} cases (1 per category), these numbers are a smoke test, "
        f"not a statistically meaningful benchmark. Treat tests/run_production50.py as the real signal.[/dim]",
        title="Aggregate RAG Quality Grades",
        expand=False
    ))

if __name__ == "__main__":
    default_pdf = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "uploads",
        "1cf6b031-c99_NIPS-2017-attention-is-all-you-need-Paper.pdf"
    )
    
    pdf_arg = sys.argv[1] if len(sys.argv) > 1 else default_pdf
    run_benchmarks(pdf_arg)
