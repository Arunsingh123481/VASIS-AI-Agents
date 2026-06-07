"""
Main Pipeline — PageIndex + RE-MSE Hybrid RAG System with 11-Agent CRDB Engine.
Orchestrates all phases: ingestion, indexing, reasoning, expansion, reconstruction.
"""
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from console_helper import print_msg, tqdm, print_panel


import json
import os
from typing import Dict, Optional

from ingest.pdf_loader import load_pdf, get_pdf_metadata
from ingest.atomic_decomposer import decompose_to_atoms, cross_reference_atoms_to_tree
from ingest.tree_builder import build_tree, print_tree
from reconstruction.stitcher import stitch, get_provenance
from storage.store import get_doc_id, save_index, load_index, index_exists, save_query_result
from config import AGENT_MODEL, REASONING_MODEL


class PageIndexREMSE:
    """
    PageIndex-RE-MSE Hybrid RAG System with 11-Agent CRDB Orchestration.

    Combines:
    - PageIndex: reasoning-based hierarchical tree navigation (macro layer)
    - RE-MSE: multi-segment stateful atomic expansion (micro layer)
    - CRDB: autonomous multi-agent planner, auditor, contradiction, and causal synthesis engine
    """

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.tree_nodes = []
        self.atoms = []
        self.triples = []
        self.doc_id = None
        self.pdf_path = None
        self._ready = False

    def ingest(self, pdf_path: str, force_reindex: bool = False) -> None:
        """
        Ingest a PDF document: load, decompose, build tree, extract triples, cross-reference.
        Uses cached index if available (unless force_reindex=True).
        """
        self.pdf_path = pdf_path
        self.doc_id = get_doc_id(pdf_path)

        print_panel(
            f"[bold cyan]PageIndex-RE-MSE 11-Agent CRDB Engine[/bold cyan]\n"
            f"Document: {os.path.basename(pdf_path)}\n"
            f"Doc ID: {self.doc_id}",
            title="System Ingestion Startup"
        )

        # Check Ollama connections for routed local models
        from llm.ollama_client import check_ollama_connection
        if not check_ollama_connection(AGENT_MODEL):
            raise RuntimeError(
                "Cannot connect to Ollama or missing agent model.\n"
                f"Please run: ollama pull {AGENT_MODEL}"
            )
        if not check_ollama_connection(REASONING_MODEL):
            raise RuntimeError(
                "Cannot connect to Ollama or missing reasoning model.\n"
                f"Please run: ollama pull {REASONING_MODEL}"
            )

        # Try loading cached index
        if not force_reindex and index_exists(self.doc_id):
            print_msg("[green]Found cached index. Loading...[/green]")
            cached = load_index(self.doc_id)
            if cached:
                self.tree_nodes = cached["tree"]
                self.atoms = cached["atoms"]
                self.triples = cached.get("triples", [])
                self._ready = True
                print_msg(f"[green]Ready. {len(self.tree_nodes)} tree nodes, {len(self.atoms)} atoms, {len(self.triples)} triples loaded from cache.[/green]")
                return

        # Full ingestion pipeline
        print_msg("[cyan]No cache found. Running full ingestion pipeline...[/cyan]")

        # Phase 1a: Load PDF
        pages = load_pdf(pdf_path)
        if not pages:
            raise ValueError("No text extracted from PDF. Is it a scanned/image-only PDF?")

        # Phase 1b: Build PageIndex tree (macro layer)
        self.tree_nodes = build_tree(pages, self.doc_id)

        # Phase 1c: Decompose to atoms (micro layer)
        self.atoms = decompose_to_atoms(pages, self.doc_id)

        # Phase 1d: Cross-reference atoms to tree nodes
        self.atoms = cross_reference_atoms_to_tree(self.atoms, self.tree_nodes)

        # Extract knowledge triples for causal graph
        from ingest.triple_extractor import extract_all_triples
        self.triples = extract_all_triples(self.atoms)

        # Save to cache
        save_index(self.doc_id, self.tree_nodes, self.atoms, self.triples)

        self._ready = True
        print_msg(f"\n[bold green]Ingestion complete.[/bold green]")
        print_msg(f"  Tree nodes: {len(self.tree_nodes)}")
        print_msg(f"  Atoms: {len(self.atoms)}")
        print_msg(f"  Causal Triples: {len(self.triples)}")

    def query(
        self,
        question: str,
        top_k_anchors: int = 5,
        expansion_passes: int = 4,
        show_provenance: bool = True,
        save_result: bool = True,
        generate_answer: bool = True
    ) -> Dict:
        """
        Run a full query through the advanced 11-agent CRDB pipeline.

        Returns dict with:
          - answer: the final validated answer
          - provenance: full audit trail (sections, atoms, reasoning)
          - narrative: the reconstructed narrative context
        """
        if not self._ready:
            raise RuntimeError("System not ready. Call ingest() first.")

        print_panel(f"[bold]Query via Advanced Multi-Agent Engine:[/bold] {question}", title="New Query")

        # Load database stores
        from db.atom_store import AtomStore
        from db.bm25_index import BM25Index
        from db.triple_store import TripleStore
        from db.causal_store import CausalStore
        from learning.feedback_index import FeedbackIndex
        from agents.agent10_super import SuperAgent

        atom_store   = AtomStore(self.atoms)
        bm25_index   = BM25Index(self.atoms)
        triple_store = TripleStore(self.triples)
        causal_store = CausalStore(self.triples)
        feedback     = FeedbackIndex()

        # Instantiate and run autonomous orchestrator SuperAgent
        super_agent = SuperAgent(
            tree=self.tree_nodes,
            atom_store=atom_store,
            bm25_index=bm25_index,
            triple_store=triple_store,
            causal_store=causal_store,
            feedback_index=feedback
        )

        result = super_agent.execute(question, doc_id=self.doc_id)

        # Handle aborted pipeline gracefully
        if result.get("aborted", False):
            return {
                "answer": result["answer"],
                "provenance": {"reasoning_path": "Aborted", "sections_used": [], "pages_referenced": [], "total_atoms": 0},
                "narrative": "",
                "selected_sections": [],
                "atoms_used": 0,
                "ordered_atoms": []
            }

        # Build dual-layer audit provenance record
        ordered_atoms = result.get("ordered_atoms", [])
        selected_nodes_ids = result.get("selected_sections", [])
        selected_sections = [n for n in self.tree_nodes if n["node_id"] in selected_nodes_ids]
        
        reasoning_trail = result.get("review_report", {}).get("per_agent_scores", [])
        provenance = get_provenance(ordered_atoms, self.tree_nodes, json.dumps(reasoning_trail))

        # Display result
        if show_provenance:
            self._display_result(question, result["answer"], provenance, show_provenance)

        # Save to query log
        if save_result and self.doc_id:
            save_query_result(self.doc_id, question, result["answer"], provenance)

        return {
            "answer": result["answer"],
            "provenance": provenance,
            "narrative": result.get("narrative", ""),
            "selected_sections": [s["title"] for s in selected_sections] if selected_sections else selected_nodes_ids,
            "atoms_used": len(ordered_atoms),
            "ordered_atoms": ordered_atoms,
            # CRDB extra metadata for advanced features
            "confidence": result.get("confidence", 0.0),
            "trust_level": result.get("trust_level", "low"),
            "novel_connections": result.get("novel_connections", []),
            "contradictions_found": result.get("contradictions_found", False),
            "contradiction_details": result.get("contradiction_details", []),
            "pipeline_grade": result.get("pipeline_grade", "F"),
            "elapsed_seconds": result.get("elapsed_seconds", 0.0)
        }

    def _display_result(self, question: str, answer: str, provenance: Dict, show_provenance: bool) -> None:
        """Pretty print the final result."""
        print_panel(
            f"[bold green]{answer}[/bold green]",
            title="Answer"
        )

        if show_provenance:
            print_msg("\n[bold cyan]Provenance (Audit Trail):[/bold cyan]")
            print_msg(f"  Reasoning Trail: [dim]{provenance['reasoning_path'][:300]}...[/dim]")

            if provenance.get("sections_used"):
                print_msg("\n  Sections referenced:")
                for s in provenance["sections_used"]:
                    print_msg(f"    • [{s['node_id']}] {s['title']} (pages {s['pages']})")

            if provenance.get("pages_referenced"):
                pages = provenance["pages_referenced"]
                print_msg(f"\n  Pages referenced: {pages}")
                print_msg(f"  Total atoms used: {provenance['total_atoms']}")

    def show_tree(self) -> None:
        """Display the PageIndex tree structure."""
        if not self.tree_nodes:
            print_msg("[yellow]No tree loaded. Run ingest() first.[/yellow]")
            return
        print_tree(self.tree_nodes)

    def get_stats(self) -> Dict:
        """Return system statistics."""
        return {
            "doc_id": self.doc_id,
            "pdf_path": self.pdf_path,
            "tree_nodes": len(self.tree_nodes),
            "total_atoms": len(self.atoms),
            "total_triples": len(self.triples),
            "model": f"Qwen & DeepSeek (Ollama)",
            "ready": self._ready
        }
