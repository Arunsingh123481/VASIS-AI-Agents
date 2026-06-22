"""
vault.py — Multi-document session manager for the CRDB engine.

Lets you load several papers into one session and ask questions that span
all of them, including a real cross-document contradiction check.

Why this is a separate module instead of changes to pipeline.py:
Every PageIndexREMSE instance numbers its own atoms starting from 0
(see ingest/atomic_decomposer.py). That's fine for a single document, but
it means atom_id alone is NOT unique across documents — paper A's atom #5
and paper B's atom #5 are unrelated. Naively concatenating triples from
multiple PageIndexREMSE instances into one TripleStore would silently
scramble TripleStore's by-atom index and any atom_id-based scoping.

VaultSession keeps each paper's PageIndexREMSE fully independent (so the
existing single-document agents — which assume atom_id is locally unique —
are never at risk) and only builds a merged, remapped copy of the relevant
triples when a cross-document check is actually requested.
"""
import os
from typing import Dict, List

from pipeline import PageIndexREMSE
from console_helper import print_msg


# Keyword-based intent detection for "compare these papers" style questions.
# Deliberately simple/transparent rather than another LLM call — this just
# decides which code path to run, it doesn't need to be a judgment call.
COMPARISON_KEYWORDS = [
    "contradict", "contradiction", "disagree", "disagreement", "conflict",
    "consistent", "inconsistent", "differ", "difference", "compare",
    "comparison", "agree", "agreement", "versus", " vs ", "align", "same as",
    "both papers", "across papers", "all papers", "these papers",
]


def is_comparison_question(question: str) -> bool:
    q = f" {question.lower()} "
    return any(kw in q for kw in COMPARISON_KEYWORDS)


class VaultSession:
    """Holds multiple loaded papers and answers single-paper or
    cross-paper questions against them."""

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        # Ordered so doc_index (used in the synthetic ID remap) is stable.
        self.papers: "Dict[str, PageIndexREMSE]" = {}  # label -> rag instance
        self._order: List[str] = []

    # ── Loading ──────────────────────────────────────────────────────────
    def load(self, pdf_path: str, force_reindex: bool = False, on_status=None) -> str:
        """Ingest (or load from cache) one paper. Returns its session label."""
        rag = PageIndexREMSE(model=self.model)
        if on_status:
            on_status(f"Loading {os.path.basename(pdf_path)}...")
        rag.ingest(pdf_path, force_reindex=force_reindex)

        label = os.path.splitext(os.path.basename(pdf_path))[0]
        # Disambiguate if two files share a basename
        base_label, i = label, 2
        while label in self.papers:
            label = f"{base_label}_{i}"
            i += 1

        self.papers[label] = rag
        self._order.append(label)
        return label

    def doc_index(self, label: str) -> int:
        return self._order.index(label)

    # ── Per-paper question (used for both normal and comparison questions) ─
    def _ask_each_paper(self, question: str) -> Dict[str, dict]:
        """Query every loaded paper independently. Each query runs entirely
        inside that paper's own PageIndexREMSE/TripleStore — no merging here,
        so this is exactly as safe as the existing single-document `chat`."""
        results = {}
        for label, rag in self.papers.items():
            try:
                results[label] = rag.query(
                    question, show_provenance=False, save_result=False
                )
            except Exception as e:
                results[label] = {"answer": f"(error querying {label}: {e})",
                                   "narrative": "", "ordered_atoms": []}
        return results

    # ── Cross-document contradiction / comparison handler ──────────────────
    def compare(self, question: str) -> dict:
        """Runs the real cross-document check: structural triple conflicts
        via Agent7 (on an ID-remapped merged store), plus the existing
        narrative-level debate detector as a softer fallback for
        disagreements that aren't a clean subject/relation collision."""
        if len(self.papers) < 2:
            return {"error": "Need at least 2 papers loaded to compare. Use 'list' to see what's loaded."}

        per_paper = self._ask_each_paper(question)

        merged_triples = []
        for label, result in per_paper.items():
            rag = self.papers[label]
            d_idx = self.doc_index(label)
            used_atom_ids = {a["atom_id"] for a in result.get("ordered_atoms", [])}
            if not used_atom_ids:
                continue
            for t in rag.triples:
                if t.get("atom_id") in used_atom_ids:
                    # Synthetic globally-unique id, scoped to this merged
                    # view only. Original doc_id is left untouched so
                    # Agent7's per-document grouping still reports the
                    # real paper, not the synthetic index.
                    remapped = dict(t)
                    remapped["atom_id"] = d_idx * 1_000_000 + int(t["atom_id"])
                    remapped["_source_label"] = label
                    merged_triples.append(remapped)

        from db.triple_store import TripleStore
        from agents.agent7_contradiction import detect as agent7_detect

        merged_store = TripleStore(merged_triples)
        merged_atom_ids = [t["atom_id"] for t in merged_triples]

        combined_narrative = "\n\n".join(
            f"[{label}]\n{result.get('narrative', '')[:1500]}"
            for label, result in per_paper.items()
        )

        audit = agent7_detect(merged_atom_ids, merged_store, combined_narrative) if merged_triples else {
            "triple_conflicts": [], "cross_doc_conflicts": [], "llm_contradictions": [],
            "contradictions_found": False, "llm_contradictions_found": False, "consistency_score": 1.0,
        }

        # Narrative-level debate check (paper-summary LLM comparison) —
        # only worth the extra calls when the structural check found
        # nothing, since it's a coarser, ungrounded signal.
        debates = []
        if not audit["contradictions_found"]:
            try:
                from intelligence.novelty_pipeline import intelligence_engine
                debates = intelligence_engine.detect_debates(self.papers)
            except Exception as e:
                print_msg(f"[yellow]Debate-narrative fallback failed: {e}[/yellow]")

        return {
            "per_paper_answers": {label: r.get("answer", "") for label, r in per_paper.items()},
            "triple_conflicts": audit["triple_conflicts"],
            "cross_doc_conflicts": audit["cross_doc_conflicts"],
            "llm_contradictions": audit["llm_contradictions"],
            "structural_conflict_found": audit["contradictions_found"],
            "narrative_debates": debates,
            "consistency_score": audit["consistency_score"],
        }

    # ── Normal (non-comparison) multi-paper question ───────────────────────
    def ask_all(self, question: str) -> Dict[str, dict]:
        return self._ask_each_paper(question)

    def stats(self) -> List[dict]:
        return [
            {"label": label, **self.papers[label].get_stats()}
            for label in self._order
        ]
