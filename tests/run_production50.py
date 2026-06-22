"""
Production-50 Benchmark Suite — PageIndex-RE-MSE CRDB RAG System
Runs 50 diverse queries across 7 categories, detects failures,
logs root causes, and exports full results for production1.md.

Run with: python tests/run_production50.py
"""

import sys
import os
import time
import json
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pipeline import PageIndexREMSE

console = Console(highlight=False)

# ─── 50-QUERY BENCHMARK DATASET ───────────────────────────────────────────────
# Ground truth pages are 1-indexed to match the Attention Is All You Need paper.
# Categories:
#   F  = Factual Retrieval
#   M  = Mathematical / Formula
#   C  = Causal Multi-Hop
#   CP = Comparative Analysis
#   H  = Hallucination Rejection (negative control)
#   S  = Structural / Architectural
#   AE = Adversarial Edge Case

PRODUCTION_CASES = [
    # ─── CATEGORY 1: FACTUAL RETRIEVAL (12 cases) ────────────────────────────
    {
        "id": "F-01", "category": "Factual Retrieval",
        "question": "What is the dimension of the keys d_k in Scaled Dot-Product Attention?",
        "ground_truth_pages": {4, 5},
        "expected_keywords": ["64", "d_k", "dimension"],
        "description": "Direct lookup: d_k = 64."
    },
    {
        "id": "F-02", "category": "Factual Retrieval",
        "question": "How many attention heads does the Transformer model use?",
        "ground_truth_pages": {4, 5},
        "expected_keywords": ["8", "heads", "parallel"],
        "description": "h = 8 attention heads."
    },
    {
        "id": "F-03", "category": "Factual Retrieval",
        "question": "What is the model dimension d_model used in the Transformer?",
        "ground_truth_pages": {4, 5},
        "expected_keywords": ["512", "d_model", "dimension"],
        "description": "d_model = 512."
    },
    {
        "id": "F-04", "category": "Factual Retrieval",
        "question": "How many encoder layers are stacked in the base Transformer model?",
        "ground_truth_pages": {3, 4},
        "expected_keywords": ["6", "encoder", "stack", "layer"],
        "description": "N = 6 stacked encoder layers."
    },
    {
        "id": "F-05", "category": "Factual Retrieval",
        "question": "What optimizer is used to train the Transformer model?",
        "ground_truth_pages": {7},
        "expected_keywords": ["adam", "optimizer", "beta"],
        "description": "Adam optimizer with warmup."
    },
    {
        "id": "F-06", "category": "Factual Retrieval",
        "question": "What dataset is used for English-to-German translation?",
        "ground_truth_pages": {6},
        "expected_keywords": ["WMT", "English", "German", "training"],
        "description": "WMT 2014 English-German dataset."
    },
    {
        "id": "F-07", "category": "Factual Retrieval",
        "question": "What BLEU score does the Transformer achieve on English-to-German translation?",
        "ground_truth_pages": {6, 7},
        "expected_keywords": ["28.4", "BLEU", "German"],
        "description": "28.4 BLEU on EN-DE."
    },
    {
        "id": "F-08", "category": "Factual Retrieval",
        "question": "What is the dropout rate applied in the base Transformer model?",
        "ground_truth_pages": {7},
        "expected_keywords": ["0.1", "dropout", "residual"],
        "description": "P_drop = 0.1."
    },
    {
        "id": "F-09", "category": "Factual Retrieval",
        "question": "What is the inner-layer dimensionality d_ff of the feed-forward network?",
        "ground_truth_pages": {4, 5},
        "expected_keywords": ["2048", "d_ff", "feed-forward", "inner"],
        "description": "d_ff = 2048 for position-wise FFN."
    },
    {
        "id": "F-10", "category": "Factual Retrieval",
        "question": "How many decoder layers are stacked in the Transformer?",
        "ground_truth_pages": {3, 4},
        "expected_keywords": ["6", "decoder", "stack"],
        "description": "N = 6 stacked decoder layers."
    },
    {
        "id": "F-11", "category": "Factual Retrieval",
        "question": "What training hardware and how many GPUs were used to train the base model?",
        "ground_truth_pages": {7},
        "expected_keywords": ["GPU", "P100", "8", "training"],
        "description": "8 NVIDIA P100 GPUs."
    },
    {
        "id": "F-12", "category": "Factual Retrieval",
        "question": "What label smoothing value is used during training?",
        "ground_truth_pages": {7},
        "expected_keywords": ["0.1", "label smoothing", "epsilon"],
        "description": "epsilon_ls = 0.1 label smoothing."
    },

    # ─── CATEGORY 2: MATHEMATICAL / FORMULA (8 cases) ───────────────────────
    {
        "id": "M-01", "category": "Mathematical/Formula",
        "question": "Write the Scaled Dot-Product Attention formula.",
        "ground_truth_pages": {4},
        "expected_keywords": ["softmax", "Q", "K", "V", "sqrt"],
        "description": "Core attention formula: softmax(QK^T/sqrt(d_k))V."
    },
    {
        "id": "M-02", "category": "Mathematical/Formula",
        "question": "Why is the dot product divided by the square root of d_k in the attention formula?",
        "ground_truth_pages": {4},
        "expected_keywords": ["scale", "gradient", "vanishing", "large", "magnitude"],
        "description": "Scaling prevents softmax gradient vanishing for large d_k."
    },
    {
        "id": "M-03", "category": "Mathematical/Formula",
        "question": "What is the positional encoding formula used in the Transformer?",
        "ground_truth_pages": {5, 6},
        "expected_keywords": ["sin", "cos", "pos", "frequency", "position"],
        "description": "Sinusoidal PE: sin(pos/10000^(2i/d_model))."
    },
    {
        "id": "M-04", "category": "Mathematical/Formula",
        "question": "What is the formula for the position-wise feed-forward network in the Transformer?",
        "ground_truth_pages": {4, 5},
        "expected_keywords": ["FFN", "ReLU", "W1", "W2", "bias"],
        "description": "FFN(x) = max(0, xW1+b1)W2+b2."
    },
    {
        "id": "M-05", "category": "Mathematical/Formula",
        "question": "How is Multi-Head Attention defined mathematically?",
        "ground_truth_pages": {4},
        "expected_keywords": ["MultiHead", "Concat", "head", "W_O", "projection"],
        "description": "MultiHead(Q,K,V) = Concat(head1..h)W^O."
    },
    {
        "id": "M-06", "category": "Mathematical/Formula",
        "question": "What is the learning rate schedule formula used for training?",
        "ground_truth_pages": {7},
        "expected_keywords": ["warmup", "d_model", "step", "learning rate"],
        "description": "Warmup learning rate schedule based on d_model and step_num."
    },
    {
        "id": "M-07", "category": "Mathematical/Formula",
        "question": "What is the computational complexity of self-attention per layer?",
        "ground_truth_pages": {6},
        "expected_keywords": ["O", "n^2", "n", "d", "complexity"],
        "description": "Self-attention: O(n^2 * d) per layer."
    },
    {
        "id": "M-08", "category": "Mathematical/Formula",
        "question": "What is the maximum path length for self-attention versus recurrence?",
        "ground_truth_pages": {6},
        "expected_keywords": ["O(1)", "O(n)", "path", "maximum", "recurrence"],
        "description": "Self-attention O(1) vs recurrence O(n) max path length."
    },

    # ─── CATEGORY 3: CAUSAL MULTI-HOP (8 cases) ─────────────────────────────
    {
        "id": "C-01", "category": "Causal Multi-Hop",
        "question": "Why does the Transformer utilize masking in the decoder self-attention?",
        "ground_truth_pages": {2, 3, 5},
        "expected_keywords": ["subsequent", "prevent", "attend", "future", "mask"],
        "description": "Masking preserves auto-regressive property by blocking future tokens."
    },
    {
        "id": "C-02", "category": "Causal Multi-Hop",
        "question": "Why were recurrent architectures replaced by attention in sequence-to-sequence tasks?",
        "ground_truth_pages": {1, 2, 6},
        "expected_keywords": ["sequential", "parallel", "long-range", "dependency"],
        "description": "RNNs cannot be parallelized; attention handles long-range dependencies directly."
    },
    {
        "id": "C-03", "category": "Causal Multi-Hop",
        "question": "How does positional encoding allow the Transformer to handle sequence order?",
        "ground_truth_pages": {5, 6},
        "expected_keywords": ["position", "encoding", "frequency", "inject", "order"],
        "description": "Injects positional signal since attention has no inherent order."
    },
    {
        "id": "C-04", "category": "Causal Multi-Hop",
        "question": "Why is residual connection important in each sub-layer of the Transformer?",
        "ground_truth_pages": {3, 4},
        "expected_keywords": ["residual", "gradient", "vanishing", "add", "normalize"],
        "description": "Residuals prevent vanishing gradients in deep stacked layers."
    },
    {
        "id": "C-05", "category": "Causal Multi-Hop",
        "question": "Why does the encoder-decoder attention use keys and values from the encoder?",
        "ground_truth_pages": {3, 4},
        "expected_keywords": ["encoder", "keys", "values", "cross-attention", "output"],
        "description": "Decoder attends to full encoder output to align source with target."
    },
    {
        "id": "C-06", "category": "Causal Multi-Hop",
        "question": "Why does using multiple attention heads improve learning compared to single attention?",
        "ground_truth_pages": {4},
        "expected_keywords": ["jointly", "subspace", "different", "representation", "position"],
        "description": "Multiple heads learn diverse representation subspaces jointly."
    },
    {
        "id": "C-07", "category": "Causal Multi-Hop",
        "question": "Why does label smoothing hurt perplexity but improve BLEU?",
        "ground_truth_pages": {7},
        "expected_keywords": ["uncertainty", "smooth", "perplexity", "BLEU", "translation"],
        "description": "Smoothing teaches uncertainty, reducing overconfidence."
    },
    {
        "id": "C-08", "category": "Causal Multi-Hop",
        "question": "How does the Transformer achieve constant path length between positions?",
        "ground_truth_pages": {6},
        "expected_keywords": ["O(1)", "path", "distance", "constant", "signal"],
        "description": "Self-attention connects all positions with a single O(1) operation."
    },

    # ─── CATEGORY 4: COMPARATIVE ANALYSIS (6 cases) ─────────────────────────
    {
        "id": "CP-01", "category": "Comparative Analysis",
        "question": "How does the Transformer's approach to sequence modeling differ from RNNs and LSTMs?",
        "ground_truth_pages": {1, 2, 6},
        "expected_keywords": ["recurrent", "sequential", "parallel", "attention", "long-range"],
        "description": "Attention replaces recurrence with parallelizable global computation."
    },
    {
        "id": "CP-02", "category": "Comparative Analysis",
        "question": "What are the advantages of self-attention over convolutional layers for sequence tasks?",
        "ground_truth_pages": {6},
        "expected_keywords": ["convolution", "path length", "resolution", "kernel", "attention"],
        "description": "Attention achieves O(1) path; convolutions need O(log n) or O(n) layers."
    },
    {
        "id": "CP-03", "category": "Comparative Analysis",
        "question": "How does sinusoidal positional encoding compare to learned positional encoding?",
        "ground_truth_pages": {5, 6},
        "expected_keywords": ["sinusoidal", "learned", "extrapolate", "relative", "position"],
        "description": "Sinusoidal allows extrapolation; learned is equivalent in experiments."
    },
    {
        "id": "CP-04", "category": "Comparative Analysis",
        "question": "How does the large Transformer model differ from the base model in architecture?",
        "ground_truth_pages": {7},
        "expected_keywords": ["big", "base", "h=16", "d_ff", "512", "dropout"],
        "description": "Big model: h=16, d_model=1024, d_ff=4096, dropout=0.3."
    },
    {
        "id": "CP-05", "category": "Comparative Analysis",
        "question": "How does the Transformer's BLEU score compare to prior state-of-the-art models?",
        "ground_truth_pages": {6, 7},
        "expected_keywords": ["BLEU", "outperform", "previous", "state-of-the-art"],
        "description": "Transformer big: 28.4 BLEU EN-DE, surpassing previous SOTA."
    },
    {
        "id": "CP-06", "category": "Comparative Analysis",
        "question": "How does restricted self-attention differ from full self-attention?",
        "ground_truth_pages": {6},
        "expected_keywords": ["restricted", "neighborhood", "r", "local", "complexity"],
        "description": "Restricted self-attention limits neighborhood to reduce O(n^2) cost."
    },

    # ─── CATEGORY 5: HALLUCINATION REJECTION (8 cases) ──────────────────────
    {
        "id": "H-01", "category": "Hallucination Rejection",
        "question": "What are the quantum computing experiments mentioned in this paper?",
        "ground_truth_pages": set(), "is_negative": True,
        "expected_keywords": [],
        "description": "Quantum computing is not in this paper — must be rejected."
    },
    {
        "id": "H-02", "category": "Hallucination Rejection",
        "question": "What does the paper say about reinforcement learning from human feedback?",
        "ground_truth_pages": set(), "is_negative": True,
        "expected_keywords": [],
        "description": "RLHF is not discussed in this 2017 paper — must be rejected."
    },
    {
        "id": "H-03", "category": "Hallucination Rejection",
        "question": "What is the Transformer's performance on image classification benchmarks?",
        "ground_truth_pages": set(), "is_negative": True,
        "expected_keywords": [],
        "description": "Paper is NLP only — no image tasks. Must reject."
    },
    {
        "id": "H-04", "category": "Hallucination Rejection",
        "question": "What is GPT-4's architecture described in this paper?",
        "ground_truth_pages": set(), "is_negative": True,
        "expected_keywords": [],
        "description": "GPT-4 did not exist in 2017 — must reject."
    },
    {
        "id": "H-05", "category": "Hallucination Rejection",
        "question": "Describe the graph neural network component of the Transformer architecture.",
        "ground_truth_pages": set(), "is_negative": True,
        "expected_keywords": [],
        "description": "No GNN in this paper — must reject."
    },
    {
        "id": "H-06", "category": "Hallucination Rejection",
        "question": "What blockchain technology does this paper propose for secure model training?",
        "ground_truth_pages": set(), "is_negative": True,
        "expected_keywords": [],
        "description": "Blockchain not in this paper — must reject."
    },
    {
        "id": "H-07", "category": "Hallucination Rejection",
        "question": "What is the paper's proposed method for federated learning across edge devices?",
        "ground_truth_pages": set(), "is_negative": True,
        "expected_keywords": [],
        "description": "Federated learning is not in this paper — must reject."
    },
    {
        "id": "H-08", "category": "Hallucination Rejection",
        "question": "What speech recognition results does this paper report?",
        "ground_truth_pages": set(), "is_negative": True,
        "expected_keywords": [],
        "description": "No speech recognition results — paper is MT only. Must reject."
    },

    # ─── CATEGORY 6: STRUCTURAL / ARCHITECTURAL (5 cases) ───────────────────
    {
        "id": "S-01", "category": "Structural/Architectural",
        "question": "Describe the overall encoder-decoder structure of the Transformer.",
        "ground_truth_pages": {2, 3, 4},
        "expected_keywords": ["encoder", "decoder", "stack", "attention", "feed-forward"],
        "description": "High-level encoder-decoder architecture overview."
    },
    {
        "id": "S-02", "category": "Structural/Architectural",
        "question": "What are the three different ways attention is used within the Transformer?",
        "ground_truth_pages": {4, 5},
        "expected_keywords": ["encoder", "decoder", "cross", "self-attention", "three"],
        "description": "Three attention uses: encoder self, decoder self (masked), encoder-decoder cross."
    },
    {
        "id": "S-03", "category": "Structural/Architectural",
        "question": "What sub-layers does each encoder layer contain?",
        "ground_truth_pages": {3},
        "expected_keywords": ["multi-head", "feed-forward", "residual", "layer norm"],
        "description": "Each encoder layer: multi-head attention + FFN + residual + LN."
    },
    {
        "id": "S-04", "category": "Structural/Architectural",
        "question": "Where is layer normalization applied within each sub-layer?",
        "ground_truth_pages": {3},
        "expected_keywords": ["layer norm", "residual", "each", "output", "sublayer"],
        "description": "LayerNorm applied after residual connection in each sub-layer."
    },
    {
        "id": "S-05", "category": "Structural/Architectural",
        "question": "How are the Transformer weights shared between the embedding layers and the final linear layer?",
        "ground_truth_pages": {5},
        "expected_keywords": ["shared", "embedding", "linear", "weight", "pre-softmax"],
        "description": "Input/output embedding and pre-softmax linear share weights."
    },

    # ─── CATEGORY 7: ADVERSARIAL EDGE CASES (3 cases) ───────────────────────
    {
        "id": "AE-01", "category": "Adversarial Edge Case",
        "question": "What does this paper say about attention is all you need?",
        "ground_truth_pages": {1, 2},
        "expected_keywords": ["attention", "recurrence", "convolution", "solely"],
        "description": "Tricky: the title itself is the answer — attention replacing recurrence."
    },
    {
        "id": "AE-02", "category": "Adversarial Edge Case",
        "question": "What happens when d_k is very large in the dot product attention?",
        "ground_truth_pages": {4},
        "expected_keywords": ["vanishing", "gradient", "large", "magnitude", "softmax"],
        "description": "Edge case: large d_k causes softmax saturation and gradient issues."
    },
    {
        "id": "AE-03", "category": "Adversarial Edge Case",
        "question": "Is additive attention better than dot-product attention according to this paper?",
        "ground_truth_pages": {4},
        "expected_keywords": ["additive", "dot-product", "faster", "practice", "similar"],
        "description": "Nuanced comparison — dot-product wins in practice due to matrix multiplication speed."
    },
]


# ─── FAILURE DETECTION ────────────────────────────────────────────────────────

REJECTION_WORDS = [
    "not mention", "not discuss", "not explicitly", "no mention",
    "not found", "does not contain", "does not provide", "not described",
    "not covered", "no information", "not addressed", "not included",
    "not present", "not referenced", "not related", "cannot find",
    "no reference", "no evidence", "outside the scope", "not in the",
    "does not address", "not available in", "not part of", "not an aspect",
    "is not discussed", "is not mentioned", "is not provided",
]

HALLUCINATION_TRIGGERS = [
    # Phrases that indicate the model is describing an out-of-scope topic as if real
    "quantum computing experiments", "quantum hardware",
    "speech recognition results", "speech recognition systems",
    "federated learning across edge", "blockchain technology",
    "image classification benchmark", "reinforcement learning from human",
    "gpt-4\'s architecture is", "graph neural network component",
]

FAILURE_REASONS = {
    "zero_recall": "0% page overlap (RRF / Navigator navigational failure)",
    "low_accuracy": "Low factual overlap score (< 40% keyword match)",
    "false_acceptance": "Unsafe hallucination or out-of-scope query acceptance",
    "pipeline_failure": "Empty text response generated by Agent10 / Synthesizer",
    "timeout": "Query latency exceeded safety threshold (> 200 seconds)",
    "inflated_recall": "Recall >= 99% achieved by over-retrieving most of the document "
                        "(precision < 30%) rather than precise targeting",
}

# Recall is trivially gameable by returning most/all pages. A precision floor
# catches that: high recall + low precision means the system dumped the
# document rather than navigating to it.
INFLATED_RECALL_PRECISION_FLOOR = 0.3


def classify_failure(case: dict, r: dict) -> list:
    """Returns list of failure codes for a result."""
    failures = []
    answer_text = r.get("answer_snippet", "").lower()  # FIX: was r.get("answer") which is always missing
    if case.get("is_negative"):
        if not r["hallucination_pass"]:
            failures.append("false_acceptance")
    else:
        if r["recall"] == 0.0:
            failures.append("zero_recall")
        if r["accuracy"] < 0.4 and not case.get("is_negative"):
            failures.append("low_accuracy")
        precision = r.get("precision")
        if precision is not None and r["recall"] >= 0.99 and precision < INFLATED_RECALL_PRECISION_FLOOR:
            failures.append("inflated_recall")
    if r["elapsed"] > 200:
        failures.append("timeout")
    if not answer_text.strip():  # FIX: now checks the correct key
        failures.append("pipeline_failure")
    return failures


# ─── MAIN RUNNER ──────────────────────────────────────────────────────────────

def run_production50(pdf_path: str):
    run_time = datetime.datetime.now()

    console.print(Panel(
        "[bold cyan]PageIndex-RE-MSE CRDB -- Production-50 Benchmark[/bold cyan]\n"
        f"Target PDF: [bold]{os.path.basename(pdf_path)}[/bold]\n"
        f"Cases: [bold green]50 queries across 7 categories[/bold green]\n"
        f"Started: {run_time.strftime('%Y-%m-%d %H:%M:%S')}",
        title="[*] Production-Level Evaluation Console",
        expand=False
    ))

    if not os.path.exists(pdf_path):
        console.print(f"[bold red]Error: PDF not found: {pdf_path}[/bold red]")
        sys.exit(1)

    rag = PageIndexREMSE()
    with console.status("[bold yellow]Loading document index...[/bold yellow]"):
        rag.ingest(pdf_path)

    results = []
    all_failures = []
    category_stats = {}

    total = len(PRODUCTION_CASES)

    for i, case in enumerate(PRODUCTION_CASES):
        console.print(
            "\n[bold magenta]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold magenta]"
        )
        console.print(
            f"[bold blue][{case['id']}] ({i+1}/{total}) {case['category']}[/bold blue]"
        )
        console.print(f"[white]Q:[/white] \"{case['question']}\"")
        console.print(f"[dim]{case['description']}[/dim]\n")

        start = time.time()
        try:
            res = rag.query(
                question=case["question"],
                show_provenance=False,
                save_result=False
            )
            elapsed = time.time() - start
            answer = res.get("answer", "")
            provenance = res.get("provenance", {})
            pages_retrieved = set(provenance.get("pages_referenced", []))
            trust_level = res.get("trust_level", "low")
            pipeline_grade = res.get("pipeline_grade", "F")
            confidence = res.get("confidence", 0.0)
        except Exception as e:
            elapsed = time.time() - start
            answer = ""
            pages_retrieved = set()
            trust_level = "low"
            pipeline_grade = "F"
            confidence = 0.0
            console.print(f"[bold red]⚠ Exception: {e}[/bold red]")

        # ── Compute Metrics ───────────────────────────────────────────────────
        is_neg = case.get("is_negative", False)

        # Recall
        if is_neg:
            retrieval_pass = (len(pages_retrieved) <= 3 or trust_level in ("low", "medium"))
            recall_score = 1.0 if retrieval_pass else 0.0
            precision_score = None  # not meaningful for negative controls (no ground-truth pages)
            f1_score = None
        else:
            gt = case["ground_truth_pages"]
            hit = pages_retrieved.intersection(gt)
            recall_score = len(hit) / len(gt) if gt else 1.0
            # Precision: of the pages actually retrieved, how many were relevant?
            # This is what keeps recall honest — without it, retrieving the whole
            # document trivially maxes out recall.
            if len(pages_retrieved) == 0:
                precision_score = 1.0 if not gt else 0.0
            else:
                precision_score = len(hit) / len(pages_retrieved)
            f1_score = (
                2 * precision_score * recall_score / (precision_score + recall_score)
                if (precision_score + recall_score) > 0 else 0.0
            )

        # Accuracy
        if is_neg:
            accuracy_pass = any(w in answer.lower() for w in REJECTION_WORDS) or trust_level == "low"
            accuracy_score = 1.0 if accuracy_pass else 0.0
        else:
            kws = case.get("expected_keywords", [])
            matches = [kw for kw in kws if kw.lower() in answer.lower()]
            accuracy_score = len(matches) / len(kws) if kws else 1.0

        # ── Hallucination
        if is_neg:
            answer_lower = answer.lower()
            # PRIMARY CHECK: Does the answer contain explicit rejection language?
            clearly_rejects = any(w in answer_lower for w in REJECTION_WORDS)
            # SECONDARY CHECK: Does the answer describe the hallucinated topic as real?
            clearly_hallucinates = any(t in answer_lower for t in HALLUCINATION_TRIGGERS)
            # An answer passes if it clearly rejects OR has low confidence and doesn't hallucinate
            hallucination_pass = (clearly_rejects and not clearly_hallucinates) or (
                confidence < 0.65 and not clearly_hallucinates
            )
        else:
            hallucination_pass = True

        r = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "is_negative": is_neg,
            "recall": recall_score,
            "precision": precision_score,
            "f1": f1_score,
            "accuracy": accuracy_score,
            "hallucination_pass": hallucination_pass,
            "elapsed": elapsed,
            "grade": pipeline_grade,
            "trust": trust_level,
            "confidence": confidence,
            "pages": sorted(list(pages_retrieved)),
            "num_pages_retrieved": len(pages_retrieved),
            "ground_truth_pages": sorted(list(case.get("ground_truth_pages", set()))),
            "answer_snippet": answer[:300],
        }

        # ── Failure Classification ────────────────────────────────────────────
        failures = classify_failure(case, r)
        r["failures"] = failures
        if failures:
            all_failures.append({
                "id": case["id"],
                "category": case["category"],
                "failures": failures,
                "recall": recall_score,
                "pages_retrieved": sorted(list(pages_retrieved)),
                "ground_truth": sorted(list(case.get("ground_truth_pages", set()))),
                "answer_snippet": answer[:200],
            })

        results.append(r)

        # ── Category rollup ──────────────────────────────────────────────────
        cat = case["category"]
        if cat not in category_stats:
            category_stats[cat] = {
                "count": 0, "recall_sum": 0.0, "accuracy_sum": 0.0, "safety_pass": 0,
                "precision_sum": 0.0, "f1_sum": 0.0, "precision_n": 0,
            }
        category_stats[cat]["count"] += 1
        category_stats[cat]["recall_sum"] += recall_score
        category_stats[cat]["accuracy_sum"] += accuracy_score
        if hallucination_pass:
            category_stats[cat]["safety_pass"] += 1
        if precision_score is not None:
            category_stats[cat]["precision_sum"] += precision_score
            category_stats[cat]["f1_sum"] += f1_score
            category_stats[cat]["precision_n"] += 1

        # ── Per-query output ─────────────────────────────────────────────────
        status_icon = "✅" if not failures else "❌"
        console.print(f"{status_icon} [bold]Pages retrieved:[/bold] {sorted(list(pages_retrieved))} "
                       f"(Target: {sorted(list(case.get('ground_truth_pages', set())))})")
        prec_str = f"{precision_score*100:.0f}%" if precision_score is not None else "N/A"
        console.print(f"   Recall:     [bold]{recall_score*100:.0f}%[/bold]  "
                       f"Precision: [bold]{prec_str}[/bold]  "
                       f"Accuracy: [bold]{accuracy_score*100:.0f}%[/bold]  "
                       f"Safety: {'[green]SAFE[/green]' if hallucination_pass else '[red]RISKY[/red]'}  "
                       f"Grade: {pipeline_grade}  Time: {elapsed:.1f}s")
        if failures:
            for f in failures:
                console.print(f"   [red]⚠ FAILURE: {f} — {FAILURE_REASONS.get(f, 'Unknown')}[/red]")

    # ─── FINAL SCORECARD ──────────────────────────────────────────────────────
    console.print(f"\n[bold magenta]{'━'*60}[/bold magenta]")
    console.print(Panel("[bold green]PRODUCTION-50 FINAL SCORECARD[/bold green]", expand=False))

    score_table = Table(
        title="Production-50 Benchmark — Category Breakdown",
        show_header=True, header_style="bold cyan"
    )
    score_table.add_column("Category", width=28)
    score_table.add_column("Count", justify="center")
    score_table.add_column("Avg Recall", justify="center")
    score_table.add_column("Avg Precision", justify="center")
    score_table.add_column("Avg Accuracy", justify="center")
    score_table.add_column("Safety Rate", justify="center")

    for cat, st in category_stats.items():
        n = st["count"]
        avg_r = st["recall_sum"] / n
        avg_a = st["accuracy_sum"] / n
        safety = st["safety_pass"] / n
        avg_p_str = f"{(st['precision_sum'] / st['precision_n'])*100:.1f}%" if st["precision_n"] else "N/A"
        score_table.add_row(
            cat,
            str(n),
            f"{avg_r*100:.1f}%",
            avg_p_str,
            f"{avg_a*100:.1f}%",
            f"{safety*100:.1f}%"
        )
    console.print(score_table)

    # Aggregates
    total_n = len(results)
    avg_recall   = sum(r["recall"]  for r in results) / total_n
    avg_accuracy = sum(r["accuracy"] for r in results) / total_n
    total_safe   = sum(1 for r in results if r["hallucination_pass"])
    total_fail   = len(all_failures)
    total_time   = sum(r["elapsed"] for r in results)

    precision_results = [r["precision"] for r in results if r["precision"] is not None]
    f1_results = [r["f1"] for r in results if r["f1"] is not None]
    avg_precision = (sum(precision_results) / len(precision_results)) if precision_results else None
    avg_f1 = (sum(f1_results) / len(f1_results)) if f1_results else None
    avg_precision_str = f"{avg_precision*100:.1f}%" if avg_precision is not None else "N/A"
    avg_f1_str = f"{avg_f1*100:.1f}%" if avg_f1 is not None else "N/A"

    console.print(Panel(
        f"[bold white]PRODUCTION-50 AGGREGATE RESULTS[/bold white]\n\n"
        f"  Total Queries Run         : [bold yellow]{total_n}[/bold yellow]\n"
        f"  Average Page Recall       : [bold yellow]{avg_recall*100:.1f}%[/bold yellow]\n"
        f"  Average Page Precision    : [bold yellow]{avg_precision_str}[/bold yellow] (of retrieved pages, % actually relevant — "
        f"computed over {len(precision_results)} non-negative-control cases)\n"
        f"  Average Recall/Precision F1: [bold yellow]{avg_f1_str}[/bold yellow]\n"
        f"  Average Factual Accuracy  : [bold yellow]{avg_accuracy*100:.1f}%[/bold yellow]\n"
        f"  Hallucination Safety Rate : [bold green]{total_safe/total_n*100:.1f}%[/bold green]\n"
        f"  Total Failures Detected   : [bold yellow]{total_fail}[/bold yellow]\n"
        f"  Total Execution Time      : [bold white]{total_time:.1f}s ({total_time/60:.1f} min)[/bold white]",
        title="[bold] Aggregate RAG Quality[/bold]",
        expand=False
    ))

    neg_results = [r for r in results if r["is_negative"]]
    neg_safety_rate = (
        sum(1 for r in neg_results if r["hallucination_pass"]) / len(neg_results)
        if neg_results else None
    )
    if neg_results:
        console.print(Panel(
            f"[bold white]Note:[/bold white] hallucination_pass defaults to True for the "
            f"{total_n - len(neg_results)} non-negative-control questions (it isn't evaluated for them), "
            f"so the {total_safe/total_n*100:.1f}% overall safety rate above is diluted.\n"
            f"The real signal is the negative-control-only rate: "
            f"[bold {'green' if neg_safety_rate >= 0.8 else 'red'}]{neg_safety_rate*100:.1f}%[/bold] "
            f"across {len(neg_results)} actual hallucination-rejection cases.",
            title="[bold]Hallucination Safety — Negative Controls Only[/bold]",
            expand=False
        ))

    if all_failures:
        console.print(Panel(
            "[bold red]FAILURES REQUIRING ATTENTION[/bold red]\n\n" +
            "\n".join([
                f"  [{f['id']}] {f['category']}: {', '.join(f['failures'])}"
                for f in all_failures
            ]),
            title="⚠ Failure Report",
            expand=False
        ))

    # ─── EXPORT JSON FOR REPORT GENERATION ───────────────────────────────────
    output = {
        "run_timestamp": run_time.isoformat(),
        "pdf": os.path.basename(pdf_path),
        "total_queries": total_n,
        "avg_recall": round(avg_recall, 4),
        "avg_precision": round(avg_precision, 4) if avg_precision is not None else None,
        "avg_f1": round(avg_f1, 4) if avg_f1 is not None else None,
        "avg_accuracy": round(avg_accuracy, 4),
        "safety_rate": round(total_safe / total_n, 4),
        "negative_control_safety_rate": round(neg_safety_rate, 4) if neg_safety_rate is not None else None,
        "negative_control_count": len(neg_results),
        "total_failures": total_fail,
        "total_time_seconds": round(total_time, 1),
        "category_stats": {
            cat: {
                "count": st["count"],
                "avg_recall": round(st["recall_sum"] / st["count"], 4),
                "avg_precision": round(st["precision_sum"] / st["precision_n"], 4) if st["precision_n"] else None,
                "avg_accuracy": round(st["accuracy_sum"] / st["count"], 4),
                "safety_rate": round(st["safety_pass"] / st["count"], 4),
            }
            for cat, st in category_stats.items()
        },
        "failures": all_failures,
        "results": results,
    }

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "production50_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    console.print(f"\n[bold green]✅ JSON results saved to:[/bold green] {json_path}")
    return output


if __name__ == "__main__":
    default_pdf = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "uploads",
        "1cf6b031-c99_NIPS-2017-attention-is-all-you-need-Paper.pdf"
    )
    pdf_arg = sys.argv[1] if len(sys.argv) > 1 else default_pdf
    run_production50(pdf_arg)
