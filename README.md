# Vasis AI: PageIndex-RE-MSE CRDB Hybrid RAG Engine

A production-ready, highly secure, **100% offline local RAG system** combining macro-level document tree navigation, micro-level atomic text reconstruction, and an **autonomous 11-Agent Contextual Reconstruction Database (CRDB)** orchestration engine.

The system is fully self-contained, requiring zero external APIs, keys, or cloud dependencies, and runs entirely locally via **Ollama**.

---

## ─── KEY TECHNOLOGIES USED ───────────────────────────────────────

The system is engineered using state-of-the-art local AI architecture components:

### 1. Model Orchestration & Local LLM Host
*   **Ollama (Local Server @ Port 11435):** Acts as the offline inference gateway, running open-weights LLMs with hardware-optimized acceleration.
*   **Agentic Model (`qwen2.5-coder:3b`):** Highly optimized for low-latency instruction following, query decomposition, and structured JSON parsing.
*   **Reasoning Model (`deepseek-llm:7b`):** Deep reasoning model dynamically routed for factual synthesis, contradiction audits, and final grounding verification.

### 2. Micro & Macro Indexing Layer (Vectorless RAG)
*   **PageIndex Tree Builder:** Builds a macro-level logical tree representing section summaries, titles, and page boundaries (avoiding global vector searches that cause context drift).
*   **`AtomStore`:** An in-memory, bi-directionally linked data structure containing 50-100 token text segments (atoms) with sequential pointers.
*   **`BM25Index`:** A high-precision keyword retrieval index scoped explicitly within the PageIndex-validated tree nodes to eliminate irrelevant pages.
*   **`TripleStore` & `CausalStore`:** Extract semantic Subject-Relation-Object knowledge triples and form causal reasoning graphs for multi-hop link traversal.

### 3. Core Software Stack & Libraries
*   **Python 3.9+ / 3.11:** Core execution environment.
*   **PyMuPDF (`fitz`):** Light, ultra-fast local PDF processing and text extraction engine.
*   **FastAPI & Uvicorn (`api.py`):** Serves backend REST endpoints supporting UI integration.
*   **Rich CLI Dashboard & Panels:** Delivers modern visual terminal interfaces, real-time agent logging, and benchmark reporting.

### 4. Experience & Self-Correction Engine
*   **`ExperienceStore` & `FeedbackIndex`:** Persists previous user query signatures and agent audit grades to enable instantaneous similarity warm starts.

---

## ─── THE 11-AGENT CRDB ARCHITECTURE ──────────────────────────────

At the heart of the system is the **Contextual Reconstruction Database (CRDB)**, managed by an autonomous multi-agent swarm:

```
[User Query]
    │
    ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ AGENT 10: SuperAgent (Master Planner & Router)              │
  └──────┬──────────────────────┬────────────────────────┬──────┘
         │                      │                        │
         ▼                      ▼                        ▼
┌─────────────────┐    ┌─────────────────┐      ┌─────────────────┐
│ Agent 1: Router │    │ Agent 2: Decomp │      │ Agent 3: Nav    │
│ rewrite/intent  │    │ sub-queries     │      │ macro tree navigation
└─────────────────┘    └─────────────────┘      └─────────────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │ Agent 4: BM25   │
                                                │ anchor retrieval│
                                                └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │ Agent 5: RE-MSE │
                                                │ stateful expand │
                                                └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │ Answer Synthesizer
                                                └────────┬────────┘
                                                         │
                                 ┌───────────────────────┴───────────────────────┐
                                 ▼                                               ▼
                      ┌──────────────────────┐                       ┌──────────────────────┐
                      │ Agent 6: Grounding   │                       │ Agent 7: Contradict  │
                      │ validator            │                       │ triple conflict audit│
                      └──────────┬───────────┘                       └──────────────────────┘
                                 │
                   (If ungrounded: triggers Requery Loop)
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │ Agent 8: Novelty     │
                      │ causal synthesis     │
                      └──────────┬───────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │ Agent 9: Calibrator  │
                      │ trust calibration    │
                      └──────────┬───────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │ Agent 11: Feedback   │
                      │ experience logging   │
                      └──────────────────────┘
```

1.  **Agent 1: Router** — Rewrites queries, normalizes terminology, and identifies query intent.
2.  **Agent 2: Decomposer** — Splits multi-part questions into atomic sub-questions.
3.  **Agent 3: Navigator** — Guides logical traversal down the PageIndex section tree.
4.  **Agent 4: Retrieval Agent** — Fetches primary "anchor" atoms from the index.
5.  **Agent 5: Stateful Expander** — Performs progressive RE-MSE passes using a shared state cache.
6.  **Agent 6: Grounding Validator** — Verifies the final response against exact atoms; raises a `RequerySignal` to retry retrieval upon failure.
7.  **Agent 7: Contradiction Auditor** — Scans the generated text for semantic contradictions and structural anomalies.
8.  **Agent 8: Novelty Synthesizer** — Identifies new multi-hop causal connections along the knowledge graph.
9.  **Agent 9: Trust Calibrator** — Evaluates and compiles an objective confidence and trust rating.
10. **Agent 10: SuperAgent** — Serves as the central workflow orchestrator, managing retries, routing, and error loops.
11. **Agent 11: Feedback Compiler** — Logs execution metrics and grades to persistent storage.

---

## ─── INSTALLATION & QUICK START ──────────────────────────────────

### Prerequisites
*   Install [Ollama](https://ollama.ai)
*   Pull the models:
    ```bash
    ollama pull qwen2.5-coder:3b
    ollama pull deepseek-llm:7b
    ```

### Setup
1.  Navigate to the workspace and install requirements:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the CLI Commands:
    *   **Index a PDF:**
        ```bash
        python main.py index uploads/your_document.pdf
        ```
    *   **Ask a Question:**
        ```bash
        python main.py ask uploads/your_document.pdf "What are the main findings?"
        ```
    *   **Interactive Chat Panel:**
        ```bash
        python main.py chat uploads/your_document.pdf
        ```

3.  **Windows clickable Control Panel (`start_system.bat`):**
    Double-click the `start_system.bat` file in Windows Explorer to open a visual CLI launcher where you can run the Web API, start chats, index new PDFs, run diagnostic benchmarks, and monitor Ollama.

---

## ─── DIAGNOSTIC BENCHMARKS ────────────────────────────────────────

The system includes a dedicated, ASCII-safe RAG evaluation suite to verify **recall recall**, **precision accuracy**, and **hallucination rejection**.

```bash
python tests/run_benchmarks.py
```

This sequentially tests:
1.  **Factual accuracy** against target values (e.g. verifying exact key dimensions like $d_k=64$).
2.  **Hallucination safety** via out-of-scope negative controls (forcing the grounding and calibration agents to reject quantum computing queries in standard papers).
3.  **Multi-Hop Traversal recall** across adjacent document sections.
