# Vasis AI: PageIndex-RE-MSE CRDB Hybrid RAG Engine

A production-ready, highly secure, **100% offline local RAG system** combining macro-level document tree navigation, micro-level atomic text reconstruction, and an **autonomous 14-Agent Contextual Reconstruction Database (CRDB)** orchestration engine.

The system is fully self-contained, requiring zero external APIs, keys, or cloud dependencies, and runs entirely locally via **Ollama**.

---

## SYSTEM FEATURES & CAPABILITIES 

*   **Vectorless RAG Architecture:** Replaces traditional vector embeddings and flat chunking with a dual-layer index that preserves document hierarchy and sequence.
*   **Dual-Model Orchestration Swarm:** Dynamically routes tasks between lightweight agentic models (`qwen2.5-coder:3b` for fast, structured planning and parsing) and deep reasoning models (`deepseek-llm:7b` for qualitative synthesis and grounding audits).
*   **RE-MSE Progressive Context Reconstruction:** Walks sentence-level sequence nodes bidirectionally to stitch seamless, gap-free reading contexts.
*   **Self-Healing JSON Hardening:** Employs Ollama token grammar schemas and a regex-based healing parser to ensure 100% stable execution.
*   **Automatic Output Ingestion:** Automatically saves generated papers and implementation guides into clean markdown formats under `outputs/` with timestamped, query-derived filenames.
*   **Interactive Control Panel:** Clickable Windows Batch Control Panel (`start_system.bat`) to launch servers, run benchmarks, open chat clients, and monitor VRAM.
*   **Advanced Terminal Interfaces:** Includes a styled, autocomplete-enabled Interactive CLI Shell (`vasis_cli.py`) and a full-screen, asynchronous Textual Terminal User Interface (TUI, `vasis_shell.py`) for comprehensive RAG querying, agent inspection, and multi-document vault operations.

---

## KEY TECHNOLOGIES USED 

### 1. Model Orchestration & Local LLM Host
*   **Ollama (Local Server @ Port 11435):** Offline inference gateway running open-weights LLMs with hardware acceleration.
*   **Agentic Model (`qwen2.5-coder:3b`):** Optimized for structured planning, schema compliance, tree routing, and parsing.
*   **Reasoning Model (`deepseek-llm:7b`):** Routed dynamically for natural language audit critiques, contradictions, and final answer generation.

### 2. Micro & Macro Indexing Layer
*   **PageIndex Tree Builder:** Builds a macro-level logical tree representing section summaries, topics, and page boundaries (avoiding semantic drift).
*   **`AtomStore`:** An in-memory, doubly-linked data store containing 50-100 token text segments (atoms) with sequential pointers.
*   **`BM25Index`:** A high-precision keyword retrieval index scoped explicitly within PageIndex-validated nodes to eliminate irrelevant pages.
*   **`TripleStore` & `CausalStore`:** Extract semantic Subject-Relation-Object knowledge triples and form causal reasoning graphs for multi-hop link traversal.

### 3. Software Stack & Core Libraries
*   **Python 3.9+ / 3.11:** Core runtime.
*   **PyMuPDF (`fitz`):** Ultra-fast local PDF processing and text extraction.
*   **FastAPI & Uvicorn (`api.py`):** REST API endpoints supporting frontend and control panel integrations.
*   **Rich CLI Dashboard, Autocomplete Shell, & Textual TUI:** Provides advanced visual terminal panels (`rich`), an autocomplete-enabled interactive command line interface (`prompt_toolkit`), a full-screen asynchronous terminal dashboard (`textual`), and real-time agent routing visualization.

---

## THE 14-AGENT CRDB SWARM ORCHESTRATION 
At the heart of the system is the **Contextual Reconstruction Database (CRDB)**, managed by an autonomous multi-agent swarm:

```mermaid
graph TD
    query["User Query"] --> super["Agent 10: SuperAgent"]
    
    subgraph Ingestion & Cache
        warm["FeedbackIndex (Warm Start)"]
    end

    subgraph Query Understanding
        a1["Agent 1: Router (Intent & Rewrite)"]
        a2["Agent 2: Decomposer (Sub-Queries)"]
    end

    subgraph Retrieval & Expansion
        a3["Agent 3: Navigator (Tree Filter)"]
        a4["Agent 4: Retrieval (Anchor Search)"]
        a5["Agent 5: Expansion (RE-MSE Walk)"]
    end

    subgraph Answer & Grounding Validation
        gen["Answer Synthesizer"]
        a6["Agent 6: Grounding Validator"]
        requery{"Requery Loop Needed?"}
    end

    subgraph Post-Generation Audits
        a7["Agent 7: Contradiction Auditor"]
        a8["Agent 8: Temporal (Timeline Auditor)"]
        a11["Agent 11: Causal Synthesis"]
        a9["Agent 9: Trust Calibrator"]
    end

    super --> warm
    super --> a1
    super --> a2
    super --> a3
    a3 --> |"Scoped Page Ranges"| a4
    a4 --> |"Anchor Atoms"| a5
    a5 --> |"Stitched Context"| gen
    gen --> a6
    a6 --> requery
    requery --> |"Yes: RequerySignal"| super
    requery --> |"No"| a7
    a7 --> a8
    a8 --> a11
    a11 --> a9
    a9 --> |"Trust Level Tag"| out["Final Answer & Audit Trail"]
```

### Agent Directory & Workflow Roles
1.  **Agent 1: Router** — Identifies intent, normalizes terminology, and classifies bibliography targets.
2.  **Agent 2: Decomposer** — Splits multi-part questions into atomic sub-queries.
3.  **Agent 3: Navigator** — Scopes search ranges top-down via the PageIndex summary tree.
4.  **Agent 4: Retrieval Agent** — Fetches primary "anchor" atoms from scoped candidate ranges.
5.  **Agent 5: Stateful Expander** — Walks the doubly-linked `AtomStore` bidirectionally to stitch gaps.
6.  **Agent 6: Grounding Validator** — Compares output against retrieved atoms; triggers the **Requery Loop** on failure.
7.  **Agent 7: Contradiction Auditor** — Scans the generated text against triples to isolate factual conflicts.
8.  **Agent 8: Temporal Agent** — Analyzes chronological timeline consistency.
9.  **Agent 9: Trust Calibrator** — Compares grounding, gaps, and conflicts using a mathematical deduction matrix to calculate overall trust.
10. **Agent 10: SuperAgent** — Central orchestrator, planner, dynamic agent scheduler, and quality reviewer.
11. **Agent 11: Causal Synthesizer** — Traverses causal graph triples to extract multi-hop inference links.
12. **Agent 12: Web Search** — Fallback external search agent (activated for paper writing and implementation guides).
13. **Agent 13: Paper Writer** — Autonomous drafting agent for formatted, objective scientific articles.
14. **Agent 14: Implementation Guide** — Autonomous drafting agent for detailed step-by-step developer manuals.

---

## CORE SYSTEM INNOVATIONS 

Detailed documentation of our architectural innovations can be found in [innovation.md](file:///e:/Vasis%20AI/innovation.md):
1.  **RE-MSE Progressive Expansion:** Eliminates context gaps by statefully growing retrieval regions around key facts.
2.  **Tree-Summary Navigation:** Prevents global semantic drift by scoping search pages before parsing raw text.
3.  **Cooperative Dual-Model Review Loop:** Decouples reasoning (`DeepSeek-7B`) from structured extraction (`Qwen-Coder-3B`) for crash-free JSON parsing.
4.  **Causal Traversal Knowledge Graph:** Traces multi-hop relationships across separated chapters.
5.  **Calibrated Trust Level Tagging:** Defensive confidence scoring (High/Medium/Low) using a penalty matrix (gaps, conflicts, requery cycles).
6.  **Bibliography Bypass Fast-Path:** Targets terminal citation nodes directly, bypassing BM25 matching.
7.  **Adaptive Paragraph Atomization:** Adjusts chunk borders dynamically based on layouts instead of double-newlines.

---

## `/learn` — ADAPTIVE LEARNING ENGINE

VASIS AI includes a built-in learning engine that **compounds knowledge over time**. Every paper you generate, every query you ask, and every correction you make teaches the system to produce better results on future runs.

### How It Works

The learn engine operates in **three modes** that work together automatically:

| Mode | Trigger | What It Does |
|------|---------|--------------|
| **Passive** | Automatic | Every `/paper` and `/query` run is silently recorded. The engine tracks grounding ratios, failure patterns, and which sub-queries retrieved the most atoms. |
| **Active** | `/learn <topic>` | Crawls the web via Agent 12 and permanently ingests high-quality atoms into a local vault. Next paper on that topic starts pre-loaded. |
| **Feedback** | `/learn feedback` | Rate or correct the last generated paper section-by-section. Corrections marked "good" become high-trust learned atoms; hallucination flags become negative examples. |

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     LearnEngine                         │
│                                                         │
│   ┌─────────────┐   ┌──────────────┐   ┌────────────┐  │
│   │  RunRecords  │   │ LearnedAtoms │   │ Corrections│  │
│   │  (passive)   │   │  (active)    │   │ (feedback) │  │
│   └──────┬──────┘   └──────┬───────┘   └─────┬──────┘  │
│          │                 │                  │         │
│          └─────────┬───────┘──────────────────┘         │
│                    ▼                                    │
│            TopicMatcher (Jaccard / Embeddings)           │
│                    │                                    │
│          ┌─────────┴─────────┐                          │
│          ▼                   ▼                          │
│   PreflightHints       ReviewDashboard                  │
│   (before /paper)      (/learn review)                  │
│                                                         │
│   Store: .vasis_learn.json (auto-created)               │
└─────────────────────────────────────────────────────────┘
```

### CLI Commands

```bash
❯ /learn                         # Show learning status and available modes
❯ /learn <topic>                 # Crawl web and ingest atoms on a topic
❯ /learn feedback                # Rate/correct the last generated paper
❯ /learn review                  # Full learning dashboard
```

### When to Use Each Mode

#### `/learn <topic>` — Pre-load before writing
Use **before** running `/paper` on a topic you know the vault doesn't cover well:
```bash
❯ /learn attention transformer mechanisms
  Learning about: attention transformer mechanisms
  Searching web and ingesting results into vault…

  ✓  Ingested 12 atoms on 'attention transformer mechanisms'
     Trust: 8 high, 4 medium
     These atoms will auto-load next time you write a paper on this topic.

❯ /paper limitations of attention mechanisms
  ┌─ learn  —  from 3 previous runs ─────────────────────────────┐
  │  Seen 3 similar runs before  ·  avg grounding 61%            │
  │  ⚠  Grounding-risk topic — citation injector will run        │
  │  12 pre-loaded atoms from /learn vault                       │
  │  Tip: Attention topics: cite Vaswani et al. [2017]           │
  └──────────────────────────────────────────────────────────────┘
```

#### `/learn feedback` — After spotting issues
Use **after** `/paper` when you notice hallucinations or weak sections:
```bash
❯ /learn feedback
  Feedback mode — rate the last paper
  For each section, mark: good / hallucinated / wrong_citation / unclear

  ── Abstract ──
  The Transformer model uses attention mechanisms…
  [1] g=good  h=hallucinated  w=wrong_citation  u=unclear  skip: g

  ── Methodology ──
  We propose a hybrid approach combining MoE with dense layers…
  [1] g=good  h=hallucinated  w=wrong_citation  u=unclear  skip: h

  ✓ 2 corrections recorded. They'll improve the next paper on this topic.
```

#### `/learn review` — Monitor learning health
Use **periodically** to check system learning patterns and identify weak topics:
```bash
❯ /learn review
  Runs: 12  ·  Atoms learned: 47  ·  Corrections: 8  ·  Grounding fails: 5
  Store: .vasis_learn.json  ·  Embeddings: off (pip install sentence-transformers)

   Topic                          Runs   Avg grounding   Fails   Last seen
   attention is all you need         4              0%       4   2026-06-22
   convolutional neural networks     3             61%       2   2026-06-22
   mixture of experts transformer    2             91%       0   2026-06-08

  Grounding trend (last 10 runs):
  ▁▁▃▃▅▆▅▆▃▃   06-08 → 06-22

  Top failure causes:
    5×  grounding_fail
    3×  context_retrieval_failed

  Pre-loaded atoms by topic:
  ✓ attention transformer - 12 atoms   web   high-trust
  ✓ convolutional neural networks - 8 atoms   web
  ✓ mixture of experts - 6 atoms   feedback
```

### Data Store

All learning data is persisted in `.vasis_learn.json` (auto-created on first run). This file stores:
- **Run records** — every `/paper` and `/query` execution with grounding metrics
- **Learned atoms** — web-sourced knowledge atoms with trust levels and usage counts
- **Corrections** — user feedback mapped to specific sections and sentences
- **Topic cache** — aggregated per-topic statistics for fast pre-flight lookups

> **Tip:** Install `sentence-transformers` for semantic topic matching instead of keyword overlap:
> ```bash
> pip install sentence-transformers
> ```

---

## `/loop` — MULTI-AGENT LOOP ORCHESTRATION ENGINE

The **Loop Engine** (`loop_engine.py`) turns the linear 14-agent swarming pipeline into a **reactive multi-agent graph**. Rather than executing a one-pass generation, loops continuously monitor agent outputs, audit quality, perform deeper web crawling when gaps are found, check logic consistency, and trigger self-improvement loops.

For detailed, step-by-step instructions and architecture details, refer to the [loop.md](file:///e:/Vasis%20AI/loop.md) user guide.

### Commands Syntax

```bash
❯ /loop paper "<topic>"                 # Runs quality + chain loops (smart default)
❯ /loop paper "<topic>" chain           # Generates paper then immediately fires implementation guide writer
❯ /loop paper "<topic>" quality         # Loops writing + citation fixes until grounding score >= 85%
❯ /loop paper "<topic>" critique        # Runs contradiction audit, then performs section-level revisions
❯ /loop paper "<topic>" deep            # Accumulates web sources/atoms prior to writing
❯ /loop paper "<topic>" consensus       # Builds two distinct drafts and merges them using Agent 11
❯ /loop paper "<topic>" full            # Runs all six loops (research -> drafts -> grounding -> consistency -> guide -> learn)
❯ /loop paper "<topic>" deep quality chain --max 3 # Custom combination of loops with a retry limit
❯ /loop config                          # Show active loop settings
❯ /loop status                          # Show metrics and completed loops of the last execution
```

### Visual TUI Highlights
When running a loop inside the full-screen terminal TUI (`vasis_shell.py`), the loop executes in a background thread, and the **sidebar dot indicators automatically light up and transition** in real-time between agents (e.g., Agent 12 for `deep`, Agent 13 for `consensus`/`_write`, Agent 7 for `critique`, and Agent 14 for `chain`) to visualize the swarm's focus.

---

## INSTALLATION & QUICK START 

### Prerequisites
*   Install [Ollama](https://ollama.ai)
*   Pull the required models:
    ```bash
    ollama pull qwen2.5-coder:3b
    ollama pull deepseek-llm:7b
    ```

### Local Setup
1.  Clone the repository and navigate to the directory:
    ```bash
    cd "Vasis AI"
    ```
2.  Create and activate a virtual environment, then install dependencies:
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Linux/Mac:
    source .venv/bin/activate
    
    pip install -r requirements.txt
    ```

### Running Commands & Interactive Clients

#### 🖥️ Interactive Terminal Interfaces
*   **Launch the Interactive CLI Shell (Rich + Autocomplete):**
    ```bash
    python vasis_cli.py
    ```
    *(Or run via `start_system.bat` option `1`)*. The shell supports autocomplete for commands, venues, levels, and local vault documents.
*   **Launch the Full-Screen Textual TUI (Textual + Rich):**
    ```bash
    python vasis_shell.py [optional_path_to_pdf.pdf]
    ```
    *(See the detailed keyboard shortcut and command reference in [TUI.md](file:///e:/Vasis%20AI/TUI.md))*. It features asynchronous task processing (`@work`), thread-safe UI updates, a live agent activation sidebar, and a vault visualizer.

#### ⚙️ Single-Line CLI Commands
*   **Index a PDF Document:**
    ```bash
    python main.py index uploads/your_document.pdf
    ```
*   **Ask a Single Fact-check Question:**
    ```bash
    python main.py ask uploads/your_document.pdf "What are the core metrics of this system?"
    ```
*   **Launch Interactive Swarm Chat CLI (Legacy):**
    ```bash
    python main.py chat uploads/your_document.pdf
    ```
*   **Start Backend REST API Server:**
    ```bash
    python api.py
    ```

### Windows Control Panel Launcher
Double-click [start_system.bat](file:///e:/Vasis%20AI/start_system.bat) in Windows Explorer to open the interactive dashboard. You can start the Interactive CLI Shell, run benchmark validations, index new documents, launch the API backend, and monitor local VRAM health directly from this dashboard.

---

## DIAGNOSTIC BENCHMARKS & EVALUATION 

The system includes a dedicated, ASCII-safe RAG evaluation suite to verify **factual recall**, **precision accuracy**, and **hallucination rejection** under offline conditions.

```bash
python tests/run_benchmarks.py
```

This sequentially tests:
1.  **Factual accuracy** against target values (e.g. verifying exact key dimensions like $d_k=64$).
2.  **Hallucination safety** via out-of-scope negative controls (forcing the grounding and calibration agents to reject quantum computing queries in standard transformer papers).
3.  **Multi-Hop Traversal recall** across adjacent document sections.
4.  **Bibliography retrieval validation** verifying last-page extraction.

---

## INTERFACE SCREENSHOTS & GALLERY 

Below is a visual walkthrough of the system interfaces, showing the autocomplete CLI, the full-screen Textual TUI dashboard, and multi-agent consensus workflows.

![Interface Screenshot 1](images/1.png)
*Figure 1: Swarm orchestration dashboard showing real-time logs and active agent indicators.*

![Interface Screenshot 2](images/2.png)
*Figure 2: Intent routing analysis and document indexing flow.*

![Interface Screenshot 3](images/3.png)
*Figure 3: Interactive CLI shell prompt-toolkit autocomplete interface.*

![Interface Screenshot 4](images/4.png)
*Figure 4: Bidirectional stateful context expansion logs in the terminal.*

![Interface Screenshot 5](images/5.png)
*Figure 5: High-resolution detail of dynamic reasoning steps.*

![Interface Screenshot 6](images/6.png)
*Figure 6: Cross-document consensus and contradiction validator output.*

![Interface Screenshot 7](images/7.png)
*Figure 7: Causal relationship mapping and knowledge graph traversal.*

![Interface Screenshot 8](images/8.png)
*Figure 8: Mathematical confidence scoring and response calibration.*

![Interface Screenshot 9](images/9.png)
*Figure 9: Autonomous literature search and paper drafting results.*
