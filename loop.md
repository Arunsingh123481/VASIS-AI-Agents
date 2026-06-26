# VASIS AI — /loop Orchestration Guide

The **Loop Engine** (`loop_engine.py`) turns the linear 14-agent swarming pipeline into a **reactive multi-agent graph**. Rather than executing a one-pass generation, loops continuously monitor agent outputs, audit quality, perform deeper web crawling when gaps are found, check logic consistency, and trigger self-improvement loops.

---

## 🚀 Step-by-Step Usage

### Step 1: Start the Swarm System
Make sure your Ollama instance is running and has the models pulled:
```powershell
ollama pull qwen2.5-coder:3b
ollama pull deepseek-llm:7b
```
Launch either the interactive CLI or the full-screen terminal TUI:
* **Interactive CLI**:
  ```powershell
  python vasis_cli.py
  ```
* **Full-Screen TUI**:
  ```powershell
  python vasis_shell.py
  ```

---

### Step 2: Load Your Context Documents
Before running loop commands, you must load reference PDFs so that the paper writer can pull grounded factual claims.
* **To index a single PDF**:
  ```text
  /index uploads/1cf6b031-c99_NIPS-2017-attention-is-all-you-need-Paper.pdf
  ```
* **To load multiple PDFs at once**:
  ```text
  /vault paper1.pdf paper2.pdf
  ```

---

### Step 3: Run /loop Commands

You can run loops using different presets or combination flags depending on your requirements.

| Loop Command | Execution Preset / Behavior |
| :--- | :--- |
| `/loop paper "attention mechanism"` | **Smart Default**: Runs `quality` (grounding gate) and `chain` (auto-triggers guide). |
| `/loop paper "topic" chain` | **Chain Presets**: Generates paper then immediately fires implementation guide writer. |
| `/loop paper "topic" quality` | **Quality Gate**: Loops writing + citation injection until grounding ratio is $\ge 85\%$. |
| `/loop paper "topic" critique` | **Critique & Revise**: Runs Agent 7 contradiction auditor, then rewrites flagged sections. |
| `/loop paper "topic" deep` | **Deep Research**: Ingests $\ge 15$ atoms via targeted search passes before writing. |
| `/loop paper "topic" consensus` | **Consensus Swarm**: Generates two distinct drafts (formal vs. deep) and merges them via Agent 11. |
| `/loop paper "topic" full` | **Full swarming pipeline**: Runs `deep` $\to$ `consensus` $\to$ `quality` $\to$ `critique` $\to$ `chain` $\to$ `learn`. |
| `/loop paper "topic" deep quality chain --max 3` | **Custom combination**: Combines specific loops with a maximum iteration retry budget. |

---

### Step 4: Inspect Loop Settings and State

* **View Active Loop Configuration**:
  To inspect current threshold settings (e.g., target grounding percentage, critique retry counts):
  ```text
  /loop config
  ```
* **Check Live / Last Loop Status**:
  To view metrics, elapsed execution time, and completed loops for the last execution:
  ```text
  /loop status
  ```

---

## 🎨 Interactive CLI vs. Full-Screen TUI

### 1. In the Interactive CLI (`vasis_cli.py`)
* Output is printed step-by-step in the terminal with colored status indicators:
  * `◆` loop started
  * `·` loop iteration
  * `✓` loop successfully finished
  * `⚡` auto-triggered actions (e.g., grounding-risk preloads)
* Papers and guides are automatically saved to the `outputs/` directory.

### 2. In the Full-Screen TUI (`vasis_shell.py`)
* The loop runs in a background thread so the interface remains fully responsive.
* Dynamic status updates are streamed to the main chat logs.
* **Live Sidebar Highlights**: The agent sidebar dot dynamically transitions (e.g., lighting up Agent 12 during `deep`, Agent 13 during `consensus`/`_write`, Agent 7 during `critique`, and Agent 14 during `chain` loops) so you can visually watch the swarms work.
