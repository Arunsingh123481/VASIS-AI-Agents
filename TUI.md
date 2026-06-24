# VASIS AI — Terminal User Interface (TUI) User Guide

The VASIS AI Textual TUI is a terminal client built with the `textual` and `rich` libraries. It provides a full-screen dashboard interface to query, analyze, and generate papers/guides utilizing the 14-Agent Consensus Engine.

---

## 🚀 How to Launch

You can start the TUI in two ways:

1. **Via the Control Panel** (Recommended):
   Run the batch script:
   ```cmd
   start_system.bat
   ```
   Select Option **`1`** (`Launch Interactive Textual TUI Chat`). The script will automatically check for and install `textual` if it is missing.

2. **Direct CLI Launch**:
   Install dependencies and run the script:
   ```cmd
   pip install textual
   python vasis_shell.py [optional_path_to_pdf.pdf]
   ```

---

## 🛠️ TUI Command Reference

Type any command into the prompt input line at the bottom (`╰──>`) and press **Enter**.

### 1. Document Management
* **`/index <path>`**  
  Decomposes and indexes a new PDF paper, building the causal graph and PageIndex tree.  
  *Example:* `/index E:\papers\transformer.pdf`
* **`/vault <path1> <path2> ...`**  
  Ingests multiple PDFs into a shared vault session for cross-document contradiction check and consensus comparison.  
  *Example:* `/vault E:\papers\p1.pdf E:\papers\p2.pdf`
* **`/status`** or **`/stats`**  
  Displays metrics for the loaded document(s) (node count, atom count, triple count, cached files).
* **`/tree`**  
  Shows the PageIndex section-level navigation tree for single-document structures.
* **`/history`**  
  Lists the last 10 queries executed against the currently active document.

### 2. Target Configurations
* **`/venue <name>`**  
  Configures the target venue template (influences grounding rules, section outlines, and reference formatting).  
  *Options:* `IEEE`, `DSJ`, `Elsevier`, `Springer`, `ACM`, `NeurIPS`, `ICML`, `ICLR`
* **`/type <type>`**  
  Configures the target article structure for paper generation.  
  *Options:* `research_article`, `review_article`, `systematic_review`, `short_communication`, `perspective_article`, `technical_note`, `case_study`, `letter_to_editor`
* **`/level <level>`**  
  Configures the researcher target level for implementation guides.  
  *Options:* `beginner`, `masters`, `phd`

### 3. Agent Prompts & Queries
* **`<any raw question>`** or **`/query <question>`**  
  Dispatches the question to the consensus pipeline. If multiple documents are loaded in the vault, comparison keywords automatically route into a cross-document contradiction check.
* **`/paper <topic>`**  
  Dispatches Agent 13 (Research Paper Writer) to draft a citation-grounded paper. The output will automatically save to the `outputs/` folder.
* **`/guide <topic>`**  
  Dispatches Agent 14 (Implementation Guide Writer) to generate a step-by-step framework tailored to the set researcher level. The output will automatically save to the `outputs/` folder.
* **`/outputs`**  
  Lists the 20 most recently saved Markdown papers and guides generated in your `outputs/` directory.

---

## 💡 Keyboard Shortcuts
* **`F1`** — Toggles the Command Help modal overlay.
* **`F2`** — Toggles the visibility of the left Agents Sidebar.
* **`Ctrl+L`** or `/clear` — Clears the chat log screen.
* **`Ctrl+C`** or `/exit` — Closes the terminal dashboard.

---

## ⚙️ Under the Hood: How the TUI Works

### 1. Asynchronous Task Processing (`@work`)
Because local LLM generation (via Ollama or APIs) is a blocking operation, executing queries on the main UI thread would cause the terminal layout to freeze.  
To prevent this, all command processors (`_cmd_index`, `_cmd_query`, `_cmd_paper`, `_cmd_guide`, etc.) are decorated with Textual's `@work(exclusive=True, thread=True)` decorator. This spawns background worker threads while keeping the UI fully interactive and responsive (e.g., animations continue, inputs aren't blocked).

### 2. Thread-Safe UI Updates (`call_from_thread`)
Background worker threads cannot directly mutate UI widgets. The TUI uses `self.call_from_thread()` to delegate updates back to Textual's main event loop:
* Updating status bar indicators (`STATUS: RUNNING`, `STATUS: WRITING PAPER`).
* Disabling the user input box while processing is active.
* Printing logs and rendering output results inside the `RichLog` console.

### 3. Agent Activation Sidebar
The left side panel displays the 14 agents. When a `/query` is run, the TUI animates the agent panel in real-time (`agent_panel.set_active(index)`) as the pipeline moves from routing (Agent 1), navigation (Agent 3), context expansion (Agent 5), validation (Agent 6), and synthesis (Agent 11).

### 4. Advanced Vault Visualizer
When cross-paper comparison checks are run in `/vault` mode, the TUI prints:
* Individual answers side-by-side for comparison.
* Red warning flags (`⚠ STRUCTURAL CONTRADICTIONS DETECTED`) if conflicting RDF triples are found.
* Soft logical conflict bullet points.
* Causal graph paths discovered across the texts.
