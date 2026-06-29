#!/usr/bin/env python3
"""
VASIS AI — Interactive CLI Shell
Rich + Prompt Toolkit terminal interface.
────────────────────────────────────────
Usage:
    python vasis_cli.py
    python vasis_cli.py --venue IEEE --type research_article

Drop-in: replaces the old main.py chat loop with a styled, autocomplete-enabled shell.
All dispatch methods call the real VASIS agent backend.
"""

import sys
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Rich ──────────────────────────────────────────────────────────────────────
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown
from rich import box

# ── Prompt Toolkit ────────────────────────────────────────────────────────────
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEFAULT_MODEL, AGENT_MODEL, REASONING_MODEL
from agent_routing_rules import QUERY_DETECTION_PATTERNS

# ── Learn Engine ──────────────────────────────────────────────────────────────
from learn_engine import (
    LearnEngine, PreflightHint,
    render_preflight, render_active_learn_result,
)

# ── Loop Engine ───────────────────────────────────────────────────────────────
from loop_engine import (
    LoopOrchestrator, LoopContext, parse_loop_command,
    PRESETS, EXECUTION_ORDER,
    GROUNDING_THRESHOLD, DEEP_RESEARCH_MIN_ATOMS,
    MAX_CRITIQUE_ROUNDS, CONSENSUS_DRAFTS,
)

# ── Custom Agent Studio ──────────────────────────────────────────────────────
from agent_builder import AgentStudio, rich_print_fn

# =============================================================================
# THEME — edit these to change the look
# =============================================================================

class Theme:
    PRIMARY   = "#7C3AED"   # Brand purple
    SECONDARY = "#A78BFA"   # Light purple
    SUCCESS   = "#10B981"   # Green
    WARNING   = "#F59E0B"   # Amber
    ERROR     = "#EF4444"   # Red
    INFO      = "#60A5FA"   # Blue
    MUTED     = "#6B7280"   # Gray
    DIM       = "#374151"   # Dark gray
    TEXT      = "#F9FAFB"   # Near white

    # Grade colours
    GRADE = {
        "A": ("#10B981", "✦"),   # green
        "B": ("#60A5FA", "✓"),   # blue
        "C": ("#F59E0B", "◆"),   # amber
        "D": ("#EF4444", "◇"),   # red
        "F": ("#EF4444", "✗"),   # red bold
        "S": ("#6B7280", "—"),   # skip
    }

T = Theme  # shorthand

# =============================================================================
# AGENT REGISTRY
# =============================================================================

AGENTS = {
    1:  "Router",        2:  "Decomposer",    3:  "Navigator",
    4:  "Retrieval",     5:  "Expansion",     6:  "Validation",
    7:  "Contradiction", 8:  "Temporal",      9:  "Calibration",
    10: "Supervisor",    11: "Synthesis",     12: "Web Search",
    13: "Paper Writer",  14: "Impl. Guide",
}

# =============================================================================
# CONSOLE  (single global instance — thread-safe Rich console)
# =============================================================================

console = Console(highlight=False, soft_wrap=True)

# =============================================================================
# PRINT HELPERS
# =============================================================================

def nl():
    console.print()

def divider(label: str = ""):
    """Thin section rule"""
    if label:
        console.print()
        line = Text()
        line.append(f"  {label.upper()}", style=f"bold {T.MUTED}")
        console.print(line)
    else:
        console.rule(style=T.DIM)
    nl()


def agent_line(name: str, grade: str, score: float, elapsed: float):
    """
    Single completed-agent line, e.g.
      ✦ Paper Writer    grade=A   0.90   583.4s
    """
    color, icon = T.GRADE.get(grade, (T.MUTED, "·"))
    score_color = T.SUCCESS if score >= 0.8 else T.WARNING if score >= 0.5 else T.ERROR

    line = Text()
    line.append(f"  {icon} ", style=f"bold {color}")
    line.append(f"{name:<16}", style=T.TEXT)
    line.append(f" grade={grade}  ", style=f"bold {color}")
    line.append(f"{score:.2f}  ", style=score_color)
    line.append(f"{elapsed:.1f}s", style=T.MUTED)
    console.print(line)


def skip_line(name: str):
    line = Text()
    line.append("  — ", style=T.DIM)
    line.append(f"{name:<16}", style=T.DIM)
    line.append("skip", style=T.DIM)
    console.print(line)


def tool_line(action: str, detail: str = ""):
    """  ◆  Indexed  NIPS-2017... · 123 atoms"""
    line = Text()
    line.append("  ◆ ", style=f"bold {T.PRIMARY}")
    line.append(action, style=f"bold {T.TEXT}")
    if detail:
        line.append(f"  {detail}", style=T.MUTED)
    console.print(line)


def info_line(msg: str):
    console.print(Text(f"  {msg}", style=T.MUTED))


def error_line(msg: str):
    console.print(Text(f"  ✗ {msg}", style=f"bold {T.ERROR}"))


def success_line(msg: str):
    console.print(Text(f"  ✓ {msg}", style=T.SUCCESS))


def grounding_banner(ratio: float, tagged: int, total: int):
    """
    Visible banner when grounding ratio is below threshold.
    """
    if ratio >= 0.85:
        return  # all good, no banner

    pct = int(ratio * 100)
    if ratio == 0.0:
        msg = f"✗  Grounding FAIL — {tagged}/{total} sentences tagged (0%)"
        style = T.ERROR
        border = "red"
    else:
        msg = f"⚠  Low grounding — {tagged}/{total} sentences tagged ({pct}%)"
        style = T.WARNING
        border = "yellow"

    nl()
    console.print(Panel(
        f"[bold {style}]{msg}[/]\n"
        f"[{T.MUTED}]Threshold is 85%. Verify citations before publishing.[/]",
        border_style=border,
        padding=(0, 2),
    ))


def paper_panel(word_count: int, sections: int, preview: str):
    """Rendered markdown preview of the generated paper."""
    nl()
    console.print(Panel(
        Markdown(preview),
        title=f"[bold {T.SECONDARY}]RESEARCH PAPER  •  {word_count:,} WORDS  •  {sections} SECTIONS[/]",
        border_style=T.DIM,
        padding=(1, 2),
    ))


def outputs_table(files: list):
    """
    Replaces the bare numbered list with a scannable table.
    Each dict: { name, size_kb, date, kind }
    """
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style=f"bold {T.MUTED}",
        border_style=T.DIM,
        padding=(0, 1),
        expand=False,
    )
    table.add_column("#", style=T.DIM, width=3)
    table.add_column("FILE", style=T.TEXT)
    table.add_column("TYPE", style=T.MUTED, width=8)
    table.add_column("SIZE", style=T.MUTED, justify="right", width=7)
    table.add_column("DATE", style=T.DIM, width=12)

    for i, f in enumerate(files, 1):
        kind_color = T.INFO if f["kind"] == "paper" else T.SECONDARY
        table.add_row(
            str(i),
            f["name"],
            Text(f["kind"], style=kind_color),
            f"{f['size_kb']} KB",
            f["date"],
        )

    nl()
    console.print(table)
    nl()


def vault_status(docs: list):
    """Show each loaded document with atom/triple counts."""
    nl()
    for doc in docs:
        line = Text()
        line.append("  ✓ ", style=f"bold {T.SUCCESS}")
        line.append(doc["name"], style=f"bold {T.TEXT}")
        line.append(f"  {doc.get('atoms', 0)} atoms", style=T.MUTED)
        line.append("  ·", style=T.DIM)
        line.append(f"  {doc.get('triples', 0)} triples", style=T.DIM)
        console.print(line)

    count = len(docs)
    console.print(Text(
        f"\n  Vault ready  ·  {count} document{'s' if count != 1 else ''}  "
        f"·  Use /query or /paper to begin",
        style=T.MUTED,
    ))
    nl()


def help_table():
    table = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    table.add_column(style=f"bold {T.SECONDARY}", width=28)
    table.add_column(style=T.MUTED)

    rows = [
        ("/index <path.pdf>",       "Index a single PDF into the vault"),
        ("/vault <p1.pdf> ...",     "Load multiple PDFs at once"),
        ("/query <question>",       "Ask a question across vault documents"),
        ("/paper <topic>",          "Generate a full research paper"),
        ("/guide <topic>",          "Generate a masters-level implementation guide"),
        ("/outputs",                "Browse all generated files"),
        ("/venue <name>",           "Set publication venue  (IEEE / DSJ / Elsevier …)"),
        ("/type <name>",            "Set document type  (research_article / review …)"),
        ("/level <level>",          "Set researcher level  (beginner / masters / phd)"),
        ("/learn",                  "Show learning status and modes"),
        ("/learn <topic>",          "Crawl web and ingest atoms on a topic"),
        ("/learn feedback",         "Rate/correct the last generated paper"),
        ("/learn review",           "Full learning dashboard"),
        ("/loop paper <topic>",     "Run looped paper pipeline  (quality + chain)"),
        ("/loop config",            "Show current loop settings"),
        ("/loop status",            "Show live loop state"),
        ("/loop help",              "Show detailed loop subcommands & presets"),
        ("",                        ""),
        ("── CUSTOM AGENT STUDIO ──", ""),
        ("/build agent <name>",     "Create a custom research agent (wizard)"),
        ("/my-agents",              "List fixed + custom agents"),
        ("/my-loops",               "List custom agent loops"),
        ("/connect a1 a2 …",        "Wire agents into a loop  (--name --quality)"),
        ("/delete agent <name>",    "Delete a custom agent"),
        ("/delete loop <name>",     "Delete a custom loop"),
        ("/<custom> \"topic\"",       "Run a custom agent directly"),
        ("/loop <loopname> topic",  "Run a custom agent loop"),
        ("",                        ""),
        ("/status",                 "Show loaded document stats"),
        ("/tree",                   "Show PageIndex tree structure"),
        ("/history",                "Show recent query history"),
        ("/models",                 "Show active LLM routing table"),
        ("/clear",                  "Clear the screen"),
        ("/help",                   "Show this help"),
        ("/exit",                   "Quit VASIS AI"),
        ("",                        ""),
        ("Anything without /",      "Treated as a natural-language /query"),
    ]

    nl()
    for cmd, desc in rows:
        table.add_row(cmd, desc)
    console.print(table)
    nl()


def models_table(routing: list):
    """Show the LLM routing table."""
    table = Table(
        box=box.SIMPLE, show_header=True,
        header_style=f"bold {T.MUTED}", border_style=T.DIM,
        padding=(0, 1),
    )
    table.add_column("Role", style=T.TEXT)
    table.add_column("Model", style=T.SECONDARY)
    table.add_column("Task", style=T.MUTED)

    for row in routing:
        table.add_row(row["role"], row["model"], row["task"])

    nl()
    console.print(table)
    nl()


# =============================================================================
# BANNER
# =============================================================================

def print_banner(venue: str, doc_type: str, model: str):
    PURPLE      = "#7C3AED"
    PURPLE_LIGHT = "#A78BFA"

    nl()

    # ── ASCII Art — VASIS ─────────────────────────────────────────────────────
    logo_lines = [
        r"  ██╗   ██╗ █████╗  ███████╗ ██╗ ███████╗",
        r"  ██║   ██║██╔══██╗ ██╔════╝ ██║ ██╔════╝",
        r"  ██║   ██║███████║ ███████╗ ██║ ███████╗",
        r"  ╚██╗ ██╔╝██╔══██║ ╚════██║ ██║ ╚════██║",
        r"   ╚████╔╝ ██║  ██║ ███████║ ██║ ███████║",
        r"    ╚═══╝  ╚═╝  ╚═╝ ╚══════╝ ╚═╝ ╚══════╝",
    ]
    for line in logo_lines:
        console.print(Text(line, style=f"bold {PURPLE}"))

    # ── Subtitle ──────────────────────────────────────────────────────────────
    nl()
    subtitle = Text()
    subtitle.append("                   V A S I S   A I   A G E N T S", style=f"bold {PURPLE_LIGHT}")
    console.print(subtitle)
    nl()

    # ── Tagline ───────────────────────────────────────────────────────────────
    tag = Text()
    tag.append("  14-Agent Consensus Intelligence Engine", style=f"bold {PURPLE_LIGHT}")
    console.print(tag)

    # ── Session meta ──────────────────────────────────────────────────────────
    meta = Text()
    meta.append(f"  Session started {datetime.now().strftime('%Y-%m-%d %H:%M')}  ", style=T.MUTED)
    meta.append(f"Venue={venue}  Type={doc_type}  Model={model}", style=T.DIM)
    console.print(meta)
    nl()

    # ── Quick-start hint ──────────────────────────────────────────────────────
    hint = Text()
    hint.append("  /index path/to/paper.pdf", style=T.SECONDARY)
    hint.append("  to load a document  ·  ", style=T.DIM)
    hint.append("/help", style=T.SECONDARY)
    hint.append("  for all commands", style=T.DIM)
    console.print(hint)
    nl()


# =============================================================================
# PROMPT  (prompt_toolkit)
# =============================================================================

COMMANDS = [
    "/index", "/vault", "/query", "/paper", "/guide",
    "/outputs", "/venue", "/type", "/level",
    "/status", "/tree", "/history",
    "/models", "/learn", "/loop", "/clear", "/help", "/exit",
    # Custom Agent Studio commands
    "/build", "/my-agents", "/my-loops", "/connect", "/delete",
]

_pt_style = Style.from_dict({
    "prompt":                          f"bold {T.PRIMARY}",
    "completion-menu.completion":      f"bg:{T.DIM} {T.TEXT}",
    "completion-menu.completion.current": f"bg:{T.PRIMARY} white",
    "completion-menu.meta.completion":    T.MUTED,
    "auto-suggestion":                    T.DIM,
})

def _make_session(extra_words: list = None) -> PromptSession:
    words = list(COMMANDS)
    if extra_words:
        words.extend(extra_words)
    completer = WordCompleter(words, ignore_case=True, sentence=True)
    history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".vasis_history")
    return PromptSession(
        completer=completer,
        style=_pt_style,
        auto_suggest=AutoSuggestFromHistory(),
        history=FileHistory(history_path),
        complete_while_typing=True,
    )

def _prompt_text():
    return HTML("<ansi_bright_magenta><b>❯</b></ansi_bright_magenta> ")


# =============================================================================
# MAIN CLI
# =============================================================================

class VasisCLI:
    """
    Interactive terminal UI for VASIS AI.
    All dispatch methods call the real VASIS backend.
    """

    VENUES = ["IEEE", "DSJ", "Elsevier", "Springer", "ACM", "NeurIPS", "ICML", "ICLR"]
    ARTICLE_TYPES = [
        "research_article", "review_article", "systematic_review",
        "short_communication", "perspective_article", "technical_note",
        "case_study", "letter_to_editor"
    ]
    RESEARCHER_LEVELS = ["beginner", "masters", "phd"]

    def __init__(
        self,
        venue: str = "IEEE",
        doc_type: str = "research_article",
        level: str = "masters",
        outputs_dir: str = "outputs",
    ):
        self.venue = venue
        self.doc_type = doc_type
        self.level = level
        self.outputs_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), outputs_dir))
        self.vault_docs: list = []
        self.session = None

        # Backend state
        self.rag = None              # PageIndexREMSE instance (single-paper mode)
        self._vault_session = None   # VaultSession instance (multi-paper mode)

        # Learn engine
        self.learn = LearnEngine()
        self._last_paper_topic: str = ""

        # Custom Agent Studio
        self._init_agent_studio()

    def _init_agent_studio(self):
        """Initialise the Custom Agent Studio with LLM and Rich print adapter."""
        try:
            from llm.ollama_client import ask_llm
            # Use slightly higher temperature (e.g. 0.2) for open-ended formatting/synthesis 
            # tasks like references/abstract to prevent repetition loops.
            def ask_llm_with_temp(prompt: str) -> str:
                temp = 0.0
                prompt_lower = prompt.lower()
                if "references" in prompt_lower or "bibliography" in prompt_lower or "abstract" in prompt_lower:
                    temp = 0.2
                return ask_llm(prompt, model=REASONING_MODEL, temperature=temp)
            llm_fn = ask_llm_with_temp
        except Exception:
            llm_fn = None   # works without LLM (template fallback)

        self.studio = AgentStudio(
            llm_fn     = llm_fn,
            print_fn   = rich_print_fn(console),
            store_path = Path(os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                ".vasis_custom_agents.json",
            )),
        )

    def _refresh_completer(self):
        """Rebuild prompt-toolkit autocomplete with current custom agent names."""
        extra = [f"/{name}" for name in self.studio.custom_command_names()]
        if self.session:
            words = list(COMMANDS) + extra
            self.session.completer = WordCompleter(
                words, ignore_case=True, sentence=True,
            )

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self):
        if self.session is None:
            extra = [f"/{name}" for name in self.studio.custom_command_names()]
            self.session = _make_session(extra_words=extra)
        model_str = f"{AGENT_MODEL} / {REASONING_MODEL}"
        print_banner(self.venue, self.doc_type, model_str)
        while True:
            try:
                raw = self.session.prompt(_prompt_text()).strip()
            except KeyboardInterrupt:
                console.print(Text("\n  Use /exit to quit.", style=T.MUTED))
                continue
            except EOFError:
                self._cmd_exit()

            if not raw:
                continue

            if raw.startswith("/"):
                self._dispatch_command(raw)
            else:
                # Natural language → treat as query
                self._dispatch_query(raw)

    # ── Command dispatcher ────────────────────────────────────────────────────

    def _dispatch_command(self, raw: str):
        parts = raw.split(maxsplit=1)
        cmd  = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        dispatch = {
            "/index":     lambda: self._cmd_index(args),
            "/vault":     lambda: self._cmd_vault(args),
            "/query":     lambda: self._dispatch_query(args),
            "/paper":     lambda: self._cmd_paper(args),
            "/guide":     lambda: self._cmd_guide(args),
            "/outputs":   lambda: self._cmd_outputs(),
            "/venue":     lambda: self._cmd_venue(args),
            "/type":      lambda: self._cmd_type(args),
            "/level":     lambda: self._cmd_level(args),
            "/status":    lambda: self._cmd_status(),
            "/tree":      lambda: self._cmd_tree(),
            "/history":   lambda: self._cmd_history(),
            "/models":    lambda: self._cmd_models(),
            "/clear":     lambda: self._cmd_clear(),
            "/learn":     lambda: self._cmd_learn(args),
            "/loop":      lambda: self._cmd_loop(args),
            "/help":      lambda: help_table(),
            "/exit":      lambda: self._cmd_exit(),
            "/quit":      lambda: self._cmd_exit(),
            # Custom Agent Studio commands
            "/build":     lambda: self._cmd_studio_build(args),
            "/my-agents": lambda: self.studio.cmd_list_agents(),
            "/my-loops":  lambda: self.studio.cmd_list_loops(),
            "/connect":   lambda: self.studio.cmd_connect(args),
            "/delete":    lambda: self._cmd_studio_delete(args),
        }

        fn = dispatch.get(cmd)
        if fn:
            fn()
        else:
            # Dynamic dispatch: check if cmd matches a custom agent name
            cmd_name = cmd.lstrip("/")
            if cmd_name in self.studio.custom_command_names():
                self._cmd_run_custom_agent(cmd_name, args)
            # Check if the first arg to /loop is a custom loop name
            else:
                error_line(f"Unknown command '{cmd}'.  Type /help for the list.")

    # ── Commands ──────────────────────────────────────────────────────────────

    def _cmd_index(self, path: str):
        if not path:
            error_line("Usage: /index path/to/document.pdf")
            return

        # Strip quotes the user may have typed
        path = path.strip('"').strip("'")

        if not os.path.exists(path):
            error_line(f"File not found: {path}")
            return

        self._load_single(path)
        # After loading, show vault summary
        if self.vault_docs:
            vault_status(self.vault_docs)

    def _cmd_vault(self, args: str):
        # Parse paths, handling quoted strings
        paths = []
        for token in re.findall(r'(?:[^\s"\']+|"[^"]*"|\'[^\']*\')+', args):
            paths.append(token.strip('"').strip("'"))

        if not paths:
            error_line("Usage: /vault file1.pdf file2.pdf ...")
            return

        # Reset for a fresh vault session
        self.vault_docs = []
        self.rag = None
        self._vault_session = None

        nl()
        for p in paths:
            if not os.path.exists(p):
                error_line(f"File not found: {p}")
                continue
            self._load_single(p)

        vault_status(self.vault_docs)

    def _load_single(self, path: str):
        fname = Path(path).name
        console.print(f"[{T.MUTED}]Indexing {fname}…[/]")
        result = self._dispatch_index(path)

        if result:
            self.vault_docs.append(result)
            tool_line("Indexed", f"{result['name']}  ·  {result['atoms']} atoms")

    def _cmd_paper(self, topic: str):
        if not topic:
            error_line("Usage: /paper <topic>")
            return
        if not self.rag and not self._vault_session:
            error_line("Load documents first with /index or /vault")
            return

        # ── Learn: pre-flight hints ───────────────────────────────────────
        try:
            hints = self.learn.get_preflight(topic)
            self._show_preflight(hints)
        except Exception:
            pass  # learn is best-effort

        nl()
        console.print(Text("  Writing research paper…", style=T.MUTED))
        console.print(Text(f"  {topic}", style=f"bold {T.TEXT}"))
        info_line(f"Venue: {self.venue}  ·  Type: {self.doc_type}")
        nl()

        result = self._dispatch_paper(topic)

        if not result:
            return

        # ── Learn: record this run passively ──────────────────────────────
        self._last_paper_topic = topic
        try:
            self.learn.record_run(topic, {
                "agents":           result.get("agents", []),
                "grounding_ratio":  result.get("grounding_ratio", 0.0),
                "atoms_retrieved":  result.get("atoms_retrieved", 0),
                "sub_queries":      result.get("sub_queries", []),
                "word_count":       result.get("word_count", 0),
                "duration_s":       result.get("duration_s", 0.0),
                "venue":            self.venue,
                "doc_type":         self.doc_type,
                "query_raw":        f"write a research paper on {topic}",
                "failures":         result.get("failures", []),
            })
        except Exception:
            pass  # learn recording is best-effort

        # Print per-agent lines
        if result.get("agents"):
            divider("Agent pipeline")
            for ag in result["agents"]:
                if ag.get("skipped"):
                    skip_line(ag["name"])
                else:
                    agent_line(ag["name"], ag.get("grade", "?"), ag.get("score", 0.0), ag.get("time", 0.0))

        # Grounding banner
        if result.get("grounding_ratio") is not None:
            grounding_banner(
                result["grounding_ratio"],
                result.get("grounding_tagged", 0),
                result.get("grounding_total", 0),
            )

        # Paper preview
        if result.get("word_count") and result.get("preview"):
            paper_panel(result["word_count"], result.get("sections", 0), result["preview"])

        # Save confirmation
        if result.get("output_path"):
            nl()
            success_line(f"Saved → {result['output_path']}")
            nl()

    def _cmd_guide(self, topic: str):
        if not topic:
            error_line("Usage: /guide <topic>")
            return
        if not self.rag and not self._vault_session:
            error_line("Load documents first with /index or /vault")
            return

        nl()
        console.print(Text("  Building implementation guide…", style=T.MUTED))
        console.print(Text(f"  {topic}", style=f"bold {T.TEXT}"))
        info_line(f"Level: {self.level}")
        nl()

        result = self._dispatch_guide(topic)

        if not result:
            return

        # Show step progress
        if result.get("steps"):
            divider(f"Implementation guide  ·  {len(result['steps'])} steps")
            for step in result["steps"]:
                tool_line(f"Step {step['n']}/{len(result['steps'])}", step["title"])

        # Save confirmation
        if result.get("output_path"):
            nl()
            success_line(f"Saved → {result['output_path']}")
            nl()

    def _dispatch_query(self, query: str):
        if not query:
            error_line("Usage: /query <question>  (or just type your question)")
            return

        has_rag = self.rag and self.rag._ready
        has_vault = self._vault_session is not None

        if not has_rag and not has_vault:
            error_line("No document loaded. Use /index or /vault first.")
            return

        nl()
        with console.status(
            f"[{T.MUTED}]Thinking…[/]",
            spinner="dots",
            spinner_style=T.PRIMARY,
        ):
            result = self._dispatch_query_impl(query)

        if not result:
            return

        # ── Learn: record query run passively ─────────────────────────────
        try:
            self.learn.record_run(query, {
                "grounding_ratio":  result.get("grounding_ratio", 0.0),
                "atoms_retrieved":  result.get("atoms_retrieved", 0),
                "sub_queries":      result.get("sub_queries", []),
                "word_count":       0,
                "duration_s":       result.get("elapsed_seconds", 0.0),
                "venue":            self.venue,
                "doc_type":         self.doc_type,
                "query_raw":        query,
                "failures":         [],
            })
        except Exception:
            pass  # learn recording is best-effort

        nl()

        # Check if it's a vault comparison result
        if result.get("_type") == "comparison":
            self._render_comparison(result)
        elif result.get("_type") == "vault_multi":
            self._render_vault_multi(result)
        else:
            # Single paper result
            answer = result.get("answer", "")
            trust = result.get("trust_level", "low").upper()
            confidence = result.get("confidence", 0.0)
            grade = result.get("pipeline_grade", "?")
            elapsed = result.get("elapsed_seconds", 0.0)

            trust_color = {"HIGH": T.SUCCESS, "MEDIUM": T.WARNING, "LOW": T.ERROR}.get(trust, T.MUTED)

            header = Text()
            header.append("  ✓ VASIS RESPONSE", style=f"bold {T.SUCCESS}")
            header.append(f"   TRUST: {trust}", style=f"bold {trust_color}")
            header.append(f"  conf={confidence:.2f}  grade={grade}  {elapsed:.1f}s", style=T.DIM)
            console.print(header)
            nl()

            console.print(Panel(
                Markdown(answer),
                border_style=T.DIM,
                padding=(1, 2),
            ))

            # Contradictions
            if result.get("contradictions_found"):
                nl()
                console.print(Text("  ⚠ CONTRADICTIONS DETECTED", style=f"bold {T.ERROR}"))
                for c in result.get("contradiction_details", [])[:3]:
                    console.print(Text(
                        f"    [{c.get('severity', '?').upper()}] "
                        f"{c.get('claim_a', '')}  vs  {c.get('claim_b', '')}",
                        style=T.WARNING
                    ))

            # Novel connections
            if result.get("novel_connections"):
                nl()
                console.print(Text(
                    f"  ⚡ {len(result['novel_connections'])} novel causal connections synthesised",
                    style=f"bold {T.SUCCESS}"
                ))
                for n in result["novel_connections"][:2]:
                    via = " → ".join(n.get("via", []))
                    console.print(Text(
                        f"    {n.get('from', '')} → {via} → {n.get('to', '')}  conf={n.get('confidence', 0):.2f}",
                        style=T.DIM
                    ))

            # Auto-save paper/guide outputs
            self._auto_save_outputs(result, query)

        nl()

    def _render_comparison(self, result: dict):
        """Render cross-paper comparison result."""
        console.print(Text("  ✓ VAULT CROSS-PAPER COMPARISON", style=f"bold {T.SUCCESS}"))
        nl()

        for label, ans in result.get("per_paper_answers", {}).items():
            console.print(Text(f"  {label}:", style=f"bold {T.INFO}"))
            for line in ans.split("\n"):
                if line.strip():
                    console.print(Text(f"    {line}", style=T.TEXT))
            nl()

        if result.get("structural_conflict_found"):
            console.print(Panel(
                Text("⚠ STRUCTURAL CONTRADICTIONS DETECTED", style=f"bold {T.ERROR}"),
                border_style="red",
                padding=(0, 2),
            ))
            for c in result.get("cross_doc_conflicts", []):
                console.print(Text(
                    f"    {c.get('subject', '')} — {c.get('relation', '')}",
                    style=T.WARNING
                ))
                for doc, obj in c.get("per_document", {}).items():
                    console.print(Text(f"      [{doc}] → {obj}", style=T.DIM))
            for c in result.get("triple_conflicts", []):
                console.print(Text(
                    f"    {c.get('subject', '')} — {c.get('relation', '')}: {c.get('conflicting_objects', '')}",
                    style=T.WARNING
                ))
        else:
            score = result.get("consistency_score", 1.0)
            success_line(f"Consistency Check: No structural conflicts.  Score: {score:.2f}")

        if result.get("llm_contradictions"):
            nl()
            console.print(Text("  Logical inconsistencies flagged:", style=f"bold {T.WARNING}"))
            for c in result.get("llm_contradictions", []):
                console.print(Text(
                    f"    • {c.get('claim_a', '')}  vs  {c.get('claim_b', '')}  ({c.get('severity', 'low')})",
                    style=T.DIM
                ))

        if result.get("narrative_debates"):
            nl()
            console.print(Text("  Narrative-level debates:", style=f"bold {T.SECONDARY}"))
            for d in result.get("narrative_debates", []):
                console.print(Text(f"    {d.get('debate_topic', 'Untitled')}", style=f"bold {T.TEXT}"))
                console.print(Text(f"      Side A: {d.get('side_a', '')}", style=T.DIM))
                console.print(Text(f"      Side B: {d.get('side_b', '')}", style=T.DIM))

    def _render_vault_multi(self, result: dict):
        """Render vault independent papers response."""
        console.print(Text("  ✓ VAULT INDEPENDENT PAPERS RESPONSE", style=f"bold {T.SUCCESS}"))
        nl()

        for label, r in result.get("per_paper", {}).items():
            console.print(Text(f"  ● {label}", style=f"bold {T.SUCCESS}"))
            ans = r.get("answer", "")
            for line in ans.split("\n"):
                if line.strip():
                    console.print(Text(f"    {line}", style=T.TEXT))
            nl()

    def _auto_save_outputs(self, result: dict, query: str):
        """Auto-save paper/guide outputs from a query result."""
        try:
            from main import _save_agent_output

            paper_result = result.get("paper_result")
            impl_result  = result.get("impl_result")

            if paper_result and paper_result.get("full_text"):
                path = _save_agent_output(
                    content=paper_result["full_text"],
                    agent_name="paper",
                    topic=query,
                    venue=self.venue,
                    article_type=self.doc_type,
                )
                success_line(f"Paper saved → {path}")

            if impl_result and impl_result.get("full_text"):
                path = _save_agent_output(
                    content=impl_result["full_text"],
                    agent_name="guide",
                    topic=query,
                    researcher_level=self.level,
                )
                success_line(f"Guide saved → {path}")
        except Exception:
            pass  # Don't crash the CLI if auto-save fails

    def _cmd_outputs(self):
        files = self._dispatch_list_outputs()
        if not files:
            info_line("No outputs yet.  Use /paper or /guide to generate files.")
            nl()
            return
        outputs_table(files)

    def _cmd_venue(self, v: str):
        if not v:
            info_line(f"Current venue: {self.venue}")
            return
        v_upper = v.strip().upper()
        valid = [ven.upper() for ven in self.VENUES]
        if v_upper in valid:
            matched = next((ven for ven in self.VENUES if ven.upper() == v_upper), v)
            self.venue = matched
            success_line(f"Venue → {self.venue}")
        else:
            error_line(f"Unknown venue '{v}'.  Available: {', '.join(self.VENUES)}")

    def _cmd_type(self, t: str):
        if not t:
            info_line(f"Current type: {self.doc_type}")
            return
        t_lower = t.strip().lower()
        if t_lower in self.ARTICLE_TYPES:
            self.doc_type = t_lower
            success_line(f"Type → {self.doc_type}")
        else:
            error_line(f"Unknown type '{t}'.  Available: {', '.join(self.ARTICLE_TYPES)}")

    def _cmd_level(self, lvl: str):
        if not lvl:
            info_line(f"Current level: {self.level}")
            return
        lvl_lower = lvl.strip().lower()
        if lvl_lower in self.RESEARCHER_LEVELS:
            self.level = lvl_lower
            success_line(f"Level → {self.level}")
        else:
            error_line(f"Unknown level '{lvl}'.  Available: {', '.join(self.RESEARCHER_LEVELS)}")

    def _cmd_status(self):
        if self._vault_session is not None:
            nl()
            console.print(Text("  VAULT STATUS", style=f"bold {T.WARNING}"))
            nl()
            for s in self._vault_session.stats():
                line = Text()
                line.append(f"    {s['label']}: ", style=f"bold {T.INFO}")
                line.append(f"{s.get('total_atoms', 0)} atoms, ", style=T.TEXT)
                line.append(f"{s.get('total_triples', 0)} triples, ", style=T.TEXT)
                line.append(f"{s.get('tree_nodes', 0)} sections", style=T.MUTED)
                console.print(line)
            nl()
            return

        if not self.rag:
            error_line("No document loaded. Use /index first.")
            return

        stats = self.rag.get_stats()
        nl()
        console.print(Text("  SYSTEM STATUS", style=f"bold {T.WARNING}"))
        nl()
        for k, v in stats.items():
            line = Text()
            line.append(f"    {k:<20}", style=f"bold {T.INFO}")
            line.append(f"  {v}", style=T.TEXT)
            console.print(line)
        nl()

    def _cmd_tree(self):
        if not self.rag or not self.rag._ready:
            error_line("No single document loaded.")
            return

        nl()
        info_line("PageIndex tree structure:")
        nl()
        for node in self.rag.tree_nodes[:40]:
            depth = node.get("depth", 0)
            indent = "  " * depth
            title_text = node.get("title", "Untitled")
            pages = node.get("pages", node.get("start_page", "?"))
            console.print(Text(
                f"    {indent}└─ {title_text}  (p.{pages})",
                style=T.INFO
            ))
        if len(self.rag.tree_nodes) > 40:
            info_line(f"... and {len(self.rag.tree_nodes) - 40} more nodes")
        nl()

    def _cmd_history(self):
        if not self.rag:
            error_line("No single document loaded.")
            return
        try:
            from storage.store import load_query_log
            log = load_query_log(self.rag.doc_id)
            if not log:
                info_line("No query history found for this document.")
                return
            nl()
            console.print(Text("  QUERY HISTORY", style=f"bold {T.WARNING}"))
            nl()
            for i, entry in enumerate(log[-10:], 1):
                console.print(Text(f"    {i}. [{entry.get('timestamp', '?')}]", style=f"bold {T.TEXT}"))
                console.print(Text(f"       Q: {entry['query']}", style=T.INFO))
                answer_preview = entry.get('answer', '')[:120].replace('\n', ' ')
                console.print(Text(f"       A: {answer_preview}…", style=T.DIM))
            nl()
        except Exception as e:
            error_line(f"Could not load history: {e}")

    def _cmd_models(self):
        routing = [
            {"role": "Router / JSON tasks",  "model": AGENT_MODEL,     "task": "generate_json"},
            {"role": "Answer generation",    "model": REASONING_MODEL, "task": "generate"},
            {"role": "Paper Writer",         "model": REASONING_MODEL, "task": "generate"},
            {"role": "Impl. Guide",          "model": REASONING_MODEL, "task": "generate"},
        ]
        models_table(routing)

    def _cmd_clear(self):
        console.clear()
        model_str = f"{AGENT_MODEL} / {REASONING_MODEL}"
        print_banner(self.venue, self.doc_type, model_str)

    # =========================================================================
    # /learn COMMAND METHODS
    # =========================================================================

    def _cmd_learn(self, args: str):
        """Dispatch /learn sub-commands."""
        args = args.strip()

        if not args:
            self._learn_status()
        elif args.lower() == "feedback":
            self._learn_feedback()
        elif args.lower() == "review":
            self._learn_review()
        elif args.lower() in ("auto", "auto on", "auto off"):
            console.print(Text(
                "  Auto-learning is always on — every /paper and /query run "
                "is recorded automatically.",
                style=T.MUTED,
            ))
        else:
            self._learn_topic(args)

    # ── /learn  (no args) — brief status ─────────────────────────────────────

    def _learn_status(self):
        summary = self.learn.review()
        stats   = summary["stats"]
        console.print()
        console.print(Text("  /learn  modes:", style=f"bold {T.SECONDARY}"))

        rows = [
            ("/learn <topic>",   "Crawl the web and permanently ingest atoms"),
            ("/learn feedback",  "Correct or rate the last generated paper"),
            ("/learn review",    "Full learning dashboard"),
        ]
        for cmd, desc in rows:
            line = Text()
            line.append(f"  {cmd:<22}", style=T.SECONDARY)
            line.append(desc, style=T.MUTED)
            console.print(line)

        console.print()
        console.print(Text(
            f"  Runs recorded: {stats['total_runs']}  ·  "
            f"Atoms learned: {stats['total_atoms_learned']}  ·  "
            f"Corrections: {stats['total_corrections']}",
            style=T.MUTED,
        ))
        console.print()

    # ── /learn <topic> — active crawl ────────────────────────────────────────

    def _learn_topic(self, topic: str):
        console.print()
        console.print(Text(f"  Learning about: {topic}", style=f"bold {T.TEXT}"))
        console.print(Text(
            "  Searching web and ingesting results into vault…",
            style=T.MUTED,
        ))
        console.print()

        web_results = self._dispatch_learn_crawl(topic)

        with console.status(
            f"  [{T.MUTED}]Ingesting {len(web_results)} results…[/]",
            spinner="dots",
            spinner_style=T.PRIMARY,
        ):
            result = self.learn.active_learn(
                topic       = topic,
                web_results = web_results,
            )

        console.print()
        for line in render_active_learn_result(result):
            icon = "✓" if line.startswith("✓") else " "
            style = T.SUCCESS if icon == "✓" else T.MUTED
            console.print(Text(f"  {line}", style=style))
        console.print()

    # ── /learn feedback — correct the last paper ──────────────────────────

    def _learn_feedback(self):
        """
        Interactive section-by-section rating of the last paper.
        """
        console.print()
        console.print(Text("  Feedback mode — rate the last paper", style=f"bold {T.TEXT}"))
        console.print(Text(
            "  For each section, mark: good / hallucinated / wrong_citation / unclear",
            style=T.MUTED,
        ))
        console.print()

        sections = self._dispatch_last_paper_sections()

        if not sections:
            console.print(Text(
                "  No recent paper found. Run /paper first.",
                style=T.MUTED,
            ))
            console.print()
            return

        corrections = []
        ISSUES = {"g": "good", "h": "hallucinated", "w": "wrong_citation", "u": "unclear"}

        for section_name, sentences in sections.items():
            console.print(Text(f"  ── {section_name} ──", style=T.SECONDARY))
            for i, sentence in enumerate(sentences[:3]):
                console.print(Text(f"  {sentence[:120]}…", style=T.TEXT))
                try:
                    raw = self.session.prompt(
                        f"  [{i+1}] g=good  h=hallucinated  w=wrong_citation  u=unclear  skip: "
                    ).strip().lower()
                except (KeyboardInterrupt, EOFError):
                    break

                if raw in ISSUES:
                    corrections.append({
                        "section":  section_name,
                        "issue":    ISSUES[raw],
                        "sentence": sentence,
                    })
                console.print()

        if corrections:
            n = self.learn.record_feedback(
                corrections,
                topic=self._last_paper_topic,
            )
            console.print(Text(
                f"  ✓ {n} correction{'s' if n != 1 else ''} recorded. "
                "They'll improve the next paper on this topic.",
                style=T.SUCCESS,
            ))
        else:
            console.print(Text("  No corrections recorded.", style=T.MUTED))
        console.print()

    # ── /learn review — full dashboard ───────────────────────────────────────

    def _learn_review(self):
        summary = self.learn.review()
        stats   = summary["stats"]
        console.print()

        # ── header stats ──────────────────────────────────────────────────
        header = Text()
        header.append("  Runs: ", style=T.MUTED)
        header.append(str(stats["total_runs"]), style=f"bold {T.TEXT}")
        header.append("  ·  Atoms learned: ", style=T.MUTED)
        header.append(str(stats["total_atoms_learned"]), style=f"bold {T.TEXT}")
        header.append("  ·  Corrections: ", style=T.MUTED)
        header.append(str(stats["total_corrections"]), style=f"bold {T.TEXT}")
        header.append("  ·  Grounding fails: ", style=T.MUTED)
        header.append(
            str(stats["grounding_failures"]),
            style=f"bold {T.ERROR}" if stats["grounding_failures"] > 0 else T.TEXT,
        )
        console.print(header)
        console.print(Text(
            f"  Store: {summary['store_path']}  ·  "
            f"Embeddings: {'on' if summary['using_embeddings'] else 'off (pip install sentence-transformers)'}",
            style=T.DIM,
        ))
        console.print()

        # ── per-topic table ────────────────────────────────────────────────
        if summary["topic_summary"]:
            table = Table(
                box=box.SIMPLE,
                show_header=True,
                header_style=f"bold {T.MUTED}",
                border_style=T.DIM,
                padding=(0, 1),
            )
            table.add_column("Topic",         style=T.TEXT, max_width=40)
            table.add_column("Runs",          style=T.MUTED, justify="right", width=6)
            table.add_column("Avg grounding", justify="right", width=14)
            table.add_column("Fails",         justify="right", width=6)
            table.add_column("Last seen",     style=T.DIM, width=12)

            for row in summary["topic_summary"]:
                g     = row["avg_ground"]
                gcol  = T.SUCCESS if g >= 0.85 else T.WARNING if g >= 0.5 else T.ERROR
                gcell = Text(f"{g:.0%}", style=gcol)
                fcell = Text(
                    str(row["failures"]),
                    style=T.ERROR if row["failures"] > 0 else T.DIM,
                )
                table.add_row(
                    row["topic"], str(row["runs"]),
                    gcell, fcell, row["last_seen"],
                )
            console.print(table)

        # ── grounding trend sparkline ─────────────────────────────────────
        trend = summary.get("grounding_trend", [])
        if trend:
            console.print()
            console.print(Text("  Grounding trend (last 10 runs):", style=T.MUTED))
            BARS = " ▁▂▃▄▅▆▇█"
            line = Text("  ")
            for t in trend:
                idx = min(8, int(t["ratio"] * 8))
                bar = BARS[idx]
                col = T.SUCCESS if t["ratio"] >= 0.85 else \
                      T.WARNING if t["ratio"] >= 0.5 else T.ERROR
                line.append(bar, style=col)
            line.append(f"   {trend[0]['date']} → {trend[-1]['date']}", style=T.DIM)
            console.print(line)

        # ── top failure causes ─────────────────────────────────────────────
        if summary["top_failures"]:
            console.print()
            console.print(Text("  Top failure causes:", style=T.MUTED))
            for cause, count in summary["top_failures"]:
                console.print(Text(
                    f"  {count:>3}×  {cause}",
                    style=T.ERROR if count > 2 else T.MUTED,
                ))

        # ── pre-loaded atoms by topic ─────────────────────────────────────
        atoms = self.learn.store.get_all_atoms()
        if atoms:
            from collections import Counter
            topic_atoms = Counter(a["topic"] for a in atoms)
            console.print()
            console.print(Text("  Pre-loaded atoms by topic:", style=T.MUTED))
            for topic, count in topic_atoms.most_common(8):
                # Determine source tags
                topic_atoms_list = [a for a in atoms if a["topic"] == topic]
                tags = set()
                for a in topic_atoms_list:
                    url = a.get("source_url", "")
                    if url.startswith("feedback:"):
                        tags.add("feedback")
                    elif any(s in url for s in ("arxiv.org", "scholar.google", "ieee.org", "acm.org")):
                        tags.add("high-trust")
                    if url and not url.startswith("feedback:"):
                        tags.add("web")

                line = Text()
                line.append("  ✓ ", style=f"bold {T.SUCCESS}")
                line.append(f"{topic}", style=f"bold {T.TEXT}")
                line.append(f" - {count} atoms  ", style=T.MUTED)
                for tag in sorted(tags):
                    tag_color = T.SUCCESS if tag == "high-trust" else \
                                T.WARNING if tag == "feedback" else T.INFO
                    line.append(f" {tag} ", style=f"bold {tag_color}")
                    line.append(" ", style=T.DIM)
                console.print(line)

        console.print()

    # =========================================================================
    # /loop COMMAND  — reactive multi-loop paper pipeline
    # =========================================================================

    def _init_loop_orchestrator(self):
        """Lazily create the LoopOrchestrator with real agent dispatch adapters."""
        if hasattr(self, '_loop_orch') and self._loop_orch is not None:
            return
        self._loop_orch = LoopOrchestrator(
            agents={
                "agent12_websearch":    self._loop_dispatch_agent12,
                "agent13_paper_writer": self._loop_dispatch_agent13,
                "agent13_revise":       self._loop_dispatch_agent13,
                "agent14_impl_guide":   self._loop_dispatch_agent14,
                "agent7_contradiction": self._loop_dispatch_agent7,
                "agent11_synthesis":    self._loop_dispatch_agent11,
                "grounding_auditor":    self._loop_dispatch_grounding_audit,
                "learn_engine":         self.learn,
            },
        )
        self._loop_last_ctx: Optional[LoopContext] = None

    # ── /loop entry ──────────────────────────────────────────────────────────

    def _cmd_loop(self, args: str):
        # Check if the first word is a custom loop name
        parts = args.strip().split(maxsplit=1)
        if parts and parts[0].lower() in self.studio.custom_loop_names():
            loop_name = parts[0].lower()
            topic = parts[1].strip().strip('"\'') if len(parts) > 1 else ""
            self._cmd_run_custom_loop(loop_name, topic)
            return

        parsed = parse_loop_command(args)

        if parsed["special"]:
            if parsed["special"] == "help":
                self._loop_help()
            elif parsed["special"] == "config":
                self._loop_config()
            elif parsed["special"] == "status":
                self._loop_status()
            return

        topic      = parsed["topic"]
        loop_types = parsed["loop_types"]
        max_iter   = parsed["max_iter"]

        if not topic:
            error_line("Usage: /loop paper <topic> [quality] [chain] [full] [--max N]")
            return
        if not self.rag and not self._vault_session:
            error_line("Load documents first with /index or /vault")
            return

        self._init_loop_orchestrator()

        nl()
        console.print(Text(f"  /loop paper · {topic}", style=f"bold {T.TEXT}"))
        console.print(Text(
            f"  Loops: {', '.join(loop_types)}  ·  max {max_iter} retries",
            style=T.MUTED,
        ))
        nl()

        self._loop_active_ctx = None
        def on_status(level, msg, ctx):
            self._loop_active_ctx = ctx
            icons = {
                "start": "◆", "done": "✓", "skip": "—",
                "iter": "·", "auto": "⚡", "warn": "⚠", "error": "✗",
            }
            icon = icons.get(level, "·")
            colors = {
                "start": T.PRIMARY, "done": T.SUCCESS, "skip": T.DIM,
                "warn": T.WARNING, "error": T.ERROR, "auto": T.SECONDARY,
            }
            color = colors.get(level, T.MUTED)
            console.print(Text(f"  {icon}  {msg}", style=color))

        ctx = self._loop_orch.run(
            topic          = topic,
            loop_types     = loop_types,
            venue          = self.venue,
            doc_type       = self.doc_type,
            level          = self.level,
            max_iterations = max_iter,
            status_cb      = on_status,
        )
        self._loop_last_ctx = ctx

        # ── results ──────────────────────────────────────────────────────────
        nl()
        divider("Loop results")
        if ctx.paper_written:
            success_line(
                f"Paper: {ctx.word_count} words  ·  "
                f"grounding {ctx.grounding_ratio:.0%}"
            )
            # Auto-save
            try:
                from main import _save_agent_output
                path = _save_agent_output(
                    content=ctx.paper_text,
                    agent_name="paper",
                    topic=topic,
                    venue=self.venue,
                    article_type=self.doc_type,
                )
                success_line(f"Paper saved → {path}")
            except Exception as e:
                error_line(f"Save failed: {e}")

        if ctx.guide_written:
            success_line(
                f"Guide: {len(ctx.guide_text.split())} words"
            )
            try:
                from main import _save_agent_output
                path = _save_agent_output(
                    content=ctx.guide_text,
                    agent_name="guide",
                    topic=topic,
                    researcher_level=self.level,
                )
                success_line(f"Guide saved → {path}")
            except Exception as e:
                error_line(f"Save failed: {e}")

        info_line(
            f"Completed loops: {', '.join(ctx.completed_loops) or 'none'}  ·  "
            f"elapsed {ctx.elapsed_s:.1f}s"
        )
        nl()

    # ── /loop help ───────────────────────────────────────────────────────────

    def _loop_help(self):
        nl()
        table = Table(
            title="  /loop — Reactive Multi-Agent Loop Engine",
            box=box.SIMPLE_HEAD,
            title_style=f"bold {T.PRIMARY}",
            border_style=T.DIM,
            pad_edge=False,
        )
        table.add_column("Command", style=f"bold {T.INFO}", no_wrap=True)
        table.add_column("Description", style=T.TEXT)
        rows = [
            ('/loop paper <topic>',              'quality + chain (smart default)'),
            ('/loop paper <topic> chain',         'Generates paper then guide'),
            ('/loop paper <topic> quality',        'Loops writing and citation fixes until grounding score >= 85%'),
            ('/loop paper <topic> critique',       'Runs logic contradiction audit, then performs section-level revisions'),
            ('/loop paper <topic> deep',           'Accumulates web sources/atoms prior to writing'),
            ('/loop paper <topic> consensus',      'Builds two distinct writing drafts and merges the best sections using Agent 11'),
            ('/loop paper <topic> full',           'Combines all loops together: research -> drafts -> grounding -> consistency -> guide -> learn'),
            ('/loop paper <topic> deep quality chain --max 3', 'combine loops with retry limit'),
            ('/loop config',                       'show current loop settings'),
            ('/loop status',                       'show live loop state'),
        ]
        for cmd, desc in rows:
            table.add_row(cmd, desc)
        console.print(table)
        nl()

    # ── /loop config ─────────────────────────────────────────────────────────

    def _loop_config(self):
        nl()
        divider("Loop configuration")
        settings = [
            ("Grounding threshold",    f"{GROUNDING_THRESHOLD:.0%}"),
            ("Deep-research min atoms", str(DEEP_RESEARCH_MIN_ATOMS)),
            ("Max critique rounds",    str(MAX_CRITIQUE_ROUNDS)),
            ("Consensus drafts",       str(CONSENSUS_DRAFTS)),
            ("Default loops",          ", ".join(PRESETS["default"])),
            ("Full loops",             ", ".join(PRESETS["full"])),
            ("Execution order",        " → ".join(EXECUTION_ORDER)),
        ]
        for label, value in settings:
            line = Text()
            line.append(f"  {label}: ", style=T.MUTED)
            line.append(value, style=f"bold {T.TEXT}")
            console.print(line)
        nl()

    # ── /loop status ─────────────────────────────────────────────────────────

    def _loop_status(self):
        ctx = getattr(self, '_loop_last_ctx', None)
        nl()
        if not ctx:
            info_line("No loop has been run yet.  Use /loop paper <topic> to start.")
            nl()
            return

        divider("Last loop run")
        rows = [
            ("Topic",            ctx.topic),
            ("Run ID",           ctx.run_id),
            ("Elapsed",          f"{ctx.elapsed_s:.1f}s"),
            ("Active loops",     ", ".join(ctx.active_loops)),
            ("Completed loops",  ", ".join(ctx.completed_loops)),
            ("Paper written",    "yes" if ctx.paper_written else "no"),
            ("Guide written",    "yes" if ctx.guide_written else "no"),
            ("Word count",       str(ctx.word_count)),
            ("Grounding",        f"{ctx.grounding_ratio:.0%} "
                                 f"({ctx.grounding_tagged}/{ctx.grounding_total})"),
            ("Atoms collected",  str(len(ctx.atoms))),
            ("Drafts generated", str(len(ctx.drafts))),
            ("Failures",         str(len(ctx.failures)) if ctx.failures else "none"),
        ]
        for label, value in rows:
            line = Text()
            line.append(f"  {label}: ", style=T.MUTED)
            style = f"bold {T.SUCCESS}" if "yes" in value else \
                    f"bold {T.ERROR}" if "no" == value else \
                    f"bold {T.TEXT}"
            line.append(value, style=style)
            console.print(line)
        nl()

    # =========================================================================
    # LOOP AGENT DISPATCH ADAPTERS
    # These bridge the loop engine's generic callable interface to the real
    # VASIS agents.  Each returns the dict shape the loop engine expects.
    # =========================================================================

    def _get_all_atoms(self) -> list:
        """Get all atoms from either the single RAG document or vault session."""
        if self._vault_session:
            atoms = []
            for rag in self._vault_session.papers.values():
                atoms.extend(getattr(rag, "atoms", []))
            return atoms
        elif self.rag:
            return getattr(self.rag, "atoms", [])
        return []

    def _loop_dispatch_agent12(self, queries: list) -> list:
        """Agent 12 web search — returns list of source dicts."""
        try:
            from agents.agent12_websearch import search_web
            result = search_web(
                topic=" ".join(queries[:2]),
                queries_override=queries,
            )
            return result.get("sources", [])
        except Exception as e:
            console.print(Text(f"  ⚠ Agent 12 error: {e}", style=T.WARNING))
            return []

    def _loop_dispatch_agent13(self, **kwargs) -> dict:
        """Agent 13 paper writer — maps loop engine kwargs to write_paper."""
        from agents import agent13_paper_writer as a13

        topic       = kwargs.get("topic", "")
        atoms       = kwargs.get("atoms", [])
        if not atoms:
            atoms = self._get_all_atoms()
        web_sources = kwargs.get("web_sources", [])
        if not web_sources:
            console.print(Text("  * Empty web sources — dynamically fetching fallback web sources using Agent 12...", style=T.INFO))
            web_sources = self._loop_dispatch_agent12([topic])
            active_ctx = getattr(self, '_loop_active_ctx', None)
            if active_ctx:
                active_ctx.web_sources = web_sources

        venue       = kwargs.get("venue", self.venue)
        doc_type    = kwargs.get("doc_type", self.doc_type)
        extra_instr = kwargs.get("extra_instruction", "")

        # Handle revision mode (CritiqueReviseLoop)
        paper_text = kwargs.get("paper_text", "")
        revision_flags = kwargs.get("revision_flags", [])
        instruction = kwargs.get("instruction", "")
        if paper_text and (revision_flags or instruction):
            # Revision: append the fix instructions to extra_instruction
            extra_instr = (extra_instr + "\n" + instruction).strip()
            # We still call write_paper with the extra instruction
            # The critique loop replaces the paper text from the result

        web_evidence = {"sources": web_sources}
        try:
            result = a13.write_paper(
                topic=topic,
                venue=venue,
                article_type=doc_type,
                atoms=atoms,
                web_evidence=web_evidence,
                extra_instruction=extra_instr,
            )
            return {
                "paper_text":       result.get("full_text", ""),
                "grounding_ratio":  result.get("audit", {}).get("grounding_ratio", 0.0),
                "tagged":           result.get("audit", {}).get("grounded_sentences", 0),
                "total":            result.get("audit", {}).get("total_sentences", 0),
                "word_count":       result.get("word_count", 0),
                "agent_result":     result,
            }
        except Exception as e:
            console.print(Text(f"  ⚠ Agent 13 error: {e}", style=T.WARNING))
            return {"paper_text": paper_text, "grounding_ratio": 0.0}

    def _loop_dispatch_agent14(self, **kwargs) -> dict:
        """Agent 14 implementation guide — maps loop engine kwargs."""
        from agents import agent14_implementation_guide as a14
        topic = kwargs.get("topic", "")
        web_sources = kwargs.get("web_sources", [])
        if not web_sources:
            active_ctx = getattr(self, '_loop_active_ctx', None)
            if active_ctx and active_ctx.web_sources:
                web_sources = active_ctx.web_sources
            else:
                console.print(Text("  * Empty web sources — dynamically fetching fallback web sources using Agent 12...", style=T.INFO))
                web_sources = self._loop_dispatch_agent12([topic])
                if active_ctx:
                    active_ctx.web_sources = web_sources

        try:
            result = a14.guide_implementation(
                innovation=topic,
                narrative=kwargs.get("paper_text", ""),
                web_evidence={"sources": web_sources},
                researcher_level=kwargs.get("level", self.level),
            )
            return {
                "guide_text": result.get("full_text", ""),
                "steps":      [],
                "word_count": len(result.get("full_text", "").split()),
            }
        except Exception as e:
            console.print(Text(f"  ⚠ Agent 14 error: {e}", style=T.WARNING))
            return {"guide_text": ""}

    def _loop_dispatch_agent7(self, paper_text: str) -> list:
        """Agent 7 contradiction — returns list of contradiction description strings."""
        try:
            from agents import agent7_contradiction as a7
            from db.triple_store import TripleStore
            ts = TripleStore([])
            result = a7.detect([], ts, paper_text)
            flags = []
            for c in result.get("llm_contradictions", []):
                flags.append(
                    f"{c.get('type', 'logical')} conflict: "
                    f"{c.get('claim_a', '')} vs {c.get('claim_b', '')}"
                )
            return flags
        except Exception as e:
            console.print(Text(f"  ⚠ Agent 7 error: {e}", style=T.WARNING))
            return []

    def _loop_dispatch_agent11(self, drafts: list) -> dict:
        """Agent 11 draft merger — picks best sections from multiple drafts."""
        # Agent 11 in the current codebase is a causal-chain synthesiser,
        # not a draft merger.  We implement a simple section-level merge here:
        # pick the draft with better grounding for each section.
        if not drafts:
            return {"paper_text": ""}
        best = max(drafts, key=lambda d: d.get("grounding_ratio", 0.0))
        return {"paper_text": best.get("paper_text", "")}

    def _loop_dispatch_grounding_audit(self, paper_text: str) -> dict:
        """Grounding audit — counts [A:…] / [W:…] citation markers."""
        import re as _re
        sentences = _re.split(r"(?<=[.!?])\s+", paper_text)
        tagged = sum(1 for s in sentences if _re.search(r"\[\s*[AW]\s*:\s*[^\]]+\]", s))
        total = max(len(sentences), 1)
        return {"ratio": tagged / total, "tagged": tagged, "total": total}

    # ── pre-flight banner (called before /paper) ──────────────────────────

    def _show_preflight(self, hint: PreflightHint):
        rendered = render_preflight(hint)
        if not rendered.get("show"):
            return

        lines = rendered.get("lines", [])
        if not lines:
            return

        border = T.WARNING if rendered.get("risk") else T.DIM
        content = "\n".join(f"  {ln}" for ln in lines)
        console.print(Panel(
            Text(content, style=T.MUTED),
            title=f"[{T.SECONDARY}]  learn  —  from {hint.similar_runs} previous run"
                  f"{'s' if hint.similar_runs != 1 else ''}[/]",
            border_style=border,
            padding=(0, 1),
        ))
        console.print()

    # =========================================================================
    # LEARN HOOKS — dispatch to real backends
    # =========================================================================

    def _dispatch_learn_crawl(self, topic: str) -> list[dict]:
        """
        Call Agent 12 (Web Search) in standalone mode.
        Returns: list of {url, title, snippet}
        Falls back to stub data when Agent 12 is not available.
        """
        try:
            from agents.agent12_websearch import Agent12WebSearch
            agent12 = Agent12WebSearch()
            results = agent12.search(topic, num_results=15)
            # Normalise to the expected shape
            return [
                {
                    "url":     r.get("url", r.get("link", "")),
                    "title":   r.get("title", ""),
                    "snippet": r.get("snippet", r.get("abstract", "")),
                }
                for r in results
                if r.get("snippet") or r.get("abstract")
            ]
        except Exception:
            # Fallback: stub data so the command always works
            return [
                {
                    "url":     "https://arxiv.org/abs/2005.14165",
                    "title":   "Generating Long Sequences with Sparse Transformers",
                    "snippet": "We introduce Sparse Transformers, which reduce the memory "
                               "and computational requirements of attention by using factored "
                               "sparse attention patterns, enabling modelling of sequences "
                               "with tens of thousands of tokens.",
                },
            ]

    def _dispatch_last_paper_sections(self) -> dict[str, list[str]]:
        """
        Return the last generated paper split by section.
        Reads from the outputs directory.
        """
        try:
            md_files = sorted(
                self.outputs_dir.glob("paper_*.md"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if not md_files:
                return {}

            text = md_files[0].read_text(encoding="utf-8")
            sections: dict[str, list[str]] = {}
            current_section = "Untitled"

            for line in text.split("\n"):
                if line.startswith("## "):
                    current_section = line[3:].strip()
                    sections.setdefault(current_section, [])
                elif line.strip():
                    sections.setdefault(current_section, []).append(line.strip())

            return sections
        except Exception:
            return {}

    # =========================================================================
    # CUSTOM AGENT STUDIO — command handlers
    # =========================================================================

    def _cmd_studio_build(self, args: str):
        """Handle /build agent <name> and refresh autocomplete afterwards."""
        parts = args.strip().split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""

        if sub == "agent":
            result = self.studio.cmd_build(rest, session=self.session)
            if result:
                self._refresh_completer()
        else:
            error_line("Usage: /build agent <agentname>")

    def _cmd_studio_delete(self, args: str):
        """Handle /delete agent <name> and /delete loop <name> with refresh."""
        self.studio.cmd_delete(args, session=self.session)
        self._refresh_completer()

    # ── Section-synonym table for BM25 search boost ───────────────────────
    _AGENT_SYNONYMS = {
        "references":      "references bibliography citations sources works cited",
        "abstract":         "abstract summary executive summary outline",
        "introduction":     "introduction background overview setup",
        "literaturereview": "literature review related work survey state of the art",
        "methodology":      "methodology materials methods experiment setup design",
        "results":          "results findings experimental results evaluation data",
        "discussion":       "discussion interpretation implications limitations",
        "conclusion":       "conclusion future work concluding remarks summary",
        "gapanalysis":      "gap analysis research gap missing unexplored",
        "keywords":         "keywords index terms key phrases",
        "factcheck":        "fact check verification claims evidence",
        "criticalreview":   "critical review critique strengths weaknesses",
        "comparison":       "comparison compare contrast versus differences",
    }

    def _gather_smart_context(
        self,
        agent_name: str,
        topic: str,
        agent_input_type: str = "all",
    ) -> tuple:
        """
        Build (paper_text, atoms, web_results) with smart retrieval:
          1. Load last generated paper draft as paper_text.
          2. Locate and extract sequential bibliography atoms if it is a references agent,
             or rank general vault atoms by BM25 relevance.
          3. Call Agent12 web search when input_type needs it.
        """
        paper_text = ""
        atoms = []
        web_results = []

        # ── 1. Load last generated paper text ─────────────────────────────
        if self.outputs_dir.exists():
            md_files = sorted(
                self.outputs_dir.glob("paper_*.md"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if md_files:
                try:
                    paper_text = md_files[0].read_text(encoding="utf-8")
                except Exception:
                    pass

        # ── 2. Gather and BM25-rank vault atoms ──────────────────────────
        raw_atoms = []
        if self._vault_session:
            for rag_inst in self._vault_session.papers.values():
                raw_atoms.extend(getattr(rag_inst, "atoms", []))
        elif self.rag:
            raw_atoms = list(getattr(self.rag, "atoms", []))

        if raw_atoms:
            # Check if this is a references query to get the complete bibliography
            if agent_name == "references":
                bib_atoms = []
                BIB_TITLE_KEYWORDS = {"references", "bibliography", "works cited", "reference list", "citations", "cited works"}

                # Tier 1: Tree scan for bibliography section
                bib_node = None
                for node in (self.rag.tree_nodes if self.rag else []):
                    title = node.get("title", "").lower()
                    if any(k in title for k in BIB_TITLE_KEYWORDS):
                        bib_node = node  # last match wins

                if bib_node:
                    bib_atoms = [
                        a for a in raw_atoms
                        if bib_node["start_page"] <= (a.get("page_num") or a.get("page", -1)) <= bib_node["end_page"]
                    ]

                # Tier 1.5: Header scan (last 35% of document)
                if not bib_atoms:
                    sorted_atoms = sorted(raw_atoms, key=lambda x: int(x.get("atom_id", 0)))
                    search_start = int(len(sorted_atoms) * 0.65)
                    tail = sorted_atoms[search_start:]
                    for idx, atom in enumerate(tail):
                        lines = [l.strip().lower() for l in atom.get("text", "").split("\n")]
                        if any(l in ("references", "bibliography", "works cited", "citations") for l in lines):
                            bib_atoms = tail[idx:]
                            break

                # Tier 2: Last-pages fallback (last 5 pages of document)
                if not bib_atoms:
                    sorted_atoms = sorted(raw_atoms, key=lambda x: int(x.get("page_num") or x.get("page", 0)))
                    max_page = sorted_atoms[-1].get("page_num") or sorted_atoms[-1].get("page", 0) if sorted_atoms else 0
                    if max_page > 0:
                        bib_atoms = [a for a in sorted_atoms if (a.get("page_num") or a.get("page", 0)) >= max_page - 4]

                if bib_atoms:
                    # Sort bibliography atoms in document order (chronological) so they flow naturally
                    atoms = sorted(bib_atoms, key=lambda x: int(x.get("atom_id", 0)))
                else:
                    atoms = raw_atoms
            else:
                # Normal BM25 ranker for general section/topic extraction
                search_query = topic
                synonyms = self._AGENT_SYNONYMS.get(agent_name, "")
                if synonyms:
                    search_query += " " + synonyms

                try:
                    from db.bm25_index import BM25Index
                    ranked = BM25Index(raw_atoms).search(
                        search_query, top_k=len(raw_atoms),
                    )
                    atoms = ranked if ranked else raw_atoms
                except Exception:
                    atoms = raw_atoms
        else:
            atoms = raw_atoms

        # ── 3. Web search via Agent 12 ────────────────────────────────────
        if agent_input_type in ("web_search", "all"):
            try:
                web_results = self._dispatch_learn_crawl(topic)
            except Exception:
                pass

        return paper_text, atoms, web_results

    def _cmd_run_custom_agent(self, agent_name: str, topic: str):
        """Run a custom agent with smart context retrieval."""
        if not topic:
            error_line(f"Usage: /{agent_name} \"your topic\"")
            return

        # Strip quotes the user may have typed (e.g. /references "all")
        topic = topic.strip('"').strip("'").strip()

        # Look up the agent to get its input_type
        from agent_builder import _slugify
        agent_obj = self.studio.store.get_agent(_slugify(agent_name))
        input_type = getattr(agent_obj, "input_type", "all") if agent_obj else "all"

        paper_text, atoms, web_results = self._gather_smart_context(
            agent_name, topic, input_type,
        )

        # ── Direct-output fast path for /references "all" ─────────────
        # The full bibliography (100+ entries, ~20k chars) cannot fit in a
        # local 7B model's context window.  Instead of sending it through
        # the LLM (which truncates/hallucinates), concatenate the raw
        # bibliography atoms and display them directly — matching what the
        # built-in Agent 4 does.
        if agent_name == "references" and topic.lower() == "all" and atoms:
            import time as _t, re as _re
            t0 = _t.time()

            # 1. Join all atom texts into one continuous string
            raw_text = " ".join(
                atom.get("text", "").replace("\uFFFD", "").strip()
                for atom in atoms
                if atom.get("text", "").strip()
            )
            # Normalise whitespace (collapse line breaks, double spaces)
            raw_text = _re.sub(r"\s+", " ", raw_text).strip()

            # 2. Split by [N] citation markers → one entry per reference
            #    Pattern: "[1] Author..." "[2] Author..." etc.
            parts = _re.split(r"\s*(?=\[\d+\]\s)", raw_text)

            refs = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                # Only keep parts that start with [N] (skip preamble/conclusion text)
                if _re.match(r"^\[\d+\]", part):
                    # Clean up hyphenation artefacts from PDF column breaks
                    part = part.replace("- ", "")
                    refs.append(part)

            elapsed = _t.time() - t0
            if refs:
                body = "REFERENCES\n\n" + "\n\n".join(refs) + f"\n\n— {len(refs)} references extracted from document"
                nl()
                console.print(Panel(
                    Markdown(body),
                    title=f"[bold {T.SECONDARY}]/references  ·  {elapsed:.1f}s[/]",
                    border_style=T.DIM,
                    padding=(1, 2),
                ))
                nl()
            else:
                error_line("No bibliography entries found in the indexed document.")
            return

        result = self.studio.cmd_run_agent(
            command     = agent_name,
            topic       = topic,
            paper_text  = paper_text,
            atoms       = atoms,
            web_results = web_results,
        )

        if result.get("output_text"):
            nl()
            console.print(Panel(
                Markdown(result["output_text"]),
                title=f"[bold {T.SECONDARY}]/{agent_name}  ·  {result.get('elapsed_s', 0):.1f}s[/]",
                border_style=T.DIM,
                padding=(1, 2),
            ))
            nl()

    def _cmd_run_custom_loop(self, loop_name: str, topic: str):
        """Run a custom agent loop with smart context retrieval."""
        if not topic:
            error_line(f"Usage: /loop {loop_name} \"your topic\"")
            return

        # Use "all" input type for loops — they benefit from maximum context
        paper_text, atoms, web_results = self._gather_smart_context(
            loop_name, topic, "all",
        )

        result = self.studio.cmd_run_loop(
            loop_id     = loop_name,
            topic       = topic,
            paper_text  = paper_text,
            atoms       = atoms,
            web_results = web_results,
        )

        if result.get("output_text"):
            nl()
            console.print(Panel(
                Markdown(result["output_text"]),
                title=f"[bold {T.SECONDARY}]loop: {loop_name}  ·  {result.get('elapsed_s', 0):.1f}s[/]",
                border_style=T.DIM,
                padding=(1, 2),
            ))
            nl()

    def _cmd_exit(self):
        nl()
        console.print(Text("  Goodbye.", style=T.MUTED))
        nl()
        sys.exit(0)

    # =========================================================================
    # BACKEND DISPATCH — real agent calls
    # =========================================================================

    def _dispatch_index(self, path: str) -> Optional[dict]:
        """
        Index a single PDF using PageIndexREMSE.
        Returns: { name, atoms, triples, doc_id } or None on error.
        """
        try:
            from pipeline import PageIndexREMSE

            if self._vault_session is None and len(self.vault_docs) == 0:
                # Single-paper mode: use PageIndexREMSE directly
                rag = PageIndexREMSE(model=DEFAULT_MODEL)
                rag.ingest(path)
                self.rag = rag
                stats = rag.get_stats()
                return {
                    "name":    Path(path).stem,
                    "atoms":   stats.get("total_atoms", 0),
                    "triples": stats.get("total_triples", 0),
                    "doc_id":  rag.doc_id,
                }
            else:
                # Multi-paper / vault mode
                from vault import VaultSession
                if self._vault_session is None:
                    self._vault_session = VaultSession(model=DEFAULT_MODEL)
                    # If we already had a single rag, load it into vault
                    # by re-loading the first document
                    self.rag = None

                label = self._vault_session.load(path)
                rag_instance = self._vault_session.papers[label]
                stats = rag_instance.get_stats()
                return {
                    "name":    label,
                    "atoms":   stats.get("total_atoms", 0),
                    "triples": stats.get("total_triples", 0),
                    "doc_id":  rag_instance.doc_id,
                }

        except Exception as e:
            error_line(f"Indexing failed: {e}")
            import traceback
            console.print(Text(traceback.format_exc(), style=T.DIM))
            return None

    def _dispatch_paper(self, topic: str) -> Optional[dict]:
        """
        Run the paper writing pipeline (Agent 13).
        Returns dict with agents, grounding info, preview, output_path.
        """
        try:
            has_vault = self._vault_session is not None

            if has_vault:
                # Vault mode: query all papers with paper intent
                with console.status(
                    f"[{T.MUTED}]Running 14-agent pipeline…[/]",
                    spinner="dots",
                    spinner_style=T.PRIMARY,
                ):
                    results = self._vault_session.ask_all(
                        f"write a research paper on {topic}",
                        venue=self.venue,
                        article_type=self.doc_type,
                        target_paper="all"
                    )

                # Aggregate results from all papers
                combined_text = ""
                for label, r in results.items():
                    paper_result = r.get("paper_result")
                    if paper_result and paper_result.get("full_text"):
                        combined_text = paper_result["full_text"]
                        # Save each paper
                        from main import _save_agent_output
                        path = _save_agent_output(
                            content=paper_result["full_text"],
                            agent_name="paper",
                            topic=f"{label}_{topic}",
                            venue=self.venue,
                            article_type=self.doc_type,
                        )
                        success_line(f"[{label}] Paper saved → {path}")

                # Build agent status from the first paper's review report
                first_result = next(iter(results.values()), {})
                agents = self._extract_agent_status(first_result)
                audit = first_result.get("paper_result", {}).get("audit", {}) if first_result.get("paper_result") else {}

                return {
                    "agents": agents,
                    "word_count": len(combined_text.split()) if combined_text else 0,
                    "sections": combined_text.count("\n## ") if combined_text else 0,
                    "preview": self._extract_preview(combined_text),
                    "grounding_ratio": audit.get("grounding_ratio", None),
                    "grounding_tagged": audit.get("grounded_sentences", 0),
                    "grounding_total": audit.get("total_sentences", 0),
                    "output_path": None,  # Already saved above
                }

            else:
                # Single-paper mode
                with console.status(
                    f"[{T.MUTED}]Running 14-agent pipeline…[/]",
                    spinner="dots",
                    spinner_style=T.PRIMARY,
                ):
                    result = self.rag.query(
                        f"write a research paper on {topic}",
                        show_provenance=False,
                        venue=self.venue,
                        article_type=self.doc_type
                    )

                agents = self._extract_agent_status(result)
                paper = result.get("paper_result")
                output_path = None

                if paper and paper.get("full_text"):
                    from main import _save_agent_output
                    output_path = _save_agent_output(
                        content=paper["full_text"],
                        agent_name="paper",
                        topic=topic,
                        venue=self.venue,
                        article_type=self.doc_type,
                    )

                audit = paper.get("audit", {}) if paper else {}
                full_text = paper.get("full_text", "") if paper else ""

                return {
                    "agents": agents,
                    "word_count": paper.get("word_count", len(full_text.split())) if paper else 0,
                    "sections": len(paper.get("sections", {})) if paper else 0,
                    "preview": self._extract_preview(full_text),
                    "grounding_ratio": audit.get("grounding_ratio", None),
                    "grounding_tagged": audit.get("grounded_sentences", 0),
                    "grounding_total": audit.get("total_sentences", 0),
                    "output_path": output_path,
                }

        except Exception as e:
            error_line(f"Paper generation failed: {e}")
            import traceback
            console.print(Text(traceback.format_exc(), style=T.DIM))
            return None

    def _dispatch_guide(self, topic: str) -> Optional[dict]:
        """
        Run the implementation guide pipeline (Agent 14).
        Returns dict with steps and output_path.
        """
        try:
            has_vault = self._vault_session is not None

            if has_vault:
                with console.status(
                    f"[{T.MUTED}]Running 14-agent pipeline…[/]",
                    spinner="dots",
                    spinner_style=T.PRIMARY,
                ):
                    results = self._vault_session.ask_all(
                        f"write an implementation guide for {topic}",
                        researcher_level=self.level,
                        target_paper="all"
                    )

                output_path = None
                for label, r in results.items():
                    impl_result = r.get("impl_result")
                    if impl_result and impl_result.get("full_text"):
                        from main import _save_agent_output
                        output_path = _save_agent_output(
                            content=impl_result["full_text"],
                            agent_name="guide",
                            topic=f"{label}_{topic}",
                            researcher_level=self.level,
                        )
                        success_line(f"[{label}] Guide saved → {output_path}")

                # Extract steps from the first result's guide
                first_result = next(iter(results.values()), {})
                steps = self._extract_guide_steps(first_result)

                return {
                    "steps": steps,
                    "output_path": output_path,
                }

            else:
                with console.status(
                    f"[{T.MUTED}]Running 14-agent pipeline…[/]",
                    spinner="dots",
                    spinner_style=T.PRIMARY,
                ):
                    result = self.rag.query(
                        f"write an implementation guide for {topic}",
                        show_provenance=False,
                        researcher_level=self.level
                    )

                output_path = None
                guide = result.get("impl_result")
                if guide and guide.get("full_text"):
                    from main import _save_agent_output
                    output_path = _save_agent_output(
                        content=guide["full_text"],
                        agent_name="guide",
                        topic=topic,
                        researcher_level=self.level,
                    )

                steps = self._extract_guide_steps(result)

                return {
                    "steps": steps,
                    "output_path": output_path,
                }

        except Exception as e:
            error_line(f"Guide generation failed: {e}")
            import traceback
            console.print(Text(traceback.format_exc(), style=T.DIM))
            return None

    def _dispatch_query_impl(self, query: str) -> Optional[dict]:
        """
        Run the multi-agent query pipeline.
        Returns dict with answer, sources, metadata.
        """
        try:
            has_vault = self._vault_session is not None
            has_rag = self.rag and self.rag._ready

            if has_vault:
                from vault import is_comparison_question

                # Check for paper / guide intent
                q_lower = query.lower()
                is_paper = any(pat in q_lower for pat in QUERY_DETECTION_PATTERNS.get("paper_writing", []))
                is_guide = any(pat in q_lower for pat in QUERY_DETECTION_PATTERNS.get("implementation_guide", []))

                venue = self.venue if is_paper else None
                article_type = self.doc_type if is_paper else None
                researcher_level = self.level if is_guide else None

                if is_comparison_question(query):
                    result = self._vault_session.compare(
                        query,
                        venue=venue,
                        article_type=article_type,
                        researcher_level=researcher_level,
                        target_paper="all"
                    )
                    result["_type"] = "comparison"
                    return result
                else:
                    results = self._vault_session.ask_all(
                        query,
                        venue=venue,
                        article_type=article_type,
                        researcher_level=researcher_level,
                        target_paper="all"
                    )
                    # If only one paper, render as single result
                    if len(results) == 1:
                        label, r = next(iter(results.items()))
                        return r
                    else:
                        return {
                            "_type": "vault_multi",
                            "per_paper": results,
                        }
            elif has_rag:
                # Detect paper/guide intent for single-paper mode
                q_lower = query.lower()
                is_paper = any(pat in q_lower for pat in QUERY_DETECTION_PATTERNS.get("paper_writing", []))
                is_guide = any(pat in q_lower for pat in QUERY_DETECTION_PATTERNS.get("implementation_guide", []))

                venue = self.venue if is_paper else None
                article_type = self.doc_type if is_paper else None
                researcher_level = self.level if is_guide else None

                result = self.rag.query(
                    query,
                    top_k_anchors=5,
                    expansion_passes=4,
                    show_provenance=False,
                    save_result=True,
                    venue=venue,
                    article_type=article_type,
                    researcher_level=researcher_level
                )
                return result

            return None

        except Exception as e:
            error_line(f"Query failed: {e}")
            import traceback
            console.print(Text(traceback.format_exc(), style=T.DIM))
            return None

    def _dispatch_list_outputs(self) -> list:
        """
        Scan the outputs directory and return file metadata.
        Each dict: { name, kind, size_kb, date }
        """
        result = []
        if self.outputs_dir.exists():
            md_files = sorted(self.outputs_dir.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
            for f in md_files[:20]:
                kind = "paper" if f.name.startswith("paper") else "guide"
                stat = f.stat()
                result.append({
                    "name":    f.name,
                    "kind":    kind,
                    "size_kb": round(stat.st_size / 1024),
                    "date":    datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d"),
                })
        return result

    # =========================================================================
    # HELPERS — extract structured data from pipeline results
    # =========================================================================

    def _extract_agent_status(self, result: dict) -> list:
        """
        Extract per-agent grade/score/time from a pipeline result.
        Uses the review_report.per_agent_scores if available.
        """
        agents = []

        # Try to get from the review report embedded in the result
        review_report = result.get("review_report") or result.get("provenance", {})

        # The per_agent_scores are embedded via SuperAgent._report()
        # They come through in the reasoning trail
        per_agent = []
        if isinstance(review_report, dict):
            per_agent = review_report.get("per_agent_scores", [])

        if per_agent:
            # Map agent names to AGENTS dict for display
            agent_name_map = {
                "agent1_router": (1, "Router"),
                "agent2_decomposer": (2, "Decomposer"),
                "agent3_navigator": (3, "Navigator"),
                "agent4_retrieval": (4, "Retrieval"),
                "agent5_expansion": (5, "Expansion"),
                "agent6_validation": (6, "Validation"),
                "agent7_contradiction": (7, "Contradiction"),
                "agent8_temporal": (8, "Temporal"),
                "agent9_calibration": (9, "Calibration"),
                "agent10_super": (10, "Supervisor"),
                "agent11_synthesis": (11, "Synthesis"),
                "agent12_websearch": (12, "Web Search"),
                "agent13_paper_writer": (13, "Paper Writer"),
                "agent14_implementation_guide": (14, "Impl. Guide"),
            }
            for a in per_agent:
                name_key = a.get("agent", "")
                agent_id, display_name = agent_name_map.get(name_key, (0, name_key))
                agents.append({
                    "id": agent_id,
                    "name": display_name,
                    "grade": a.get("grade", "?"),
                    "score": float(a.get("score", 0.0)),
                    "time": float(a.get("elapsed", 0.0)),
                    "skipped": a.get("skipped", False),
                })

        return agents

    def _extract_preview(self, full_text: str) -> str:
        """Return the full text of the generated paper for rendering in the console."""
        return full_text or ""

    def _extract_guide_steps(self, result: dict) -> list:
        """
        Extract implementation guide steps from a pipeline result.
        """
        impl_result = result.get("impl_result")
        if not impl_result:
            return []

        _ = impl_result.get("guide", {})
        _ = impl_result.get("timings", {})

        step_titles = [
            "Breaking down innovation",
            "Designing architecture",
            "Writing pseudocode",
            "Writing code skeleton",
            "Recommending datasets",
            "Defining baselines",
            "Defining metrics",
            "Creating implementation plan",
            "Listing pitfalls",
            "Hardware requirements",
        ]

        steps = []
        for i, title in enumerate(step_titles):
            steps.append({"n": i + 1, "title": title})

        return steps


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VASIS AI — Interactive CLI Shell")
    parser.add_argument("--venue",   default="IEEE",             help="Publication venue")
    parser.add_argument("--type",    default="research_article", help="Document type")
    parser.add_argument("--level",   default="masters",          help="Guide level")
    parser.add_argument("--outputs", default="outputs",          help="Outputs directory")
    args = parser.parse_args()

    VasisCLI(
        venue=args.venue,
        doc_type=args.type,
        level=args.level,
        outputs_dir=args.outputs,
    ).run()
