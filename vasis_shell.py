"""
vasis_shell.py — VASIS AI Agents Textual TUI
Claude Code-style full-screen terminal interface.

Drop this file into the root of your VASIS-AI-Agents-main folder and run:
    pip install textual
    python vasis_shell.py

Or with a PDF pre-loaded:
    python vasis_shell.py path/to/paper.pdf
"""

import sys
import os
import re
from datetime import datetime

from textual.app import App, ComposeResult
from textual.widgets import (
    Header, Footer, Static, Input, RichLog,
    Button
)
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from textual import work
from textual.binding import Binding
from textual.screen import ModalScreen

# Import Agent-specific routing patterns
from agent_routing_rules import QUERY_DETECTION_PATTERNS

# ── Learn Engine ──────────────────────────────────────────────────────────────
from learn_engine import LearnEngine
# ── Loop Engine ───────────────────────────────────────────────────────────────
from loop_engine import (
    LoopOrchestrator, parse_loop_command,
    PRESETS, EXECUTION_ORDER,
    GROUNDING_THRESHOLD, DEEP_RESEARCH_MIN_ATOMS,
    MAX_CRITIQUE_ROUNDS, CONSENSUS_DRAFTS,
)


# ─── AGENT DEFINITIONS ───────────────────────────────────────────────────────

AGENTS = [
    ("01", "Router",           "Query routing & intent detection"),
    ("02", "Decomposer",       "Sub-query decomposition"),
    ("03", "Navigator",        "PageIndex tree navigation"),
    ("04", "Retrieval",        "Atom anchor selection"),
    ("05", "Expansion",        "RE-MSE stateful expansion"),
    ("06", "Validation",       "Evidence validation"),
    ("07", "Contradiction",    "Cross-claim conflict detection"),
    ("08", "Temporal",         "Timeline & recency reasoning"),
    ("09", "Calibration",      "Confidence calibration"),
    ("10", "Supervisor",       "SuperAgent orchestrator"),
    ("11", "Synthesis",        "Consensus synthesis"),
    ("12", "Web Search",       "Live web retrieval"),
    ("13", "Paper Writer",     "Academic paper generation"),
    ("14", "Impl. Guide",      "Implementation guide generation"),
]

VENUES = ["IEEE", "DSJ", "Elsevier", "Springer", "ACM", "NeurIPS", "ICML", "ICLR"]
ARTICLE_TYPES = [
    ("research_article",    "Research Article"),
    ("review_article",      "Review Article"),
    ("systematic_review",   "Systematic Review"),
    ("short_communication", "Short Communication"),
    ("perspective_article", "Perspective"),
    ("technical_note",      "Technical Note"),
    ("case_study",          "Case Study"),
    ("letter_to_editor",    "Letter to Editor"),
]
RESEARCHER_LEVELS = ["beginner", "masters", "phd"]


# ─── STDOUT REDIRECTOR ───────────────────────────────────────────────────────

class TuiStdoutRedirector:
    """Redirects stdout printing (from agents and query execution) to the TUI chat log."""
    def __init__(self, app):
        self.app = app
        self.buffer = ""

    def write(self, data: str):
        self.buffer += data
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip()
            if line:
                # Strip raw ANSI escape sequences
                clean_line = re.sub(r'\x1b\[[0-9;]*[mK]', '', line)
                if clean_line:
                    # Color-code outputs based on the emitting Agent prefix
                    if clean_line.startswith("[Agent13]"):
                        clean_line = f"[bold magenta]{clean_line}[/bold magenta]"
                    elif clean_line.startswith("[Agent14]"):
                        clean_line = f"[bold blue]{clean_line}[/bold blue]"
                    elif clean_line.startswith("[Agent10]") or clean_line.startswith("[SuperAgent]"):
                        clean_line = f"[bold cyan]{clean_line}[/bold cyan]"
                    elif any(clean_line.startswith(f"[Agent{i}]") for i in range(1, 13)):
                        clean_line = f"[bold green]{clean_line}[/bold green]"
                    elif "error" in clean_line.lower() or "fail" in clean_line.lower() or "exception" in clean_line.lower():
                        clean_line = f"[bold red]{clean_line}[/bold red]"
                    elif "warning" in clean_line.lower() or "warn" in clean_line.lower() or "skip" in clean_line.lower():
                        clean_line = f"[bold yellow]{clean_line}[/bold yellow]"
                    
                    self.app.call_from_thread(self.app.chat_log.write, clean_line)

    def flush(self):
        pass


# ─── HELP MODAL ──────────────────────────────────────────────────────────────

class HelpModal(ModalScreen):
    BINDINGS = [("escape", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("[bold orange]VASIS COMMANDS[/bold orange]\n", id="help-title"),
            Static(
                "[bold cyan]DOCUMENT COMMANDS[/bold cyan]\n"
                "  [bold]/index [path][/bold]       Index a PDF document\n"
                "  [bold]/vault p1.pdf p2.pdf[/bold] Load multiple PDFs for cross-paper analysis\n"
                "  [bold]/status[/bold]             Show document stats (tree nodes, atoms, triples)\n"
                "  [bold]/tree[/bold]               Display PageIndex tree structure\n"
                "  [bold]/history[/bold]            Show query history for current document\n\n"
                "[bold cyan]QUERY COMMANDS[/bold cyan]\n"
                "  [bold]/query [question][/bold]    Ask any question about the loaded document\n"
                "  [bold]/paper [topic][/bold]       Generate an academic research paper (Agent 13)\n"
                "  [bold]/guide [topic][/bold]       Generate an implementation guide (Agent 14)\n\n"
                "[bold cyan]SETTINGS COMMANDS[/bold cyan]\n"
                "  [bold]/venue [name][/bold]        Set paper writing target journal/venue (e.g. IEEE, DSJ)\n"
                "  [bold]/type [type][/bold]          Set article type (e.g. research_article, review_article)\n"
                "  [bold]/level [level][/bold]        Set researcher level (beginner, masters, phd)\n\n"
                "[bold cyan]LOOP COMMANDS[/bold cyan]\n"
                "  [bold]/loop paper [topic][/bold]   Run loops on paper writing\n"
                "  [bold]/loop config[/bold]          Show current loop config\n"
                "  [bold]/loop status[/bold]          Show live loop status\n\n"
                "[bold cyan]SYSTEM COMMANDS[/bold cyan]\n"
                "  [bold]/agents[/bold]             Show all 14 agents and their roles\n"
                "  [bold]/outputs[/bold]            List generated papers and guides\n"
                "  [bold]/clear[/bold]              Clear the chat log\n"
                "  [bold]/help[/bold]               Show this help\n"
                "  [bold]/exit[/bold]               Exit VASIS\n\n"
                "[bold cyan]KEYBOARD SHORTCUTS[/bold cyan]\n"
                "  [bold]Ctrl+C[/bold]  Exit\n"
                "  [bold]Ctrl+L[/bold]  Clear chat log\n"
                "  [bold]F1[/bold]      Toggle help\n"
                "  [bold]F2[/bold]      Toggle agent panel\n",
                id="help-body"
            ),
            Button("Close  [dim][Esc][/dim]", id="help-close", variant="primary"),
            id="help-modal"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss()

    DEFAULT_CSS = """
    HelpModal {
        align: center middle;
    }
    #help-modal {
        background: #0f1524;
        border: solid #1e293b;
        padding: 1 2;
        width: 76;
        height: auto;
        max-height: 95vh;
    }
    #help-title {
        color: orange;
        text-align: center;
        padding-bottom: 1;
    }
    #help-body {
        padding-bottom: 1;
    }
    #help-close {
        width: 100%;
    }
    """


# ─── CUSTOMIZER MODALS ────────────────────────────────────────────────────────

class PaperCustomizer(ModalScreen):
    def __init__(self, topic: str):
        super().__init__()
        self.topic = topic
        self.selected_venue = "IEEE"
        self.selected_type = "research_article"

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("[bold orange]📄 CUSTOMISE RESEARCH PAPER[/bold orange]\n", id="modal-title"),
            Static("[bold cyan]1. Select Target Venue / Journal[/bold cyan]"),
            Horizontal(
                *[Button(v, id=f"venue-{v}", classes="opt-btn venue-btn" + (" active-btn" if v == "IEEE" else "")) for v in VENUES],
                id="venue-row"
            ),
            Static("\n[bold cyan]2. Select Article Type[/bold cyan]"),
            ScrollableContainer(
                *[Button(label, id=f"type-{t_key}", classes="opt-btn type-btn" + (" active-btn" if t_key == "research_article" else "")) for t_key, label in ARTICLE_TYPES],
                id="type-container"
            ),
            Horizontal(
                Button("Cancel", id="btn-cancel", variant="error"),
                Button("Generate Paper 🚀", id="btn-submit", variant="success"),
                id="btn-row"
            ),
            id="customizer-modal"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "btn-cancel":
            self.dismiss()
        elif button_id == "btn-submit":
            self.dismiss((self.selected_venue, self.selected_type))
        elif button_id.startswith("venue-"):
            v = button_id[6:]
            self.selected_venue = v
            for btn in self.query(".venue-btn"):
                btn.remove_class("active-btn")
            event.button.add_class("active-btn")
        elif button_id.startswith("type-"):
            t = button_id[5:]
            self.selected_type = t
            for btn in self.query(".type-btn"):
                btn.remove_class("active-btn")
            event.button.add_class("active-btn")

    DEFAULT_CSS = """
    PaperCustomizer {
        align: center middle;
    }
    #customizer-modal {
        background: #0f1524;
        border: solid #1e293b;
        padding: 1 2;
        width: 80;
        height: auto;
        max-height: 95vh;
    }
    #modal-title {
        text-align: center;
        padding-bottom: 1;
    }
    #venue-row {
        height: 3;
        margin: 1 0;
        align: center middle;
    }
    #type-container {
        height: 10;
        border: solid #1e293b;
        background: #0b0f17;
        padding: 1 1;
        margin: 1 0;
    }
    .opt-btn {
        min-width: 10;
        height: 3;
        margin: 0 1;
    }
    .type-btn {
        width: 100%;
        margin-bottom: 1;
        content-align: left middle;
    }
    .active-btn {
        background: #38bdf8 !important;
        color: #0b0f17 !important;
        text-style: bold;
    }
    #btn-row {
        height: 3;
        margin-top: 1;
        align: center middle;
    }
    #btn-row Button {
        width: 25;
        margin: 0 2;
    }
    """


class GuideCustomizer(ModalScreen):
    def __init__(self, topic: str):
        super().__init__()
        self.topic = topic
        self.selected_level = "masters"

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("[bold orange]🔧 CUSTOMISE IMPLEMENTATION GUIDE[/bold orange]\n", id="modal-title"),
            Static("[bold cyan]Select Researcher Level[/bold cyan]\n"),
            Horizontal(
                Button("Beginner", id="lvl-beginner", classes="opt-btn lvl-btn"),
                Button("Masters", id="lvl-masters", classes="opt-btn lvl-btn active-btn"),
                Button("PhD", id="lvl-phd", classes="opt-btn lvl-btn"),
                id="level-row"
            ),
            Horizontal(
                Button("Cancel", id="btn-cancel", variant="error"),
                Button("Generate Guide 🚀", id="btn-submit", variant="success"),
                id="btn-row"
            ),
            id="guide-modal-container"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "btn-cancel":
            self.dismiss()
        elif button_id == "btn-submit":
            self.dismiss(self.selected_level)
        elif button_id.startswith("lvl-"):
            lvl = button_id[4:]
            self.selected_level = lvl
            for btn in self.query(".lvl-btn"):
                btn.remove_class("active-btn")
            event.button.add_class("active-btn")

    DEFAULT_CSS = """
    GuideCustomizer {
        align: center middle;
    }
    #guide-modal-container {
        background: #0f1524;
        border: solid #1e293b;
        padding: 1 2;
        width: 70;
        height: auto;
    }
    #modal-title {
        text-align: center;
        padding-bottom: 1;
    }
    #level-row {
        height: 3;
        margin: 1 0;
        align: center middle;
    }
    .opt-btn {
        min-width: 15;
        height: 3;
        margin: 0 1;
    }
    .active-btn {
        background: #38bdf8 !important;
        color: #0b0f17 !important;
        text-style: bold;
    }
    #btn-row {
        height: 3;
        margin-top: 2;
        align: center middle;
    }
    #btn-row Button {
        width: 20;
        margin: 0 2;
    }
    """


# ─── AGENT STATUS PANEL ──────────────────────────────────────────────────────

class AgentPanel(Static):
    """Left sidebar showing all 14 agents with live status dots."""

    active_agent: reactive[int] = reactive(-1)

    def compose(self) -> ComposeResult:
        yield Static("[bold orange]AGENTS[/bold orange]", id="agent-header")
        for i, (num, name, _role) in enumerate(AGENTS):
            yield Static(
                f"[dim]●[/dim] {num} {name}",
                id=f"agent-{i}",
                classes="agent-row"
            )

    def set_active(self, index: int) -> None:
        """Highlight the currently running agent."""
        for i in range(len(AGENTS)):
            widget = self.query_one(f"#agent-{i}")
            num, name, _ = AGENTS[i]
            widget.update(f"[dim]●[/dim] {num} {name}")
            widget.remove_class("agent-active")

        if 0 <= index < len(AGENTS):
            widget = self.query_one(f"#agent-{index}")
            num, name, _ = AGENTS[index]
            widget.update(f"[bold cyan]▶[/bold cyan] {num} [bold]{name}[/bold]")
            widget.add_class("agent-active")
            self.active_agent = index

    def reset_all(self) -> None:
        for i in range(len(AGENTS)):
            widget = self.query_one(f"#agent-{i}")
            num, name, _ = AGENTS[i]
            widget.update(f"[dim]●[/dim] {num} {name}")
            widget.remove_class("agent-active")
        self.active_agent = -1


# ─── STATUS BAR ──────────────────────────────────────────────────────────────

class StatusBar(Static):
    """Bottom status bar showing doc info and pipeline state."""

    def __init__(self):
        super().__init__()
        self._doc = "No document loaded"
        self._state = "[dim]IDLE[/dim]"
        self._model = "qwen2.5-coder:3b"

    def set_doc(self, doc: str) -> None:
        self._doc = doc
        self._refresh()

    def set_state(self, state: str, color: str = "green") -> None:
        self._state = f"[bold {color}]{state}[/bold {color}]"
        self._refresh()

    def set_model(self, model: str) -> None:
        self._model = model
        self._refresh()

    def _refresh(self) -> None:
        self.update(
            f" [bold cyan]DOC:[/bold cyan] {self._doc}   "
            f"[bold cyan]MODEL:[/bold cyan] [dim]{self._model}[/dim]   "
            f"[bold cyan]STATUS:[/bold cyan] {self._state}"
        )


# ─── MAIN APP ────────────────────────────────────────────────────────────────

class VasisApp(App):
    TITLE = "VASIS AI AGENTS"
    SUB_TITLE = "14-Agent Consensus Intelligence Engine"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Exit"),
        Binding("ctrl+l", "clear_log", "Clear"),
        Binding("f1",     "toggle_help",   "Help",   show=True),
        Binding("f2",     "toggle_agents", "Agents", show=True),
    ]

    CSS = """
    /* ── Layout ────────────────────────────────────────────── */
    Screen {
        layout: vertical;
        background: #0b0f17;
    }

    #body {
        layout: horizontal;
        height: 1fr;
    }

    /* ── Agent Sidebar ─────────────────────────────────────── */
    AgentPanel {
        width: 25;
        min-width: 25;
        height: 100%;
        background: #0f1524;
        border-right: solid #1e293b;
        padding: 0 1;
        overflow-y: auto;
    }

    #agent-header {
        color: orange;
        text-align: center;
        padding: 1 0;
        border-bottom: solid #1e293b;
        margin-bottom: 1;
    }

    .agent-row {
        color: #64748b;
        content-align: left middle;
        padding-left: 1;
        height: 1;
    }

    .agent-active {
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.08);
        text-style: bold;
    }

    /* ── Chat area ─────────────────────────────────────────── */
    #chat-area {
        layout: vertical;
        width: 1fr;
    }

    /* ── Chat area ─────────────────────────────────────────── */
    RichLog {
        height: 1fr;
        padding: 0 2;
        background: #0d121f;
        border: none;
        scrollbar-gutter: stable;
    }

    /* ── Input row ─────────────────────────────────────────── */
    #input-row {
        height: 3;
        layout: horizontal;
        background: #0f1524;
        padding: 0 2;
        align: left middle;
    }

    #prompt-label {
        width: auto;
        color: orange;
        padding: 0 1 0 0;
        content-align: left middle;
    }

    #user-input {
        width: 1fr;
        border: none;
        background: transparent;
        color: #f8fafc;
    }

    #user-input:focus {
        border: none;
        background: transparent;
    }

    /* ── Status bar ────────────────────────────────────────── */
    StatusBar {
        height: 1;
        background: #0f1524;
        color: #64748b;
        padding: 0 2;
        content-align: left middle;
    }

    /* ── Streamlined Top Header ────────────────────────────── */
    #header-bar {
        height: 13;
        background: #0f1524;
        border-bottom: solid #1e293b;
        content-align: center middle;
        text-align: center;
    }
    """

    # ── State ────────────────────────────────────────────────

    def __init__(self, pdf_path: str = None):
        super().__init__()
        self.pdf_path = pdf_path
        self.rag = None          # PageIndexREMSE instance
        self._vault_session = None
        self._busy = False
        self._show_agents = True
        self._venue = "IEEE"
        self._article_type = "research_article"
        self._researcher_level = "masters"

        # Learn engine
        self.learn = LearnEngine()

    # ── Compose ──────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # Beautiful ASCII banner top header
        banner_text = (
            "[bold cyan]=====================================================================[/bold cyan]\n"
            "[bold orange]  ██╗   ██╗ █████╗ ███████╗██╗███████╗[/bold orange]\n"
            "[bold orange]  ██║   ██║██╔══██╗██╔════╝██║██╔════╝[/bold orange]\n"
            "[bold orange]  ██║   ██║███████║███████╗██║███████╗[/bold orange]\n"
            "[bold orange]  ╚██╗ ██╔╝██╔══██║╚════██║██║╚════██║[/bold orange]\n"
            "[bold orange]   ╚████╔╝ ██║  ██║███████║██║███████║[/bold orange]\n"
            "[bold orange]    ╚═══╝  ╚═╝  ╚═╝╚══════╝╚═╝╚══════╝[/bold orange]\n"
            "              [bold cyan]VASIS AI AGENTS[/bold cyan]\n"
            "[bold cyan]=====================================================================[/bold cyan]"
        )
        yield Static(banner_text, id="header-bar")

        with Horizontal(id="body"):
            self.agent_panel = AgentPanel()
            yield self.agent_panel

            with Vertical(id="chat-area"):
                self.chat_log = RichLog(highlight=True, markup=True, wrap=True)
                yield self.chat_log

                with Horizontal(id="input-row"):
                    yield Static("[bold orange]╰──>[/bold orange]", id="prompt-label")
                    yield Input(
                        placeholder='/index paper.pdf  |  /query  |  /paper  |  /guide  |  /help',
                        id="user-input"
                    )

        self.status_bar = StatusBar()
        yield self.status_bar
        yield Footer()

    # ── Lifecycle ────────────────────────────────────────────

    def on_mount(self) -> None:
        # Hook stdout redirector to capture background prints in real-time
        self._old_stdout = sys.stdout
        sys.stdout = TuiStdoutRedirector(self)

        self._print_welcome()
        self.query_one("#user-input").focus()

        # Auto-load if PDF passed as argument
        if self.pdf_path:
            self.set_timer(0.3, lambda: self._dispatch_command(f"/index {self.pdf_path}"))

    def on_unmount(self) -> None:
        # Restore stdout
        sys.stdout = self._old_stdout

    def _print_welcome(self) -> None:
        self.chat_log.write("")
        self.chat_log.write("[bold green]●[/bold green] [bold]VASIS AI AGENTS[/bold]  —  14-Agent Consensus Intelligence Engine")
        self.chat_log.write(f"[dim]Session started {datetime.now().strftime('%Y-%m-%d %H:%M')}[/dim]")
        self.chat_log.write(f"  [dim]Settings: Venue=[bold]{self._venue}[/bold] | Type=[bold]{self._article_type}[/bold] | Level=[bold]{self._researcher_level}[/bold][/dim]")
        self.chat_log.write("")
        self.chat_log.write("  Type [bold cyan]/index path/to/paper.pdf[/bold cyan] to load a document")
        self.chat_log.write("  Type [bold cyan]/vault p1.pdf p2.pdf[/bold cyan] to load multiple documents")
        self.chat_log.write("  Type [bold cyan]/help[/bold cyan] for all commands")
        self.chat_log.write("")

    # ── Input handling ───────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        self._dispatch_command(text)

    def _dispatch_command(self, text: str) -> None:
        # Echo user input
        self.chat_log.write("\n[bold orange]╭── you[/bold orange]")
        self.chat_log.write(f"[bold orange]╰──>[/bold orange] {text}")
        self.chat_log.write("")

        if self._busy:
            self._warn("Pipeline is busy. Please wait.")
            return

        cmd = text.strip()
        lower = cmd.lower()

        if lower in ("/exit", "/quit", "exit", "quit"):
            self.exit()

        elif lower in ("/help", "help", "?"):
            self.push_screen(HelpModal())

        elif lower == "/clear":
            self.action_clear_log()

        elif lower == "/agents":
            self._show_agents_list()

        elif lower in ("/status", "/stats"):
            self._show_status()

        elif lower == "/tree":
            self._cmd_tree()

        elif lower == "/history":
            self._cmd_history()

        elif lower == "/outputs":
            self._cmd_outputs()

        elif lower.startswith("/index "):
            tokens = []
            for token in re.findall(r'(?:[^\s"\']|"[^"]*"|\'[^\']*\')+', cmd[7:]):
                tokens.append(token.strip('"').strip("'"))
            if len(tokens) > 1:
                self._warn("`/index` only accepts a single document. To load multiple documents, please use: [bold cyan]/vault path1.pdf path2.pdf[/bold cyan]")
            elif len(tokens) == 1:
                self._cmd_index(tokens[0])
            else:
                self._warn("Please specify a document path. Example: [bold cyan]/index paper.pdf[/bold cyan]")

        elif lower.startswith("/vault "):
            # Split properly, handling potential quotes
            paths = []
            for token in re.findall(r'(?:[^\s"\']|"[^"]*"|\'[^\']*\')+', cmd[7:]):
                paths.append(token.strip('"').strip("'"))
            self._cmd_vault(paths)

        elif lower.startswith("/venue "):
            v = cmd[7:].strip().upper()
            valid_venues = [ven.upper() for ven in VENUES]
            if v in valid_venues:
                matched = next((ven for ven in VENUES if ven.upper() == v), v)
                self._venue = matched
                self._ok(f"Target venue set to [bold]{self._venue}[/bold]")
            else:
                self._warn(f"Unknown venue '{v}'. Available: {', '.join(VENUES)}")

        elif lower.startswith("/type "):
            t = cmd[6:].strip().lower()
            valid_types = [pair[0] for pair in ARTICLE_TYPES]
            if t in valid_types:
                self._article_type = t
                self._ok(f"Article type set to [bold]{t}[/bold]")
            else:
                display_types = ", ".join(valid_types)
                self._warn(f"Unknown type '{t}'. Available: {display_types}")

        elif lower.startswith("/level "):
            lvl = cmd[7:].strip().lower()
            if lvl in RESEARCHER_LEVELS:
                self._researcher_level = lvl
                self._ok(f"Researcher level set to [bold]{lvl.title()}[/bold]")
            else:
                self._warn(f"Unknown level '{lvl}'. Available: {', '.join(RESEARCHER_LEVELS)}")

        elif lower.startswith("/paper "):
            topic = cmd[7:].strip()
            self._cmd_paper(topic)

        elif lower.startswith("/guide "):
            topic = cmd[7:].strip()
            self._cmd_guide(topic)

        elif lower.startswith("/loop ") or lower == "/loop":
            args = cmd[5:].strip()
            self._cmd_loop(args)

        elif lower.startswith("/query "):
            question = cmd[7:].strip()
            self._cmd_query(question)

        else:
            # Bare text → treat as query if doc or vault is loaded
            has_rag = self.rag and self.rag._ready
            has_vault = self._vault_session is not None
            if has_rag or has_vault:
                self._cmd_query(cmd)
            else:
                self._warn(
                    "No document loaded. Use [bold cyan]/index path/to/paper.pdf[/bold cyan] first.\n"
                    "Or type [bold cyan]/help[/bold cyan] for all commands."
                )

    # ── Helpers ──────────────────────────────────────────────

    def _warn(self, msg: str) -> None:
        self.chat_log.write(f"[bold yellow]⚠[/bold yellow]  {msg}")

    def _err(self, msg: str) -> None:
        self.chat_log.write(f"[bold red]✗[/bold red]  {msg}")

    def _ok(self, msg: str) -> None:
        self.chat_log.write(f"[bold green]✓[/bold green]  {msg}")

    def _info(self, msg: str) -> None:
        self.chat_log.write(f"[bold cyan]→[/bold cyan]  {msg}")

    def _separator(self) -> None:
        self.chat_log.write("[dim]─────────────────────────────────────────────────[/dim]")

    def _set_busy(self, busy: bool, state: str = "RUNNING", color: str = "yellow") -> None:
        self._busy = busy
        if busy:
            self.status_bar.set_state(state, color)
            self.query_one("#user-input").disabled = True
        else:
            self.status_bar.set_state("READY", "green")
            self.query_one("#user-input").disabled = False
            self.query_one("#user-input").focus()
            self.agent_panel.reset_all()

    # ── /agents ──────────────────────────────────────────────

    def _show_agents_list(self) -> None:
        self.chat_log.write("[bold orange]AGENT STACK  —  14-Agent CRDB Engine[/bold orange]")
        self._separator()
        for num, name, role in AGENTS:
            self.chat_log.write(f"  [bold cyan]Agent {num}[/bold cyan]  [bold]{name:<18}[/bold]  [dim]{role}[/dim]")
        self._separator()
        self.chat_log.write("")

    # ── /status ──────────────────────────────────────────────

    def _show_status(self) -> None:
        if self._vault_session is not None:
            self.chat_log.write("[bold orange]VAULT STATUS[/bold orange]")
            self._separator()
            for s in self._vault_session.stats():
                self.chat_log.write(f"  [bold cyan]{s['label']}:[/bold cyan] {s['total_atoms']} atoms, {s['total_triples']} triples, {s['tree_nodes']} sections")
            self._separator()
            self.chat_log.write("")
            return

        if not self.rag:
            self._warn("No document loaded. Use [bold cyan]/index path/to/paper.pdf[/bold cyan] first.")
            return
        stats = self.rag.get_stats()
        self.chat_log.write("[bold orange]SYSTEM STATUS[/bold orange]")
        self._separator()
        for k, v in stats.items():
            self.chat_log.write(f"  [bold cyan]{k:<20}[/bold cyan]  {v}")
        self._separator()
        self.chat_log.write("")

    # ── /tree ────────────────────────────────────────────────

    def _cmd_tree(self) -> None:
        if not self.rag or not self.rag._ready:
            self._warn("No single document loaded.")
            return
        self._info("PageIndex tree structure:")
        self._separator()
        for node in self.rag.tree_nodes[:40]:
            depth = node.get("depth", 0)
            indent = "  " * depth
            title = node.get("title", "Untitled")
            pages = node.get("pages", "?")
            self.chat_log.write(f"{indent}[dim]└─[/dim] [cyan]{title}[/cyan]  [dim](p.{pages})[/dim]")
        if len(self.rag.tree_nodes) > 40:
            self.chat_log.write(f"  [dim]... and {len(self.rag.tree_nodes) - 40} more nodes[/dim]")
        self._separator()

    # ── /history ─────────────────────────────────────────────

    def _cmd_history(self) -> None:
        if not self.rag:
            self._warn("No single document loaded.")
            return
        try:
            from storage.store import load_query_log
            log = load_query_log(self.rag.doc_id)
            if not log:
                self._info("No query history found for this document.")
                return
            self.chat_log.write("[bold orange]QUERY HISTORY[/bold orange]")
            self._separator()
            for i, entry in enumerate(log[-10:], 1):
                self.chat_log.write(f"  [bold]{i}.[/bold] [{entry.get('timestamp', '?')}]")
                self.chat_log.write(f"     [cyan]Q:[/cyan] {entry['query']}")
                answer_preview = entry['answer'][:120].replace('\n', ' ')
                self.chat_log.write(f"     [dim]A: {answer_preview}...[/dim]")
            self._separator()
        except Exception as e:
            self._err(f"Could not load history: {e}")

    # ── /outputs ─────────────────────────────────────────────

    def _cmd_outputs(self) -> None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        if not os.path.exists(output_dir):
            self._info("No outputs directory found yet.")
            return
        files = [f for f in os.listdir(output_dir) if f.endswith(".md")]
        if not files:
            self._info("No generated outputs yet. Use [bold cyan]/paper[/bold cyan] or [bold cyan]/guide[/bold cyan].")
            return
        self.chat_log.write("[bold orange]GENERATED OUTPUTS[/bold orange]")
        self._separator()
        for i, f in enumerate(sorted(files, reverse=True)[:20], 1):
            size = os.path.getsize(os.path.join(output_dir, f))
            self.chat_log.write(f"  [bold cyan]{i:02d}.[/bold cyan]  {f}  [dim]({size//1024} KB)[/dim]")
        self._separator()
        self.chat_log.write(f"  [dim]Location: {output_dir}[/dim]")
        self.chat_log.write("")

    # ── /index ───────────────────────────────────────────────

    @work(exclusive=True, thread=True)
    def _cmd_index(self, path: str) -> None:
        self.call_from_thread(self._set_busy, True, "INDEXING", "yellow")
        self.call_from_thread(self._info, f"Loading document: [bold]{path}[/bold]")

        if not os.path.exists(path):
            self.call_from_thread(self._err, f"File not found: {path}")
            self.call_from_thread(self._set_busy, False)
            return

        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from pipeline import PageIndexREMSE
            from config import DEFAULT_MODEL

            self.call_from_thread(self._info, "Checking Ollama connection...")
            rag = PageIndexREMSE(model=DEFAULT_MODEL)

            self.call_from_thread(self._info, "Building PageIndex tree + atom decomposition + causal triples...")
            self.call_from_thread(self.agent_panel.set_active, 2)  # Navigator
            rag.ingest(path)

            self.rag = rag
            self._vault_session = None # Reset vault if loading single paper
            stats = rag.get_stats()

            self.call_from_thread(self.status_bar.set_doc, os.path.basename(path))
            self.call_from_thread(self.status_bar.set_model, DEFAULT_MODEL)
            self.call_from_thread(self._ok, "Document indexed successfully")
            self.call_from_thread(self._separator)
            self.call_from_thread(self.chat_log.write, f"  [bold cyan]Tree nodes:[/bold cyan]  {stats['tree_nodes']}")
            self.call_from_thread(self.chat_log.write, f"  [bold cyan]Atoms:[/bold cyan]       {stats['total_atoms']}")
            self.call_from_thread(self.chat_log.write, f"  [bold cyan]Triples:[/bold cyan]     {stats['total_triples']}")
            self.call_from_thread(self._separator)
            self.call_from_thread(self.chat_log.write, "  Now type your question or use [bold cyan]/query[/bold cyan]  [bold cyan]/paper[/bold cyan]  [bold cyan]/guide[/bold cyan]\n")

        except Exception as e:
            self.call_from_thread(self._err, f"Indexing failed: {e}")
            import traceback
            tb = traceback.format_exc()
            self.call_from_thread(self.chat_log.write, f"[dim]{tb}[/dim]")

        finally:
            self.call_from_thread(self._set_busy, False)

    # ── /vault ───────────────────────────────────────────────

    @work(exclusive=True, thread=True)
    def _cmd_vault(self, paths: list) -> None:
        self.call_from_thread(self._set_busy, True, "LOADING VAULT", "yellow")
        self.call_from_thread(self._info, f"Loading {len(paths)} documents into vault...")
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from vault import VaultSession
            from config import DEFAULT_MODEL

            session = VaultSession(model=DEFAULT_MODEL)
            for p in paths:
                if not os.path.exists(p):
                    self.call_from_thread(self._warn, f"File not found: {p}, skipping.")
                    continue
                self.call_from_thread(self._info, f"Loading [cyan]{os.path.basename(p)}[/cyan]...")
                try:
                    label = session.load(p)
                    self.call_from_thread(self._ok, f"Loaded → {label}")
                except Exception as e:
                    self.call_from_thread(self._err, f"Failed: {p} — {e}")

            if len(session.papers) < 1:
                self.call_from_thread(self._err, "No documents loaded. Check file paths.")
                return

            self._vault_session = session
            self.rag = None # Reset single rag
            labels = list(session.papers.keys())
            self.call_from_thread(self.status_bar.set_doc, f"VAULT: {', '.join(labels)}")
            self.call_from_thread(self._ok, f"Vault ready with {len(labels)} papers")
            self.call_from_thread(self.chat_log.write,
                "  Use query to ask single/multi-paper questions naturally (e.g. 'compare the methods')\n"
                "  You can set target venue/style via [bold cyan]/venue [name][/bold cyan] and [bold cyan]/type [type][/bold cyan]\n"
            )

        except Exception as e:
            self.call_from_thread(self._err, f"Vault failed: {e}")
        finally:
            self.call_from_thread(self._set_busy, False)

    # ── /query ───────────────────────────────────────────────

    @work(exclusive=True, thread=True)
    def _cmd_query(self, question: str) -> None:
        has_rag = self.rag and self.rag._ready
        has_vault = self._vault_session is not None

        if not has_rag and not has_vault:
            self.call_from_thread(self._warn, "No document or vault loaded. Use [bold cyan]/index[/bold cyan] or [bold cyan]/vault[/bold cyan] first.")
            return

        self.call_from_thread(self._set_busy, True, "SWARM ACTIVE", "yellow")

        # Animate through agents
        agent_sequence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for i, agent_idx in enumerate(agent_sequence):
            self.call_from_thread(self.agent_panel.set_active, agent_idx)
            if i == 0:
                self.call_from_thread(self._info, "Routing query through 14-agent CRDB engine...")

        try:
            if has_vault:
                from vault import is_comparison_question
                
                # Check for paper / guide requests
                q_lower = question.lower()
                is_paper = any(pat in q_lower for pat in QUERY_DETECTION_PATTERNS.get("paper_writing", []))
                is_guide = any(pat in q_lower for pat in QUERY_DETECTION_PATTERNS.get("implementation_guide", []))

                venue = self._venue if is_paper else None
                article_type = self._article_type if is_paper else None
                researcher_level = self._researcher_level if is_guide else None

                if is_comparison_question(question):
                    self.call_from_thread(self._info, "Running cross-document comparison and contradiction audit...")
                    result = self._vault_session.compare(
                        question,
                        venue=venue,
                        article_type=article_type,
                        researcher_level=researcher_level,
                        target_paper="all"
                    )
                    self.call_from_thread(self._render_vault_compare_result, result, question)
                else:
                    self.call_from_thread(self._info, "Querying all papers in vault...")
                    results = self._vault_session.ask_all(
                        question,
                        venue=venue,
                        article_type=article_type,
                        researcher_level=researcher_level,
                        target_paper="all"
                    )
                    self.call_from_thread(self._render_vault_ask_all_result, results, question)
            else:
                # Normal single-paper query
                q_lower = question.lower()
                is_paper = any(pat in q_lower for pat in QUERY_DETECTION_PATTERNS.get("paper_writing", []))
                is_guide = any(pat in q_lower for pat in QUERY_DETECTION_PATTERNS.get("implementation_guide", []))

                venue = self._venue if is_paper else None
                article_type = self._article_type if is_paper else None
                researcher_level = self._researcher_level if is_guide else None

                result = self.rag.query(
                    question,
                    top_k_anchors=5,
                    expansion_passes=4,
                    show_provenance=False,
                    save_result=True,
                    venue=venue,
                    article_type=article_type,
                    researcher_level=researcher_level
                )
                self.call_from_thread(self._render_result, result, question)

                # Auto-save outputs
                paper_result = result.get("paper_result")
                impl_result  = result.get("impl_result")

                if paper_result and paper_result.get("full_text"):
                    from main import _save_agent_output
                    path = _save_agent_output(
                        content=paper_result["full_text"],
                        agent_name="paper",
                        topic=question,
                        venue=venue,
                        article_type=article_type
                    )
                    self.call_from_thread(self._ok, f"Paper saved to {path}")

                if impl_result and impl_result.get("full_text"):
                    from main import _save_agent_output
                    path = _save_agent_output(
                        content=impl_result["full_text"],
                        agent_name="guide",
                        topic=question,
                        researcher_level=researcher_level
                    )
                    self.call_from_thread(self._ok, f"Guide saved to {path}")

        except Exception as e:
            self.call_from_thread(self._err, f"Query failed: {e}")
            import traceback
            self.call_from_thread(self.chat_log.write, f"[dim]{traceback.format_exc()}[/dim]")
        finally:
            self.call_from_thread(self._set_busy, False)

    # ── /paper ───────────────────────────────────────────────

    def _cmd_paper(self, topic: str) -> None:
        has_rag = self.rag and self.rag._ready
        has_vault = self._vault_session is not None

        if not has_rag and not has_vault:
            self._warn("Load a document first with [bold cyan]/index[/bold cyan] or [bold cyan]/vault[/bold cyan].")
            return

        self.push_screen(PaperCustomizer(topic), callback=lambda res: self._on_paper_customizer_done(topic, res))

    def _on_paper_customizer_done(self, topic: str, result: tuple) -> None:
        if not result:
            self._info("Paper generation cancelled.")
            return

        venue, article_type = result
        self._venue = venue
        self._article_type = article_type

        self.chat_log.write(f"[dim]Selected target: [bold]{self._venue}[/bold] | [bold]{self._article_type.replace('_', ' ').title()}[/bold][/dim]")
        self._run_paper_worker(topic)

    @work(exclusive=True, thread=True)
    def _run_paper_worker(self, topic: str) -> None:
        has_vault = self._vault_session is not None
        self.call_from_thread(self._set_busy, True, "WRITING PAPER", "magenta")
        self.call_from_thread(self.agent_panel.set_active, 12)  # Agent 13
        self.call_from_thread(self._info,
            f"[bold magenta]📄 Agent 13 — Paper Writer[/bold magenta]\n"
            f"   Topic: {topic}\n"
            f"   Venue: {self._venue} | Type: {self._article_type}\n"
            f"   [dim]Generating academic paper...[/dim]"
        )

        try:
            if has_vault:
                # Vault mode paper writer
                results = self._vault_session.ask_all(
                    f"write a research paper on {topic}",
                    venue=self._venue,
                    article_type=self._article_type,
                    target_paper="all"
                )
                self.call_from_thread(self._render_vault_ask_all_result, results, topic)
            else:
                result = self.rag.query(
                    f"write a research paper on {topic}",
                    show_provenance=False,
                    venue=self._venue,
                    article_type=self._article_type
                )
                paper = result.get("paper_result")
                if paper and paper.get("full_text"):
                    from main import _save_agent_output
                    path = _save_agent_output(
                        content=paper["full_text"],
                        agent_name="paper",
                        topic=topic,
                        venue=self._venue,
                        article_type=self._article_type
                    )
                    self.call_from_thread(self._ok, "Paper generated and saved")
                    self.call_from_thread(self.chat_log.write, f"  [bold cyan]Path:[/bold cyan]  {path}")
                    preview = paper["full_text"][:600].replace('\n', ' ')
                    self.call_from_thread(self._separator)
                    self.call_from_thread(self.chat_log.write, f"[dim]{preview}...[/dim]")
                    self.call_from_thread(self._separator)
                else:
                    self.call_from_thread(self._render_result, result, topic)

        except Exception as e:
            self.call_from_thread(self._err, f"Paper generation failed: {e}")
        finally:
            self.call_from_thread(self._set_busy, False)

    # ── /guide ───────────────────────────────────────────────

    def _cmd_guide(self, topic: str) -> None:
        has_rag = self.rag and self.rag._ready
        has_vault = self._vault_session is not None

        if not has_rag and not has_vault:
            self._warn("Load a document first with [bold cyan]/index[/bold cyan] or [bold cyan]/vault[/bold cyan].")
            return

        self.push_screen(GuideCustomizer(topic), callback=lambda res: self._on_guide_customizer_done(topic, res))

    def _on_guide_customizer_done(self, topic: str, researcher_level: str) -> None:
        if not researcher_level:
            self._info("Guide generation cancelled.")
            return

        self._researcher_level = researcher_level
        self.chat_log.write(f"[dim]Selected target: [bold]{self._researcher_level.title()}[/bold] level researcher[/dim]")
        self._run_guide_worker(topic)

    @work(exclusive=True, thread=True)
    def _run_guide_worker(self, topic: str) -> None:
        has_vault = self._vault_session is not None
        self.call_from_thread(self._set_busy, True, "WRITING GUIDE", "blue")
        self.call_from_thread(self.agent_panel.set_active, 13)  # Agent 14
        self.call_from_thread(self._info,
            f"[bold blue]🔧 Agent 14 — Implementation Guide[/bold blue]\n"
            f"   Topic: {topic}\n"
            f"   Level: {self._researcher_level}\n"
            f"   [dim]Generating implementation guide...[/dim]"
        )

        try:
            if has_vault:
                results = self._vault_session.ask_all(
                    f"write an implementation guide for {topic}",
                    researcher_level=self._researcher_level,
                    target_paper="all"
                )
                self.call_from_thread(self._render_vault_ask_all_result, results, topic)
            else:
                result = self.rag.query(
                    f"write an implementation guide for {topic}",
                    show_provenance=False,
                    researcher_level=self._researcher_level
                )
                guide = result.get("impl_result")
                if guide and guide.get("full_text"):
                    from main import _save_agent_output
                    path = _save_agent_output(
                        content=guide["full_text"],
                        agent_name="guide",
                        topic=topic,
                        researcher_level=self._researcher_level
                    )
                    self.call_from_thread(self._ok, "Guide generated and saved")
                    self.call_from_thread(self.chat_log.write, f"  [bold cyan]Path:[/bold cyan]  {path}")
                    preview = guide["full_text"][:600].replace('\n', ' ')
                    self.call_from_thread(self._separator)
                    self.call_from_thread(self.chat_log.write, f"[dim]{preview}...[/dim]")
                    self.call_from_thread(self._separator)
                else:
                    self.call_from_thread(self._render_result, result, topic)

        except Exception as e:
            self.call_from_thread(self._err, f"Guide generation failed: {e}")
        finally:
            self.call_from_thread(self._set_busy, False)

    # ── Loop Engine Integration ───────────────────────────────

    def _init_loop_orchestrator(self) -> None:
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
        self._loop_last_ctx = None

    def _cmd_loop(self, args: str) -> None:
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
            self._warn("Usage: /loop paper <topic> [quality] [chain] [full] [--max N]")
            return

        has_rag = self.rag and self.rag._ready
        has_vault = self._vault_session is not None
        if not has_rag and not has_vault:
            self._warn("Load a document first with [bold cyan]/index[/bold cyan] or [bold cyan]/vault[/bold cyan].")
            return

        self._init_loop_orchestrator()
        self._run_loop_worker(topic, loop_types, max_iter)

    @work(exclusive=True, thread=True)
    def _run_loop_worker(self, topic: str, loop_types: list, max_iter: int) -> None:
        self._loop_active_ctx = None
        self.call_from_thread(self._set_busy, True, "LOOP ACTIVE", "yellow")
        self.call_from_thread(self._info, "Running looped paper pipeline...")

        def on_status(level, msg, ctx):
            self._loop_active_ctx = ctx
            icons = {
                "start": "◆", "done": "✓", "skip": "—",
                "iter": "·", "auto": "⚡", "warn": "⚠", "error": "✗",
            }
            icon = icons.get(level, "·")
            colors = {
                "start": "purple", "done": "green", "skip": "dim",
                "warn": "yellow", "error": "red", "auto": "blue",
            }
            color = colors.get(level, "gray")

            markup_msg = f"  [bold {color}]{icon}[/bold {color}]  {msg}"
            self.call_from_thread(self.chat_log.write, markup_msg)

            # Sidebar agent mapping based on loop execution
            loop_lower = msg.split()[0].lower() if msg else ""
            agent_idx = -1
            if loop_lower == "deep":
                agent_idx = 11  # Agent 12
            elif loop_lower in ("consensus", "_write"):
                agent_idx = 12  # Agent 13
            elif loop_lower == "quality":
                agent_idx = 5   # Agent 06 (Validation)
            elif loop_lower == "critique":
                agent_idx = 6   # Agent 07 (Contradiction)
            elif loop_lower == "chain":
                agent_idx = 13  # Agent 14 (Implementation Guide)
            elif loop_lower == "learn":
                agent_idx = 9   # Agent 10 (Supervisor)

            if agent_idx != -1:
                self.call_from_thread(self.agent_panel.set_active, agent_idx)

        try:
            ctx = self._loop_orch.run(
                topic          = topic,
                loop_types     = loop_types,
                venue          = self._venue,
                doc_type       = self._article_type,
                level          = self._researcher_level,
                max_iterations = max_iter,
                status_cb      = on_status,
            )
            self._loop_last_ctx = ctx

            # Save outputs
            self.call_from_thread(self.chat_log.write, "")
            self.call_from_thread(self.chat_log.write, "[bold green]✓ LOOP EXECUTION COMPLETED[/bold green]")
            self.call_from_thread(self._separator)

            if ctx.paper_written:
                self.call_from_thread(self._ok, f"Paper: {ctx.word_count} words  ·  grounding {ctx.grounding_ratio:.0%}")
                try:
                    from main import _save_agent_output
                    path = _save_agent_output(
                        content=ctx.paper_text,
                        agent_name="paper",
                        topic=topic,
                        venue=self._venue,
                        article_type=self._article_type,
                    )
                    self.call_from_thread(self._ok, f"Paper saved to [cyan]{path}[/cyan]")
                except Exception as e:
                    self.call_from_thread(self._err, f"Paper save failed: {e}")

            if ctx.guide_written:
                self.call_from_thread(self._ok, f"Guide: {len(ctx.guide_text.split())} words")
                try:
                    from main import _save_agent_output
                    path = _save_agent_output(
                        content=ctx.guide_text,
                        agent_name="guide",
                        topic=topic,
                        researcher_level=self._researcher_level,
                    )
                    self.call_from_thread(self._ok, f"Guide saved to [cyan]{path}[/cyan]")
                except Exception as e:
                    self.call_from_thread(self._err, f"Guide save failed: {e}")

            self.call_from_thread(self._info, f"Completed loops: {', '.join(ctx.completed_loops) or 'none'}  ·  elapsed {ctx.elapsed_s:.1f}s")
            self.call_from_thread(self.chat_log.write, "")

        except Exception as e:
            self.call_from_thread(self._err, f"Loop execution failed: {e}")
            import traceback
            tb = traceback.format_exc()
            self.call_from_thread(self.chat_log.write, f"[dim]{tb}[/dim]")
        finally:
            self.call_from_thread(self._set_busy, False)

    def _loop_help(self) -> None:
        self.chat_log.write("\n[bold orange]LOOP COMMANDS[/bold orange]")
        self._separator()
        rows = [
            ("/loop paper <topic>",              "quality + chain (smart default)"),
            ("/loop paper <topic> chain",         "Generates paper then guide"),
            ("/loop paper <topic> quality",        "Loops writing and citation fixes until grounding score >= 85%"),
            ("/loop paper <topic> critique",       "Runs logic contradiction audit, then performs section-level revisions"),
            ("/loop paper <topic> deep",           "Accumulates web sources/atoms prior to writing"),
            ("/loop paper <topic> consensus",      "Builds two distinct writing drafts and merges the best sections using Agent 11"),
            ("/loop paper <topic> full",           "Combines all loops together: research -> drafts -> grounding -> consistency -> guide -> learn"),
            ("/loop paper <topic> deep quality chain --max 3", "combine loops with retry limit"),
            ("/loop config",                       "show current loop settings"),
            ("/loop status",                       "show live loop state"),
        ]
        for cmd, desc in rows:
            self.chat_log.write(f"  [bold cyan]{cmd:<48}[/bold cyan]  {desc}")
        self._separator()
        self.chat_log.write("")

    def _loop_config(self) -> None:
        self.chat_log.write("\n[bold orange]LOOP CONFIGURATION[/bold orange]")
        self._separator()
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
            self.chat_log.write(f"  [bold cyan]{label:<28}[/bold cyan]  {value}")
        self._separator()
        self.chat_log.write("")

    def _loop_status(self) -> None:
        ctx = getattr(self, '_loop_last_ctx', None)
        if not ctx:
            self._info("No loop has been run yet. Use [bold cyan]/loop paper <topic>[/bold cyan] to start.")
            return

        self.chat_log.write("\n[bold orange]LAST LOOP RUN STATUS[/bold orange]")
        self._separator()
        rows = [
            ("Topic",            ctx.topic),
            ("Run ID",           ctx.run_id),
            ("Elapsed",          f"{ctx.elapsed_s:.1f}s"),
            ("Active loops",     ", ".join(ctx.active_loops)),
            ("Completed loops",  ", ".join(ctx.completed_loops)),
            ("Paper written",    "yes" if ctx.paper_written else "no"),
            ("Guide written",    "yes" if ctx.guide_written else "no"),
            ("Word count",       str(ctx.word_count)),
            ("Grounding",        f"{ctx.grounding_ratio:.0%} ({ctx.grounding_tagged}/{ctx.grounding_total})"),
            ("Atoms collected",  str(len(ctx.atoms))),
            ("Drafts generated", str(len(ctx.drafts))),
            ("Failures",         str(len(ctx.failures)) if ctx.failures else "none"),
        ]
        for label, value in rows:
            color = "green" if "yes" in value else "red" if "no" == value else "white"
            self.chat_log.write(f"  [bold cyan]{label:<20}[/bold cyan]  [bold {color}]{value}[/bold {color}]")
        self._separator()
        self.chat_log.write("")

    def _get_all_atoms(self) -> list:
        if self._vault_session:
            atoms = []
            for rag in self._vault_session.papers.values():
                atoms.extend(getattr(rag, "atoms", []))
            return atoms
        elif self.rag:
            return getattr(self.rag, "atoms", [])
        return []

    # ── Loop Agent Dispatch Adapters ──────────────────────────

    def _loop_dispatch_agent12(self, queries: list) -> list:
        try:
            from agents.agent12_websearch import search_web
            result = search_web(
                topic=" ".join(queries[:2]),
                queries_override=queries,
            )
            return result.get("sources", [])
        except Exception as e:
            self.call_from_thread(self._warn, f"Agent 12 error: {e}")
            return []

    def _loop_dispatch_agent13(self, **kwargs) -> dict:
        from agents import agent13_paper_writer as a13

        topic       = kwargs.get("topic", "")
        atoms       = kwargs.get("atoms", [])
        if not atoms:
            atoms = self._get_all_atoms()
        web_sources = kwargs.get("web_sources", [])
        if not web_sources:
            self.call_from_thread(self._info, "Empty web sources — dynamically fetching fallback web sources using Agent 12...")
            web_sources = self._loop_dispatch_agent12([topic])
            active_ctx = getattr(self, '_loop_active_ctx', None)
            if active_ctx:
                active_ctx.web_sources = web_sources

        venue       = kwargs.get("venue", self._venue)
        doc_type    = kwargs.get("doc_type", self._article_type)
        extra_instr = kwargs.get("extra_instruction", "")

        paper_text = kwargs.get("paper_text", "")
        revision_flags = kwargs.get("revision_flags", [])
        instruction = kwargs.get("instruction", "")
        if paper_text and (revision_flags or instruction):
            extra_instr = (extra_instr + "\n" + instruction).strip()

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
            self.call_from_thread(self._warn, f"Agent 13 error: {e}")
            return {"paper_text": paper_text, "grounding_ratio": 0.0}

    def _loop_dispatch_agent14(self, **kwargs) -> dict:
        from agents import agent14_implementation_guide as a14
        topic = kwargs.get("topic", "")
        web_sources = kwargs.get("web_sources", [])
        if not web_sources:
            active_ctx = getattr(self, '_loop_active_ctx', None)
            if active_ctx and active_ctx.web_sources:
                web_sources = active_ctx.web_sources
            else:
                self.call_from_thread(self._info, "Empty web sources — dynamically fetching fallback web sources using Agent 12...")
                web_sources = self._loop_dispatch_agent12([topic])
                if active_ctx:
                    active_ctx.web_sources = web_sources

        try:
            result = a14.guide_implementation(
                innovation=topic,
                narrative=kwargs.get("paper_text", ""),
                web_evidence={"sources": web_sources},
                researcher_level=kwargs.get("level", self._researcher_level),
            )
            return {
                "guide_text": result.get("full_text", ""),
                "steps":      [],
                "word_count": len(result.get("full_text", "").split()),
            }
        except Exception as e:
            self.call_from_thread(self._warn, f"Agent 14 error: {e}")
            return {"guide_text": ""}

    def _loop_dispatch_agent7(self, paper_text: str) -> list:
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
            self.call_from_thread(self._warn, f"Agent 7 error: {e}")
            return []

    def _loop_dispatch_agent11(self, drafts: list) -> dict:
        if not drafts:
            return {"paper_text": ""}
        best = max(drafts, key=lambda d: d.get("grounding_ratio", 0.0))
        return {"paper_text": best.get("paper_text", "")}

    def _loop_dispatch_grounding_audit(self, paper_text: str) -> dict:
        import re as _re
        sentences = _re.split(r"(?<=[.!?])\s+", paper_text)
        tagged = sum(1 for s in sentences if _re.search(r"\[\s*[AW]\s*:\s*[^\]]+\]", s))
        total = max(len(sentences), 1)
        return {"ratio": tagged / total, "tagged": tagged, "total": total}

    # ── Result renderers ──────────────────────────────────────

    def _render_result(self, result: dict, question: str) -> None:
        trust = result.get("trust_level", "low").upper()
        confidence = result.get("confidence", 0.0)
        grade = result.get("pipeline_grade", "?")
        elapsed = result.get("elapsed_seconds", 0.0)

        trust_color = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}.get(trust, "dim")

        self._separator()
        self.chat_log.write(
            f"[bold green]✓ VASIS RESPONSE[/bold green]   "
            f"[bold {trust_color}]TRUST: {trust}[/bold {trust_color}]  "
            f"[dim]conf={confidence:.2f}  grade={grade}  {elapsed:.1f}s[/dim]"
        )
        self._separator()

        answer = result.get("answer", "")
        for line in answer.split("\n"):
            self.chat_log.write(line if line.strip() else "")

        self._separator()

        # Contradictions
        if result.get("contradictions_found"):
            self.chat_log.write("[bold red]⚠ CONTRADICTIONS DETECTED[/bold red]")
            for c in result.get("contradiction_details", [])[:3]:
                self.chat_log.write(
                    f"  [yellow][{c.get('severity','?').upper()}][/yellow] "
                    f"{c.get('claim_a','')}  [dim]vs[/dim]  {c.get('claim_b','')}"
                )

        # Novel connections
        if result.get("novel_connections"):
            self.chat_log.write(f"[bold green]⚡ {len(result['novel_connections'])} novel causal connections synthesised[/bold green]")
            for n in result["novel_connections"][:2]:
                via = " → ".join(n.get("via", []))
                self.chat_log.write(
                    f"  [bold]{n.get('from','')}[/bold] → {via} → [bold]{n.get('to','')}[/bold]  "
                    f"[dim]conf={n.get('confidence',0):.2f}[/dim]"
                )

        # Sections
        sections = result.get("selected_sections", [])
        if sections:
            self.chat_log.write(f"[dim]Sections: {', '.join(str(s) for s in sections[:5])}[/dim]")

        self.chat_log.write("")

    def _render_vault_compare_result(self, result: dict, question: str) -> None:
        if "error" in result:
            self.chat_log.write(f"[bold red]Error: {result['error']}[/bold red]")
            return

        self._separator()
        self.chat_log.write("[bold green]✓ VAULT CROSS-PAPER COMPARISON[/bold green]")
        self._separator()

        for label, ans in result.get("per_paper_answers", {}).items():
            self.chat_log.write(f"[bold cyan]{label}:[/bold cyan]")
            for line in ans.split("\n"):
                self.chat_log.write(f"  {line}" if line.strip() else "")
            self._separator()

        if result.get("structural_conflict_found"):
            self.chat_log.write("[bold red]⚠ STRUCTURAL CONTRADICTIONS DETECTED[/bold red]")
            for c in result.get("cross_doc_conflicts", []):
                self.chat_log.write(f"  [yellow]{c.get('subject', '')} — {c.get('relation', '')}[/yellow]")
                for doc, obj in c.get("per_document", {}).items():
                    self.chat_log.write(f"    [{doc}] → {obj}")
            for c in result.get("triple_conflicts", []):
                self.chat_log.write(f"  [yellow]{c.get('subject', '')} — {c.get('relation', '')}[/yellow]: conflicting values {c.get('conflicting_objects', '')}")
            self._separator()
        else:
            consistency = result.get("consistency_score", 1.0)
            self.chat_log.write(f"[bold green]✓ Consistency Check:[/bold green] No structural conflicts found. Score: {consistency:.2f}")
            self._separator()

        if result.get("llm_contradictions"):
            self.chat_log.write("[bold yellow]Logical inconsistencies flagged by audit LLM:[/bold yellow]")
            for c in result.get("llm_contradictions", []):
                self.chat_log.write(f"  • {c.get('claim_a', '')}  [dim]vs[/dim]  {c.get('claim_b', '')}  ({c.get('severity', 'low')})")
            self._separator()

        if result.get("narrative_debates"):
            self.chat_log.write("[bold magenta]Narrative-level debates across papers:[/bold magenta]")
            for d in result.get("narrative_debates", []):
                self.chat_log.write(f"  [bold]{d.get('debate_topic', 'Untitled')}[/bold]")
                self.chat_log.write(f"    Side A: {d.get('side_a', '')}")
                self.chat_log.write(f"    Side B: {d.get('side_b', '')}")
                self.chat_log.write(f"    Open issue: {d.get('open_issue', '')}")
            self._separator()
        self.chat_log.write("")

    def _render_vault_ask_all_result(self, results: dict, question: str) -> None:
        self._separator()
        self.chat_log.write("[bold green]✓ VAULT INDEPENDENT PAPERS RESPONSE[/bold green]")
        self._separator()

        for label, r in results.items():
            self.chat_log.write(f"[bold green]● {label}[/bold green]")
            ans = r.get("answer", "")
            for line in ans.split("\n"):
                self.chat_log.write(f"  {line}" if line.strip() else "")
            self._separator()
            
            # Auto-save outputs for each paper in vault mode
            paper_result = r.get("paper_result")
            impl_result  = r.get("impl_result")
            
            if paper_result and paper_result.get("full_text"):
                from main import _save_agent_output
                path = _save_agent_output(
                    content=paper_result["full_text"],
                    agent_name="paper",
                    topic=f"{label}_{question}",
                    venue=self._venue,
                    article_type=self._article_type
                )
                self.chat_log.write(f"  [bold green]📄 Paper saved →[/bold green] [cyan]{path}[/cyan]")
                
            if impl_result and impl_result.get("full_text"):
                from main import _save_agent_output
                path = _save_agent_output(
                    content=impl_result["full_text"],
                    agent_name="guide",
                    topic=f"{label}_{question}",
                    researcher_level=self._researcher_level
                )
                self.chat_log.write(f"  [bold green]🔧 Guide saved →[/bold green] [cyan]{path}[/cyan]")
                
        self.chat_log.write("")

    # ── Actions (keybindings) ────────────────────────────────

    def action_clear_log(self) -> None:
        self.chat_log.clear()
        self._print_welcome()

    def action_toggle_help(self) -> None:
        self.push_screen(HelpModal())

    def action_toggle_agents(self) -> None:
        panel = self.query_one(AgentPanel)
        self._show_agents = not self._show_agents
        panel.display = self._show_agents


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    app = VasisApp(pdf_path=pdf_path)
    app.run()


if __name__ == "__main__":
    main()
