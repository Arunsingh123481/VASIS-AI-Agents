#!/usr/bin/env python3
"""
VASIS AI — /loop Engine
========================
Turns your linear 14-agent pipeline into a reactive multi-agent graph.
Six orchestration loops that can run individually or combined.

USAGE
-----
/loop paper "topic"                      # quality + chain (smart default)
/loop paper "topic" chain                # auto-trigger guide after paper
/loop paper "topic" quality              # retry until grounding ≥ 85%
/loop paper "topic" critique             # critique & revise flagged sections
/loop paper "topic" deep                 # deep research before writing
/loop paper "topic" consensus            # dual-draft merge via Agent 11
/loop paper "topic" full                 # all six loops
/loop paper "topic" quality chain --max 3
/loop config                             # show active configuration
/loop status                             # show live loop state

EXECUTION ORDER (when combined)
---------------------------------
  1. deep        — accumulate atoms (pre-write)
  2. consensus   — or standard single write
  3. quality     — fix grounding (post-write)
  4. critique    — fix contradictions (post-write)
  5. chain       — trigger guide + learn
  6. learn       — self-improvement (always last)

INTEGRATION
-----------
    from loop_engine import LoopOrchestrator, LoopContext

    loop = LoopOrchestrator(learn_engine=learn)

    # one call replaces your entire /paper pipeline:
    ctx = loop.run(
        topic       = "attention transformer limitations",
        loop_types  = ["quality", "chain"],   # or ["full"]
        venue       = "IEEE",
        doc_type    = "research_article",
        status_cb   = lambda level, msg, ctx: print(f"[{level}] {msg}"),
    )

    print(ctx.paper_text)
    print(ctx.guide_text)
"""

import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable

# ── PRESETS ────────────────────────────────────────────────────────────────────

PRESETS = {
    "default": ["quality", "chain"],
    "full":    ["deep", "quality", "critique", "consensus", "chain", "learn"],
    "fast":    ["chain"],
    "quality": ["quality", "chain"],
    "research":["deep", "quality"],
}

EXECUTION_ORDER = ["deep", "consensus", "quality", "critique", "chain", "learn"]

# ── ATOM THRESHOLD ─────────────────────────────────────────────────────────────

DEEP_RESEARCH_MIN_ATOMS = 15   # deep loop target before writing
DEEP_RESEARCH_MAX_PASSES = 3   # max web search passes
GROUNDING_THRESHOLD = 0.85     # quality gate target
MAX_CRITIQUE_ROUNDS = 3        # critique loop max retries
CONSENSUS_DRAFTS = 2           # number of drafts to generate


# =============================================================================
# LOOP CONTEXT  — shared mutable state that flows between all loops
# =============================================================================

@dataclass
class LoopContext:
    # ── Input ─────────────────────────────────────────────────────────────────
    topic:          str
    venue:          str          = "IEEE"
    doc_type:       str          = "research_article"
    level:          str          = "masters"
    max_iterations: int          = 3

    # ── Accumulated knowledge ────────────────────────────────────────────────
    atoms:          list         = field(default_factory=list)
    web_sources:    list         = field(default_factory=list)

    # ── Outputs ───────────────────────────────────────────────────────────────
    paper_text:     str          = ""
    guide_text:     str          = ""
    drafts:         list         = field(default_factory=list)   # consensus

    # ── Quality metrics ───────────────────────────────────────────────────────
    grounding_ratio: float       = 0.0
    grounding_tagged: int        = 0
    grounding_total:  int        = 0
    agent_results:   dict        = field(default_factory=dict)   # id → result
    contradiction_flags: list    = field(default_factory=list)

    # ── Loop control ──────────────────────────────────────────────────────────
    iteration:      int          = 0
    active_loops:   list         = field(default_factory=list)
    completed_loops: list        = field(default_factory=list)

    # ── Timing / budget ───────────────────────────────────────────────────────
    start_ts:       float        = field(default_factory=time.time)
    time_budget_s:  float        = 7200.0   # 2 hours

    # ── Metadata ──────────────────────────────────────────────────────────────
    run_id:         str          = ""
    failures:       list         = field(default_factory=list)
    metadata:       dict         = field(default_factory=dict)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.start_ts

    @property
    def over_budget(self) -> bool:
        return self.elapsed_s > self.time_budget_s

    @property
    def paper_written(self) -> bool:
        return bool(self.paper_text.strip())

    @property
    def guide_written(self) -> bool:
        return bool(self.guide_text.strip())

    @property
    def grounding_ok(self) -> bool:
        return self.grounding_ratio >= GROUNDING_THRESHOLD

    @property
    def word_count(self) -> int:
        return len(self.paper_text.split())


# =============================================================================
# BASE LOOP  — abstract interface every loop implements
# =============================================================================

class BaseLoop(ABC):
    name: str = "base"

    @abstractmethod
    def should_enter(self, ctx: LoopContext) -> bool:
        """Return True if this loop has work to do given current context."""

    @abstractmethod
    def should_exit(self, ctx: LoopContext) -> bool:
        """Return True to stop iterating (success or max reached)."""

    @abstractmethod
    def run_iteration(self, ctx: LoopContext, agents: dict) -> LoopContext:
        """Execute one iteration. Mutate and return ctx."""

    def pre_loop(self, ctx: LoopContext) -> LoopContext:
        """Called once before first iteration."""
        return ctx

    def post_loop(self, ctx: LoopContext) -> LoopContext:
        """Called once after last iteration."""
        ctx.completed_loops.append(self.name)
        return ctx


# =============================================================================
# LOOP 1 — DeepResearchLoop
# =============================================================================

class DeepResearchLoop(BaseLoop):
    """
    Runs Agent 12 (Web Search) in multiple targeted passes BEFORE writing.
    Each pass analyses atom gaps and generates more specific queries.
    Agent 13 only fires once atom count passes DEEP_RESEARCH_MIN_ATOMS.

    When to use: topics where the vault has few atoms, or where you
    want comprehensive web knowledge ingested before writing begins.
    """
    name = "deep"

    def should_enter(self, ctx: LoopContext) -> bool:
        return len(ctx.atoms) < DEEP_RESEARCH_MIN_ATOMS

    def should_exit(self, ctx: LoopContext) -> bool:
        return (
            len(ctx.atoms) >= DEEP_RESEARCH_MIN_ATOMS
            or ctx.iteration >= DEEP_RESEARCH_MAX_PASSES
            or ctx.over_budget
        )

    def pre_loop(self, ctx: LoopContext) -> LoopContext:
        ctx.metadata["deep_start_atoms"] = len(ctx.atoms)
        return ctx

    def run_iteration(self, ctx: LoopContext, agents: dict) -> LoopContext:
        """
        Pass N:
          1. Analyse which sub-topics have < 3 atoms (gap analysis)
          2. Generate targeted queries for those gaps
          3. Run Agent 12 with the gap queries
          4. Add results to ctx.atoms
        """
        agent12 = agents.get("agent12_websearch")
        if not agent12:
            return ctx

        # ── gap analysis ──────────────────────────────────────────────────────
        covered_topics = _extract_topics_from_atoms(ctx.atoms)
        sub_topics = _decompose_topic(ctx.topic)
        gap_topics = [t for t in sub_topics if t not in covered_topics]

        if not gap_topics:
            gap_topics = [f"{ctx.topic} {suffix}" for suffix in
                          ["recent advances", "limitations", "benchmark results"]]

        # ── web search for gaps ────────────────────────────────────────────────
        # HOOK: replace with your real Agent 12 call
        # Signature: agent12(queries: list[str]) -> list[dict]
        #   returns: [{url, title, snippet, source}]
        new_sources = agent12(gap_topics[:4])

        # de-duplicate by URL
        seen_urls = {a.get("url", "") for a in ctx.atoms}
        new_atoms = [
            s for s in new_sources
            if s.get("url", "") not in seen_urls and len(s.get("snippet", "")) > 40
        ]
        ctx.atoms.extend(new_atoms)
        ctx.web_sources.extend(new_atoms)

        return ctx

    def post_loop(self, ctx: LoopContext) -> LoopContext:
        added = len(ctx.atoms) - ctx.metadata.get("deep_start_atoms", 0)
        ctx.metadata["deep_atoms_added"] = added
        return super().post_loop(ctx)


# =============================================================================
# LOOP 2 — ConsensusLoop
# =============================================================================

class ConsensusLoop(BaseLoop):
    """
    Runs Agent 13 twice with different style settings, then uses Agent 11
    (Synthesis) to pick the best Abstract, Introduction, Methodology etc.
    from each draft and merge them into one final paper.

    Draft A: conservative/formal — high citation density
    Draft B: analytical/critical — more discussion depth

    When to use: high-stakes papers where quality trumps speed.
    """
    name = "consensus"

    def should_enter(self, ctx: LoopContext) -> bool:
        return not ctx.paper_written

    def should_exit(self, ctx: LoopContext) -> bool:
        return len(ctx.drafts) >= CONSENSUS_DRAFTS or ctx.over_budget

    def run_iteration(self, ctx: LoopContext, agents: dict) -> LoopContext:
        agent13 = agents.get("agent13_paper_writer")
        if not agent13:
            return ctx

        styles = [
            {
                "style": "conservative formal",
                "instruction": "Prioritise citation density. Every claim must be cited. "
                               "Use hedged academic language (e.g., 'suggests', 'indicates').",
            },
            {
                "style": "analytical critical",
                "instruction": "Prioritise depth of analysis. Compare competing approaches. "
                               "Use direct declarative sentences. Include a strong limitations section.",
            },
        ]
        style_cfg = styles[min(ctx.iteration, len(styles) - 1)]

        # HOOK: replace with your real Agent 13 call
        # Signature: agent13(topic, atoms, web_sources, venue, doc_type,
        #                    extra_instruction) -> dict
        # Returns: {paper_text, grounding_ratio, tagged, total, word_count,
        #           agent_result}
        result = agent13(
            topic             = ctx.topic,
            atoms             = ctx.atoms,
            web_sources       = ctx.web_sources,
            venue             = ctx.venue,
            doc_type          = ctx.doc_type,
            extra_instruction = style_cfg["instruction"],
        )

        draft = {
            "style":           style_cfg["style"],
            "paper_text":      result.get("paper_text", ""),
            "grounding_ratio": result.get("grounding_ratio", 0.0),
            "word_count":      result.get("word_count", 0),
        }
        ctx.drafts.append(draft)
        ctx.agent_results[f"agent13_draft{ctx.iteration + 1}"] = result

        return ctx

    def post_loop(self, ctx: LoopContext) -> LoopContext:
        """After both drafts, run Agent 11 to merge best sections."""
        if len(ctx.drafts) < 2:
            # only one draft written — use it directly
            if ctx.drafts:
                ctx.paper_text      = ctx.drafts[0]["paper_text"]
                ctx.grounding_ratio = ctx.drafts[0]["grounding_ratio"]
            return super().post_loop(ctx)

        agent11 = getattr(self, '_agents', {}).get("agent11_synthesis")
        if agent11:
            # HOOK: agent11(drafts: list[dict]) -> {paper_text, sections_chosen}
            merged = agent11(ctx.drafts)
            ctx.paper_text      = merged.get("paper_text", ctx.drafts[0]["paper_text"])
            ctx.grounding_ratio = max(d["grounding_ratio"] for d in ctx.drafts)
        else:
            # fallback: pick whichever draft has better grounding
            best = max(ctx.drafts, key=lambda d: d["grounding_ratio"])
            ctx.paper_text      = best["paper_text"]
            ctx.grounding_ratio = best["grounding_ratio"]

        return super().post_loop(ctx)


# =============================================================================
# LOOP 3 — StandardWriteLoop  (used when consensus is NOT active)
# =============================================================================

class StandardWriteLoop(BaseLoop):
    """Single Agent 13 run. Used when consensus loop is not in active_loops."""
    name = "_write"

    def should_enter(self, ctx: LoopContext) -> bool:
        return not ctx.paper_written and "consensus" not in ctx.active_loops

    def should_exit(self, ctx: LoopContext) -> bool:
        return ctx.paper_written or ctx.over_budget

    def run_iteration(self, ctx: LoopContext, agents: dict) -> LoopContext:
        agent13 = agents.get("agent13_paper_writer")
        if not agent13:
            return ctx

        # HOOK: replace with your real Agent 13 call
        result = agent13(
            topic       = ctx.topic,
            atoms       = ctx.atoms,
            web_sources = ctx.web_sources,
            venue       = ctx.venue,
            doc_type    = ctx.doc_type,
        )
        ctx.paper_text       = result.get("paper_text", "")
        ctx.grounding_ratio  = result.get("grounding_ratio", 0.0)
        ctx.grounding_tagged = result.get("tagged", 0)
        ctx.grounding_total  = result.get("total", 0)
        ctx.agent_results["agent13"] = result

        return ctx


# =============================================================================
# LOOP 4 — QualityGateLoop
# =============================================================================

class QualityGateLoop(BaseLoop):
    """
    After writing, runs the grounding audit.
    If below 85%, applies the citation injector and re-audits.
    Exits when grounding passes or max_iterations reached.

    This loop is the direct fix for the 0% grounding failure.
    See grounding_fix.py for the injector implementation.
    """
    name = "quality"

    def should_enter(self, ctx: LoopContext) -> bool:
        return ctx.paper_written and not ctx.grounding_ok

    def should_exit(self, ctx: LoopContext) -> bool:
        return (
            ctx.grounding_ok
            or ctx.iteration >= ctx.max_iterations
            or ctx.over_budget
        )

    def run_iteration(self, ctx: LoopContext, agents: dict) -> LoopContext:
        injector = agents.get("citation_injector")
        auditor  = agents.get("grounding_auditor")

        if injector:
            # HOOK: injector(paper_text, atoms, web_sources) ->
            #   {paper_text, tagged_before, tagged_after}
            result = injector(
                paper_text  = ctx.paper_text,
                atoms       = ctx.atoms,
                web_sources = ctx.web_sources,
            )
            ctx.paper_text = result.get("paper_text", ctx.paper_text)

        if auditor:
            # HOOK: auditor(paper_text) -> {ratio, tagged, total}
            audit = auditor(ctx.paper_text)
            ctx.grounding_ratio  = audit.get("ratio", 0.0)
            ctx.grounding_tagged = audit.get("tagged", 0)
            ctx.grounding_total  = audit.get("total", 0)
        else:
            # fallback: count [N] style citation markers
            import re
            sentences = re.split(r"(?<=[.!?])\s+", ctx.paper_text)
            tagged = sum(1 for s in sentences if re.search(r"\[\d+\]", s))
            total  = max(len(sentences), 1)
            ctx.grounding_ratio  = tagged / total
            ctx.grounding_tagged = tagged
            ctx.grounding_total  = total

        return ctx


# =============================================================================
# LOOP 5 — CritiqueReviseLoop
# =============================================================================

class CritiqueReviseLoop(BaseLoop):
    """
    Agent 7 (Contradiction) audits the paper for logical conflicts.
    Only flagged sections are sent back to Agent 13 for targeted rewriting.
    This is far more efficient than a full rewrite.

    When to use: papers making multi-step arguments where internal
    consistency matters (methodology ↔ results, abstract ↔ conclusion).
    """
    name = "critique"

    def should_enter(self, ctx: LoopContext) -> bool:
        return ctx.paper_written

    def should_exit(self, ctx: LoopContext) -> bool:
        return (
            not ctx.contradiction_flags
            or ctx.iteration >= MAX_CRITIQUE_ROUNDS
            or ctx.over_budget
        )

    def pre_loop(self, ctx: LoopContext) -> LoopContext:
        """Populate initial contradiction flags before loop starts."""
        agent7 = getattr(self, '_agents', {}).get("agent7_contradiction")
        if agent7:
            ctx.contradiction_flags = agent7(ctx.paper_text)
        return ctx

    def run_iteration(self, ctx: LoopContext, agents: dict) -> LoopContext:
        agent7  = agents.get("agent7_contradiction")
        agent13 = agents.get("agent13_paper_writer")

        if not ctx.contradiction_flags or not agent13:
            ctx.contradiction_flags = []
            return ctx

        # HOOK: agent13_revise(paper_text, flags) ->
        #   {paper_text}  — rewrites only the flagged sections
        reviser = agents.get("agent13_revise", agent13)
        result = reviser(
            paper_text   = ctx.paper_text,
            revision_flags = ctx.contradiction_flags,
            instruction  = (
                "Rewrite ONLY the following sections to resolve the "
                "contradictions listed. Do not change other sections.\n"
                + "\n".join(f"- {f}" for f in ctx.contradiction_flags)
            ),
        )
        ctx.paper_text = result.get("paper_text", ctx.paper_text)

        # re-audit
        if agent7:
            # HOOK: agent7(paper_text) -> list[str] (contradiction descriptions)
            ctx.contradiction_flags = agent7(ctx.paper_text)
        else:
            ctx.contradiction_flags = []

        return ctx


# =============================================================================
# LOOP 6 — ChainLoop
# =============================================================================

class ChainLoop(BaseLoop):
    """
    After the paper is saved, automatically fires Agent 14 (Impl. Guide).
    Then optionally triggers /learn auto at the end.

    This is what the user described as the core /loop use case:
    "after Agent 13 completes, automatically run Agent 14."
    """
    name = "chain"

    def should_enter(self, ctx: LoopContext) -> bool:
        return ctx.paper_written

    def should_exit(self, ctx: LoopContext) -> bool:
        return ctx.guide_written or ctx.over_budget

    def run_iteration(self, ctx: LoopContext, agents: dict) -> LoopContext:
        agent14 = agents.get("agent14_impl_guide")
        if not agent14:
            return ctx

        # HOOK: agent14(topic, paper_text, web_sources, level) ->
        #   {guide_text, steps, word_count}
        result = agent14(
            topic       = ctx.topic,
            paper_text  = ctx.paper_text,
            web_sources = ctx.web_sources,
            level       = ctx.level,
        )
        ctx.guide_text = result.get("guide_text", "")
        ctx.agent_results["agent14"] = result

        return ctx


# =============================================================================
# LOOP 7 — SelfImprovementLoop
# =============================================================================

class SelfImprovementLoop(BaseLoop):
    """
    Always runs last. Records the entire run into the /learn store,
    then calls active_learn() to fill any knowledge gaps.
    Makes every subsequent paper on this topic smarter.
    """
    name = "learn"

    def should_enter(self, ctx: LoopContext) -> bool:
        return ctx.paper_written

    def should_exit(self, ctx: LoopContext) -> bool:
        return ctx.iteration >= 1   # single pass

    def run_iteration(self, ctx: LoopContext, agents: dict) -> LoopContext:
        learn = agents.get("learn_engine")
        if not learn:
            return ctx

        # record this run
        learn.record_run(
            topic  = ctx.topic,
            result = {
                "agents":          list(ctx.agent_results.values()),
                "grounding_ratio": ctx.grounding_ratio,
                "atoms_retrieved": len(ctx.atoms),
                "sub_queries":     ctx.metadata.get("sub_queries", []),
                "word_count":      ctx.word_count,
                "duration_s":      ctx.elapsed_s,
                "venue":           ctx.venue,
                "doc_type":        ctx.doc_type,
                "query_raw":       ctx.topic,
                "failures":        ctx.failures,
            },
        )

        # fill knowledge gaps via active learn
        if ctx.web_sources:
            learn.active_learn(
                topic       = ctx.topic,
                web_results = ctx.web_sources,
            )

        return ctx


# =============================================================================
# LOOP ORCHESTRATOR  — the main class
# =============================================================================

class LoopOrchestrator:
    """
    Runs loops in the correct order, with smart auto-additions from /learn.

    agents dict keys (replace stubs with real functions):
        "agent12_websearch"    — web search
        "agent13_paper_writer" — paper writer
        "agent13_revise"       — section-level reviser (optional, falls back to agent13)
        "agent14_impl_guide"   — implementation guide
        "agent7_contradiction" — contradiction checker
        "agent11_synthesis"    — section merger (consensus)
        "citation_injector"    — grounding fix post-processor
        "grounding_auditor"    — grounding ratio calculator
        "learn_engine"         — LearnEngine instance
    """

    LOOP_CLASSES = {
        "deep":      DeepResearchLoop,
        "consensus": ConsensusLoop,
        "_write":    StandardWriteLoop,
        "quality":   QualityGateLoop,
        "critique":  CritiqueReviseLoop,
        "chain":     ChainLoop,
        "learn":     SelfImprovementLoop,
    }

    def __init__(self, agents: Optional[dict] = None, learn_engine=None):
        self.agents = agents or {}
        if learn_engine:
            self.agents["learn_engine"] = learn_engine
        self._status_cb: Optional[Callable] = None

    def run(
        self,
        topic:          str,
        loop_types:     Optional[list] = None,
        venue:          str = "IEEE",
        doc_type:       str = "research_article",
        level:          str = "masters",
        max_iterations: int = 3,
        time_budget_s:  float = 7200.0,
        status_cb:      Optional[Callable] = None,
    ) -> LoopContext:
        """
        Run the loop pipeline.

        loop_types: list of loop names, or a preset string like "full".
        status_cb(level, message, ctx) — called at each step for live display.

        Returns the final LoopContext with paper_text, guide_text, and metrics.
        """
        self._status_cb = status_cb

        # ── resolve preset ────────────────────────────────────────────────────
        if loop_types is None:
            loop_types = PRESETS["default"]
        elif isinstance(loop_types, str):
            loop_types = PRESETS.get(loop_types, loop_types.split())
        else:
            # expand "full" if inside a list
            expanded = []
            for lt in loop_types:
                expanded.extend(PRESETS.get(lt, [lt]))
            loop_types = expanded

        # ── build context ─────────────────────────────────────────────────────
        run_id = hashlib.md5(f"{topic}{time.time()}".encode()).hexdigest()[:12]
        ctx = LoopContext(
            topic          = topic,
            venue          = venue,
            doc_type       = doc_type,
            level          = level,
            max_iterations = max_iterations,
            time_budget_s  = time_budget_s,
            active_loops   = loop_types,
            run_id         = run_id,
        )

        # ── auto-add quality gate if /learn flags grounding risk ──────────────
        learn = self.agents.get("learn_engine")
        if learn and "quality" not in ctx.active_loops:
            hint = learn.get_preflight(topic)
            if hint.grounding_risk:
                ctx.active_loops.insert(0, "quality")
                ctx.atoms = learn.get_learned_atoms_for_context(topic)
                self._emit("auto", f"Grounding-risk topic — quality gate added. "
                                   f"{len(ctx.atoms)} pre-loaded atoms.", ctx)

        # ── determine loop execution order ────────────────────────────────────
        ordered = []
        for name in EXECUTION_ORDER:
            if name in ctx.active_loops or name == "_write":
                ordered.append(name)
        # always include _write unless consensus will handle it
        if "_write" not in ordered and "consensus" not in ctx.active_loops:
            ordered.insert(0, "_write")
        # always include learn last if requested
        if "learn" in ctx.active_loops and "learn" not in ordered:
            ordered.append("learn")

        # ── run each loop ─────────────────────────────────────────────────────
        for loop_name in ordered:
            if ctx.over_budget:
                self._emit("warn", "Time budget exceeded — skipping remaining loops.", ctx)
                break

            LoopCls = self.LOOP_CLASSES.get(loop_name)
            if not LoopCls:
                continue

            loop = LoopCls()
            loop._agents = self.agents  # expose agents for pre_loop / post_loop
            if not loop.should_enter(ctx):
                self._emit("skip", f"{loop_name} loop — conditions not met, skipping.", ctx)
                continue

            self._emit("start", f"{loop_name} loop started.", ctx)
            ctx.iteration = 0
            ctx = loop.pre_loop(ctx)

            while not loop.should_exit(ctx):
                self._emit(
                    "iter",
                    f"{loop_name}  iteration {ctx.iteration + 1}  "
                    f"(elapsed {ctx.elapsed_s:.0f}s)",
                    ctx,
                )
                ctx = loop.run_iteration(ctx, self.agents)
                ctx.iteration += 1

                if ctx.over_budget:
                    self._emit("warn", "Time budget exceeded — exiting loop.", ctx)
                    break

            ctx = loop.post_loop(ctx)
            self._emit(
                "done",
                f"{loop_name} loop done  ({ctx.iteration} iteration(s)).",
                ctx,
            )

        return ctx

    def _emit(self, level: str, message: str, ctx: LoopContext):
        if self._status_cb:
            self._status_cb(level, message, ctx)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _decompose_topic(topic: str) -> list[str]:
    """Generate sub-topic strings for gap analysis."""
    import re
    clean = re.sub(
        r"(write|generate|create|make)\s+(a\s+)?(research\s+)?paper\s+(on|about)\s+",
        "", topic, flags=re.IGNORECASE,
    ).strip()
    base  = clean[:60]
    return [
        base,
        f"{base} limitations",
        f"{base} solutions",
        f"{base} benchmarks",
        f"{base} recent advances",
    ]


def _extract_topics_from_atoms(atoms: list) -> set:
    """Return the set of sub-topic strings already covered by existing atoms."""
    topics = set()
    for atom in atoms:
        text = atom.get("text", atom.get("snippet", "")).lower()
        for kw in ["limitation", "solution", "benchmark", "advance", "architecture",
                   "attention", "transformer", "training", "inference", "dataset"]:
            if kw in text:
                topics.add(kw)
    return topics


# =============================================================================
# CLI COMMAND PARSER  — parses /loop commands into run() arguments
# =============================================================================

def parse_loop_command(args: str) -> dict:
    """
    Parse the argument string from the /loop command.

    Examples:
        "paper attention transformer quality chain --max 3"
        "paper attention transformer full"
        "config"
        "status"
    """
    parts  = args.strip().split()
    result = {
        "task":         None,
        "topic":        "",
        "loop_types":   [],
        "max_iter":     3,
        "special":      None,
    }

    if not parts:
        result["special"] = "help"
        return result

    # ── special commands ──────────────────────────────────────────────────────
    if parts[0].lower() in ("config", "status", "help", "stop"):
        result["special"] = parts[0].lower()
        return result

    # ── task ──────────────────────────────────────────────────────────────────
    result["task"] = parts[0].lower()
    parts = parts[1:]

    # ── max iterations ────────────────────────────────────────────────────────
    if "--max" in parts:
        idx = parts.index("--max")
        if idx + 1 < len(parts):
            try:
                result["max_iter"] = int(parts[idx + 1])
                parts = parts[:idx] + parts[idx + 2:]
            except ValueError:
                parts = parts[:idx] + parts[idx + 1:]

    # ── loop names ────────────────────────────────────────────────────────────
    LOOP_NAMES = set(PRESETS.keys()) | set(EXECUTION_ORDER)
    loop_types = []
    topic_parts = []
    for p in parts:
        if p.lower() in LOOP_NAMES:
            loop_types.append(p.lower())
        else:
            topic_parts.append(p)

    result["loop_types"] = loop_types if loop_types else PRESETS["default"]
    result["topic"]      = " ".join(topic_parts).strip('"\'')

    return result


# =============================================================================
# QUICK INTEGRATION SNIPPET  — drop into VasisCLI
# =============================================================================

_INTEGRATION_EXAMPLE = """
# ─── Add to VasisCLI.__init__ ────────────────────────────────────────────────
from loop_engine import LoopOrchestrator, parse_loop_command

self.loop_orch = LoopOrchestrator(
    agents = {
        "agent12_websearch":    self._dispatch_agent12,
        "agent13_paper_writer": self._dispatch_agent13,
        "agent14_impl_guide":   self._dispatch_agent14,
        "agent7_contradiction": self._dispatch_agent7,
        "agent11_synthesis":    self._dispatch_agent11,
        "citation_injector":    inject_missing_citations,   # from grounding_fix.py
        "grounding_auditor":    self._dispatch_grounding_audit,
    },
    learn_engine = self.learn,   # from learn_engine.py
)

# ─── Add to VasisCLI._dispatch_command ────────────────────────────────────────
"/loop": lambda: self._cmd_loop(args),

# ─── Add to VasisCLI ──────────────────────────────────────────────────────────
def _cmd_loop(self, args: str):
    parsed = parse_loop_command(args)

    if parsed["special"]:
        if parsed["special"] == "help":
            self._loop_help()
        elif parsed["special"] == "config":
            self._loop_config()
        return

    topic      = parsed["topic"]
    loop_types = parsed["loop_types"]
    max_iter   = parsed["max_iter"]

    if not topic:
        error_line("Usage: /loop paper <topic> [quality] [chain] [full]")
        return
    if not self.vault_docs:
        error_line("Load documents first: /index or /vault")
        return

    # ── run ──────────────────────────────────────────────────────────────────
    nl()
    console.print(Text(f"  /loop paper · {topic}", style="bold"))
    console.print(Text(f"  Loops: {', '.join(loop_types)}  ·  max {max_iter} retries", style=T.MUTED))
    nl()

    def on_status(level, msg, ctx):
        icons = {"start":"◆","done":"✓","skip":"—","iter":"·","auto":"⚡","warn":"⚠","error":"✗"}
        icon  = icons.get(level, "·")
        colors= {"start":T.PRIMARY,"done":T.SUCCESS,"skip":T.DIM,"warn":T.WARNING,"error":T.ERROR,"auto":T.SECONDARY}
        color = colors.get(level, T.MUTED)
        console.print(Text(f"  {icon}  {msg}", style=color))

    ctx = self.loop_orch.run(
        topic          = topic,
        loop_types     = loop_types,
        venue          = self.venue,
        doc_type       = self.doc_type,
        max_iterations = max_iter,
        status_cb      = on_status,
    )

    # ── results ────────────────────────────────────────────────────────────────
    nl()
    if ctx.paper_written:
        success_line(f"Paper: {ctx.word_count} words  ·  grounding {ctx.grounding_ratio:.0%}")
    if ctx.guide_written:
        success_line(f"Guide: {len(ctx.guide_text.split())} words")
    nl()
"""


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    # ── test parse_loop_command ───────────────────────────────────────────────
    cases = [
        'paper "attention transformer limitations" quality chain --max 3',
        'paper attention transformer full',
        'paper transformer quality',
        'config',
        'paper some topic deep quality --max 2',
    ]
    print("─" * 60)
    print("parse_loop_command tests")
    print("─" * 60)
    for c in cases:
        p = parse_loop_command(c)
        print(f"  in:   {c[:60]}")
        print(f"  out:  task={p['task']} loops={p['loop_types']} max={p['max_iter']}"
              f" topic='{p['topic'][:40]}'\n")

    # ── test orchestrator with stub agents ────────────────────────────────────
    def stub_agent13(topic, atoms, web_sources, venue, doc_type,
                     extra_instruction="", revision_flags=None, instruction=""):
        text = (
            f"## Abstract\n\nThis paper examines {topic}. [1][2]\n\n"
            f"## Introduction\n\nThe field has advanced significantly. [3]\n\n"
            f"## Conclusion\n\nFuture work will address remaining gaps. [4]\n"
        )
        sents = len(text.split("."))
        tagged = sents - 1
        return {
            "paper_text":       text,
            "grounding_ratio":  tagged / max(sents, 1),
            "tagged":           tagged,
            "total":            sents,
            "word_count":       len(text.split()),
        }

    def stub_agent14(topic, paper_text, web_sources, level):
        return {"guide_text": f"# Implementation Guide: {topic}\n\nStep 1...", "steps": []}

    agents = {
        "agent13_paper_writer": stub_agent13,
        "agent14_impl_guide":   stub_agent14,
    }

    orch   = LoopOrchestrator(agents=agents)
    events = []

    ctx = orch.run(
        topic       = "attention transformer limitations",
        loop_types  = ["quality", "chain"],
        max_iterations = 2,
        status_cb   = lambda level, msg, ctx: events.append(f"[{level}] {msg}"),
    )

    print("─" * 60)
    print("Orchestrator smoke test")
    print("─" * 60)
    for e in events:
        print(f"  {e}")
    print()
    print(f"  paper_written : {ctx.paper_written}")
    print(f"  guide_written : {ctx.guide_written}")
    print(f"  grounding     : {ctx.grounding_ratio:.0%}")
    print(f"  loops done    : {ctx.completed_loops}")
    print(f"  elapsed       : {ctx.elapsed_s:.1f}s")
    print()
    print("  PASS")
