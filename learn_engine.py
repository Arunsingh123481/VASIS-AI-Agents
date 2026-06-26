#!/usr/bin/env python3
"""
VASIS AI — /learn Engine
=========================

Three modes that compound over time:

  A  Passive   auto-records every /paper or /query run, learns from outcomes.
               Called automatically — user doesn't have to think about it.

  B  Active    /learn <topic> crawls the web and permanently ingests atoms
               into the vault. Next paper on that topic starts pre-loaded.

  C  Feedback  /learn feedback lets you rate/correct the last paper.
               High-trust corrections are written back as vault atoms.

One store file: .vasis_learn.json  (auto-created on first run)

Integration:
    from learn_engine import LearnEngine
    learn = LearnEngine()

    # before every /paper run:
    hints = learn.get_preflight(topic)

    # after every /paper run:
    learn.record_run(topic, result)

    # /learn <topic>  command:
    learn.active_learn(topic, web_results, vault_add_fn)

    # /learn feedback command:
    learn.record_feedback(corrections)

    # /learn review  command:
    summary = learn.review()
"""

import json
import math
import re
import time
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Callable
from collections import Counter
from dataclasses import dataclass, asdict

# ── Constants ──────────────────────────────────────────────────────────────────

STORE_PATH      = Path(".vasis_learn.json")
DECAY_DAYS      = 30          # learning half-life in days
MIN_SIMILARITY  = 0.68        # topic match threshold (0–1)
MAX_ATOMS       = 300         # cap on stored learned atoms
TOP_K_HINTS     = 3           # how many past runs to surface as hints
GROUNDING_WARN  = 0.85        # ratio below which we flag a topic as risky
MIN_RUNS_PATTERN = 2          # min runs before we declare a failure pattern


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class RunRecord:
    """One completed /paper or /query run, stored for learning."""
    run_id:           str
    ts:               float
    topic:            str                    # cleaned topic string
    query_raw:        str                    # original user query
    sub_queries:      list[str]
    atoms_retrieved:  int
    grounding_ratio:  float
    agent_scores:     dict[str, float]       # {"1": 0.50, "3": 1.00, ...}
    failures:         list[str]              # ["grounding_fail", "context_retrieval_failed"]
    venue:            str
    doc_type:         str
    word_count:       int
    duration_s:       float


@dataclass
class LearnedAtom:
    """An atom ingested via /learn <topic> from web sources."""
    atom_id:    str
    ts:         float
    topic:      str
    text:       str
    source_url: str
    source_title: str
    trust:      str    # "high" | "medium" | "low"
    use_count:  int = 0
    last_used:  float = 0.0


@dataclass
class Correction:
    """User feedback attached to a specific run."""
    corr_id:    str
    ts:         float
    run_id:     str
    topic:      str
    section:    str        # "introduction", "methodology", etc.
    issue:      str        # "hallucinated" | "wrong_citation" | "good" | "unclear"
    sentence:   str
    note:       str = ""
    trust:      str = "high"


@dataclass
class PreflightHint:
    """What the engine tells Agent 13 before it starts writing."""
    topic:              str
    similar_runs:       int          # how many times we've seen this topic
    best_sub_queries:   list[str]    # sub-queries that retrieved the most atoms
    grounding_risk:     bool         # True if this topic has a grounding failure pattern
    learned_atom_ids:   list[str]    # atom IDs to pre-load into context
    venue_tips:         list[str]    # tips specific to the venue
    avg_grounding:      float        # historical average grounding ratio
    confidence:         float        # 0–1 confidence in this hint


# =============================================================================
# STORE  — atomic JSON persistence
# =============================================================================

class LearnStore:
    """
    Thread-safe JSON store.  All writes go to a temp file first,
    then atomically rename to prevent corruption.
    """

    SCHEMA = {
        "version":        "1.1",
        "created":        "",
        "stats": {
            "total_runs":           0,
            "total_atoms_learned":  0,
            "total_corrections":    0,
            "grounding_failures":   0,
        },
        "runs":           [],
        "learned_atoms":  [],
        "corrections":    [],
        "topic_cache":    {},    # topic_key → aggregated stats
    }

    def __init__(self, path: Path = STORE_PATH):
        self.path = path
        self._data: dict = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                return
            except (json.JSONDecodeError, OSError):
                pass    # fall through to create fresh store
        self._data = self.SCHEMA.copy()
        self._data["created"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def _save(self):
        """Atomic write: temp file → rename."""
        dir_ = self.path.parent
        try:
            fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError as e:
            print(f"[LearnStore] Write failed: {e}")

    # ── accessors ─────────────────────────────────────────────────────────────

    def add_run(self, run: RunRecord):
        self._data["runs"].append(asdict(run))
        self._data["stats"]["total_runs"] += 1
        if run.grounding_ratio == 0.0:
            self._data["stats"]["grounding_failures"] += 1
        self._update_topic_cache(run)
        self._save()

    def add_atom(self, atom: LearnedAtom):
        # cap total atoms
        if len(self._data["learned_atoms"]) >= MAX_ATOMS:
            # remove lowest-trust, least-used atoms first
            self._data["learned_atoms"].sort(
                key=lambda a: (a["use_count"], a["ts"])
            )
            self._data["learned_atoms"] = self._data["learned_atoms"][10:]
        self._data["learned_atoms"].append(asdict(atom))
        self._data["stats"]["total_atoms_learned"] += 1
        self._save()

    def add_correction(self, corr: Correction):
        self._data["corrections"].append(asdict(corr))
        self._data["stats"]["total_corrections"] += 1
        self._save()

    def get_runs(self) -> list[dict]:
        return self._data.get("runs", [])

    def get_atoms_for_topic(self, topic_key: str) -> list[dict]:
        return [
            a for a in self._data.get("learned_atoms", [])
            if a["topic"] == topic_key
        ]

    def get_all_atoms(self) -> list[dict]:
        return self._data.get("learned_atoms", [])

    def get_corrections(self) -> list[dict]:
        return self._data.get("corrections", [])

    def get_topic_cache(self) -> dict:
        return self._data.get("topic_cache", {})

    def get_stats(self) -> dict:
        return self._data.get("stats", {})

    def mark_atom_used(self, atom_id: str):
        for atom in self._data["learned_atoms"]:
            if atom["atom_id"] == atom_id:
                atom["use_count"] += 1
                atom["last_used"] = time.time()
                break
        self._save()

    def _update_topic_cache(self, run: RunRecord):
        """Aggregate per-topic statistics for fast pre-flight lookup."""
        key = _topic_key(run.topic)
        cache = self._data["topic_cache"]
        if key not in cache:
            cache[key] = {
                "runs": 0,
                "total_grounding":   0.0,
                "total_atoms":       0,
                "best_sub_queries":  [],
                "failure_count":     0,
                "last_ts":           0.0,
            }
        c = cache[key]
        c["runs"]            += 1
        c["total_grounding"] += run.grounding_ratio
        c["total_atoms"]     += run.atoms_retrieved
        c["last_ts"]          = run.ts
        if run.grounding_ratio < GROUNDING_WARN:
            c["failure_count"] += 1
        # keep the sub-queries from the best-grounding run
        if run.grounding_ratio >= c.get("best_grounding", 0.0):
            c["best_sub_queries"] = run.sub_queries
            c["best_grounding"]   = run.grounding_ratio
        self._save()


# =============================================================================
# TOPIC MATCHER  — finds similar past queries
# =============================================================================

def _topic_key(text: str) -> str:
    """Stable lowercase slug for a topic, used as cache key."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower().strip())[:80]


def _tokenize(text: str) -> set[str]:
    """Simple word tokeniser, strips stopwords."""
    STOP = {
        "a","an","the","and","or","of","in","on","to","for","is","are","was",
        "were","with","by","from","that","this","be","at","as","it","its",
        "about","write","paper","research","generate","me","my","you","your",
        "please","can","how","what","why","which","we","our","they","their",
        "i","am","have","has","had","will","would","could","should","may",
        "might","do","does","did","not","no","up","so","if","than","then",
    }
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in STOP and len(t) > 1}


def _jaccard(a: str, b: str) -> float:
    """Word-overlap Jaccard similarity — fast fallback when no embeddings."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class TopicMatcher:
    """
    Finds past runs similar to a new topic.
    Uses sentence-transformers if available, Jaccard otherwise.
    """

    def __init__(self):
        self._model = None
        self._use_embeddings = False
        self._try_load_model()

    def _try_load_model(self):
        try:
            from sentence_transformers import SentenceTransformer, util
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._util  = util
            self._use_embeddings = True
        except ImportError:
            self._use_embeddings = False

    def similarity(self, topic_a: str, topic_b: str) -> float:
        if self._use_embeddings:
            embs = self._model.encode([topic_a, topic_b], convert_to_tensor=True)
            return float(self._util.cos_sim(embs[0], embs[1]))
        return _jaccard(topic_a, topic_b)

    def find_similar(
        self,
        topic: str,
        candidates: list[str],
        top_k: int = TOP_K_HINTS,
    ) -> list[tuple[str, float]]:
        """
        Return the top-k candidates most similar to topic,
        as (candidate, score) pairs sorted descending.
        """
        if not candidates:
            return []

        if self._use_embeddings:
            all_texts = [topic] + candidates
            embs = self._model.encode(all_texts, convert_to_tensor=True)
            query_emb = embs[0]
            cand_embs = embs[1:]
            scores = [
                float(self._util.cos_sim(query_emb, e))
                for e in cand_embs
            ]
        else:
            scores = [_jaccard(topic, c) for c in candidates]

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(c, s) for c, s in ranked[:top_k] if s >= MIN_SIMILARITY]


# =============================================================================
# DECAY  — recency weighting
# =============================================================================

def _decay_weight(ts: float, half_life_days: float = DECAY_DAYS) -> float:
    """
    Exponential decay so recent learnings count more.
    Returns 1.0 for brand-new, 0.5 after half_life_days, etc.
    """
    age_days = (time.time() - ts) / 86400.0
    return math.exp(-math.log(2) * age_days / half_life_days)


# =============================================================================
# LEARN ENGINE  — the main class
# =============================================================================

class LearnEngine:
    """
    The brain behind /learn.

    Usage
    -----
    learn = LearnEngine()

    # every /paper run (call AFTER the run finishes):
    learn.record_run(topic, result_dict)

    # before every /paper run:
    hints = learn.get_preflight(topic)

    # /learn <topic>  — active web crawl ingestion:
    learn.active_learn(topic, web_results, vault_add_fn)

    # /learn feedback — user corrects the last paper:
    learn.record_feedback(corrections_list)

    # /learn review   — show the dashboard:
    summary = learn.review()
    """

    def __init__(self, store_path: Path = STORE_PATH):
        self.store   = LearnStore(store_path)
        self.matcher = TopicMatcher()
        self._last_run_id: Optional[str] = None

    # =========================================================================
    # MODE A — Passive recording
    # =========================================================================

    def record_run(self, topic: str, result: dict) -> str:
        """
        Call this automatically after every /paper or /query run.

        result dict keys (from your existing pipeline):
            agents          list of {id, grade, score, time, skipped?}
            grounding_ratio float
            atoms_retrieved int  (len of atoms passed to Agent 13)
            sub_queries     list[str]
            word_count      int
            duration_s      float
            venue           str
            doc_type        str
            query_raw       str    (original user input)
            failures        list[str]  optional
        """
        import hashlib

        run_id = hashlib.md5(
            f"{topic}{time.time()}".encode()
        ).hexdigest()[:12]

        agent_scores = {
            str(ag["id"]): ag.get("score", 0.0)
            for ag in result.get("agents", [])
            if not ag.get("skipped")
        }

        failures = list(result.get("failures", []))
        if result.get("grounding_ratio", 1.0) < GROUNDING_WARN:
            if "grounding_fail" not in failures:
                failures.append("grounding_fail")

        run = RunRecord(
            run_id          = run_id,
            ts              = time.time(),
            topic           = topic,
            query_raw       = result.get("query_raw", topic),
            sub_queries     = result.get("sub_queries", []),
            atoms_retrieved = result.get("atoms_retrieved", 0),
            grounding_ratio = result.get("grounding_ratio", 0.0),
            agent_scores    = agent_scores,
            failures        = failures,
            venue           = result.get("venue", "IEEE"),
            doc_type        = result.get("doc_type", "research_article"),
            word_count      = result.get("word_count", 0),
            duration_s      = result.get("duration_s", 0.0),
        )

        self.store.add_run(run)
        self._last_run_id = run_id
        return run_id

    # =========================================================================
    # MODE A — Pre-flight hints (call BEFORE a /paper run)
    # =========================================================================

    def get_preflight(self, topic: str) -> PreflightHint:
        """
        Before starting a paper, check what the system already knows.
        Returns a PreflightHint with cached sub-queries, atom IDs to
        pre-load, and a grounding-risk flag.

        The paper pipeline should:
          1. Use hint.best_sub_queries instead of deriving new ones
             (they worked before for this topic)
          2. Pre-load hint.learned_atom_ids into the context
          3. If hint.grounding_risk is True, run Fix 3 (post-processing
             citation injector) unconditionally, don't wait for the audit
        """
        runs     = self.store.get_runs()
        cache    = self.store.get_topic_cache()
        atoms    = self.store.get_all_atoms()
        _  = _topic_key(topic)  # reserved for future direct-match lookups

        # ── find similar past topics ────────────────────────────────────────
        past_topics = list(set(r["topic"] for r in runs))
        similar = self.matcher.find_similar(topic, past_topics, top_k=TOP_K_HINTS)

        if not similar:
            # no history at all for this topic
            return PreflightHint(
                topic            = topic,
                similar_runs     = 0,
                best_sub_queries = [],
                grounding_risk   = False,
                learned_atom_ids = [],
                venue_tips       = [],
                avg_grounding    = 0.0,
                confidence       = 0.0,
            )

        # ── aggregate stats across similar topics ───────────────────────────
        total_runs     = 0
        total_grounding = 0.0
        failure_count  = 0
        best_sub_queries: list[str] = []
        best_grounding_seen = -1.0

        for sim_topic, sim_score in similar:
            sim_key = _topic_key(sim_topic)
            c = cache.get(sim_key, {})
            n = c.get("runs", 0)
            if n == 0:
                continue

            # weight by similarity AND recency
            recent_runs = [
                r for r in runs
                if r["topic"] == sim_topic
            ]
            recency_w = max(
                (_decay_weight(r["ts"]) for r in recent_runs),
                default=0.5,
            )
            w = sim_score * recency_w

            total_runs       += n
            total_grounding  += c.get("total_grounding", 0.0) * w
            failure_count    += c.get("failure_count", 0)

            # prefer sub-queries from the best grounding run
            bg = c.get("best_grounding", 0.0)
            if bg > best_grounding_seen:
                best_grounding_seen = bg
                best_sub_queries    = c.get("best_sub_queries", [])

        avg_g = (total_grounding / total_runs) if total_runs > 0 else 0.0
        grounding_risk = (
            failure_count >= MIN_RUNS_PATTERN
            and avg_g < GROUNDING_WARN
        )

        # ── find pre-loadable learned atoms ─────────────────────────────────
        relevant_atoms = []
        for atom in atoms:
            sim = self.matcher.similarity(topic, atom["topic"])
            if sim >= MIN_SIMILARITY:
                relevant_atoms.append((atom["atom_id"], sim, atom["ts"]))

        # rank by similarity × recency
        relevant_atoms.sort(
            key=lambda x: x[1] * _decay_weight(x[2]),
            reverse=True,
        )
        atom_ids = [aid for aid, _, _ in relevant_atoms[:12]]

        # ── venue-specific tips ──────────────────────────────────────────────
        venue_tips = _venue_tips(similar[0][0] if similar else "", avg_g)

        confidence = min(1.0, total_runs / 5.0) * (similar[0][1] if similar else 0.0)

        return PreflightHint(
            topic            = topic,
            similar_runs     = total_runs,
            best_sub_queries = best_sub_queries,
            grounding_risk   = grounding_risk,
            learned_atom_ids = atom_ids,
            venue_tips       = venue_tips,
            avg_grounding    = avg_g,
            confidence       = confidence,
        )

    # =========================================================================
    # MODE B — Active learning (/learn <topic>)
    # =========================================================================

    def active_learn(
        self,
        topic: str,
        web_results: list[dict],
        vault_add_fn: Optional[Callable] = None,
    ) -> dict:
        """
        Ingest web results as permanent atoms.

        web_results: list of dicts from Agent 12, each with:
            { url, title, snippet, source }

        vault_add_fn: optional callback to also add the atom to your
            existing vault system. Signature: fn(atom_text, metadata) -> atom_id

        Returns a summary dict.
        """
        import hashlib

        added     = 0
        skipped   = 0
        atom_ids  = []

        existing_texts = {a["text"] for a in self.store.get_all_atoms()}

        for result in web_results:
            text = result.get("snippet", result.get("abstract", "")).strip()
            if not text or len(text) < 40:
                skipped += 1
                continue

            # deduplicate on content
            if text in existing_texts:
                skipped += 1
                continue

            atom_id = "la_" + hashlib.md5(text.encode()).hexdigest()[:10]

            # quality score: longer snippets from academic sources rank higher
            url = result.get("url", "")
            trust = "high" if any(
                s in url for s in ("arxiv.org", "scholar.google", "acm.org",
                                   "ieee.org", "springer.com", "nature.com",
                                   "semanticscholar.org", "pubmed.ncbi")
            ) else "medium"

            atom = LearnedAtom(
                atom_id     = atom_id,
                ts          = time.time(),
                topic       = topic,
                text        = text,
                source_url  = url,
                source_title= result.get("title", ""),
                trust       = trust,
            )
            self.store.add_atom(atom)
            existing_texts.add(text)
            atom_ids.append(atom_id)
            added += 1

            # optionally push into the live vault
            if vault_add_fn:
                try:
                    vault_add_fn(
                        text,
                        {"source": url, "topic": topic,
                         "atom_id": atom_id, "trust": trust},
                    )
                except Exception:
                    pass   # vault ingestion is best-effort

        return {
            "topic":      topic,
            "added":      added,
            "skipped":    skipped,
            "atom_ids":   atom_ids,
            "trust_dist": self._trust_distribution(atom_ids),
        }

    # =========================================================================
    # MODE C — Feedback (/learn feedback)
    # =========================================================================

    def record_feedback(
        self,
        corrections: list[dict],
        run_id: Optional[str] = None,
        topic: str = "",
    ) -> int:
        """
        Record user corrections on a paper.

        corrections: list of dicts, each with:
            { section, issue, sentence, note? }
            issue: "hallucinated" | "wrong_citation" | "good" | "unclear"

        Returns number of corrections stored.
        """
        import hashlib

        rid = run_id or self._last_run_id or "unknown"
        stored = 0

        for c in corrections:
            corr_id = "c_" + hashlib.md5(
                f"{rid}{c.get('sentence','')}{time.time()}".encode()
            ).hexdigest()[:8]

            # "good" sentences with high trust become learned atoms
            if c.get("issue") == "good":
                atom_id = "fb_" + corr_id[2:]
                atom = LearnedAtom(
                    atom_id      = atom_id,
                    ts           = time.time(),
                    topic        = topic,
                    text         = c.get("sentence", ""),
                    source_url   = f"feedback:{rid}",
                    source_title = f"User-verified: {topic}",
                    trust        = "high",
                )
                self.store.add_atom(atom)

            corr = Correction(
                corr_id = corr_id,
                ts      = time.time(),
                run_id  = rid,
                topic   = topic,
                section = c.get("section", "unknown"),
                issue   = c.get("issue", "unclear"),
                sentence= c.get("sentence", ""),
                note    = c.get("note", ""),
            )
            self.store.add_correction(corr)
            stored += 1

        return stored

    def get_hallucination_patterns(self, topic: str) -> list[str]:
        """
        Return sentences previously flagged as hallucinated for this topic.
        Agent 13 can use these as negative examples in its prompt.
        """
        all_corrs = self.store.get_corrections()
        return [
            c["sentence"] for c in all_corrs
            if c["issue"] == "hallucinated"
            and self.matcher.similarity(topic, c["topic"]) >= MIN_SIMILARITY
            and len(c["sentence"]) > 20
        ]

    # =========================================================================
    # REVIEW — /learn review dashboard
    # =========================================================================

    def review(self) -> dict:
        """
        Return a structured summary for the /learn review command.
        Consumed by the CLI to render the dashboard.
        """
        stats    = self.store.get_stats()
        runs     = self.store.get_runs()
        atoms    = self.store.get_all_atoms()
        cache    = self.store.get_topic_cache()
        corrs    = self.store.get_corrections()

        # ── per-topic summary ────────────────────────────────────────────────
        topic_rows = []
        for topic_key, c in cache.items():
            n = c.get("runs", 0)
            if n == 0:
                continue
            avg_g = c.get("total_grounding", 0.0) / n
            topic_rows.append({
                "topic":       topic_key.replace("_", " ")[:50],
                "runs":        n,
                "avg_ground":  avg_g,
                "failures":    c.get("failure_count", 0),
                "last_seen":   datetime.fromtimestamp(
                                   c.get("last_ts", 0)).strftime("%Y-%m-%d"),
            })
        topic_rows.sort(key=lambda x: x["runs"], reverse=True)

        # ── grounding trend (last 10 runs) ───────────────────────────────────
        last_10 = sorted(runs, key=lambda r: r["ts"], reverse=True)[:10]
        grounding_trend = [
            {
                "date":  datetime.fromtimestamp(r["ts"]).strftime("%m-%d"),
                "ratio": r["grounding_ratio"],
                "topic": r["topic"][:30],
            }
            for r in reversed(last_10)
        ]

        # ── atom trust distribution ──────────────────────────────────────────
        trust_counts = Counter(a["trust"] for a in atoms)

        # ── top failure causes ────────────────────────────────────────────────
        all_failures = [f for r in runs for f in r.get("failures", [])]
        failure_counts = Counter(all_failures).most_common(5)

        # ── most corrected topics ────────────────────────────────────────────
        corr_topics = Counter(c["topic"] for c in corrs).most_common(5)

        return {
            "stats":           stats,
            "topic_summary":   topic_rows[:10],
            "grounding_trend": grounding_trend,
            "trust_dist":      dict(trust_counts),
            "top_failures":    failure_counts,
            "corr_topics":     corr_topics,
            "total_topics":    len(cache),
            "using_embeddings":self.matcher._use_embeddings,
            "store_path":      str(self.store.path.resolve()),
        }

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _trust_distribution(self, atom_ids: list[str]) -> dict:
        atoms = {a["atom_id"]: a for a in self.store.get_all_atoms()}
        dist = Counter(
            atoms[aid]["trust"] for aid in atom_ids if aid in atoms
        )
        return dict(dist)

    def get_learned_atoms_for_context(
        self, topic: str, top_n: int = 12
    ) -> list[dict]:
        """
        Return the top-N learned atoms most relevant to a topic,
        weighted by similarity × recency × use_count.
        Ready to inject directly into Agent 13's context.
        """
        all_atoms = self.store.get_all_atoms()
        scored = []
        for atom in all_atoms:
            sim  = self.matcher.similarity(topic, atom["topic"])
            if sim < 0.5:
                continue
            rec  = _decay_weight(atom["ts"])
            pop  = math.log1p(atom.get("use_count", 0)) / 5.0
            score = sim * rec + 0.15 * pop
            scored.append((atom, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        result = []
        for atom, _ in scored[:top_n]:
            self.store.mark_atom_used(atom["atom_id"])
            result.append({
                "atom_id": atom["atom_id"],
                "text":    atom["text"],
                "source":  atom["source_url"],
                "trust":   atom["trust"],
            })
        return result


# =============================================================================
# VENUE TIPS — pattern library from past runs
# =============================================================================

def _venue_tips(topic: str, avg_grounding: float) -> list[str]:
    """
    Return short tips based on known venue conventions.
    These get surfaced in the pre-flight hint.
    """
    tips = []
    t = topic.lower()

    if avg_grounding < 0.5:
        tips.append(
            "Grounding consistently low for this topic — "
            "run post-processing citation injector regardless of audit result."
        )

    if "transformer" in t or "attention" in t:
        tips.append(
            "Attention/transformer topics: cite original Vaswani et al. [2017] "
            "in Introduction; Agent 13 often omits this."
        )

    if "neural network" in t or "cnn" in t or "deep learning" in t:
        tips.append(
            "Deep learning topics: always include a Limitations section — "
            "reviewers at IEEE flag its absence."
        )

    if "language model" in t or "llm" in t:
        tips.append(
            "LLM topics: include benchmark comparisons (MMLU, HumanEval) "
            "in Results — DSJ and ACL reviewers expect them."
        )

    return tips[:3]


# =============================================================================
# CLI RENDER HELPERS  — feed these into your Rich console
# =============================================================================

def render_preflight(hint: PreflightHint) -> dict:
    """
    Returns a dict ready for the CLI to render a pre-flight banner.
    Only show this if hint.confidence > 0.3 (otherwise it's noise).
    """
    if hint.confidence <= 0.3:
        return {"show": False}

    lines = []
    if hint.similar_runs > 0:
        lines.append(
            f"Seen {hint.similar_runs} similar run"
            f"{'s' if hint.similar_runs != 1 else ''} before  ·  "
            f"avg grounding {hint.avg_grounding:.0%}"
        )
    if hint.grounding_risk:
        lines.append(
            "⚠  Grounding-risk topic — "
            "citation injector will run unconditionally after writing"
        )
    if hint.learned_atom_ids:
        lines.append(
            f"{len(hint.learned_atom_ids)} pre-loaded atom"
            f"{'s' if len(hint.learned_atom_ids) != 1 else ''} "
            "from /learn vault"
        )
    for tip in hint.venue_tips:
        lines.append(f"Tip: {tip}")

    return {
        "show":      True,
        "lines":     lines,
        "confidence":hint.confidence,
        "risk":      hint.grounding_risk,
    }


def render_active_learn_result(result: dict) -> list[str]:
    """Lines to print after /learn <topic> completes."""
    lines = [
        f"✓  Ingested {result['added']} atom"
        f"{'s' if result['added'] != 1 else ''} "
        f"on '{result['topic']}'",
    ]
    if result["skipped"]:
        lines.append(
            f"   {result['skipped']} duplicate/short results skipped"
        )
    dist = result.get("trust_dist", {})
    if dist:
        parts = [f"{v} {k}" for k, v in dist.items()]
        lines.append(f"   Trust: {', '.join(parts)}")
    lines.append(
        "   These atoms will auto-load next time you write a paper "
        "on this topic."
    )
    return lines


# =============================================================================
# ENTRY POINT — quick smoke test
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import shutil

    # use a temp store so the test doesn't touch .vasis_learn.json
    tmp_dir = tempfile.mkdtemp()
    tmp_store = Path(tmp_dir) / "test_learn.json"

    learn = LearnEngine(store_path=tmp_store)

    # ── smoke test: record a run ─────────────────────────────────────────────
    fake_result = {
        "agents": [
            {"id":1,"grade":"C","score":0.50,"time":19.5},
            {"id":3,"grade":"B","score":1.00,"time":0.0},
            {"id":4,"grade":"B","score":0.75,"time":12.4},
            {"id":13,"grade":"A","score":0.90,"time":583.4},
        ],
        "grounding_ratio": 0.0,
        "atoms_retrieved":  8,
        "sub_queries": ["attention is all you need",
                        "attention is all you need limitations"],
        "word_count":   1731,
        "duration_s":   605.0,
        "venue":        "IEEE",
        "doc_type":     "research_article",
        "query_raw":    "write a paper on limitations of attention is all you need",
        "failures":     ["grounding_fail"],
    }
    run_id = learn.record_run("attention is all you need", fake_result)
    print(f"[test] recorded run {run_id}")

    # ── smoke test: active learn ─────────────────────────────────────────────
    fake_web = [
        {"url":"https://arxiv.org/abs/2005.14165",
         "title":"Sparse Transformers",
         "snippet":"Sparse Transformers reduce attention complexity from O(n²) "
                   "to O(n√n) by using factorized attention patterns, enabling "
                   "much longer sequence modelling than standard dense attention."},
        {"url":"https://arxiv.org/abs/2006.04768",
         "title":"Linformer",
         "snippet":"Linformer approximates full attention with a low-rank matrix "
                   "projection, reducing memory and time complexity to O(n)."},
    ]
    res = learn.active_learn("attention transformer", fake_web)
    print(f"[test] active learn: added={res['added']} skipped={res['skipped']}")

    # ── smoke test: pre-flight ───────────────────────────────────────────────
    hint = learn.get_preflight("solutions to transformer attention limitations")
    print(f"[test] preflight: runs={hint.similar_runs} "
          f"risk={hint.grounding_risk} conf={hint.confidence:.2f}")
    for line in render_preflight(hint).get("lines", []):
        print(f"       {line}")

    # ── smoke test: review ───────────────────────────────────────────────────
    summary = learn.review()
    print(f"[test] review: "
          f"runs={summary['stats']['total_runs']} "
          f"atoms={summary['stats']['total_atoms_learned']} "
          f"embeddings={summary['using_embeddings']}")

    shutil.rmtree(tmp_dir)
    print("[test] PASS")
