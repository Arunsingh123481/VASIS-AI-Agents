# Vasis AI Custom Agents Guide

Custom agents allow you to run specialized tasks against loaded vault papers or general academic topics. This guide explains how to build, run, and query custom agents, including the high-speed direct section extraction feature.

---

## 1. How to Build a Custom Agent

To build a custom agent, use the `/build agent` command in the CLI.

```bash
/build agent [agent_name]
```

### Example: Building an `/introduction` Agent
1. Type `/build agent introduction`.
2. The interactive agent wizard will fire. Because `introduction` is a recognized blueprint, it pre-fills standard academic defaults (category: `paper_section`, input: `all`, etc.).
3. The wizard asks 3 simple questions (instead of the standard 7) to finalize the agent.
4. Once completed, `/introduction` is registered as a custom command.

---

## 2. How to Query a Custom Agent

You can query a custom agent using the slash command of the agent's name followed by the topic or paper title:

```bash
/[agent_name] "[paper title or general topic]"
```

---

## 3. Query Execution Flows

Depending on the agent's category and the query topic, the system uses two distinct execution flows:

### Flow A: Direct Section Extraction (0.0s, No LLM)
This flow is triggered when:
1. The agent category is `paper_section` (e.g., `abstract`, `introduction`, `methodology`, `results`, `discussion`, `conclusion`, `references`).
2. The query topic matches a paper currently indexed in the vault (fuzzy title matching).

**Example Queries:**
* `/introduction "Attention Is All You Need"`
* `/abstract "Attention Is All You Need"`
* `/references "Attention Is All You Need"`
* `/introduction "all"` (matches the loaded paper if exactly one is active)

**Result:**
The system instantly extracts the exact section directly from the PDF's indexed atoms (using tree nodes or regex boundary detection) and prints it to the console in **0.0s**.

---

### Flow B: LLM Generation with Web Search (Fallback)
This flow is triggered when:
1. The agent category is *not* a `paper_section` (e.g., `gapanalysis`, `comparison`, etc.).
2. **OR** the query topic does *not* match any paper loaded in the vault.

**Example Queries:**
* `/introduction "Quantum Computing in 2026"` (No matching paper loaded)
* `/gapanalysis "Attention Is All You Need"` (Analysis agent, not a `paper_section` category)

**Result:**
The system runs a web search using Agent 12 (ArXiv + Semantic Scholar), gathers search results and smart context, passes them to the local LLM, and prints the generated response.

---

## 4. Summary of Commands

| Command Example | Matches Vault Paper? | Category | Action |
|---|---|---|---|
| `/introduction "Attention Is All You Need"` | Yes | `paper_section` | **Direct Extraction (0.0s)** |
| `/introduction "all"` | Yes (Single doc) | `paper_section` | **Direct Extraction (0.0s)** |
| `/introduction "Quantum Computing"` | No | `paper_section` | **Web Search + LLM Fallback** |
| `/gapanalysis "Attention Is All You Need"` | Yes/No | `analysis` | **LLM Generation** |
