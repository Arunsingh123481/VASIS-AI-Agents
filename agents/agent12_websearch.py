# agents/agent12_websearch.py
# Web Search Agent — uses Serper API (Google), ArXiv, Semantic Scholar
# Local models ONLY: Qwen 2.5 Coder 3B for query formulation/JSON parsing
#
# Called by Agent 10 when:
#   - Calibration score < 0.70 and novel connections exist
#   - Paper writing mode needs external evidence
#   - Implementation guide needs datasets/baselines

import os
import time
import requests
from typing import List, Dict, Any

from llm.router import generate_json
from config import (
    SERPER_API_KEY, WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_TIMEOUT
)
from console_helper import print_msg

# Load from .env if not in config
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

_SERPER_KEY = SERPER_API_KEY or os.environ.get("SERPER_API_KEY", "")

SYSTEM_QUERY = (
    "You are an expert academic search query formulator. "
    "Given a research topic or gap description, generate "
    "precise search queries. Return ONLY valid JSON."
)


# ── SERPER (GOOGLE SEARCH) ───────────────────────────────────────────────────

def _search_serper(query: str, num: int = 5) -> List[Dict[str, Any]]:
    """Search Google via Serper API and return structured results."""
    if not _SERPER_KEY:
        print_msg("[Agent12] No Serper API key configured — skipping Google search.")
        return []

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": _SERPER_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "num": num
    }

    try:
        resp = requests.post(
            url, json=payload, headers=headers,
            timeout=WEB_SEARCH_TIMEOUT
        )
        if resp.status_code != 200:
            print_msg(f"[Agent12] Serper error {resp.status_code}: {resp.text[:200]}")
            return []

        data = resp.json()
        results = []

        for item in data.get("organic", [])[:num]:
            results.append({
                "title":   item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url":     item.get("link", ""),
                "source":  "Google (Serper)",
                "position": item.get("position", 0),
            })

        # Also grab knowledge graph if available
        kg = data.get("knowledgeGraph", {})
        if kg.get("title"):
            results.append({
                "title":   kg.get("title", ""),
                "snippet": kg.get("description", ""),
                "url":     kg.get("website", ""),
                "source":  "Google Knowledge Graph",
                "position": 0,
            })

        return results

    except requests.exceptions.Timeout:
        print_msg("[Agent12] Serper request timed out.")
        return []
    except Exception as e:
        print_msg(f"[Agent12] Serper error: {e}")
        return []


# ── ARXIV SEARCH ─────────────────────────────────────────────────────────────

def _search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search ArXiv for recent preprints."""
    try:
        import arxiv
        client = arxiv.Client(num_retries=2, page_size=max_results)

        # Truncate query to avoid ArXiv 503 errors
        safe_query = query[:100].rsplit(' ', 1)[0] if len(query) > 100 else query

        search = arxiv.Search(
            query=safe_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for paper in client.results(search):
            results.append({
                "title":   paper.title,
                "snippet": paper.summary[:300] if paper.summary else "",
                "url":     paper.pdf_url or paper.entry_id,
                "source":  "ArXiv",
                "authors": [a.name for a in paper.authors[:3]],
                "year":    paper.published.year if paper.published else None,
            })

        time.sleep(2)  # Respect ArXiv rate limit
        return results

    except Exception as e:
        print_msg(f"[Agent12] ArXiv search error: {e}")
        return []


# ── SEMANTIC SCHOLAR SEARCH ──────────────────────────────────────────────────

def _search_semantic_scholar(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search Semantic Scholar for cited academic papers."""
    ss_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": ss_key} if ss_key else {}

    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,authors,year,citationCount,url,openAccessPdf"
    }

    try:
        resp = requests.get(
            base_url, headers=headers, params=params,
            timeout=WEB_SEARCH_TIMEOUT
        )
        if resp.status_code != 200:
            print_msg(f"[Agent12] Semantic Scholar error: {resp.status_code}")
            return []

        data = resp.json()
        results = []

        for paper in data.get("data", [])[:max_results]:
            pdf_url = None
            if paper.get("openAccessPdf"):
                pdf_url = paper["openAccessPdf"].get("url")

            results.append({
                "title":     paper.get("title", ""),
                "snippet":   (paper.get("abstract") or "")[:300],
                "url":       pdf_url or paper.get("url", ""),
                "source":    "Semantic Scholar",
                "authors":   [a.get("name", "") for a in paper.get("authors", [])[:3]],
                "year":      paper.get("year"),
                "citations": paper.get("citationCount", 0),
            })

        return results

    except Exception as e:
        print_msg(f"[Agent12] Semantic Scholar error: {e}")
        return []


# ── QUERY FORMULATION ────────────────────────────────────────────────────────

def _formulate_queries(topic: str, novel_connections: list = None) -> List[str]:
    """Use local Qwen model to generate smart search queries."""
    connections_text = ""
    if novel_connections:
        connections_text = "\n".join([
            f"  - {c.get('from', '')} -> {c.get('to', '')} "
            f"(via {c.get('via', [])})"
            for c in novel_connections[:3]
        ])

    prompt = f"""Generate 4 precise academic search queries for this topic.

Topic: {topic}
{"Novel connections found:" + chr(10) + connections_text if connections_text else ""}

Return JSON:
{{
  "queries": [
    "query 1 — most specific to the topic",
    "query 2 — related methodology or approach",
    "query 3 — benchmark datasets for this area",
    "query 4 — recent survey or review paper"
  ]
}}"""

    try:
        result = generate_json(
            "agent12_websearch", prompt,
            system=SYSTEM_QUERY
        )
        queries = result.get("queries", [])
        if queries and isinstance(queries, list):
            return [str(q) for q in queries[:4]]
    except Exception as e:
        print_msg(f"[Agent12] Query formulation error: {e}")

    # Fallback: use topic directly
    return [
        f"{topic} 2024 2025",
        f"{topic} survey review",
        f"{topic} benchmark dataset",
        f"{topic} methodology approach",
    ]


# ── DEDUPLICATION ────────────────────────────────────────────────────────────

def _deduplicate(results: List[Dict]) -> List[Dict]:
    """Remove duplicate results by URL or title."""
    seen_urls   = set()
    seen_titles = set()
    unique = []

    for r in results:
        url   = r.get("url", "").strip().lower()
        title = r.get("title", "").strip().lower()

        if url and url in seen_urls:
            continue
        if title and title in seen_titles:
            continue

        if url:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)
        unique.append(r)

    return unique


# ── MAIN SEARCH FUNCTION ────────────────────────────────────────────────────

def search_web(
    topic: str,
    novel_connections: list = None,
    max_per_source: int = 5,
    queries_override: List[str] = None
) -> Dict[str, Any]:
    """
    Main Agent 12 entry point.
    Searches multiple sources and returns aggregated results.

    Args:
        topic:             Research topic or gap description
        novel_connections: Agent 11 causal chains (optional)
        max_per_source:    Max results per API source
        queries_override:  List of predefined queries to use instead of generating them.

    Returns:
        Dict with sources, queries used, and search metadata
    """
    start_time = time.time()

    print_msg(f"\n{'='*50}")
    print_msg("[Agent12] WEB SEARCH")
    print_msg(f"[Agent12] Topic: {topic}")
    print_msg(f"{'='*50}\n")

    # Step 1: Formulate queries
    if queries_override:
        queries = queries_override
    else:
        queries = _formulate_queries(topic, novel_connections)
    print_msg(f"[Agent12] Generated {len(queries)} search queries")

    all_results = []

    # Step 2: Search each source with the primary query
    primary_query = queries[0] if queries else topic

    # Serper (Google)
    print_msg("[Agent12] Searching Google (Serper)...")
    serper_results = _search_serper(primary_query, num=max_per_source)
    all_results.extend(serper_results)
    print_msg(f"[Agent12]   → {len(serper_results)} results")

    # ArXiv
    print_msg("[Agent12] Searching ArXiv...")
    arxiv_results = _search_arxiv(primary_query, max_results=max_per_source)
    all_results.extend(arxiv_results)
    print_msg(f"[Agent12]   → {len(arxiv_results)} results")

    # Semantic Scholar
    print_msg("[Agent12] Searching Semantic Scholar...")
    ss_results = _search_semantic_scholar(primary_query, max_results=max_per_source)
    all_results.extend(ss_results)
    print_msg(f"[Agent12]   → {len(ss_results)} results")

    # Step 3: Search remaining queries via Serper only (avoid rate limits)
    for q in queries[1:]:
        extra = _search_serper(q, num=3)
        all_results.extend(extra)
        time.sleep(0.5)  # Small delay between requests

    # Step 4: Deduplicate
    unique_results = _deduplicate(all_results)

    elapsed = round(time.time() - start_time, 1)
    print_msg(f"\n[Agent12] Search complete: {len(unique_results)} unique sources in {elapsed}s")

    return {
        "sources":        unique_results,
        "queries_used":   queries,
        "total_found":    len(unique_results),
        "search_time":    elapsed,
        "serper_count":   len(serper_results),
        "arxiv_count":    len(arxiv_results),
        "scholar_count":  len(ss_results),
    }


# ── LEGACY COMPATIBILITY (called by agent10_super.py) ─────────────────────

def search_and_solve(
    gap_description: str,
    novel_connections: list = None
) -> Dict[str, Any]:
    """
    Legacy entry point for Agent 10 orchestrator.
    Wraps search_web for backward compatibility.
    """
    result = search_web(
        topic=gap_description,
        novel_connections=novel_connections
    )
    result["new_papers_added"] = 0  # No auto-ingest in basic mode
    return result


# ── PAPER-WRITING MODE (called by agent13) ────────────────────────────────

def search_for_paper(
    topic: str,
    article_type: str = "research_article",
    novel_connections: list = None
) -> Dict[str, Any]:
    """
    Enhanced search for paper writing mode.
    Uses 5 deterministic paper-writing queries for full coverage.
    """
    queries = [
        f"{topic} survey overview arxiv",
        f"{topic} SOTA 2024 2025",
        f"{topic} architecture design benchmark evaluation",
        f"{topic} implementation code guide tutorial",
        f"{topic} limitations open problems future work"
    ]
    
    query_count_map = {
        "research_article":    4,
        "review_article":      8,
        "short_communication": 2,
        "letter_to_editor":    1,
        "systematic_review":   10,
        "perspective_article": 2,
        "technical_note":      3,
        "case_study":          3,
    }
    max_per = query_count_map.get(article_type, 4)

    return search_web(
        topic=topic,
        novel_connections=novel_connections,
        max_per_source=min(max_per + 2, WEB_SEARCH_MAX_RESULTS),
        queries_override=queries
    )
