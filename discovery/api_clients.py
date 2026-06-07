import os
import time
import requests
import arxiv
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from the .env file in the root directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def reconstruct_openalex_abstract(inverted_index: Dict[str, List[int]]) -> str:
    """Helper to reconstruct the abstract string from OpenAlex's inverted index."""
    if not inverted_index:
        return ""
    try:
        max_idx = max([idx for positions in inverted_index.values() for idx in positions])
        words = [""] * (max_idx + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        return " ".join(words).strip()
    except Exception:
        return ""

class ArxivClient:
    def __init__(self):
        self.client = arxiv.Client(num_retries=3, page_size=10)
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        # Truncate query to 100 chars max — long queries cause ArXiv 503/429
        query = query[:100].rsplit(' ', 1)[0] if len(query) > 100 else query
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        try:
            for paper in self.client.results(search):
                results.append({
                    "id": paper.get_short_id(),
                    "title": paper.title,
                    "summary": paper.summary,
                    "pdf_url": paper.pdf_url,
                    "authors": [a.name for a in paper.authors],
                    "published": paper.published.isoformat(),
                    "source": "arXiv"
                })
        except Exception as e:
            print(f"[ArxivClient] Warning: {e} — returning partial results ({len(results)} papers)")
        time.sleep(3)
        return results



class SemanticScholarClient:
    def __init__(self):
        self.api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {"x-api-key": self.api_key} if self.api_key else {}

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,authors,year,citationCount,url,openAccessPdf"
        }
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code != 200:
            print(f"[SemanticScholarClient] Error: {response.status_code} - {response.text}")
            return []
            
        data = response.json()
        
        results = []
        for paper in data.get("data", []):
            pdf_url = None
            if paper.get("openAccessPdf"):
                pdf_url = paper["openAccessPdf"].get("url")
            
            results.append({
                "id": paper.get("paperId"),
                "title": paper.get("title"),
                "summary": paper.get("abstract"),
                "pdf_url": pdf_url or paper.get("url"),
                "authors": [a.get("name") for a in paper.get("authors", [])],
                "year": paper.get("year"),
                "citationCount": paper.get("citationCount"),
                "source": "Semantic Scholar"
            })
        return results


class OpenAlexClient:
    def __init__(self):
        self.api_key = os.environ.get("OPENALEX_API_KEY")
        self.base_url = "https://api.openalex.org"

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/works"
        params = {
            "search": query,
            "per-page": max_results
        }
        if self.api_key:
            params["api_key"] = self.api_key
            
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"[OpenAlexClient] Error: {response.status_code} - {response.text}")
            return []
            
        data = response.json()
        
        results = []
        for paper in data.get("results", []):
            pdf_url = paper.get("open_access", {}).get("oa_url")
            
            results.append({
                "id": paper.get("id"),
                "title": paper.get("title"),
                "summary": reconstruct_openalex_abstract(paper.get("abstract_inverted_index")),
                "pdf_url": pdf_url or paper.get("doi") or paper.get("id"),
                "authors": [a.get("author", {}).get("display_name") for a in paper.get("authorships", [])],
                "year": paper.get("publication_year"),
                "citationCount": paper.get("cited_by_count"),
                "source": "OpenAlex"
            })
        return results


class CoreClient:
    def __init__(self):
        self.api_key = os.environ.get("CORE_API_KEY")
        self.base_url = "https://api.core.ac.uk/v3"
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/search/works"
        params = {
            "q": query,
            "limit": max_results
        }
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code != 200:
            print(f"[CoreClient] Error: {response.status_code} - {response.text}")
            return []
            
        data = response.json()
        
        results = []
        for paper in data.get("results", []):
            results.append({
                "id": paper.get("id"),
                "title": paper.get("title"),
                "summary": paper.get("abstract"),
                "pdf_url": paper.get("downloadUrl"),
                "authors": [a.get("name") for a in paper.get("authors", [])],
                "year": paper.get("yearPublished"),
                "source": "CORE"
            })
        return results

class ResearchDiscoveryEngine:
    """Unified engine to query all sources and aggregate the landscape."""
    def __init__(self):
        self.arxiv = ArxivClient()
        self.semantic_scholar = SemanticScholarClient()
        self.openalex = OpenAlexClient()
        self.core = CoreClient()

    def discover(self, query: str, max_results_per_api: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Queries all APIs and returns an aggregated dictionary of results."""
        return {
            "arxiv": self.arxiv.search(query, max_results=max_results_per_api),
            "semantic_scholar": self.semantic_scholar.search(query, max_results=max_results_per_api),
            "openalex": self.openalex.search(query, max_results=max_results_per_api),
            "core": self.core.search(query, max_results=max_results_per_api)
        }
