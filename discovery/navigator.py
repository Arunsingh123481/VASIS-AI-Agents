import os
import json
import requests
import urllib.parse
from typing import List, Dict, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm.ollama_client import ask_llm
from discovery.api_clients import ResearchDiscoveryEngine

def narrow_topic(broad_topic: str) -> str:
    """Step 1 & 2: Topic Narrowing Engine"""
    prompt = f"""You are an expert academic research advisor.
A student has provided a broad research topic: "{broad_topic}".
Your task is to narrow this down into a specific, writable research niche suitable for an 8-page academic paper.
Pick a specific research angle, system type, and focus. 
Return ONLY the narrowed topic as a single clear sentence or phrase, without any extra conversation.
"""
    narrowed = ask_llm(prompt)
    return narrowed.strip('"\n ')

def map_landscape(narrow_topic: str, discovery_results: Dict[str, List[Dict[str, Any]]]) -> str:
    """Step 4: Research Landscape Mapping"""
    # Flatten results for LLM context
    papers_context = ""
    for source, papers in discovery_results.items():
        for i, p in enumerate(papers):
            summary = p.get('summary') or ''
            papers_context += f"[{source}] {p.get('title')} ({p.get('year', 'N/A')}): {summary[:300]}...\n"

    prompt = f"""You are a senior academic researcher. Based on the following recent papers retrieved for the topic "{narrow_topic}", map out the research landscape.
Identify the main branches, active debates, and open problems. 
Keep it concise, structured, and informative.

Papers Context:
{papers_context}

Output the landscape map in Markdown format:"""
    
    return ask_llm(prompt)

def recommend_starter_papers(landscape_map: str, discovery_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Step 5: Starter Paper Recommendation Engine"""
    all_papers = []
    for source, papers in discovery_results.items():
        for p in papers:
            # We only want papers that have a PDF URL so we can actually download them for the vault
            if p.get('pdf_url'):
                all_papers.append(p)
    
    # Create a minimal list for the LLM to prevent context overflow/crashes
    minimal_papers = []
    for p in all_papers[:20]:
        minimal_papers.append({
            "id": p.get("id"),
            "title": p.get("title"),
            "source": p.get("source"),
            "year": p.get("year")
        })
    papers_json = json.dumps(minimal_papers, indent=2)

    prompt = f"""Based on the Research Landscape below, select exactly 6 "starter papers" from the provided Candidate Papers list.
Choose a mix of foundational, recent, and specific papers that give a beginner the best starting point.
Return the result strictly as a JSON list of objects containing the "id" of the selected papers and a "rationale" for why it was chosen.

Research Landscape:
{landscape_map}

Candidate Papers (JSON):
{papers_json}

Return strictly a JSON array of objects: [ {{"id": "...", "rationale": "..."}}, ... ]
"""
    response_text = ask_llm(prompt, expect_json=True)
    print(f"DEBUG: all_papers count with pdf_url = {len(all_papers)}")
    print(f"DEBUG: LLM response for starter papers:\n{response_text}")
    try:
        selections = json.loads(response_text)
        selected_ids = [s.get('id') for s in selections if isinstance(s, dict)]
        # Filter the original papers
        final_papers = [p for p in all_papers if p.get('id') in selected_ids]
        print(f"DEBUG: final_papers count = {len(final_papers)}")
        if len(final_papers) == 0:
            print("DEBUG: final_papers is empty, falling back to all_papers[:6]")
            return all_papers[:6]
        return final_papers[:6]
    except Exception as e:
        print(f"Error parsing paper recommendations: {e}")
        # Fallback to the first 6 papers with PDFs
        return all_papers[:6]

def download_pdf(url: str, output_dir: str, filename: str) -> str:
    """Downloads a PDF from a URL."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # Simple fix for arxiv URLs: convert /abs/ to /pdf/
    if "arxiv.org/abs/" in url:
        url = url.replace("/abs/", "/pdf/")
    if "arxiv.org/pdf/" in url and not url.endswith(".pdf"):
        url += ".pdf"

    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Check if it's actually a PDF
        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' not in content_type:
            print(f"Warning: URL {url} did not return a PDF (Content-Type: {content_type})")
            return None

        with open(filepath, 'wb') as f:
            f.write(response.content)
        return filepath
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

def run_phase_1(broad_topic: str):
    """Executes the full Phase 1 Workflow."""
    print(f"\n--- Phase 1: Research Navigator ---")
    print(f"Broad Topic: {broad_topic}")
    
    # 1 & 2. Narrow Topic
    print("\nNarrowing topic using AI...")
    narrowed = narrow_topic(broad_topic)
    print(f"Narrowed Topic: {narrowed}")
    
    # 3. Research Discovery APIs
    print("\nQuerying Research APIs (ArXiv, Semantic Scholar, OpenAlex, CORE)...")
    engine = ResearchDiscoveryEngine()
    # Query 3 per API to keep context small for the LLM
    results = engine.discover(narrowed, max_results_per_api=3) 
    total_found = sum(len(papers) for papers in results.values())
    print(f"Found {total_found} papers across APIs.")
    
    # 4. Landscape Mapping
    print("\nMapping Research Landscape...")
    landscape = map_landscape(narrowed, results)
    print("\n--- Research Landscape Map ---")
    print(landscape)
    print("------------------------------\n")
    
    # 5. Recommend Starter Papers
    print("Selecting Starter Papers...")
    starter_papers = recommend_starter_papers(landscape, results)
    print(f"Selected {len(starter_papers)} starter papers.")
    for p in starter_papers:
        print(f" - {p.get('title')} ({p.get('source')})")
        
    # 6. Starter Vault Creation (Download PDFs)
    print("\nDownloading PDFs for Starter Vault...")
    vault_dir = os.path.join(os.path.dirname(__file__), "..", "uploads", "starter_vault")
    downloaded_files = []
    
    for i, p in enumerate(starter_papers):
        url = p.get('pdf_url')
        if not url:
            continue
            
        safe_title = "".join([c if c.isalnum() else "_" for c in p.get('title', f"paper_{i}")])
        filename = f"{safe_title[:50]}.pdf"
        print(f"Downloading {filename} from {url}...")
        
        filepath = download_pdf(url, vault_dir, filename)
        if filepath:
            downloaded_files.append(filepath)
            print(f"  -> Saved to {filepath}")
            
    print(f"\nPhase 1 Complete! Downloaded {len(downloaded_files)} PDFs to {vault_dir}")
    return downloaded_files

if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "AI Security"
    run_phase_1(topic)
