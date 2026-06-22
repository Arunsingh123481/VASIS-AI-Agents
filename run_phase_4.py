import os

from pipeline import PageIndexREMSE
from storage.store import load_index
from intelligence.novelty_pipeline import intelligence_engine

def run_phase_4():
    print("\n" + "="*50)
    print(" SHIKSHIT AI - PHASE 4: RESEARCH INTELLIGENCE ")
    print("="*50)
    
    # Load all cached indexes to simulate an active session workspace
    cache_dir = os.path.join(os.path.dirname(__file__), ".pageindex_cache")
    if not os.path.exists(cache_dir):
        print("No cache directory found. Please run Phase 1 & 2 first.")
        return
        
    rag_instances = {}
    for doc_id in os.listdir(cache_dir):
        path = os.path.join(cache_dir, doc_id)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "tree.json")):
            rag = PageIndexREMSE(model="llama3.2")
            # Mock the doc_id and ready state so we don't have to fully re-ingest
            cached = load_index(doc_id)
            if cached:
                rag.tree_nodes = cached["tree"]
                rag.atoms = cached["atoms"]
                rag.doc_id = doc_id
                rag._ready = True
                rag_instances[f"Paper_{doc_id[:6]}"] = rag

    if len(rag_instances) < 2:
        print("Need at least 2 papers in the vault to run cross-paper intelligence.")
        return

    print("\n[1] Detecting Debates Across Papers...")
    debates = intelligence_engine.detect_debates(rag_instances)
    for d in debates:
        print(f"\nDEBATE: {d.get('debate_topic')}")
        print(f"  Side A: {d.get('side_a')}")
        print(f"  Side B: {d.get('side_b')}")
        print(f"  Open Issue: {d.get('open_issue')}")

    print("\n[2] Running Novelty Search Pipeline (Limitations -> Gaps -> Hypotheses)...")
    all_limitations = []
    for title, rag in rag_instances.items():
        print(f"  -> Extracting limitations for {title}...")
        lims = intelligence_engine.extract_limitations(title, rag)
        all_limitations.extend(lims)
        
    print(f"\nFound {len(all_limitations)} total limitations. Mapping cross-paper gaps...")
    gaps = intelligence_engine.map_cross_paper_gaps(all_limitations)
    print(f"Found {len(gaps)} potential gaps. Filtering for feasibility...")
    
    gaps = intelligence_engine.filter_feasibility(gaps)
    
    print("Framing hypotheses for feasible gaps...")
    hypotheses = intelligence_engine.frame_hypothesis(gaps)
    
    print("\n--- FINAL NOVEL RESEARCH HYPOTHESES ---")
    for h in hypotheses:
        print(f"\nTITLE: {h.get('title')} ({h.get('feasibility', 'Feasible')})")
        print(f"QUESTION: {h.get('research_question')}")
        print(f"HYPOTHESIS: {h.get('hypothesis')}")

if __name__ == "__main__":
    run_phase_4()
