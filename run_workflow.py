import os
import sys

from discovery.navigator import run_phase_1
from pipeline import PageIndexREMSE

def run_full_workflow(broad_topic: str):
    print("\n" + "="*50)
    print(" SHIKSHIT AI - FULL RESEARCH WORKFLOW ")
    print("="*50)
    
    # --- PHASE 1: RESEARCH NAVIGATOR ---
    downloaded_pdfs = run_phase_1(broad_topic)
    
    if not downloaded_pdfs:
        print("\n[Phase 1] No PDFs were successfully downloaded. Cannot proceed to Phase 2.")
        return
        
    print("\n" + "="*50)
    print(" PHASE 2: RE-MSE HYBRID RAG INGESTION ")
    print("="*50)
    
    # --- PHASE 2: STARTER VAULT INGESTION ---
    for pdf_path in downloaded_pdfs:
        print(f"\nIngesting {os.path.basename(pdf_path)} into RE-MSE Engine...")
        try:
            # Initialize pipeline for each paper
            rag = PageIndexREMSE(model="llama3.2")
            rag.ingest(pdf_path, force_reindex=False)
            
            # Print stats to show it worked
            stats = rag.get_stats()
            print(f"  -> Success: {stats['tree_nodes']} tree nodes, {stats['total_atoms']} atoms created.")
        except Exception as e:
            print(f"  -> Failed to ingest {os.path.basename(pdf_path)}: {e}")

if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "Explainable AI in Medical Imaging"
    run_full_workflow(topic)
