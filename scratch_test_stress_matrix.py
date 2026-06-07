import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import PageIndexREMSE

# Define the PDF paths on disk
PDF_PATHS = {
    "moe": r"C:\Users\ACER\Downloads\2507.11181v2.pdf",
    "rnn": r"C:\Users\ACER\Downloads\Recurrent Neural Networks.pdf",
    "attention": r"C:\Users\ACER\Downloads\NIPS-2017-attention-is-all-you-need-Paper.pdf",
    "cnn": r"C:\Users\ACER\Downloads\Convolutional Neural Networks.pdf"
}

# The 10 queries
QUERIES = [
    {"id": 1, "target_key": "rnn", "question": "Explain what a Recurrent Neural Network is as if I am a 10-year-old child."},
    {"id": 2, "target_key": "rnn", "question": "What is the main difference between an RNN and a CNN according to these papers?"},
    {"id": 3, "target_key": "rnn", "question": "Compare the traditional RNN architecture with Long Short-Term Memory (LSTM) units. Why do LSTMs perform better?"},
    {"id": 4, "target_key": "cnn", "question": "Explain the role and steps of training a CNN from scratch as detailed in the paper."},
    {"id": 5, "target_key": "cnn", "question": "What is a convolutional layer, and how does it differ from a fully connected layer?"},
    {"id": 6, "target_key": "cnn", "question": "Explain the receptive field calculation and parameter reduction techniques for stacked convolutional layers."},
    {"id": 7, "target_key": "attention", "question": "What is the dimension of the keys d_k in Scaled Dot-Product Attention?"},
    {"id": 8, "target_key": "attention", "question": "Why does the Transformer utilize masking in the decoder self-attention, and what are its mathematical implications?"},
    {"id": 9, "target_key": "attention", "question": "What are the references cited in this paper?"},
    {"id": 10, "target_key": "moe", "question": "What is the Mixture-of-Experts (MoE) architecture in Large Language Models, what are its primary advantages, and what routing strategies are discussed in the paper?"}
]

def main():
    # Set UTF-8 output encoding for Windows terminal
    sys.stdout.reconfigure(encoding='utf-8')
    
    # Ingest all papers first to establish cache and verify ingestion pipeline
    print("=============================================================")
    print(" PHASE 1: INGESTING ALL 4 TARGET PAPERS")
    print("=============================================================")
    rags = {}
    for key, path in PDF_PATHS.items():
        print(f"\nIngesting {key}: {path}...")
        try:
            rag = PageIndexREMSE()
            rag.ingest(path, force_reindex=False)
            rags[key] = rag
            print(f"[OK] {key} ingestion successful!")
        except Exception as e:
            print(f"[FAIL] Failed to ingest {key}: {e}")
            sys.exit(1)
            
    print("\n=============================================================")
    print(" PHASE 2: EXECUTING 40-CASE MATRIX (10 QUERIES x 4 PAPERS)")
    print("=============================================================")
    
    results = []
    total_runs = len(PDF_PATHS) * len(QUERIES)
    run_idx = 0
    
    for pdf_key, path in PDF_PATHS.items():
        pdf_name = os.path.basename(path)
        rag = rags[pdf_key]
        
        print(f"\n--- ACTIVE DOCUMENT: {pdf_name} ({pdf_key}) ---")
        
        for case in QUERIES:
            run_idx += 1
            qid = case["id"]
            qtext = case["question"]
            target_key = case["target_key"]
            is_in_scope = (target_key == pdf_key)
            
            print(f"\n[{run_idx}/{total_runs}] Query {qid} (In-Scope: {is_in_scope})")
            print(f"QUERY: '{qtext}'")
            
            t0 = time.time()
            try:
                res = rag.query(
                    question=qtext,
                    show_provenance=False,
                    save_result=True
                )
                elapsed = time.time() - t0
                
                # Extract key metrics
                ans = res.get("answer", "")
                conf = res.get("confidence", 0.0)
                trust = res.get("trust_level", "low")
                grade = res.get("pipeline_grade", "F")
                pages = res.get("pages_referenced", [])
                atoms = res.get("atoms_used", 0)
                contr = res.get("contradictions_found", False)
                
                print(f"-> Result: Trust={trust} | Grade={grade} | Atoms={atoms} | Time={elapsed:.2f}s")
                print(f"-> Snippet: {ans[:150].replace(chr(10), ' ')}...")
                
                results.append({
                    "run_index": run_idx,
                    "query_id": qid,
                    "query": qtext,
                    "pdf_key": pdf_key,
                    "pdf_name": pdf_name,
                    "is_in_scope": is_in_scope,
                    "answer": ans,
                    "confidence": conf,
                    "trust_level": trust,
                    "grade": grade,
                    "pages": pages,
                    "atoms": atoms,
                    "contradictions": contr,
                    "time_taken": elapsed,
                    "status": "PASS"
                })
            except Exception as e:
                elapsed = time.time() - t0
                print(f"✗ Failed: {e}")
                results.append({
                    "run_index": run_idx,
                    "query_id": qid,
                    "query": qtext,
                    "pdf_key": pdf_key,
                    "pdf_name": pdf_name,
                    "is_in_scope": is_in_scope,
                    "answer": f"ERROR: {e}",
                    "confidence": 0.0,
                    "trust_level": "error",
                    "grade": "F",
                    "pages": [],
                    "atoms": 0,
                    "contradictions": False,
                    "time_taken": elapsed,
                    "status": "FAIL"
                })
                
    # Write summary scorecard
    summary_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "stress_test_matrix_results.json"
    )
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("\n=============================================================")
    print(f" MATRIX STRESS TESTS COMPLETE. Scorecard saved to: {summary_path}")
    print("=============================================================")

if __name__ == "__main__":
    main()
