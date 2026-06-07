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

# Define the 10 prompts mapped to their respective PDF keys
TEST_CASES = [
    {
        "id": 1,
        "category": "Beginner - Explanation/Definitional",
        "pdf_key": "rnn",
        "question": "Explain what a Recurrent Neural Network is as if I am a 10-year-old child."
    },
    {
        "id": 2,
        "category": "Beginner - Comparative",
        "pdf_key": "rnn",
        "question": "What is the main difference between an RNN and a CNN according to these papers?"
    },
    {
        "id": 3,
        "category": "Advanced - Comparative",
        "pdf_key": "rnn",
        "question": "Compare the traditional RNN architecture with Long Short-Term Memory (LSTM) units. Why do LSTMs perform better?"
    },
    {
        "id": 4,
        "category": "Beginner - Procedural",
        "pdf_key": "cnn",
        "question": "Explain the role and steps of training a CNN from scratch as detailed in the paper."
    },
    {
        "id": 5,
        "category": "Beginner - Definitional",
        "pdf_key": "cnn",
        "question": "What is a convolutional layer, and how does it differ from a fully connected layer?"
    },
    {
        "id": 6,
        "category": "Advanced - Methodology",
        "pdf_key": "cnn",
        "question": "Explain the receptive field calculation and parameter reduction techniques for stacked convolutional layers."
    },
    {
        "id": 7,
        "category": "Beginner - Factual",
        "pdf_key": "attention",
        "question": "What is the dimension of the keys d_k in Scaled Dot-Product Attention?"
    },
    {
        "id": 8,
        "category": "Advanced - Causal",
        "pdf_key": "attention",
        "question": "Why does the Transformer utilize masking in the decoder self-attention, and what are its mathematical implications?"
    },
    {
        "id": 9,
        "category": "Beginner - Bibliography",
        "pdf_key": "attention",
        "question": "What are the references cited in this paper?"
    },
    {
        "id": 10,
        "category": "Advanced - Methodology",
        "pdf_key": "moe",
        "question": "What is the Mixture-of-Experts (MoE) architecture in Large Language Models, what are its primary advantages, and what routing strategies are discussed in the paper?"
    }
]

def main():
    results = []
    
    # Ingest all papers first to establish cache and verify ingestion pipeline
    print("=============================================================")
    print(" PHASE 1: INGESTING ALL 4 TARGET PAPERS")
    print("=============================================================")
    rags = {}
    # Set utf-8 output encoding
    sys.stdout.reconfigure(encoding='utf-8')
    
    for key, path in PDF_PATHS.items():
        print(f"\nIngesting {key}: {path}...")
        try:
            rag = PageIndexREMSE()
            rag.ingest(path, force_reindex=False)
            rags[key] = rag
            print(f"[OK] {key} ingestion successful! Cache loaded or generated.")
        except Exception as e:
            print(f"[FAIL] Failed to ingest {key}: {e}")
            sys.exit(1)
            
    print("\n=============================================================")
    print(" PHASE 2: EXECUTING 10 STRESS TEST QUERIES")
    print("=============================================================")
    
    for case in TEST_CASES:
        qid = case["id"]
        cat = case["category"]
        pdf_key = case["pdf_key"]
        qtext = case["question"]
        
        print(f"\n[{qid}/10] CATEGORY: {cat}")
        print(f"DOCUMENT: {os.path.basename(PDF_PATHS[pdf_key])}")
        print(f"QUERY: '{qtext}'")
        
        rag = rags[pdf_key]
        
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
            
            print(f"-> Result: Trust={trust} | Grade={grade} | Atoms={atoms} | Pages={pages} | Time={elapsed:.2f}s")
            print(f"-> Snippet: {ans[:200].replace(chr(10), ' ')}...")
            
            results.append({
                "id": qid,
                "category": cat,
                "doc": os.path.basename(PDF_PATHS[pdf_key]),
                "query": qtext,
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
            print(f"✗ Query {qid} failed: {e}")
            results.append({
                "id": qid,
                "category": cat,
                "doc": os.path.basename(PDF_PATHS[pdf_key]),
                "query": qtext,
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
        "stress_test_results.json"
    )
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("\n=============================================================")
    print(f" STRESS TESTS COMPLETE. Scorecard saved to: {summary_path}")
    print("=============================================================")

if __name__ == "__main__":
    main()
