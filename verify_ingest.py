import sys
import os
import time

# Add workspace directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import PageIndexREMSE

def main():
    default_pdf = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "uploads",
        "1cf6b031-c99_NIPS-2017-attention-is-all-you-need-Paper.pdf"
    )
    
    rag = PageIndexREMSE()
    
    print("Testing Ingestion with cached index...")
    start_time = time.time()
    rag.ingest(default_pdf, force_reindex=False)
    elapsed = time.time() - start_time
    
    print("\n" + "="*50)
    print(f"Ingestion completed in: {elapsed:.2f} seconds!")
    print(f"Total Triples Extracted: {len(rag.triples)}")
    print("="*50)
    
    # Let's print some sample triples to verify correctness
    print("\nSample Triples Extracted:")
    for t in rag.triples[:10]:
        subj = str(t.get('subject', '')).encode('ascii', 'ignore').decode('ascii')
        rel = str(t.get('relation', '')).encode('ascii', 'ignore').decode('ascii')
        obj = str(t.get('object', '')).encode('ascii', 'ignore').decode('ascii')
        causal = str(t.get('causal_type', '')).encode('ascii', 'ignore').decode('ascii')
        print(f"  Subject: {subj} | Relation: {rel} | Object: {obj} | Causal: {causal}")
        
    # Let's run a bibliography query
    print("\nTesting Bibliography Query: 'What are the references cited in this paper?'")
    print("This should BYPASS Agent3 (force last section) and SKIP Agent7.")
    res = rag.query(
        question="What are the references cited in this paper?",
        show_provenance=False,
        save_result=False
    )
    print(f"Query Result Answer: {res.get('answer')[:150]}...")
    print(f"Pipeline Grade: {res.get('pipeline_grade')}")
    print(f"Trust Level: {res.get('trust_level')}")
    print(f"Contradictions Found: {res.get('contradictions_found')}")
    print(f"Selected Sections: {res.get('selected_sections')}")
    
    # Check if Agent3 and Agent7 were recorded as expected in results
    review_report = res.get("review_report", {})
    per_agent_scores = review_report.get("per_agent_scores", [])
    for agent_score in per_agent_scores:
        if agent_score.get("agent") in ("agent3_navigator", "agent7_contradiction"):
            print(f"Agent details: name={agent_score.get('agent')}, skipped={agent_score.get('skipped')}, grade={agent_score.get('grade')}, score={agent_score.get('score')}")

if __name__ == "__main__":
    main()
