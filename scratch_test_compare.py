import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import PageIndexREMSE

def main():
    default_pdf = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "uploads",
        "1cf6b031-c99_NIPS-2017-attention-is-all-you-need-Paper.pdf"
    )
    
    rag = PageIndexREMSE()
    rag.ingest(default_pdf, force_reindex=False)
    
    print("\n--- RUNNING TARGET QUERY ---")
    res = rag.query(
        question="Can you summarise this paper and tell me the advantages and disadvantages of this paper and find the novel problem in this paper?",
        show_provenance=True,
        save_result=False
    )
    print("\nFINAL ANSWER:")
    print(res.get('answer'))
    
if __name__ == "__main__":
    main()
