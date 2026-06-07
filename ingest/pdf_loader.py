"""
PDF Loader — Extracts text content page by page from PDF documents.
Uses PyMuPDF (fitz) for reliable text extraction including tables and headers.
"""
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from console_helper import print_msg, tqdm, print_panel


import fitz  # pymupdf
import os
from typing import List, Dict



def load_pdf(pdf_path: str) -> List[Dict]:
    """
    Load a PDF and return a list of page dicts.
    Each page dict contains: page_num, text, char_count, word_count.
    Skips blank pages automatically.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        import pymupdf4llm
        print_msg(f"[cyan]Loading PDF: {os.path.basename(pdf_path)} with pymupdf4llm (Markdown)[/cyan]")
        md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        pages = []
        for i, p_data in enumerate(tqdm(md_pages, desc="Extracting pages")):
            text = p_data.get("text", "").strip()
            if len(text) < 50:
                continue
            words = text.split()
            pages.append({
                "page_num": p_data.get("metadata", {}).get("page", i) + 1,
                "text": text,
                "char_count": len(text),
                "word_count": len(words)
            })
    except ImportError:
        # Graceful fallback to standard PyMuPDF
        print_msg(f"[cyan]Loading PDF: {os.path.basename(pdf_path)} with standard PyMuPDF (fitz) fallback[/cyan]")
        doc = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(tqdm(doc, desc="Extracting pages")):
            text = page.get_text("text").strip()
            if len(text) < 50:
                continue
            words = text.split()
            pages.append({
                "page_num": i + 1,
                "text": text,
                "char_count": len(text),
                "word_count": len(words)
            })
        doc.close()

    print_msg(f"[green]Loaded {len(pages)} non-blank pages.[/green]")
    return pages


def get_pdf_metadata(pdf_path: str) -> Dict:
    """Extract metadata from PDF (title, author, page count etc.)"""
    doc = fitz.open(pdf_path)
    meta = doc.metadata
    meta["page_count"] = len(doc)
    meta["file_size_kb"] = round(os.path.getsize(pdf_path) / 1024, 1)
    doc.close()
    return meta
