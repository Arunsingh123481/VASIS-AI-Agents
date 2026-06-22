"""
Document Loader — Extracts text content page-by-page from PDF, DOCX, TXT, and MD files.
Uses PyMuPDF (fitz) for PDFs, python-docx for DOCX, and plain reading for TXT/MD.
All loaders normalise output to a list of page dicts with: page_num, text, char_count, word_count.
"""
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from console_helper import print_msg, tqdm


import fitz  # pymupdf
import os
from typing import List, Dict


# ── PDF Loader ───────────────────────────────────────────────────────────────

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


# ── DOCX Loader ──────────────────────────────────────────────────────────────

def load_docx(docx_path: str) -> List[Dict]:
    """
    Load a DOCX file and return a list of page dicts.
    Headings are converted to Markdown-style headers (#, ##, ###) so the
    tree builder can detect natural section boundaries.

    Since DOCX has no physical page concept, we group paragraphs into
    synthetic ~600-word pages to match the PDF loader's output format.
    """
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"DOCX not found: {docx_path}")

    from docx import Document as DocxDocument

    print_msg(f"[cyan]Loading DOCX: {os.path.basename(docx_path)}[/cyan]")
    doc = DocxDocument(docx_path)

    # Map DOCX heading levels to Markdown markers
    _HEADING_MAP = {
        "Heading 1": "# ",
        "Heading 2": "## ",
        "Heading 3": "### ",
        "Heading 4": "#### ",
        "Title": "# ",
    }

    # Build a single Markdown-formatted text from all paragraphs
    md_lines: List[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = para.style.name if para.style else ""
        prefix = _HEADING_MAP.get(style_name, "")
        md_lines.append(f"{prefix}{text}")

    full_text = "\n\n".join(md_lines)

    # Split into synthetic pages of ~600 words each
    TARGET_WORDS_PER_PAGE = 600
    words_all = full_text.split()
    pages: List[Dict] = []
    page_num = 1

    for start in range(0, len(words_all), TARGET_WORDS_PER_PAGE):
        chunk_words = words_all[start:start + TARGET_WORDS_PER_PAGE]
        text = " ".join(chunk_words)
        if len(text) < 50:
            continue
        pages.append({
            "page_num": page_num,
            "text": text,
            "char_count": len(text),
            "word_count": len(chunk_words)
        })
        page_num += 1

    print_msg(f"[green]Loaded {len(pages)} synthetic pages from DOCX.[/green]")
    return pages


# ── Plain Text / Markdown Loader ─────────────────────────────────────────────

def load_text(text_path: str) -> List[Dict]:
    """
    Load a plain text or Markdown file and return a list of page dicts.
    Groups content into synthetic ~600-word pages.
    Markdown headers are preserved so the tree builder can detect sections.
    """
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Text file not found: {text_path}")

    print_msg(f"[cyan]Loading text file: {os.path.basename(text_path)}[/cyan]")

    with open(text_path, "r", encoding="utf-8", errors="replace") as f:
        full_text = f.read()

    TARGET_WORDS_PER_PAGE = 600
    words_all = full_text.split()
    pages: List[Dict] = []
    page_num = 1

    for start in range(0, len(words_all), TARGET_WORDS_PER_PAGE):
        chunk_words = words_all[start:start + TARGET_WORDS_PER_PAGE]
        text = " ".join(chunk_words)
        if len(text) < 50:
            continue
        pages.append({
            "page_num": page_num,
            "text": text,
            "char_count": len(text),
            "word_count": len(chunk_words)
        })
        page_num += 1

    print_msg(f"[green]Loaded {len(pages)} synthetic pages from text file.[/green]")
    return pages


# ── Unified Document Loader ──────────────────────────────────────────────────

_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_document(file_path: str) -> List[Dict]:
    """
    Unified document loader — routes to the appropriate parser based on
    file extension. Returns a list of page dicts.

    Supported formats: .pdf, .docx, .txt, .md
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext in (".txt", ".md"):
        return load_text(file_path)
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )
