"""
REST API — FastAPI server for the PageIndex-RE-MSE system.
Provides endpoints for PDF upload, indexing, querying, and chat.
Run with: python api.py
"""

import os
import sys
import io
import os
import sys

from dotenv import load_dotenv
load_dotenv()

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import shutil
import uuid
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import json

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import PageIndexREMSE
from reconstruction.tutor import tutor_engine
from storage.store import save_note_card, load_note_cards, get_doc_id, delete_index
from intelligence.novelty_pipeline import intelligence_engine
from verification.audit import verifier_engine
from verification.export import exporter_engine

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PageIndex-RE-MSE API",
    description="Hybrid RAG System — Vectorless + Atomic RAG",
    version="1.0.0"
)

# Allow all origins for local development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Storage ──────────────────────────────────────────────────────────────────

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory session store: { session_id -> PageIndexREMSE }
_sessions: Dict[str, Any] = {}

# Track which PDF files belong to which vault (keyed by vault_id / first session_id of a discover batch)
# { vault_id -> [pdf_path, ...] }
_vault_papers: Dict[str, List[str]] = {}

# Map each session_id to its vault_id (one discover batch = one vault_id)
_session_to_vault: Dict[str, str] = {}

SESSION_DB_FILE = os.path.join(os.path.dirname(__file__), "sessions_db.json")

def save_sessions():
    try:
        data = {
            "vault_papers": _vault_papers,
            "session_to_vault": _session_to_vault,
            "session_paths": {sid: rag.pdf_path for sid, rag in _sessions.items()}
        }
        with open(SESSION_DB_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Failed to save sessions: {e}")

def load_sessions():
    global _sessions, _vault_papers, _session_to_vault
    if os.path.exists(SESSION_DB_FILE):
        try:
            with open(SESSION_DB_FILE, "r") as f:
                data = json.load(f)
                _vault_papers.update(data.get("vault_papers", {}))
                _session_to_vault.update(data.get("session_to_vault", {}))
                for sid, pdf_path in data.get("session_paths", {}).items():
                    if os.path.exists(pdf_path):
                        from pipeline import PageIndexREMSE
                        rag = PageIndexREMSE(model="llama3.2")
                        rag.ingest(pdf_path, force_reindex=False)
                        _sessions[sid] = rag
            print(f"Loaded {len(_sessions)} sessions from disk.")
        except Exception as e:
            print(f"Failed to load sessions: {e}")

@app.on_event("startup")
def on_startup():
    load_sessions()

# Map session_id to its specific file path
_session_papers: Dict[str, str] = {}


# ── Security ─────────────────────────────────────────────────────────────────

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key: str = Depends(api_key_header)):
    expected_api_key = os.environ.get("PAGEINDEX_API_KEY", "remse_default_api_key_789")
    print(f"🔑 DEBUG: received='{api_key}' expected='{expected_api_key}'")
    if api_key != expected_api_key:
        raise HTTPException(status_code=403, detail="Could not validate API Key")
    return api_key


# ── Request/Response Models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    session_id: str
    question: str
    top_k_anchors: int = 5
    expansion_passes: int = 4


class QueryResponse(BaseModel):
    answer: str
    sections_used: list
    atoms_used: int
    provenance: dict


class IndexResponse(BaseModel):
    session_id: str
    message: str
    stats: dict


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    model: str


class TutorRequest(BaseModel):
    session_id: str
    message: str
    student_profile: str = "intermediate"
    history: List[Dict[str, str]] = []


class TutorResponse(BaseModel):
    answer: str
    card_data: dict


class NoteCardSaveRequest(BaseModel):
    session_id: str
    card_data: dict

class IntelligenceRequest(BaseModel):
    session_ids: List[str]

class NarrowTopicRequest(BaseModel):
    topic: str


class AutocompleteRequest(BaseModel):
    session_id: str
    text_context: str

class AuditRequest(BaseModel):
    session_id: str
    sentence: str
    vault_id: Optional[str] = None   # optional: to search across all vault papers

class ExportRequest(BaseModel):
    title: str
    sections: List[dict]
    format: str = "docx"


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API and Ollama are running."""
    from llm.ollama_client import check_ollama_connection, DEFAULT_MODEL
    connected = check_ollama_connection()
    return HealthResponse(
        status="ok" if connected else "ollama_disconnected",
        ollama_connected=connected,
        model=DEFAULT_MODEL
    )


@app.post("/upload")
async def upload_and_index(
    files: List[UploadFile] = File(...),
    model: str = "llama3.2",
    force_reindex: bool = False,
    api_key: str = Depends(get_api_key)
):
    """
    Upload one or more PDF files, index them, and return a vault ID.
    Returns the first session ID for compatibility.
    """
    vault_id = str(uuid.uuid4())[:12]
    _vault_papers[vault_id] = []
    
    sessions = []
    
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in (".pdf", ".docx", ".txt", ".md"):
            continue

        session_id = str(uuid.uuid4())[:12]
        filename = f"{session_id}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)

        with open(filepath, "wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            rag = PageIndexREMSE(model=model)
            rag.ingest(filepath, force_reindex=force_reindex)
            _sessions[session_id] = rag
            _vault_papers[vault_id].append(filepath)
            _session_to_vault[session_id] = vault_id
            save_sessions()

            sessions.append({
                "session_id": session_id,
                "vault_id": vault_id,
                "title": os.path.basename(filepath)
            })

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"Failed to index uploaded file {file.filename}: {e}")

    if not sessions:
        raise HTTPException(status_code=500, detail="Failed to index any files.")

    return {
        "session_id": sessions[0]["session_id"],
        "vault_id": vault_id,
        "message": f"{len(sessions)} documents indexed successfully.",
        "sessions": sessions
    }


@app.post("/index-local", response_model=IndexResponse)
def index_local_file(
    pdf_path: str,
    model: str = "llama3.2",
    force_reindex: bool = False,
    api_key: str = Depends(get_api_key)
):
    """
    Index a local PDF file (already on disk).
    Useful for development/testing without uploading.
    """
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"File not found: {pdf_path}")

    session_id = str(uuid.uuid4())[:12]

    try:
        rag = PageIndexREMSE(model=model)
        rag.ingest(pdf_path, force_reindex=force_reindex)
        _sessions[session_id] = rag

        return IndexResponse(
            session_id=session_id,
            message=f"Document indexed successfully.",
            stats=rag.get_stats()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DiscoverRequest(BaseModel):
    topic: str

@app.post("/discover")
def discover_topic(req: DiscoverRequest, model: str = "llama3.2", api_key: str = Depends(get_api_key)):
    """Run Phase 1 (Navigator) and Phase 2 (RE-MSE Ingestion) on a topic."""
    from discovery.navigator import run_phase_1
    
    print(f"Starting Discovery for topic: {req.topic}")
    downloaded_pdfs = run_phase_1(req.topic)
    
    if not downloaded_pdfs:
        raise HTTPException(status_code=404, detail="No PDFs found for this topic.")
    
    # vault_id groups all sessions from this single discover batch
    vault_id = str(uuid.uuid4())[:12]
    _vault_papers[vault_id] = list(downloaded_pdfs)
        
    sessions = []
    for pdf_path in downloaded_pdfs:
        session_id = str(uuid.uuid4())[:12]
        try:
            rag = PageIndexREMSE(model=model)
            rag.ingest(pdf_path, force_reindex=False)
            _sessions[session_id] = rag
            _session_to_vault[session_id] = vault_id
            save_sessions()
            sessions.append({
                "session_id": session_id,
                "vault_id": vault_id,
                "title": os.path.basename(pdf_path)
            })
        except Exception as e:
            print(f"Failed to ingest {pdf_path}: {e}")
            
    if not sessions:
        raise HTTPException(status_code=500, detail="Failed to ingest any discovered PDFs.")
        
    return {"message": "Discovery and ingestion complete", "vault_id": vault_id, "sessions": sessions}


@app.post("/query", response_model=QueryResponse)
def query_document(req: QueryRequest, api_key: str = Depends(get_api_key)):
    """
    Ask a question about an indexed document.
    Requires a valid session_id from /upload or /index-local.
    """
    rag = _sessions.get(req.session_id)
    if not rag:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{req.session_id}' not found. Upload or index a document first."
        )

    try:
        result = rag.query(
            req.question,
            top_k_anchors=req.top_k_anchors,
            expansion_passes=req.expansion_passes,
            show_provenance=False  # Don't print to console for API
        )

        return QueryResponse(
            answer=result["answer"],
            sections_used=result["selected_sections"],
            atoms_used=result["atoms_used"],
            provenance=result["provenance"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tutor/chat", response_model=TutorResponse)
def tutor_chat(req: TutorRequest, api_key: str = Depends(get_api_key)):
    """
    Phase 3: Stateful AI Tutor Chat
    """
    rag = _sessions.get(req.session_id)
    if not rag:
        raise HTTPException(status_code=404, detail="Session not found.")
        
    try:
        # Find all sessions in the same vault to query across multiple papers
        vault_id = _session_to_vault.get(req.session_id)
        vault_rags = []
        if vault_id:
            for sid, v_id in _session_to_vault.items():
                if v_id == vault_id and sid in _sessions:
                    vault_rags.append(_sessions[sid])
        else:
            vault_rags.append(rag)

        vault_titles = []
        context_atoms = []
        for v_rag in vault_rags:
            title = os.path.basename(v_rag.pdf_path) if v_rag.pdf_path else v_rag.doc_id
            vault_titles.append(title)
            try:
                result = v_rag.query(req.message, show_provenance=False, save_result=False, generate_answer=False)
                context_atoms.extend(result.get("ordered_atoms", []))
            except Exception as e:
                print(f"Failed to query a vault paper ({v_rag.doc_id}): {e}")
        
        answer = tutor_engine.chat(
            session_id=req.session_id,
            message=req.message,
            context_atoms=context_atoms,
            student_profile=req.student_profile,
            history=req.history,
            vault_titles=vault_titles
        )
        
        card_data = tutor_engine.generate_note_card(answer, context_atoms)
        
        return TutorResponse(answer=answer, card_data=card_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/tutor/history/{session_id}")
def clear_tutor_history(session_id: str, api_key: str = Depends(get_api_key)):
    if session_id in tutor_engine.sessions:
        tutor_engine.sessions[session_id] = []
        return {"status": "cleared"}
    return {"status": "not_found"}


@app.post("/tutor/cards")
def save_card(req: NoteCardSaveRequest, api_key: str = Depends(get_api_key)):
    """Save a generated note card to the vault."""
    rag = _sessions.get(req.session_id)
    if not rag:
        raise HTTPException(status_code=404, detail="Session not found.")
        
    try:
        save_note_card(rag.doc_id, req.card_data)
        return {"status": "success", "message": "Card saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tutor/cards")
def get_cards(session_id: str, api_key: str = Depends(get_api_key)):
    """Retrieve all saved note cards for the vault."""
    rag = _sessions.get(session_id)
    if not rag:
        raise HTTPException(status_code=404, detail="Session not found.")
        
    cards = load_note_cards(rag.doc_id)
    return {"cards": cards}


@app.post("/intelligence/debates")
def get_debates(req: IntelligenceRequest, api_key: str = Depends(get_api_key)):
    """Phase 4: Detect cross-paper debates."""
    rag_instances = {}
    for sid in req.session_ids:
        vault_id = _session_to_vault.get(sid, sid)
        for v_sid, v_id in _session_to_vault.items():
            if v_id == vault_id and v_sid in _sessions:
                rag = _sessions[v_sid]
                title = os.path.basename(rag.pdf_path) if rag.pdf_path else v_sid
                rag_instances[title] = rag

    if len(rag_instances) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 active sessions for debates.")
        
    debates = intelligence_engine.detect_debates(rag_instances)
    return {"debates": debates}


@app.post("/intelligence/novelty")
def get_novelty(req: IntelligenceRequest, api_key: str = Depends(get_api_key)):
    """Phase 4: Run full novelty search pipeline."""
    rag_instances = {}
    for sid in req.session_ids:
        vault_id = _session_to_vault.get(sid, sid)
        for v_sid, v_id in _session_to_vault.items():
            if v_id == vault_id and v_sid in _sessions:
                rag = _sessions[v_sid]
                title = os.path.basename(rag.pdf_path) if rag.pdf_path else v_sid
                rag_instances[title] = rag
            
    if not rag_instances:
        raise HTTPException(status_code=400, detail="No valid sessions provided.")
        
    all_limitations = []
    for title, rag in rag_instances.items():
        lims = intelligence_engine.extract_limitations(title, rag)
        all_limitations.extend(lims)
        
    gaps = intelligence_engine.analyze_novelty_gaps(rag_instances, all_limitations)
    
    return {
        "limitations": all_limitations,
        "gaps": gaps,
        "feasible_hypotheses": gaps
    }


@app.post("/intelligence/themes")
def get_themes(req: IntelligenceRequest, api_key: str = Depends(get_api_key)):
    """Phase 4: Extract major research themes and clusters."""
    rag_instances = {}
    for sid in req.session_ids:
        vault_id = _session_to_vault.get(sid, sid)
        for v_sid, v_id in _session_to_vault.items():
            if v_id == vault_id and v_sid in _sessions:
                rag = _sessions[v_sid]
                title = os.path.basename(rag.pdf_path) if rag.pdf_path else v_sid
                rag_instances[title] = rag
            
    if not rag_instances:
        raise HTTPException(status_code=400, detail="No valid sessions provided.")
        
    themes = intelligence_engine.extract_themes(rag_instances)
    return {"themes": themes}


@app.post("/intelligence/narrow-topic")
def narrow_topic(req: NarrowTopicRequest, model: str = "llama3.2", api_key: str = Depends(get_api_key)):
    """Before discovery, optionally use the LLM to narrow down a broad topic."""
    from llm.ollama_client import ask_llm
    prompt = f"""You are an expert academic advisor. The user wants to research the broad topic: "{req.topic}".
Suggest 4-5 specific, modern, and highly researchable sub-topics within this field.
Return ONLY a valid JSON array of strings. Do not include markdown formatting or explanations.
Example: ["Subtopic 1", "Subtopic 2", "Subtopic 3", "Subtopic 4"]
"""
    try:
        response = ask_llm(prompt, model=model, expect_json=True)
        import json
        subtopics = json.loads(response)
        if not isinstance(subtopics, list):
            subtopics = [str(subtopics)]
        return {"subtopics": subtopics}
    except Exception as e:
        print(f"Error narrowing topic: {e}")
        # Graceful fallback: just return the original topic
        return {"subtopics": [req.topic]}


@app.post("/drafting/autocomplete")
def drafting_autocomplete(req: AutocompleteRequest, api_key: str = Depends(get_api_key)):
    """Phase 5: Vault-grounded autocomplete."""
    rag = _sessions.get(req.session_id)
    if not rag:
        raise HTTPException(status_code=404, detail="Session not found.")
        
    completion = verifier_engine.autocomplete(rag, req.text_context)
    return completion


@app.post("/drafting/audit")
def drafting_audit(req: AuditRequest, api_key: str = Depends(get_api_key)):
    """Phase 5: Hallucination audit per sentence — full verification retrieval pipeline."""
    rag = _sessions.get(req.session_id)
    if not rag:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Resolve vault_id: use explicit vault_id, fall back to session's vault, then session_id itself
    vault_id = req.vault_id or _session_to_vault.get(req.session_id, req.session_id)

    # Collect all sessions belonging to this vault for cross-paper paper-level filtering
    vault_sessions: Dict[str, Any] = {}
    for sid, vid in _session_to_vault.items():
        if vid == vault_id and sid in _sessions:
            vault_sessions[sid] = _sessions[sid]
    if not vault_sessions:
        vault_sessions[req.session_id] = rag   # single-session fallback

    audit_result = verifier_engine.audit_sentence(
        rag_instance=rag,
        sentence=req.sentence,
        all_sessions=vault_sessions,
    )
    return audit_result


@app.post("/drafting/export")
def drafting_export(req: ExportRequest, api_key: str = Depends(get_api_key)):
    """Phase 5: Export draft to DOCX."""
    output_dir = os.path.join(UPLOAD_DIR, "exports")
    try:
        filepath = exporter_engine.export(req.title, req.sections, output_dir, req.format)
        
        # Determine media type based on format
        media_type = 'application/octet-stream'
        if req.format.lower() == 'docx' or req.format.lower() == 'word':
            media_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        elif req.format.lower() == 'pdf':
            media_type = 'application/pdf'
        elif req.format.lower() == 'markdown' or req.format.lower() == 'md':
            media_type = 'text/markdown'
        elif req.format.lower() == 'latex' or req.format.lower() == 'zip':
            media_type = 'application/zip'
            
        return FileResponse(filepath, filename=os.path.basename(filepath), media_type=media_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")


@app.get("/sessions")
def list_sessions(api_key: str = Depends(get_api_key)):
    """List all active sessions."""
    sessions = []
    for sid, rag in _sessions.items():
        sessions.append({
            "session_id": sid,
            "vault_id": _session_to_vault.get(sid, sid),
            "pdf_path": rag.pdf_path,
            "stats": rag.get_stats()
        })
    return {"sessions": sessions}


@app.get("/vault/papers")
def list_vault_papers(session_id: str = None, vault_id: str = None, api_key: str = Depends(get_api_key)):
    """List papers for a specific session or vault. Pass session_id to filter by session."""
    
    # Resolve vault_id from session_id if provided
    resolved_vault_id = vault_id
    if session_id and not vault_id:
        resolved_vault_id = _session_to_vault.get(session_id)
    
    # If we have a vault_id, return only those tracked PDF files
    if resolved_vault_id and resolved_vault_id in _vault_papers:
        papers = []
        for fpath in _vault_papers[resolved_vault_id]:
            if not os.path.exists(fpath):
                continue
            fname = os.path.basename(fpath)
            size_kb = os.path.getsize(fpath) // 1024
            raw_title = os.path.splitext(fname)[0]
            # Remove UUID prefix
            if "_" in raw_title and len(raw_title.split("_")[0]) <= 13:
                raw_title = raw_title.split("_", 1)[-1]
            clean_title = raw_title.replace("_", " ").replace("-", " ")
            parts = clean_title.split()
            clean_title = " ".join(p for p in parts if not p.isdigit() or len(p) < 4)
            
            # Build URL — files may be in starter_vault subdir or root
            rel_path = os.path.relpath(fpath, UPLOAD_DIR).replace("\\", "/")
            papers.append({
                "filename": fname,
                "title": clean_title.strip(),
                "size_kb": size_kb,
                "url": f"http://localhost:8001/uploads/{rel_path}"
            })
        return {"papers": papers, "total": len(papers)}
        
    # If they specifically asked for a session/vault and we couldn't find it, return empty
    if session_id or vault_id:
        return {"papers": [], "total": 0}
    
    # Fallback: return ALL papers ONLY if no filter was supplied
    papers = []
    vault_dir = os.path.join(UPLOAD_DIR, "starter_vault")
    if os.path.exists(vault_dir):
        for fname in sorted(os.listdir(vault_dir)):
            if fname.lower().endswith(".pdf"):
                fpath = os.path.join(vault_dir, fname)
                size_kb = os.path.getsize(fpath) // 1024
                raw_title = os.path.splitext(fname)[0]
                clean_title = raw_title.replace("_", " ").replace("-", " ")
                parts = clean_title.split()
                clean_title = " ".join(p for p in parts if not p.isdigit() or len(p) < 4)
                papers.append({
                    "filename": fname,
                    "title": clean_title.strip(),
                    "size_kb": size_kb,
                    "url": f"http://localhost:8001/uploads/starter_vault/{fname}"
                })
    for fname in sorted(os.listdir(UPLOAD_DIR)):
        fpath = os.path.join(UPLOAD_DIR, fname)
        if fname.lower().endswith(".pdf") and os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) // 1024
            raw_title = os.path.splitext(fname)[0]
            if "_" in raw_title:
                raw_title = raw_title.split("_", 1)[-1]
            clean_title = raw_title.replace("_", " ").replace("-", " ")
            papers.append({
                "filename": fname,
                "title": clean_title.strip(),
                "size_kb": size_kb,
                "url": f"http://localhost:8001/uploads/{fname}"
            })
    return {"papers": papers, "total": len(papers)}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, api_key: str = Depends(get_api_key)):
    """Delete all sessions in the same vault to free memory and remove PDFs."""
    vault_id = _session_to_vault.get(session_id, session_id)
    
    # 1. Delete associated PDF files and their cached indices
    files_deleted = 0
    indices_deleted = 0
    if vault_id in _vault_papers:
        for fpath in _vault_papers[vault_id]:
            try:
                # Delete the cached index (tree, atoms, triples, query logs, note cards)
                doc_id = get_doc_id(fpath)
                if delete_index(doc_id):
                    indices_deleted += 1
            except Exception:
                pass  # file may not exist for doc_id computation
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
                    files_deleted += 1
            except Exception as e:
                print(f"Error deleting file {fpath}: {e}")
        del _vault_papers[vault_id]
        
    # 2. Find all sessions with this vault_id
    sessions_to_delete = []
    for sid, vid in _session_to_vault.items():
        if vid == vault_id:
            sessions_to_delete.append(sid)
            
    if session_id not in sessions_to_delete:
        sessions_to_delete.append(session_id)
        
    # 3. Delete from memory and mapping
    deleted_count = 0
    for sid in sessions_to_delete:
        if sid in _sessions:
            del _sessions[sid]
            deleted_count += 1
        if sid in _session_to_vault:
            del _session_to_vault[sid]
            
    if deleted_count > 0 or files_deleted > 0:
        save_sessions()
        return {
            "message": f"Vault '{vault_id}' deleted ({deleted_count} sessions, "
                       f"{files_deleted} files, {indices_deleted} cached indices removed)."
        }
        
    raise HTTPException(status_code=404, detail="Session/Vault not found.")


# ── Frontend Serving ─────────────────────────────────────────────────────────

UI_DIR = os.path.join(os.path.dirname(__file__), "ui")


@app.get("/")
def root():
    """Redirect to the chat UI."""
    return RedirectResponse(url="/ui")


@app.get("/ui")
def serve_ui():
    """Serve the frontend chat interface."""
    return FileResponse(os.path.join(UI_DIR, "index.html"))

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 Starting PageIndex-RE-MSE API Server...")
    print("🌐 Chat UI:  http://localhost:8001/ui")
    print("📄 API Docs: http://localhost:8001/docs")
    print("📄 Redoc:    http://localhost:8001/redoc\n")
    uvicorn.run(app, host="0.0.0.0", port=8001)
