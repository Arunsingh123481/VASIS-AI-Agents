# config.py
# Central configuration for local PageIndex-RE-MSE CRDB Pipeline
import os

# ─── MODEL ROUTING (OLLAMA LOCAL ONLY) ───────────────────
MODE = "privacy"  # Offline privacy mode

OLLAMA_URL = "http://127.0.0.1:11435"

# Local models downloaded by user
AGENT_MODEL = "qwen2.5-coder:3b"          # Qwen-Coder for agentic/routing/parsing work
REASONING_MODEL = "deepseek-llm:7b"  # DeepSeek for factual reasoning and validation

# Fallbacks (if models are missing)
DEFAULT_MODEL = "qwen2.5-coder:3b"

# API Settings
API_TIMEOUT = 120
API_RETRIES = 3
MAX_TOKENS = 1500
TEMPERATURE = 0.0

# ─── ATOM SETTINGS ────────────────────────────────────────
ATOM_TARGET_WORDS = 75
ATOM_MIN_WORDS    = 50
ATOM_MAX_WORDS    = 100

# ─── TREE SETTINGS ────────────────────────────────────────
TREE_SECTION_SIZES = {
    20:   3,
    60:   5,
    150:  7,
    9999: 8,
}

# ─── RETRIEVAL & ADAPTIVE EXPANSION ───────────────────────
TOP_K_ANCHORS        = 8
GAP_THRESHOLD        = 3
MIN_PAGE_CHARS       = 50
MAX_NARRATIVE_CHARS  = 8000

# Phase 12 Adaptive Stopping RE-MSE Settings
ADAPTIVE_MIN_RELEVANCE = 0.10
ADAPTIVE_MAX_RADIUS    = 16
ADAPTIVE_MIN_NEW_ATOMS = 1
ADAPTIVE_RADIUS_STEP   = 1

# ─── AGENT PARAMETERS ─────────────────────────────────────
MAX_REQUERY_ATTEMPTS   = 2
MAX_SUB_QUERIES        = 4
CONFIDENCE_THRESHOLD   = 0.6
CONSISTENCY_THRESHOLD  = 0.5
MIN_SYNTHESIS_CONFIDENCE = 0.5
MAX_SYNTHESIS_HOPS     = 3

# ─── AGENT 12: WEB SEARCH ─────────────────────────────────
SERPER_API_KEY         = os.environ.get("SERPER_API_KEY", "")
WEB_SEARCH_MAX_RESULTS = 10
WEB_SEARCH_TIMEOUT     = 15

# ─── AGENT 13: PAPER WRITER ───────────────────────────────
PAPER_DEFAULT_WORD_LIMIT = 4000
PAPER_DEFAULT_VENUE      = "IEEE"
PAPER_DEFAULT_TYPE       = "research_article"

# ─── AGENT 14: IMPLEMENTATION GUIDE ──────────────────────
DEFAULT_RESEARCHER_LEVEL = "masters"

# ─── EXPERIENCE & FEEDBACK ───────────────────────────────
SIMILARITY_THRESHOLD     = 0.6
MAX_EXPERIENCE_ENTRIES   = 1000

# ─── STORAGE PATHS ────────────────────────────────────────
INDEX_DIR             = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pageindex_cache")
AUDIT_LOG_PATH        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "crdb_audit.jsonl")
EXPERIENCE_STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "crdb_experience.jsonl")

# Ensure logs dir exists
os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
