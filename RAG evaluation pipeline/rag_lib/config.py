"""Central configuration for the customs-law RAG pipeline.

All paths are resolved relative to the OpenRouter/ project folder (this file lives
in OpenRouter/rag_lib/, so BASE is its parent-of-parent), which keeps every data
path — the vector DB, the corpus, the datasets — next to the two drivers
(build_vector_db.py, RAG.py) regardless of the launch directory.
"""
from pathlib import Path

# This file is OpenRouter/rag_lib/config.py -> BASE is the OpenRouter/ folder,
# which holds chroma_db/, JSON/, concept_qa_by_type/, and the two drivers.
BASE = Path(__file__).resolve().parent.parent

# The corpus and its simplified form both live in JSON/ (alongside the source
# corpus the ICL setting also reads).
JSON_DIR  = BASE / "JSON"

# --- Resume-bookmark folders (auto-created on a rebuild; see save_json) --------
STATE_DIR = BASE / "state"   # embed / chunk progress files
LOGS_DIR  = BASE / "logs"    # skip / fail records

# --- Inputs / outputs ---------------------------------------------------------
SOURCE_JSON      = JSON_DIR / "merged_ec_output_canon_subentities.json"
SIMPLIFIED_JSON  = JSON_DIR / "simplified_corpus.json"
SKIPPED_JSON     = LOGS_DIR / "skipped_long_items.json"   # items over the token limit (embed later)
FAILED_JSON      = LOGS_DIR / "failed_items.json"         # items that errored for other reasons
PROGRESS_FILE    = STATE_DIR / "embed_progress.json"      # already-embedded store_ids (resumability)

# --- Vector store -------------------------------------------------------------
CHROMA_DIR       = BASE / "chroma_db"
COLLECTION_NAME  = "customs_corpus"

# --- Gemini embedding ---------------------------------------------------------
# gemini-embedding-001 is the current best/GA Gemini text-embedding model.
EMBED_MODEL          = "gemini-embedding-001"
EMBED_DIM            = 3072                 # full quality (output already L2-normalized at 3072)
DOC_TASK_TYPE        = "RETRIEVAL_DOCUMENT" # for corpus documents
QUERY_TASK_TYPE      = "RETRIEVAL_QUERY"    # for search queries
MAX_INPUT_TOKENS     = 2048                 # hard model input limit
# Payloads this big are guaranteed over the token limit; skip without wasting an API call.
PRESKIP_CHARS        = 12_000

# --- Long-item chunking (embed_long_items.py) ---------------------------------
# The ~82 clauses over the token limit are split into normal-clause-sized, overlapping
# chunks so each chunk retrieves on par with an ordinary clause (not better, not worse).
CHUNK_TARGET_CHARS   = 1200        # ~205 tokens worst case; between corpus median (495) and mean (1551)
CHUNK_OVERLAP_CHARS  = 150         # ~12.5% overlap, carried word-aligned into the next chunk
CHUNK_MIN_TAIL_CHARS = 300         # merge a tiny final remainder into the previous chunk
CHUNK_PROGRESS_FILE  = STATE_DIR / "chunk_progress.json"  # embedded chunk ids (resumability)

# --- Key loading / rotation ---------------------------------------------------
QUOTA_COOLDOWN_SECONDS = 60        # how long to rest a key after a 429 / quota error
SERVER_MAX_RETRIES     = 4         # transient 5xx retries before rotating off a key
# API keys are plaintext (no encrypted vault). Two sections:
#   gemini     -> embedding + the LLM-judge council's Gemini seat
#   openrouter -> the models under test + the council's two OpenRouter seats
# Each section is loaded FILE-FIRST (the first existing, non-empty candidate file
# below; keys one-per-line or comma-separated, '#' comment lines skipped), then
# ENV as a fallback. A couple of alternate Gemini file names are also accepted so
# common key-file names work out of the box. All key files are .gitignored.
SECTION_GEMINI     = "gemini"
SECTION_OPENROUTER = "openrouter"
GEMINI_KEYS_FILE     = BASE / "gemini_api.txt"        # primary Gemini key file
OPENROUTER_KEYS_FILE = BASE / "openrouter_api.txt"
KEY_FILES = {
    SECTION_GEMINI:     [GEMINI_KEYS_FILE,
                         BASE / "gemini_paid_api.txt",  # alternate accepted name
                         BASE / "all_api.txt"],         # alternate accepted name
    SECTION_OPENROUTER: [OPENROUTER_KEYS_FILE],
}
# Env fallbacks per section: a comma/newline multi-key var, numbered vars
# (NAME_1, NAME_2, ...), then a single-key var.
KEY_ENV_MULTI  = {SECTION_GEMINI: "GEMINI_API_KEYS",  SECTION_OPENROUTER: "OPENROUTER_API_KEYS"}
KEY_ENV_SINGLE = {SECTION_GEMINI: "GEMINI_API_KEY",   SECTION_OPENROUTER: "OPENROUTER_API_KEY"}

# Back-compat aliases (older call sites used the VAULT_SECTION_* names).
VAULT_SECTION_GEMINI     = SECTION_GEMINI
VAULT_SECTION_OPENROUTER = SECTION_OPENROUTER

# --- RAG evaluation (RAG.py) --------------------------------------------------
# Generation runs on OpenRouter (OpenAI-compatible API); judging is the 3-model
# LLM council and embedding stays on Gemini, matching the PK/ICL methodology.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL         = "gemini-3.1-flash-lite"   # Gemini seat's model in the judge council
RAG_RESULTS_ROOT    = BASE / "rag_check_results"
