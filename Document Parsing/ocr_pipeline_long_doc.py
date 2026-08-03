#!/usr/bin/env python3
from __future__ import annotations

"""
Document Intelligence Pipeline — Gemini API Whole-PDF Architecture
===================================================================

A single-agent PDF processing pipeline for Bangladeshi Statutory Regulatory
Orders (SROs, 1994–2022) and Finance Acts that:
  1. Uploads the entire PDF to Gemini via the File API
  2. Sends a single extraction prompt for the whole document
  3. Post-processes the response into clean final JSON (Schema B)

Uses Google Gemini API with automatic key rotation for rate-limit handling.

Usage:
  python document_pipeline_gemini_extended_prompt.py report.pdf
  python document_pipeline_gemini_extended_prompt.py ./pdfs/ --output-dir ./results/
  python document_pipeline_gemini_extended_prompt.py report.pdf --model gemini-2.5-flash

Output Files:
  {pdf_name}_output.json — Full structured extraction (metadata + clauses + tables)

===========================================================================
"""

import argparse
import datetime
import json
import logging
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import NamedTuple, TypedDict

import fitz  # PyMuPDF — used only for page count
from tqdm import tqdm

from google import genai
from google.genai import types


# ════════════════════════════════════════════════════════════════════════════════
# Configuration Defaults
# ════════════════════════════════════════════════════════════════════════════════

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash" #"gemini-3.1-pro-preview"  
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 18  # seconds between API calls (conservative for batch)
CHUNK_SIZE = 50          # Max pages per Gemini API call
OVERLAP_PAGES = 10        # Context overlap between consecutive chunks

# ─── API Key File Path ─────────────────────────────────────────────────────────
API_KEYS_FILE_PATH =  "paid_api.txt"


# ════════════════════════════════════════════════════════════════════════════════
# Data Contracts
# ════════════════════════════════════════════════════════════════════════════════

class MetadataBlock(TypedDict, total=False):
    """A metadata entry in parsed_text — any text not belonging to a clause."""
    type: str              # "metadata"
    metadata_category: str # "document" | "page"
    text_id: str           # assigned in Phase 2
    page_start: int
    page_end: int
    metadata_title: str
    text: str
    table_ids: list[str]
    tables: list[dict]     # working storage, popped in Phase 2
    _item_order: int       # internal, stripped in output


class ClauseBlock(TypedDict, total=False):
    """A clause entry in parsed_text — a numbered legal provision."""
    type: str              # "clause"
    text_id: str           # assigned in Phase 2
    page_start: int
    page_end: int
    clause_number: str
    clause_title: str
    text: str
    subclause_hierarchy: list  # JSON list from LLM, bracket string in Phase 2
    table_ids: list[str]
    tables: list[dict]     # working storage, popped in Phase 2
    _item_order: int       # internal, stripped in output


def _to_bracket_string(lst) -> str:
    """Convert a nested list to bracket-string format for subclause_hierarchy.
    Example: ["৫", ["ক", "খ", ["i", "ii"]]] -> "[৫ [ক খ [i ii]]]"
    """
    if not isinstance(lst, list):
        return str(lst)
    return "[" + " ".join(_to_bracket_string(i) for i in lst) + "]"


# ── Clause Number Normalization ────────────────────────────────────────────────

_BANGLA_DIGITS = {'০':'0','১':'1','২':'2','৩':'3','৪':'4','৫':'5','৬':'6','৭':'7','৮':'8','৯':'9'}

def _normalize_clause_number(raw: str) -> str:
    """Canonicalize clause_number to 'clause_N' format.
    Handles: '৩', '৩।', 'clause_3', 'clause_৩', '3', '3.' etc.
    """
    s = raw.strip().rstrip('।.')
    if s.lower().startswith('clause_'):
        s = s[7:]
    s = ''.join(_BANGLA_DIGITS.get(c, c) for c in s)
    m = re.match(r'(\d+)', s)
    if m:
        return f"clause_{m.group(1)}"
    return f"clause_{s}" if s else raw


class ChunkInfo(NamedTuple):
    """Describes a single chunk of a PDF for chunked processing.

    Attributes:
        chunk_path:     Path to the temporary chunk PDF file.
        chunk_num:      1-indexed chunk number.
        real_page_start: 1-indexed first page in the chunk PDF (including overlap).
        real_page_end:   1-indexed last page in the chunk PDF.
        primary_start:   1-indexed first page this chunk is *responsible* for extracting.
        primary_end:     1-indexed last page this chunk is responsible for.
    """
    chunk_path: str
    chunk_num: int
    real_page_start: int
    real_page_end: int
    primary_start: int
    primary_end: int


# ════════════════════════════════════════════════════════════════════════════════
# Logging Setup
# ════════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("doc_pipeline")

# Track file handlers so we can detach them cleanly between PDFs
_active_file_handler = None


def setup_logging(verbose: bool = False, log_file_path: str | None = None):
    """
    Configure structured logging to stdout and optionally to a log file.

    Args:
        verbose:        If True, set console level to DEBUG; otherwise INFO.
        log_file_path:  If provided, attach a FileHandler that captures
                        ALL output at DEBUG level to this path.
    """
    global _active_file_handler

    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (add only once)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        console.setLevel(level)
        logger.addHandler(console)
    else:
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setLevel(level)

    # File handler — capture everything at DEBUG level
    if log_file_path:
        _detach_file_handler()
        os.makedirs(os.path.dirname(log_file_path) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        _active_file_handler = fh

    logger.setLevel(logging.DEBUG)


def _detach_file_handler():
    """Remove and close the active file handler, if any."""
    global _active_file_handler
    if _active_file_handler is not None:
        logger.removeHandler(_active_file_handler)
        _active_file_handler.close()
        _active_file_handler = None


# ════════════════════════════════════════════════════════════════════════════════
# Gemini API Key Manager  (unchanged from original)
# ════════════════════════════════════════════════════════════════════════════════

class GeminiApiKeyManager:
    """
    Manages multiple Gemini API keys with automatic rotation and error handling.

    Features:
      - Round-robin rotation through keys
      - Auto-detection of 429/quota/rate-limit errors → switch to next key
      - Per-key failure tracking with cooldown periods
      - Graceful fallback when all keys are exhausted
    """

    def __init__(self, api_keys: list[str], calls_per_day: int = 400,
                 rate_limit_delay: float = RATE_LIMIT_DELAY):
        if not api_keys:
            raise ValueError("api_keys must contain at least one key")
        self.api_keys = list(api_keys)
        self.calls_per_day = calls_per_day
        self.rate_limit_delay = rate_limit_delay
        self.usage_count = {key: 0 for key in api_keys}
        self.failure_count = {key: 0 for key in api_keys}
        self.cooldown_until = {key: 0.0 for key in api_keys}
        self.exhausted_keys = set()
        self._current_index = 0
        self._client = genai.Client(api_key=self.api_keys[0])
        logger.info(f"GeminiApiKeyManager initialized with {len(api_keys)} key(s), "
                     f"delay={rate_limit_delay}s")

    @property
    def current_key(self) -> str:
        return self.api_keys[self._current_index]

    @property
    def current_key_masked(self) -> str:
        key = self.current_key
        return key[:4] + "..." + key[-4:] if len(key) > 8 else "****"

    def get_client(self) -> genai.Client:
        return self._client

    def rotate_key(self) -> bool:
        attempts = 0
        while attempts < len(self.api_keys):
            self._current_index = (self._current_index + 1) % len(self.api_keys)
            attempts += 1
            key = self.api_keys[self._current_index]
            if key in self.exhausted_keys:
                continue
            if time.time() < self.cooldown_until.get(key, 0):
                continue
            if self.usage_count.get(key, 0) >= self.calls_per_day:
                self.exhausted_keys.add(key)
                continue
            self._client = genai.Client(api_key=key)
            logger.info(f"Rotated to key [{self._current_index}] "
                        f"(masked: {self.current_key_masked}, "
                        f"usage: {self.usage_count.get(key, 0)})")
            return True
        logger.error("All API keys are exhausted or in cooldown!")
        return False

    def report_success(self):
        key = self.current_key
        self.usage_count[key] = self.usage_count.get(key, 0) + 1
        self.failure_count[key] = 0

    def report_error(self, error: Exception, is_rate_limit: bool = False):
        key = self.current_key
        self.failure_count[key] = self.failure_count.get(key, 0) + 1
        if is_rate_limit:
            self.exhausted_keys.add(key)
            logger.warning(f"Key [{self._current_index}] rate-limited. Error: {error}")
        else:
            cooldown = min(60 * (2 ** (self.failure_count[key] - 1)), 600)
            self.cooldown_until[key] = time.time() + cooldown
            logger.warning(f"Key [{self._current_index}] error, cooldown {cooldown}s. Error: {error}")

    def classify_error(self, error: Exception) -> bool:
        msg = str(error).lower()
        return any(ind in msg for ind in [
            "429", "quota", "rate limit", "rate_limit",
            "resource_exhausted", "too many requests", "resourceexhausted",
        ])

    def wait_rate_limit(self):
        time.sleep(self.rate_limit_delay)

    def get_usage_stats(self) -> dict:
        total_used = sum(self.usage_count.values())
        total_available = len(self.api_keys) * self.calls_per_day
        return {
            "total_used": total_used,
            "total_available": total_available,
            "active_keys": len(self.api_keys) - len(self.exhausted_keys),
        }


def load_api_keys(file_path: str | None = None) -> list[str]:
    """Load Gemini API keys from file or environment variable."""
    keys = []
    paths_to_try = []
    if file_path:
        paths_to_try.append(file_path)
    paths_to_try.append(API_KEYS_FILE_PATH)

    for path in paths_to_try:
        if os.path.exists(path):
            try:
                with open(path, "r") as fh:
                    content = fh.read().strip()
                if content:
                    raw_keys = re.split(r"[,\n]+", content)
                    keys = [k.strip() for k in raw_keys if k.strip()]
                    if keys:
                        logger.info(f"Loaded {len(keys)} API key(s) from {path}")
                        return keys
            except Exception as e:
                logger.warning(f"Failed to read API keys from {path}: {e}")

    env_val = os.environ.get("GEMINI_API_KEYS", "").strip()
    if env_val:
        keys = [k.strip() for k in env_val.split(",") if k.strip()]
        if keys:
            logger.info(f"Loaded {len(keys)} API key(s) from GEMINI_API_KEYS env var")
            return keys

    raise ValueError(
        "No API keys found. Provide keys via:\n"
        f"  1. File: {API_KEYS_FILE_PATH}\n"
        "  2. CLI: --api-keys-file <path>\n"
        "  3. Env: GEMINI_API_KEYS=key1,key2,..."
    )


# ════════════════════════════════════════════════════════════════════════════════
# Gemini Client Wrapper — File API for whole-PDF upload
# ════════════════════════════════════════════════════════════════════════════════

class GeminiClient:
    """High-level wrapper around Gemini API with PDF upload and key rotation."""

    def __init__(self, key_manager: GeminiApiKeyManager, model_name: str = DEFAULT_GEMINI_MODEL):
        self.key_manager = key_manager
        self.model_name = model_name

    def query_with_pdf(self, pdf_path: str, prompt: str,
                       max_retries: int = MAX_RETRIES,
                       _recitation_retry: bool = False) -> str:
        """Upload a PDF via the File API and query the model with it."""
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                client = self.key_manager.get_client()
                # Upload PDF using File API
                uploaded_file = client.files.upload(file=Path(pdf_path))
                logger.info(f"PDF uploaded successfully via File API")

                # On recitation retry, prepend an anti-recitation frame
                effective_prompt = prompt
                if _recitation_retry:
                    effective_prompt = (
                        "IMPORTANT: Output ONLY structured JSON field values. "
                        "Do NOT reproduce the document verbatim — extract and "
                        "restructure each field independently.\n\n"
                    ) + prompt
                    logger.info("Using anti-recitation prompt modifier")

                # Generate content with uploaded file + prompt
                # response_mime_type="application/json" tells Gemini this is
                # structured data generation, reducing RECITATION triggers.
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=[uploaded_file, effective_prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )

                raw_text = ""
                # Try to extract text, with diagnostics for empty/blocked responses
                try:
                    if hasattr(response, "text") and response.text:
                        raw_text = response.text
                    elif isinstance(response, dict) and "text" in response:
                        raw_text = response["text"]
                except Exception as text_err:
                    logger.warning(f"Could not access response.text: {text_err}")

                # Diagnostic: inspect WHY response might be empty
                if not raw_text:
                    logger.warning("Empty response from Gemini. Diagnosing...")
                    is_recitation = False
                    # Check candidates
                    if hasattr(response, "candidates") and response.candidates:
                        for ci, cand in enumerate(response.candidates):
                            finish = getattr(cand, "finish_reason", "UNKNOWN")
                            logger.warning(f"  Candidate {ci}: finish_reason={finish}")
                            if "RECITATION" in str(finish):
                                is_recitation = True
                            # Check safety ratings (may be None)
                            ratings = getattr(cand, "safety_ratings", None)
                            if ratings:
                                for sr in ratings:
                                    cat = getattr(sr, "category", "?")
                                    prob = getattr(sr, "probability", "?")
                                    logger.warning(f"    Safety: {cat} = {prob}")
                            # Try to get partial text from parts
                            content = getattr(cand, "content", None)
                            if content and hasattr(content, "parts") and content.parts:
                                for pi, part in enumerate(content.parts):
                                    part_text = getattr(part, "text", "")
                                    if part_text:
                                        logger.info(f"  Found text in candidate {ci} part {pi} "
                                                    f"({len(part_text)} chars)")
                                        raw_text = part_text
                    else:
                        logger.warning("  No candidates in response.")

                    # Check prompt feedback (blocking)
                    pfb = getattr(response, "prompt_feedback", None)
                    if pfb:
                        block_reason = getattr(pfb, "block_reason", None)
                        if block_reason:
                            logger.error(f"  PROMPT BLOCKED: {block_reason}")
                        pfb_ratings = getattr(pfb, "safety_ratings", None)
                        if pfb_ratings:
                            for sr in pfb_ratings:
                                cat = getattr(sr, "category", "?")
                                prob = getattr(sr, "probability", "?")
                                logger.warning(f"  Prompt safety: {cat} = {prob}")

                    # RECITATION — retry with anti-recitation framing
                    if is_recitation:
                        if not _recitation_retry:
                            logger.warning(
                                "  RECITATION detected. Retrying with "
                                "anti-recitation prompt modifier..."
                            )
                            return self.query_with_pdf(
                                pdf_path, prompt,
                                max_retries=max_retries,
                                _recitation_retry=True,
                            )
                        else:
                            logger.error(
                                "  RECITATION persists even with modified prompt. "
                                "This document may require page-range chunking."
                            )
                            raise RuntimeError(
                                "Document blocked by Gemini (RECITATION) after retry. "
                                "Try processing fewer pages at a time."
                            )

                    # If still empty, log the raw response repr
                    if not raw_text:
                        resp_str = str(response)[:1000]
                        logger.debug(f"  Full response repr: {resp_str}")

                if isinstance(raw_text, bytes):
                    raw_text = raw_text.decode("utf-8", errors="ignore")
                raw_text = (raw_text or "").strip()
                self.key_manager.report_success()
                self.key_manager.wait_rate_limit()
                return raw_text
            except Exception as e:
                last_error = e
                is_rate_limit = self.key_manager.classify_error(e)
                self.key_manager.report_error(e, is_rate_limit=is_rate_limit)
                if is_rate_limit:
                    if self.key_manager.rotate_key():
                        logger.info(f"Retrying with new key (attempt {attempt}/{max_retries})")
                        continue
                    else:
                        raise RuntimeError("All API keys exhausted.") from e
                else:
                    logger.warning(f"API call attempt {attempt}/{max_retries} failed: {e}")
                    if attempt < max_retries:
                        backoff = min(2 ** (attempt - 1) * 5, 60)
                        time.sleep(backoff)
        raise RuntimeError(f"All {max_retries} attempts failed. Last error: {last_error}")





# ════════════════════════════════════════════════════════════════════════════════
# Robust JSON Parser  (unchanged from original)
# ════════════════════════════════════════════════════════════════════════════════

def _find_matching_brace(text: str, open_char: str, close_char: str) -> str | None:
    start = text.find(open_char)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            if in_string:
                escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _sanitize_for_json(text: str) -> str:
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    out = []
    in_str = False
    esc = False
    for ch in text:
        if esc:
            out.append(ch); esc = False; continue
        if ch == '\\' and in_str:
            out.append(ch); esc = True; continue
        if ch == '"':
            in_str = not in_str; out.append(ch); continue
        if in_str and ch == '\n':
            out.append('\\n'); continue
        if in_str and ch == '\r':
            out.append('\\r'); continue
        out.append(ch)
    return ''.join(out)


def _try_parse(text: str) -> dict | None:
    try:
        result = json.loads(text)
        if isinstance(result, (dict, list)):
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _ensure_dict(result) -> dict:
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        return {"items": result}
    raise TypeError(f"Expected dict or list, got {type(result).__name__}")


def _try_recover_truncated_json(text: str) -> dict | None:
    for suffix in ['"}]}', '"}]', '"}', '"]}'  , '"]}', ']}', '}]', '}']:
        candidate = text.rstrip() + suffix
        r = _try_parse(candidate)
        if r is not None:
            return r
        r = _try_parse(_sanitize_for_json(candidate))
        if r is not None:
            return r
    return None


def parse_json_from_response(text: str) -> dict:
    """Robustly extract and parse a JSON object from model response text."""
    text = text.strip()
    if not text:
        raise ValueError("Empty response from model -- cannot parse JSON.")
    r = _try_parse(text)
    if r is not None:
        return _ensure_dict(r)
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```", text)
    if fence_match:
        block = fence_match.group(1).strip()
        r = _try_parse(block)
        if r is not None:
            return _ensure_dict(r)
        r = _try_parse(_sanitize_for_json(block))
        if r is not None:
            return _ensure_dict(r)
    json_str = _find_matching_brace(text, "{", "}")
    if json_str:
        r = _try_parse(json_str)
        if r is not None:
            return _ensure_dict(r)
        r = _try_parse(_sanitize_for_json(json_str))
        if r is not None:
            return _ensure_dict(r)
    json_str = _find_matching_brace(text, "[", "]")
    if json_str:
        r = _try_parse(json_str)
        if r is not None:
            return _ensure_dict(r)
        r = _try_parse(_sanitize_for_json(json_str))
        if r is not None:
            return _ensure_dict(r)
    json_start = text.find("{")
    if json_start != -1:
        partial = _sanitize_for_json(text[json_start:])
        r = _try_recover_truncated_json(partial)
        if r is not None:
            logger.warning("Recovered JSON from truncated response (data may be partial).")
            return _ensure_dict(r)
    raise ValueError(
        f"Could not parse JSON from model response.\n"
        f"Response length: {len(text)} chars.\nFirst 500 chars:\n{text[:500]}"
    )



def build_whole_document_prompt(file_name: str, total_pages: int) -> str:
    """Build the extraction prompt for whole-PDF processing via Gemini File API."""
    return (
        # ── Task framing ──
        "<task>\n"
        "You are a legal-document transcription and structured-extraction agent for "
        "Bangladeshi government PDFs (SROs 1994–2022, Finance Acts, "
        "customs/tax notifications, gazette notifications, and "
        "international treaties or bilateral agreements between the "
        "Government of Bangladesh and foreign governments or entities). "
        "Documents are in Bangla, English, or mixed Bangla-English.\n"
        "\n"
        f"Document: {file_name} ({total_pages} pages).\n"
        "\n"
        "This is a government gazette document for structured indexing. "
        "Output is a field-level transcription for research, not republication.\n"
        "\n"
        "Extract all visible text from the attached PDF in reading order. "
        "For scanned or image-based pages, perform OCR. "
        "Preserve Bangla and English scripts exactly as printed.\n"
        "\n"
        "Use visual structure (headings, numbering, tables) to decide "
        "metadata vs clause boundaries.\n"
        "\n"
        # ── Steps ──
        "Follow these steps in order:\n"
        "  1. EXTRACT every page — capture ALL visible text in reading order.\n"
        "  2. CLASSIFY each text block as METADATA or CLAUSE.\n"
        "  3. MERGE clause text that continues across page boundaries "
        "into a single clause item.\n"
        "  4. MERGE table rows that span multiple pages into a single "
        "table entry.\n"
        "  5. BUILD the JSON output with all items in document order.\n"
        "</task>\n"
        "\n"
        # ── Rules ──
        "<rules>\n"
        "\n"
        # ── R1: OCR fidelity ──
        "R1  OCR FIDELITY\n"
        "    • Extract ALL text exactly as printed — every header, footer, "
        "page number, footnote, section number, and body text.\n"
        "    • Bangla → Bengali Unicode. English → Latin script. "
        "Mixed → keep both scripts exactly as they appear.\n"
        "    • Do NOT translate, transliterate, paraphrase, or invent text.\n"
        "    • ZERO TEXT LOSS: every visible character MUST appear "
        "in your JSON.\n"
        "\n"
        # ── R2: Item types ──
        'R2  ITEM TYPES — exactly two: "metadata" and "clause".\n'
        '    • type field is the string "metadata" or "clause".\n'
        '    • Never use "clauses" (plural) or any other value.\n'
        "\n"
        # ── R3: Metadata ──
        'R3  METADATA (type="metadata")\n'
        "    • DEFINITION: Text whose function is to IDENTIFY, LABEL, or\n"
        "      NAVIGATE the document — it describes or organises the document\n"
        "      but carries no legal obligation, right, or definition in itself.\n"
        "      A metadata block could be removed without changing any legal rule.\n"
        "    • Includes: document titles, SRO/act numbers, ministry names,\n"
        "      chapter headings, schedule titles, preamble recitals, amendment\n"
        "      titles, page headers/footers, page numbers, footnotes, running\n"
        "      headers, and attestation/signature blocks.\n"
        "    • metadata_category — exactly one of:\n"
        '      – "document": anything that identifies or structures the document\n'
        "        at a content level (titles, chapter headings (পঞ্চম অধ্যায়),\n"
        "        schedule headings (তৃতীয় তফসিল), preambles).\n"
        '      – "page": anything that identifies or structures a physical page\n'
        "        (headers, footers, page numbers, footnotes).\n"
        "    • metadata_title: the heading/title text if this block IS a "
        'heading; otherwise "".\n'
        "    • Metadata items NEVER have tables. tables field = [].\n"
        "\n"
        # ── R4: Clause identification ──
        'R4  CLAUSES (type="clause")\n'
        "    • DEFINITION: A numbered unit of legal text whose function is to\n"
        "      CREATE, MODIFY, or REMOVE a legal rule — an obligation, right,\n"
        "      prohibition, penalty, definition, or procedural requirement.\n"
        "      A clause can be independently cited and has normative force on\n"
        "      its own. It would remain meaningful if extracted from the document.\n"
        "    • Primary markers (use to locate clause boundaries):\n"
        "      Bengali numeral + daari: ৪।, ৫।, ১০।\n"
        "      Arabic numeral + period: 4., 5., 10.\n"
        "    • When a block carries legal force AND bears a clause numeral, it\n"
        "      is always a clause — regardless of its visual formatting or\n"
        "      position on the page.\n"
        "    • Sub-clauses (ক)(খ)(গ)(a)(b)(c) and sub-sub-clauses\n"
        "      (i)(ii)(iii)(১)(২)(৩) qualify or elaborate the parent clause's\n"
        "      legal rule. They are ALWAYS part of the parent clause's text.\n"
        "      They are NEVER separate items in the output.\n"
        "\n"
        # ── R5: clause_number format — GLOBAL SEQUENTIAL ──
        "R5  CLAUSE_NUMBER — GLOBAL SEQUENTIAL NUMBERING\n"
        '    • Format: always "clause_N" where N is an Arabic numeral.\n'
        "    • N must be the GLOBAL sequential position of the clause in "
        "the ENTIRE document, NOT the chapter-local number.\n"
        "    • Documents with multiple chapters/parts often restart clause "
        "numbering (e.g. Chapter 6 starts with ১।). You MUST continue "
        "the global count from where the previous chapter ended.\n"
        "    • Example: if Chapters 1-5 had clauses 1-43, then Chapter 6's "
        "first clause (৩।) becomes clause_44, its second (৪।) becomes "
        "clause_45, and so on.\n"
        "    • The clause's original printed numeral (৩।) still appears "
        "in the text field verbatim — only clause_number uses the global N.\n"
        '    • Never raw Bangla ("৩"), bare numbers ("3"), or without prefix.\n'
        "\n"
        # ── R6: clause_title extraction ──
        "R6  CLAUSE_TITLE\n"
        "    • The clause_title is the heading phrase between the "
        "clause numeral and the em-dash (—).\n"
        "    • Example: \"৪। section 25B এর প্রতিস্থাপন।—\" "
        "→ clause_title = \"section 25B এর প্রতিস্থাপন\"\n"
        "    • If no heading phrase exists before the body text, "
        'clause_title = "".\n'
        "\n"
        # ── R7: Cross-page merging ──
        "R7  CROSS-PAGE MERGING (CRITICAL)\n"
        "    • You see the ENTIRE document. If a clause continues across "
        "pages, merge ALL its text into ONE clause item.\n"
        "    • page_start = first page where the clause appears. "
        "page_end = last page.\n"
        "    • Literal strings spanning pages: all text stays in the "
        "parent clause's text field.\n"
        "\n"
        # ── R8: Subclause hierarchy ──
        "R8  SUBCLAUSE HIERARCHY\n"
        "    • subclause_hierarchy: a JSON array capturing the FULL nested "
        "structure of all sub-clauses for the entire clause.\n"
        "    • Flat items = top-level sub-clauses. A nested sub-array "
        "IMMEDIATELY after a parent = that parent's children.\n"
        "    • Examples:\n"
        '      ক has children অ, আ; then খ:\n'
        '        → ["ক", ["অ", "আ"], "খ"]\n'
        '      ক has child অ, which has children i, ii; then আ; then খ:\n'
        '        → ["ক", ["অ", ["i", "ii"], "আ"], "খ"]\n'
        "    • Use ORIGINAL numbering signs from the PDF.\n"
        "    • EXCLUDE sub-clause markers inside quoted literal strings "
        '("..."). Those are part of quoted law text, not structural.\n'
        "    • No sub-clauses → empty array [].\n"
        "\n"
        # ── R9: Tables ──
        "R9  TABLE EXTRACTION\n"
        "    • Tables ALWAYS belong to a CLAUSE — embed in the clause's "
        "tables array.\n"
        "    • Detect ALL table types: bordered, partially bordered, "
        "and borderless (aligned columns of data).\n"
        "    • For each table:\n"
        "      – headers: array of column header strings as printed.\n"
        "      – rows: array of arrays [[cell, cell], ...]. "
        "NO dict rows. Empty cells → \"\".\n"
        "      – page_start / page_end: pages the table spans.\n"
        "    • Extract ALL rows — never truncate, even for 100+ rows.\n"
        "    • MULTI-PAGE TABLES: merge all rows into ONE table entry.\n"
        "    • Schedule/তফসিল tables: attach to the clause that "
        "introduces the schedule.\n"
        "    • INLINE TABLE MARKERS (CRITICAL): In the clause's text "
        "field, insert the marker [TABLE_K] at the EXACT position "
        "where the K-th table appears in the document flow. "
        "K = 1, 2, 3... per clause, matching the order of objects "
        "in the tables array. Example: if a clause has text before "
        "a table, then more text, then a second table:\n"
        '      text: "... preceding text [TABLE_1] middle text '
        '[TABLE_2] trailing text"\n'
        "    • Place [TABLE_K] at the exact spot in the text where "
        "the table visually appears on the page. If the table appears "
        "at the end of the clause text, place the marker at the end.\n"
        "\n"
        # ── R10: Literal strings ──
        "R10 LITERAL STRINGS\n"
        '    • Verbatim legislative text between outermost "..."।\n'
        "    • Inside a literal string: (ক)(খ)(a)(b)(i)(ii) are sub-sections "
        "of QUOTED LAW. Do NOT add them to subclause_hierarchy. "
        "Do NOT create new clause items from them.\n"
        "\n"
        # ── R11: Dates ──
        "R11 DATES\n"
        "    • Extract the document date from page 1 header/metadata. "
        "Report as document_date.\n"
        "\n"
        # ── R12: Ordering ──
        "R12 ORDERING\n"
        "    • Items in the output MUST follow document order: "
        "top-to-bottom, page-by-page.\n"
        "\n"
        # ── Output format ──
        "</rules>\n"
        "\n"
        "<output_format>\n"
        "Return ONLY this JSON. No markdown fences. No explanation.\n"
        "\n"
        "{\n"
        '  "document_date": "",\n'
        '  "items": [\n'
        "    {\n"
        '      "type": "metadata",\n'
        '      "metadata_category": "document",\n'
        '      "metadata_title": "",\n'
        '      "text": "full text",\n'
        '      "page_start": 1,\n'
        '      "page_end": 1,\n'
        '      "tables": []\n'
        "    },\n"
        "    {\n"
        '      "type": "clause",\n'
        '      "clause_number": "clause_N",\n'
        '      "clause_title": "heading before em-dash",\n'
        '      "text": "full clause text with all sub-clauses merged across pages, [TABLE_K] markers where tables appear",\n'
        '      "subclause_hierarchy": [],\n'
        '      "page_start": 1,\n'
        '      "page_end": 3,\n'
        '      "tables": [\n'
        "        {\n"
        '          "headers": ["col1", "col2"],\n'
        '          "rows": [["cell1", "cell2"]],\n'
        '          "page_start": 2,\n'
        '          "page_end": 3\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        # ── Example ──
        "</output_format>\n"
        "\n"
        "<example>\n"
        "Chapters 1-5 contained 43 clauses. Chapter 6 begins on page 6 with: "
        "chapter heading → local clause ৪। spans pages 6-7 → "
        "local clause ৫। on page 7 with sub-clauses → "
        "local clause ৮। has a 2-column tax table.\n"
        "\n"
        "{\n"
        '  "document_date": "২০১৩",\n'
        '  "items": [\n'
        "    {\n"
        '      "type": "metadata",\n'
        '      "metadata_category": "document",\n'
        '      "metadata_title": "Customs Act, 1969 এর সংশোধন",\n'
        '      "text": "Customs Act, 1969 এর সংশোধন",\n'
        '      "page_start": 6, "page_end": 6, "tables": []\n'
        "    },\n"
        "    {\n"
        '      "type": "clause",\n'
        '      "clause_number": "clause_44",\n'
        '      "clause_title": "section 25B এর প্রতিস্থাপন",\n'
        '      "text": "৪। section 25B এর প্রতিস্থাপন।— উক্ত আইনের section 25B এর পরিবর্তে নিম্নরূপ section 25B প্রতিস্থাপিত হইবে।",\n'
        '      "subclause_hierarchy": [],\n'
        '      "page_start": 6, "page_end": 7, "tables": []\n'
        "    },\n"
        "    {\n"
        '      "type": "clause",\n'
        '      "clause_number": "clause_45",\n'
        '      "clause_title": "section 91 এর সংশোধন",\n'
        '      "text": "৫। section 91 এর সংশোধন।— উক্ত আইনের section 91 এর—\\n(ক) sub-section (1) এ—\\n(অ) প্রথম পরিবর্তন...\\n(আ) দ্বিতীয় পরিবর্তন...\\n(খ) sub-section (3) তে...",\n'
        '      "subclause_hierarchy": ["ক", ["অ", "আ"], "খ"],\n'
        '      "page_start": 7, "page_end": 7, "tables": []\n'
        "    },\n"
        "    {\n"
        '      "type": "clause",\n'
        '      "clause_number": "clause_48",\n'
        '      "clause_title": "তফসিলের সংশোধন",\n'
        '      "text": "৮। তফসিলের সংশোধন।— উক্ত আইনের তফসিলে নিম্নরূপ সংশোধন করা হইবে। [TABLE_1]",\n'
        '      "subclause_hierarchy": [],\n'
        '      "page_start": 9, "page_end": 10,\n'
        '      "tables": [{\n'
        '        "headers": ["শিরোনাম সংখ্যা", "শুল্কের হার"],\n'
        '        "rows": [["৯০.১০", "১৫%"], ["৯০.১১", "৫%"]],\n'
        '        "page_start": 9, "page_end": 10\n'
        "      }]\n"
        "    },\n"
        "    {\n"
        '      "type": "metadata",\n'
        '      "metadata_category": "page",\n'
        '      "metadata_title": "",\n'
        '      "text": "বাংলাদেশ গেজেট, ২০১৩ সন — ৬",\n'
        '      "page_start": 6, "page_end": 6, "tables": []\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        # ── Checklist ──
        "</example>\n"
        "\n"
        "<checklist>\n"
        "Verify before returning:\n"
        "- Every visible text character appears in the JSON\n"
        '- type is "metadata" or "clause" (never plural)\n'
        '- clause_number = "clause_N" — N is GLOBAL sequential, NOT chapter-local\n'
        "- No two clause items share the same clause_number\n"
        "- clause_title extracted from text before em-dash (—)\n"
        "- metadata items have NO clause_number field\n"
        "- subclause_hierarchy is a nested JSON array, not a string\n"
        "- Table rows are [[...], ...] arrays, never dicts\n"
        "- [TABLE_K] marker in text for every table in the clause\n"
        "- Cross-page clauses merged into single items\n"
        "- Cross-page tables merged into single table entries\n"
        "- Items ordered by document appearance\n"
        "</checklist>\n"
        "\n"
        "Return ONLY the raw JSON object.\n"
    )


# ════════════════════════════════════════════════════════════════════════════════
# Chunked PDF Processing — Split, Extract, Merge
# ════════════════════════════════════════════════════════════════════════════════

def build_chunk_prompt(file_name: str, total_pages: int,
                       chunk_info: ChunkInfo,
                       prev_clause_count: int) -> str:
    """Build the extraction prompt for a single chunk of a larger PDF.

    Wraps the base whole-document prompt with chunk-specific context:
      - Page-number mapping (chunk-local → original document pages)
      - Overlap handling: which pages are context-only vs. primary extraction
      - Clause numbering continuation from previous chunks

    Args:
        file_name:        Original PDF filename (for the base prompt).
        total_pages:      Total pages in the full document.
        chunk_info:       ChunkInfo describing this chunk's page ranges.
        prev_clause_count: Number of clauses extracted by all previous chunks.
    """
    base_prompt = build_whole_document_prompt(file_name, total_pages)

    has_overlap = (chunk_info.primary_start != chunk_info.real_page_start)
    page_offset = chunk_info.real_page_start - 1  # Add to chunk-local page nums

    # ── Chunk context preamble ──────────────────────────────────────────────
    parts = [
        "<chunk_context>",
        f"CRITICAL CONTEXT: You are processing a SEGMENT of a {total_pages}-page document.",
        f"This PDF segment contains pages {chunk_info.real_page_start}"
        f"–{chunk_info.real_page_end} of the original document.",
        "",
        "PAGE NUMBER MAPPING:",
        f"  Page 1 of this PDF = Page {chunk_info.real_page_start} of the original document.",
        f"  Page 2 of this PDF = Page {chunk_info.real_page_start + 1} of the original document.",
        f"  ... (add {page_offset} to each chunk-local page number)",
        f"  ALWAYS report page_start and page_end using ORIGINAL document page numbers.",
        "",
    ]

    if has_overlap:
        overlap_count = chunk_info.primary_start - chunk_info.real_page_start
        parts.extend([
            "OVERLAP HANDLING:",
            f"  The first {overlap_count} pages of this PDF "
            f"(original pages {chunk_info.real_page_start}"
            f"–{chunk_info.primary_start - 1}) are provided as "
            f"CONTEXT from the previous segment.",
            f"  Your PRIMARY extraction range is pages "
            f"{chunk_info.primary_start}–{chunk_info.primary_end}.",
            "",
            f"  • Items FULLY within pages {chunk_info.real_page_start}"
            f"–{chunk_info.primary_start - 1}: "
            f"DO NOT extract (already processed by the previous segment).",
            f"  • Items that START before page {chunk_info.primary_start} "
            f"but CONTINUE past page {chunk_info.primary_start - 1}: "
            f"Extract the COMPLETE item with its true page_start and page_end.",
            f"  • Items starting on page {chunk_info.primary_start} or later: "
            f"Extract normally.",
            "",
        ])

    parts.extend([
        "CLAUSE NUMBERING CONTINUATION:",
        f"  Previous segments extracted {prev_clause_count} clauses.",
        f"  Start numbering from clause_{prev_clause_count + 1}.",
        f"  The first NEW clause you find should be clause_{prev_clause_count + 1}.",
        "</chunk_context>",
        "",
    ])

    return "\n".join(parts) + "\n" + base_prompt


def split_pdf_into_chunks(pdf_path: str,
                          chunk_size: int = CHUNK_SIZE,
                          overlap: int = OVERLAP_PAGES,
                          temp_dir: str | None = None) -> list[ChunkInfo]:
    """Split a PDF into overlapping chunks for sequential Gemini processing.

    Each chunk is a standalone PDF file containing at most ``chunk_size`` primary
    pages plus up to ``overlap`` context pages from the end of the previous chunk.

    Args:
        pdf_path:   Path to the original PDF.
        chunk_size: Maximum number of *primary* pages per chunk.
        overlap:    Number of context pages prepended from the previous chunk.
        temp_dir:   Directory to write temporary chunk PDFs into.

    Returns:
        Ordered list of ChunkInfo describing each chunk.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    if temp_dir is None:
        stem = Path(pdf_path).stem
        temp_dir = os.path.join(os.path.dirname(pdf_path),
                                f".{stem}_chunks_tmp")
    os.makedirs(temp_dir, exist_ok=True)

    chunks: list[ChunkInfo] = []
    primary_start_0 = 0  # 0-indexed start of the primary range
    chunk_num = 0

    while primary_start_0 < total_pages:
        chunk_num += 1

        # Overlap: include pages from the previous chunk for boundary context
        overlap_start_0 = max(0, primary_start_0 - overlap)
        # Primary end: at most chunk_size pages from primary_start
        primary_end_0 = min(primary_start_0 + chunk_size, total_pages)

        # Create the chunk PDF (overlap pages + primary pages)
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc,
                             from_page=overlap_start_0,
                             to_page=primary_end_0 - 1)

        chunk_path = os.path.join(temp_dir, f"chunk_{chunk_num}.pdf")
        chunk_doc.save(chunk_path)
        chunk_doc.close()

        info = ChunkInfo(
            chunk_path=chunk_path,
            chunk_num=chunk_num,
            real_page_start=overlap_start_0 + 1,   # 1-indexed
            real_page_end=primary_end_0,            # 1-indexed
            primary_start=primary_start_0 + 1,      # 1-indexed
            primary_end=primary_end_0,              # 1-indexed
        )
        chunks.append(info)

        logger.info(
            f"  Chunk {chunk_num}: PDF pages {info.real_page_start}"
            f"–{info.real_page_end}  "
            f"(primary: {info.primary_start}–{info.primary_end}, "
            f"overlap: {info.real_page_start}–{info.primary_start - 1 if info.primary_start > info.real_page_start else 'none'})"
        )

        # Advance to the next primary range
        primary_start_0 = primary_end_0

    doc.close()
    logger.info(
        f"Split PDF into {len(chunks)} chunk(s) "
        f"(chunk_size={chunk_size}, overlap={overlap}, "
        f"total_pages={total_pages})"
    )
    return chunks


def _validate_extraction_result(result: dict) -> dict:
    """Validate and normalize a raw Gemini extraction result.

    Handles common LLM output quirks: missing fields, wrong type names,
    dict-format table rows, string-encoded subclause hierarchies, etc.

    This is used by both single-call and chunked extraction paths.
    """
    result.setdefault("document_date", "")

    # ── Normalize items list ────────────────────────────────────────────────
    if not isinstance(result.get("items"), list):
        items = []
        for mb in result.get("metadata_blocks", []):
            mb["type"] = "metadata"
            items.append(mb)
        for cb in result.get("clause_blocks", []):
            cb["type"] = "clause"
            items.append(cb)
        result["items"] = items

    # ── Per-item validation ─────────────────────────────────────────────────
    for item in result["items"]:
        item_type = item.get("type", "")
        if item_type == "clauses":
            item["type"] = "clause"
        item.setdefault("text", "")
        item.setdefault("tables", [])
        item.setdefault("page_start", 1)
        item.setdefault("page_end", item.get("page_start", 1))
        # Drop transient LLM fields
        item.pop("provisional_id", None)
        item.pop("notes", None)
        item.pop("has_table", None)
        item.pop("is_complete", None)
        item.pop("has_subclauses", None)

        if item.get("type") == "metadata":
            item.pop("clause_number", None)
            item.setdefault("metadata_title", "")
            item.setdefault("metadata_category", "document")
            if item["metadata_category"] not in ("document", "page"):
                item["metadata_category"] = "document"
        elif item.get("type") == "clause":
            item.setdefault("clause_number", "")
            item.setdefault("clause_title", "")
            item.setdefault("subclause_hierarchy", [])
            raw_cn = item.get("clause_number", "")
            if raw_cn:
                item["clause_number"] = _normalize_clause_number(raw_cn)
            if isinstance(item["subclause_hierarchy"], str):
                try:
                    item["subclause_hierarchy"] = json.loads(
                        item["subclause_hierarchy"]
                    )
                except (json.JSONDecodeError, ValueError):
                    item["subclause_hierarchy"] = []

        # ── Table row validation ────────────────────────────────────────
        for tbl in item.get("tables", []):
            tbl.setdefault("headers", [])
            tbl.setdefault("rows", [])
            tbl.setdefault("page_start", item.get("page_start", 1))
            tbl.setdefault("page_end", item.get("page_end", 1))
            validated_rows = []
            for row in tbl.get("rows", []):
                if isinstance(row, list):
                    validated_rows.append([str(v) for v in row])
                elif isinstance(row, dict):
                    validated_rows.append([str(v) for v in row.values()])
            tbl["rows"] = validated_rows

    return result


def _replace_boundary_item(merged_items: list[dict],
                           new_item: dict,
                           boundary_page: int) -> None:
    """Replace a partial boundary item from the previous chunk with the
    complete version from the current chunk.

    Matching strategy (in order of priority):
      1. Same type + same page_start → exact match (most common case:
         both chunks see the clause starting on the same page).
      2. Same type + existing item's page_end is at/near the chunk boundary
         AND page_starts are within 2 pages → fuzzy boundary match.
      3. No match found → insert the item at the correct position (by page
         order) as a new cross-boundary discovery.

    Args:
        merged_items:   The growing merged list (mutated in place).
        new_item:       The complete item from the current chunk.
        boundary_page:  The last primary page of the *previous* chunk.
    """
    new_type = new_item.get("type", "")
    new_ps = new_item.get("page_start", 0)
    new_pe = new_item.get("page_end", 0)

    # Search backward — boundary items are near the end of merged_items
    for j in range(len(merged_items) - 1, -1, -1):
        ex = merged_items[j]
        if ex.get("type") != new_type:
            continue

        ex_ps = ex.get("page_start", 0)
        ex_pe = ex.get("page_end", 0)

        # Criterion 1: same page_start → definite match
        exact_match = (ex_ps == new_ps)

        # Criterion 2: existing ends at boundary, starts near new item
        at_boundary = (ex_pe >= boundary_page - 1 and ex_pe <= boundary_page)
        near_start = (abs(ex_ps - new_ps) <= 2)
        fuzzy_match = (at_boundary and near_start)

        if exact_match or fuzzy_match:
            logger.info(
                f"  Cross-boundary merge: replacing partial "
                f"(type={new_type}, pages {ex_ps}–{ex_pe}) with complete "
                f"(pages {new_ps}–{new_pe})"
            )
            merged_items[j] = new_item
            return

    # No match found — insert at the correct position to maintain page order
    insert_pos = len(merged_items)
    for j in range(len(merged_items)):
        if merged_items[j].get("page_start", 0) > new_ps:
            insert_pos = j
            break
    merged_items.insert(insert_pos, new_item)
    logger.info(
        f"  Cross-boundary: inserted new item "
        f"(type={new_type}, pages {new_ps}–{new_pe}) at position {insert_pos}"
    )


def merge_chunk_results(chunk_results: list[dict],
                        chunk_infos: list[ChunkInfo]) -> dict:
    """Merge extraction results from multiple chunks into a single unified result.

    The merged output is structurally identical to what a single-call extraction
    would have produced:
      - document_date: taken from the first chunk that provides one.
      - items: concatenated in document order, with cross-boundary items
        resolved (partial items from chunk N replaced by complete items from
        chunk N+1), overlap-only items discarded, and clause numbers
        renumbered globally.

    Args:
        chunk_results: List of extraction dicts (one per chunk, in order).
        chunk_infos:   Corresponding ChunkInfo objects.

    Returns:
        Unified extraction result dict with ``document_date`` and ``items``.
    """
    merged_items: list[dict] = []
    document_date = ""

    for i, (result, info) in enumerate(zip(chunk_results, chunk_infos)):
        if not result:
            logger.warning(f"  Chunk {info.chunk_num}: empty result, skipping")
            continue

        # Take document_date from the first chunk that provides one
        if not document_date and result.get("document_date"):
            document_date = result["document_date"]

        items = result.get("items", [])
        if not items:
            logger.warning(
                f"  Chunk {info.chunk_num}: zero items extracted, skipping"
            )
            continue

        if i == 0:
            # First chunk: include everything unconditionally
            merged_items.extend(items)
            logger.info(
                f"  Chunk {info.chunk_num}: added {len(items)} items "
                f"(first chunk, no overlap filtering)"
            )
        else:
            # Subsequent chunks: filter out overlap-only items
            primary_start = info.primary_start
            # boundary_page = previous chunk's primary_end
            prev_primary_end = chunk_infos[i - 1].primary_end
            added = 0
            boundary_merged = 0

            for item in items:
                ps = item.get("page_start", 0)
                pe = item.get("page_end", ps)

                if ps >= primary_start:
                    # Item starts within this chunk's primary range → include
                    merged_items.append(item)
                    added += 1
                elif pe >= primary_start:
                    # Cross-boundary item: starts in overlap, continues into
                    # primary range.  This version is COMPLETE (the previous
                    # chunk only saw a truncated version).  Replace the partial.
                    _replace_boundary_item(
                        merged_items, item, prev_primary_end
                    )
                    boundary_merged += 1
                # else: item fully within overlap zone → discard (already
                #        captured by the previous chunk)

            logger.info(
                f"  Chunk {info.chunk_num}: added {added} new items, "
                f"merged {boundary_merged} cross-boundary items"
            )

    # ── Global sequential clause renumbering ────────────────────────────────
    # Override whatever Gemini assigned per-chunk to ensure a single clean
    # sequence: clause_1, clause_2, ..., clause_N across the whole document.
    clause_counter = 0
    for item in merged_items:
        if item.get("type") == "clause":
            clause_counter += 1
            item["clause_number"] = f"clause_{clause_counter}"

    logger.info(
        f"Merge complete: {len(merged_items)} total items, "
        f"{clause_counter} clauses renumbered globally"
    )

    return {"document_date": document_date, "items": merged_items}


class DocumentExtractionAgent:
    """Extracts structured data from a whole PDF using Gemini File API."""

    def __init__(self, gemini_client: GeminiClient):
        self.client = gemini_client

    def process_whole_pdf(self, pdf_path: str, total_pages: int, file_name: str) -> dict:
        """Upload and extract the entire PDF in a single API call."""
        prompt = build_whole_document_prompt(file_name, total_pages)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = self.client.query_with_pdf(pdf_path, prompt)
                result = parse_json_from_response(raw)
                result.setdefault("document_date", "")
                if not isinstance(result.get("items"), list):
                    items = []
                    for mb in result.get("metadata_blocks", []):
                        mb["type"] = "metadata"
                        items.append(mb)
                    for cb in result.get("clause_blocks", []):
                        cb["type"] = "clause"
                        items.append(cb)
                    result["items"] = items

                # Validate each item
                for item in result["items"]:
                    item_type = item.get("type", "")
                    if item_type == "clauses":
                        item["type"] = "clause"
                    item.setdefault("text", "")
                    item.setdefault("tables", [])
                    item.setdefault("page_start", 1)
                    item.setdefault("page_end", item.get("page_start", 1))
                    item.pop("provisional_id", None)
                    item.pop("notes", None)
                    item.pop("has_table", None)
                    item.pop("is_complete", None)
                    item.pop("has_subclauses", None)

                    if item.get("type") == "metadata":
                        item.pop("clause_number", None)
                        item.setdefault("metadata_title", "")
                        item.setdefault("metadata_category", "document")
                        if item["metadata_category"] not in ("document", "page"):
                            item["metadata_category"] = "document"
                    elif item.get("type") == "clause":
                        item.setdefault("clause_number", "")
                        item.setdefault("clause_title", "")
                        item.setdefault("subclause_hierarchy", [])
                        raw_cn = item.get("clause_number", "")
                        if raw_cn:
                            item["clause_number"] = _normalize_clause_number(raw_cn)
                        if isinstance(item["subclause_hierarchy"], str):
                            try:
                                item["subclause_hierarchy"] = json.loads(item["subclause_hierarchy"])
                            except (json.JSONDecodeError, ValueError):
                                item["subclause_hierarchy"] = []

                    # Validate table rows
                    for tbl in item.get("tables", []):
                        tbl.setdefault("headers", [])
                        tbl.setdefault("rows", [])
                        tbl.setdefault("page_start", item.get("page_start", 1))
                        tbl.setdefault("page_end", item.get("page_end", 1))
                        validated_rows = []
                        for row in tbl.get("rows", []):
                            if isinstance(row, list):
                                validated_rows.append([str(v) for v in row])
                            elif isinstance(row, dict):
                                validated_rows.append([str(v) for v in row.values()])
                        tbl["rows"] = validated_rows

                total_chars = sum(len(it.get("text", "")) for it in result["items"])
                if total_chars < 50 and result["items"]:
                    logger.warning(f"LOW TEXT — only {total_chars} chars extracted. Possible text loss.")

                return result

            except Exception as exc:
                logger.warning(f"Extraction attempt {attempt}/{MAX_RETRIES} failed: {exc}")
                # Save raw response to disk for debugging failed parses
                if 'raw' in dir() and raw:
                    debug_path = os.path.join(
                        os.path.dirname(pdf_path),
                        f"{Path(file_name).stem}_raw_response_attempt{attempt}.txt"
                    )
                    try:
                        with open(debug_path, "w", encoding="utf-8") as dbg:
                            dbg.write(raw)
                        logger.info(f"  Raw response saved to: {debug_path}")
                    except Exception:
                        pass
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** (attempt - 1))
                else:
                    logger.error(f"All {MAX_RETRIES} attempts failed for {file_name}")
                    return {"document_date": "", "items": []}

    def process_pdf_chunked(
        self,
        pdf_path: str,
        total_pages: int,
        file_name: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = OVERLAP_PAGES,
        checkpoint_callback=None,
        completed_chunks: set | None = None,
        existing_chunk_results: list | None = None,
    ) -> dict:
        """Process a long PDF by splitting it into overlapping chunks.

        Each chunk is uploaded to Gemini independently, and the results are
        merged into a single extraction result identical to what a single-call
        extraction would have produced.

        Args:
            pdf_path:               Path to the original PDF.
            total_pages:            Total page count.
            file_name:              Original filename.
            chunk_size:             Max primary pages per chunk.
            overlap:                Context overlap pages.
            checkpoint_callback:    Optional callable(chunk_idx, chunk_results)
                                    invoked after each successful chunk for
                                    incremental checkpointing.
            completed_chunks:       Set of already-completed chunk indices
                                    (for resume support).
            existing_chunk_results: Previously extracted chunk results
                                    (parallel to completed_chunks).

        Returns:
            Unified extraction result dict with ``document_date`` and ``items``.
        """
        # ── Step 1: Split the PDF into chunks ───────────────────────────────
        stem = Path(file_name).stem
        temp_dir = os.path.join(
            os.path.dirname(pdf_path), f".{stem}_chunks_tmp"
        )
        chunk_infos = split_pdf_into_chunks(
            pdf_path, chunk_size, overlap, temp_dir
        )

        # ── Step 2: Initialize results tracking ────────────────────────────
        num_chunks = len(chunk_infos)
        completed = set(completed_chunks or set())

        # Pre-fill with existing results or None placeholders
        chunk_results: list[dict | None] = [None] * num_chunks
        if existing_chunk_results:
            for idx, res in enumerate(existing_chunk_results):
                if idx < num_chunks:
                    chunk_results[idx] = res

        # Count clauses from already-completed chunks for prompt continuity
        prev_clause_count = 0
        for idx in sorted(completed):
            if idx < num_chunks and chunk_results[idx]:
                for item in chunk_results[idx].get("items", []):
                    if item.get("type") == "clause":
                        prev_clause_count += 1

        # ── Step 3: Process each chunk ──────────────────────────────────────
        for idx, info in enumerate(chunk_infos):
            if idx in completed:
                logger.info(
                    f"  Chunk {info.chunk_num}/{num_chunks}: "
                    f"already completed (from checkpoint)"
                )
                continue

            logger.info(
                f"\n  ── Chunk {info.chunk_num}/{num_chunks} ──"
                f"\n     Pages: {info.real_page_start}–{info.real_page_end}"
                f"\n     Primary range: {info.primary_start}–{info.primary_end}"
                f"\n     Previous clauses: {prev_clause_count}"
            )

            # Build chunk-aware prompt
            prompt = build_chunk_prompt(
                file_name, total_pages, info, prev_clause_count
            )

            # Attempt extraction with retries
            result = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    raw = self.client.query_with_pdf(info.chunk_path, prompt)
                    result = parse_json_from_response(raw)
                    result = _validate_extraction_result(result)

                    # Sanity check
                    total_chars = sum(
                        len(it.get("text", ""))
                        for it in result.get("items", [])
                    )
                    if total_chars < 50 and result["items"]:
                        logger.warning(
                            f"  LOW TEXT in chunk {info.chunk_num}: "
                            f"only {total_chars} chars"
                        )
                    break  # Success — exit retry loop

                except Exception as exc:
                    logger.warning(
                        f"  Chunk {info.chunk_num} attempt "
                        f"{attempt}/{MAX_RETRIES} failed: {exc}"
                    )
                    # Save raw response for debugging if available
                    if "raw" in dir() and raw:
                        debug_path = os.path.join(
                            os.path.dirname(pdf_path),
                            f"{stem}_chunk{info.chunk_num}"
                            f"_raw_attempt{attempt}.txt",
                        )
                        try:
                            with open(debug_path, "w", encoding="utf-8") as dbg:
                                dbg.write(raw)
                            logger.info(f"  Raw response saved to: {debug_path}")
                        except Exception:
                            pass
                    if attempt < MAX_RETRIES:
                        time.sleep(2 ** (attempt - 1))

            if result is None:
                logger.error(
                    f"  Chunk {info.chunk_num} FAILED after {MAX_RETRIES} "
                    f"attempts. Using empty result."
                )
                result = {"document_date": "", "items": []}

            # Track clauses for the next chunk's prompt
            for item in result.get("items", []):
                if item.get("type") == "clause":
                    prev_clause_count += 1

            chunk_results[idx] = result
            completed.add(idx)

            logger.info(
                f"  Chunk {info.chunk_num} complete: "
                f"{len(result.get('items', []))} items extracted"
            )

            # Save incremental checkpoint
            if checkpoint_callback:
                checkpoint_callback(idx, chunk_results)

        # ── Step 4: Cleanup temporary chunk PDFs ────────────────────────────
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temporary chunks: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up {temp_dir}: {e}")

        # ── Step 5: Merge all chunk results ─────────────────────────────────
        logger.info("")
        logger.info("─" * 60)
        logger.info("Merging chunk results into unified extraction")
        logger.info("─" * 60)

        # Filter out None entries (shouldn't happen, but defensive)
        valid_results = [r if r else {"document_date": "", "items": []}
                         for r in chunk_results]

        merged = merge_chunk_results(valid_results, chunk_infos)
        return merged


class OrchestratorAgent:
    """Post-processes Gemini's whole-document output into Schema B."""

    def __init__(self):
        self._working_metadata: dict[tuple, dict] = {}
        self._working_clauses: dict[str, dict] = {}
        self._flat_tables: list[dict] = []
        self._item_order_counter: int = 0
        self.document_date: str = ""
        self.file_name: str = ""

    @staticmethod
    def _renumber_duplicate_clauses(items: list[dict]) -> list[dict]:
        """Detect and fix duplicate clause_numbers caused by chapter-local numbering.

        If Gemini outputs clause_1 twice (once in chapter 1, once in chapter 6),
        this method renumbers the second occurrence so that every clause has a
        unique global sequential clause_number.
        """
        seen: dict[str, int] = {}  # clause_number -> index of first occurrence
        global_counter = 0
        needs_renumber = False

        # First pass: detect duplicates and find the max clause number
        for i, item in enumerate(items):
            if item.get("type") != "clause":
                continue
            cn = item.get("clause_number", "")
            if not cn:
                continue
            # Extract the numeric part
            m = re.match(r'clause_(\d+)', cn)
            if m:
                n = int(m.group(1))
                global_counter = max(global_counter, n)
            if cn in seen:
                needs_renumber = True
            else:
                seen[cn] = i

        if not needs_renumber:
            return items

        # Second pass: renumber duplicates
        logger.warning("Detected duplicate clause_numbers (chapter-local numbering). "
                       "Renumbering to global sequential.")
        assigned: set[str] = set()
        for item in items:
            if item.get("type") != "clause":
                continue
            cn = item.get("clause_number", "")
            if not cn:
                continue
            if cn in assigned:
                # This is a duplicate — assign next global number
                global_counter += 1
                new_cn = f"clause_{global_counter}"
                logger.info(f"  Renumbered duplicate {cn} → {new_cn} "
                            f"(page {item.get('page_start', '?')})")
                item["clause_number"] = new_cn
                assigned.add(new_cn)
            else:
                assigned.add(cn)

        return items

    def process_whole_document(self, extraction_result: dict):
        """Ingest the whole-document extraction result into working storage."""
        self.document_date = extraction_result.get("document_date", "")
        items = extraction_result.get("items", [])

        # Fix chapter-local numbering collisions before merging
        items = self._renumber_duplicate_clauses(items)

        for item in items:
            item_type = item.get("type", "")
            self._item_order_counter += 1
            page_start = item.get("page_start", 1)
            page_end = item.get("page_end", page_start)

            if item_type == "metadata":
                category = item.get("metadata_category", "document")
                merge_key = (category, self._item_order_counter)
                entry = {
                    "type": "metadata",
                    "metadata_category": category,
                    "metadata_title": item.get("metadata_title", ""),
                    "text": item.get("text", ""),
                    "tables": list(item.get("tables", [])),
                    "page_start": page_start,
                    "page_end": page_end,
                    "_item_order": self._item_order_counter,
                }
                self._working_metadata[merge_key] = entry

            elif item_type == "clause":
                cn = _normalize_clause_number(item.get("clause_number", ""))
                if not cn:
                    logger.warning(f"Clause with empty clause_number, skipping")
                    continue

                if cn in self._working_clauses:
                    existing = self._working_clauses[cn]
                    if item.get("text"):
                        existing["text"] += "\n" + item["text"]
                    existing["page_end"] = page_end
                    existing["tables"].extend(item.get("tables", []))
                    new_hier = item.get("subclause_hierarchy", [])
                    if new_hier:
                        existing["subclause_hierarchy"] = new_hier
                else:
                    entry = {
                        "type": "clause",
                        "clause_number": cn,
                        "clause_title": item.get("clause_title", ""),
                        "text": item.get("text", ""),
                        "subclause_hierarchy": list(item.get("subclause_hierarchy", [])),
                        "tables": list(item.get("tables", [])),
                        "page_start": page_start,
                        "page_end": page_end,
                        "_item_order": self._item_order_counter,
                    }
                    self._working_clauses[cn] = entry

                # Extract tables
                for tbl in item.get("tables", []):
                    self._flat_tables.append({
                        "headers": tbl.get("headers", []),
                        "rows": tbl.get("rows", []),
                        "page_start": tbl.get("page_start", page_start),
                        "page_end": tbl.get("page_end", page_end),
                        "parent_merge_key": cn,
                        "parent_type": "clause",
                    })
            else:
                logger.warning(f"Unknown item type '{item_type}', skipping")

        logger.info(
            f"Working storage: {len(self._working_metadata)} metadata, "
            f"{len(self._working_clauses)} clauses, {len(self._flat_tables)} tables"
        )

    def determine_document_type(self) -> str:
        """Analyze extracted content to classify document type."""
        all_text = ""
        for item in list(self._working_metadata.values()) + list(self._working_clauses.values()):
            all_text += " " + item.get("text", "")
        all_text = all_text.lower()

        if "finance act" in all_text or "অর্থ আইন" in all_text:
            return "Finance Act"
        elif "sro" in all_text or "এস.আর.ও" in all_text:
            return "Statutory Regulatory Order (SRO)"
        elif "customs" in all_text or "শুল্ক" in all_text:
            return "Customs Notification"
        elif "income tax" in all_text or "আয়কর" in all_text:
            return "Income Tax Regulation"
        elif "vat" in all_text or "মূল্য সংযোজন কর" in all_text:
            return "VAT Regulation"
        else:
            return "Government Gazette Document"

    def finalize_output(self, total_pages: int) -> dict:
        """Assign sequential IDs, flatten tables, produce Schema B output."""
        doc_stem = Path(self.file_name).stem if self.file_name else "doc"
        doc_type = self.determine_document_type()

        all_items = list(self._working_metadata.values()) + list(self._working_clauses.values())
        all_items.sort(key=lambda x: (x.get("page_start", 0), x.get("_item_order", 0)))

        metadata_counter = 0
        clause_counter = 0
        clause_id_map: dict[str, str] = {}

        for item in all_items:
            if item.get("type") == "metadata":
                metadata_counter += 1
                item["text_id"] = f"metadata_{metadata_counter}"
            elif item.get("type") == "clause":
                clause_counter += 1
                effective_date = self.document_date or "unknown"
                item["text_id"] = f"{doc_stem}_{effective_date}_clause_{clause_counter}"
                clause_id_map[item.get("clause_number", "")] = item["text_id"]

        meta_id_map: dict[tuple, str] = {}
        for key, item in self._working_metadata.items():
            meta_id_map[key] = item.get("text_id", "")

        final_tables = []
        table_counter = 0
        for item in all_items:
            item["table_ids"] = []

        for tbl in self._flat_tables:
            table_counter += 1
            global_table_id = f"table_{table_counter}"
            parent_key = tbl.get("parent_merge_key")
            parent_type = tbl.get("parent_type", "")
            parent_text_id = ""

            if parent_type == "clause" and isinstance(parent_key, str):
                parent_text_id = clause_id_map.get(parent_key, "")
                clause_data = self._working_clauses.get(parent_key)
                if clause_data:
                    clause_data.setdefault("table_ids", []).append(global_table_id)
            elif parent_type == "metadata" and isinstance(parent_key, tuple):
                parent_text_id = meta_id_map.get(parent_key, "")
                meta_data = self._working_metadata.get(parent_key)
                if meta_data:
                    meta_data.setdefault("table_ids", []).append(global_table_id)

            final_tables.append({
                "table_id": global_table_id,
                "parent_text_id": parent_text_id,
                "page_start": tbl.get("page_start", 0),
                "page_end": tbl.get("page_end", 0),
                "headers": tbl.get("headers", []),
                "rows": tbl.get("rows", []),
            })

        # ── Normalize [TABLE_K] → [table_N] in clause text ──────────────
        for item in all_items:
            if item.get("type") == "clause" and item.get("table_ids"):
                text = item.get("text", "")
                has_markers = bool(re.search(r'\[TABLE_\d+\]', text))
                if has_markers:
                    # Replace clause-local [TABLE_K] with global [table_N]
                    for local_idx, global_tid in enumerate(item["table_ids"], start=1):
                        text = text.replace(f"[TABLE_{local_idx}]", f"[{global_tid}]")
                    # Clean up any unreplaced markers (defensive)
                    text = re.sub(r'\[TABLE_\d+\]', '', text)
                else:
                    # Fallback: LLM didn't insert markers.
                    # Append all table tags at the end of the text.
                    tags = " ".join(f"[{tid}]" for tid in item["table_ids"])
                    text = text.rstrip() + " " + tags
                    logger.debug(
                        f"Fallback table tag insertion for "
                        f"{item.get('clause_number', '?')}: appended {tags}"
                    )
                item["text"] = text

        parsed_text = []
        for item in all_items:
            entry = dict(item)
            if entry.get("type") == "clause":
                hierarchy = entry.get("subclause_hierarchy", [])
                if isinstance(hierarchy, list) and hierarchy:
                    entry["subclause_hierarchy"] = _to_bracket_string(hierarchy)
                elif isinstance(hierarchy, list):
                    entry["subclause_hierarchy"] = ""
            entry.setdefault("table_ids", [])
            entry.pop("_item_order", None)
            entry.pop("tables", None)
            parsed_text.append(entry)

        output = {
            "document_info": {
                "file_name": self.file_name,
                "document_type": doc_type,
                "document_date": self.document_date,
                "total_pages": total_pages,
                "extraction_timestamp": datetime.datetime.now().isoformat(),
                "summary": {
                    "total_metadata": metadata_counter,
                    "total_clauses": clause_counter,
                    "total_tables": table_counter,
                },
            },
            "parsed_text": parsed_text,
            "tables": final_tables,
        }
        return output


def process_single_pdf(
    pdf_path: str,
    output_dir: str | None = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
    api_keys_file: str | None = None,
    resume: bool = True,
    cleanup: bool = False,
    verbose: bool = False,
) -> str:
    """Execute the whole-PDF document-intelligence pipeline on one PDF."""
    pdf_path = os.path.abspath(pdf_path)
    pdf_name = Path(pdf_path).stem

    if output_dir is None:
        output_dir = os.path.dirname(pdf_path)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, f"{pdf_name}_pipeline_log.txt")
    output_path = os.path.join(output_dir, f"{pdf_name}_output.json")
    ckpt_path = os.path.join(output_dir, f".{pdf_name}.checkpoint.json")

    setup_logging(verbose=verbose, log_file_path=log_path)

    pipeline_start = time.time()

    logger.info("")
    logger.info("═" * 60)
    logger.info(f"  PDF:    {os.path.basename(pdf_path)}")
    logger.info(f"  Model:  {model_name}")
    logger.info(f"  Output: {output_dir}")
    logger.info("═" * 60)

    # Load API keys & create client
    api_keys = load_api_keys(api_keys_file)
    key_manager = GeminiApiKeyManager(api_keys)
    gemini_client = GeminiClient(key_manager, model_name=model_name)

    # Page count via fitz
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    doc.close()
    logger.info(f"Page count: {page_count}")

    # Check for existing checkpoint (resume support)
    extraction_result = None
    if resume and os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                ckpt_data = json.load(f)
            if ckpt_data.get("_meta", {}).get("extraction_complete"):
                extraction_result = ckpt_data.get("extraction_result")
                logger.info("Loaded extraction result from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    # STEP 1: PDF Extraction — route based on document length
    if extraction_result is None:
        agent = DocumentExtractionAgent(gemini_client)

        if page_count <= CHUNK_SIZE:
            # ── SHORT DOCUMENT: single API call (original path) ─────────
            logger.info("")
            logger.info("─" * 60)
            logger.info("STEP 1 — Whole-PDF Extraction (Single API Call)")
            logger.info(f"  Document is {page_count} pages "
                        f"(≤ {CHUNK_SIZE}), using single-call path")
            logger.info("─" * 60)

            extraction_result = agent.process_whole_pdf(
                pdf_path, page_count, os.path.basename(pdf_path)
            )

            # Save checkpoint
            try:
                ckpt_data = {
                    "_meta": {"extraction_complete": True, "chunked": False},
                    "extraction_result": extraction_result,
                }
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump(ckpt_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")

        else:
            # ── LONG DOCUMENT: chunked extraction ───────────────────────
            logger.info("")
            logger.info("─" * 60)
            logger.info("STEP 1 — Chunked PDF Extraction")
            logger.info(f"  Document is {page_count} pages "
                        f"(> {CHUNK_SIZE}), splitting into chunks")
            logger.info(f"  Chunk size: {CHUNK_SIZE} pages, "
                        f"Overlap: {OVERLAP_PAGES} pages")
            logger.info("─" * 60)

            # Check for partial chunked checkpoint (resume mid-extraction)
            completed_chunks = None
            existing_chunk_results = None
            if resume and os.path.exists(ckpt_path):
                try:
                    with open(ckpt_path, "r", encoding="utf-8") as f:
                        ckpt_data = json.load(f)
                    meta = ckpt_data.get("_meta", {})
                    if (meta.get("chunked") and
                            not meta.get("extraction_complete")):
                        completed_chunks = set(
                            meta.get("completed_chunks", [])
                        )
                        existing_chunk_results = ckpt_data.get(
                            "chunk_results", []
                        )
                        logger.info(
                            f"  Resuming chunked extraction: "
                            f"{len(completed_chunks)} chunk(s) "
                            f"already completed"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to load chunked checkpoint: {e}"
                    )

            # Checkpoint callback — saves after each chunk
            def _save_chunk_checkpoint(chunk_idx, chunk_results):
                """Save incremental checkpoint after each chunk."""
                try:
                    completed_so_far = [
                        i for i, r in enumerate(chunk_results) if r is not None
                    ]
                    ckpt = {
                        "_meta": {
                            "extraction_complete": False,
                            "chunked": True,
                            "chunk_size": CHUNK_SIZE,
                            "overlap": OVERLAP_PAGES,
                            "total_chunks": len(chunk_results),
                            "completed_chunks": completed_so_far,
                        },
                        "chunk_results": [
                            r if r else None for r in chunk_results
                        ],
                    }
                    with open(ckpt_path, "w", encoding="utf-8") as f:
                        json.dump(ckpt, f, ensure_ascii=False, indent=2)
                    logger.debug(
                        f"  Checkpoint saved after chunk {chunk_idx + 1}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to save chunk checkpoint: {e}")

            extraction_result = agent.process_pdf_chunked(
                pdf_path,
                page_count,
                os.path.basename(pdf_path),
                chunk_size=CHUNK_SIZE,
                overlap=OVERLAP_PAGES,
                checkpoint_callback=_save_chunk_checkpoint,
                completed_chunks=completed_chunks,
                existing_chunk_results=existing_chunk_results,
            )

            # Save final checkpoint (extraction complete)
            try:
                ckpt_data = {
                    "_meta": {"extraction_complete": True, "chunked": True},
                    "extraction_result": extraction_result,
                }
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump(ckpt_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")

    # STEP 2: Finalize Output (Schema B)
    logger.info("")
    logger.info("─" * 60)
    logger.info("STEP 2 — Finalize Output (IDs + Flat Tables + Schema B)")
    logger.info("─" * 60)

    orchestrator = OrchestratorAgent()
    orchestrator.file_name = os.path.basename(pdf_path)
    orchestrator.process_whole_document(extraction_result)
    output_dict = orchestrator.finalize_output(page_count)

    # Write output JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)

    summary = output_dict.get("document_info", {}).get("summary", {})
    doc_type = output_dict.get("document_info", {}).get("document_type", "")

    logger.info(f"  Document type: {doc_type}")
    logger.info(f"  Output: {output_path}")

    pipeline_elapsed = time.time() - pipeline_start
    usage_stats = key_manager.get_usage_stats()

    logger.info("")
    logger.info("═" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Time:           {pipeline_elapsed:.1f}s ({pipeline_elapsed/60:.1f} min)")
    logger.info(f"  Pages:          {page_count}")
    logger.info(f"  Metadata:       {summary.get('total_metadata', 0)}")
    logger.info(f"  Clauses:        {summary.get('total_clauses', 0)}")
    logger.info(f"  Tables:         {summary.get('total_tables', 0)}")
    logger.info(f"  API calls:      {usage_stats['total_used']}")
    logger.info(f"  Output:         {output_path}")
    logger.info("═" * 60)

    if cleanup and os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    _detach_file_handler()
    return output_path


def process_directory(input_dir: str, output_dir: str | None = None,
                      model_name: str = DEFAULT_GEMINI_MODEL,
                      api_keys_file: str | None = None, **kwargs):
    """Process every PDF in a directory.

    Creates a separate subfolder for each PDF inside the output directory:
        results/
        ├── document1/
        │   ├── document1_output.json
        │   └── document1_pipeline_log.txt
        ├── document2/
        │   └── ...
    """
    pdf_files = sorted(Path(input_dir).glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return
    batch_out_dir = output_dir if output_dir else input_dir
    os.makedirs(batch_out_dir, exist_ok=True)

    logger.info(f"Found {len(pdf_files)} PDF(s) to process in {input_dir}")

    for i, pdf_path in enumerate(tqdm(pdf_files, desc="Batch", unit="pdf", ncols=90), 1):
        logger.info(f"\n  ── PDF {i}/{len(pdf_files)}: {pdf_path.name} ──")
        try:
            # Create a per-PDF subfolder inside the output directory
            pdf_subfolder = os.path.join(batch_out_dir, pdf_path.stem)
            os.makedirs(pdf_subfolder, exist_ok=True)
            process_single_pdf(
                str(pdf_path), output_dir=pdf_subfolder, model_name=model_name,
                api_keys_file=api_keys_file,
                resume=kwargs.get("resume", True),
                cleanup=kwargs.get("cleanup", False),
                verbose=kwargs.get("verbose", False),
            )
        except Exception:
            logger.error(f"Failed for {pdf_path.name}:")
            traceback.print_exc()
    _detach_file_handler()


def main():
    parser = argparse.ArgumentParser(
        description="Document Intelligence Pipeline — Whole-PDF Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python document_pipeline_gemini_extended_prompt.py report.pdf
  python3 ocr_pipeline_v1_copy.py ./pdfs/ -o ./gemini-3-flash-preview-results/
  python document_pipeline_gemini_extended_prompt.py report.pdf --model gemini-2.5-flash
  python document_pipeline_gemini_extended_prompt.py report.pdf --no-resume
        """,
    )
    parser.add_argument("input", help="Path to a single PDF or a directory of PDFs.")
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL,
                        help=f"Gemini model name. Default: {DEFAULT_GEMINI_MODEL}")
    parser.add_argument("--api-keys-file", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    input_path = os.path.abspath(args.input)
    resume = not args.no_resume

    if os.path.isfile(input_path):
        if not input_path.lower().endswith(".pdf"):
            logger.error(f"Not a PDF file: {input_path}")
            sys.exit(1)
        process_single_pdf(
            input_path, output_dir=args.output_dir, model_name=args.model,
            api_keys_file=args.api_keys_file,
            resume=resume, cleanup=args.cleanup, verbose=args.verbose,
        )
    elif os.path.isdir(input_path):
        process_directory(
            input_path, output_dir=args.output_dir, model_name=args.model,
            api_keys_file=args.api_keys_file,
            resume=resume, cleanup=args.cleanup, verbose=args.verbose,
        )
    else:
        logger.error(f"Path does not exist: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
