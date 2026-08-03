"""
rubric_scoring.py — deterministic date normalization and rubric-v2 scoring.

Pure functions only: inputs are text / parsed judge atoms / GT metadata, outputs
are canonical dates and scores. No I/O, no API calls, no google-SDK imports —
everything here is unit-testable offline (tests/test_rubric_scoring.py).

Division of labor (rubric v2):
  - GT dates are extracted deterministically HERE (`normalize_dates` on the
    ground-truth answer) and handed TO the LLM judge as a canonical list; the
    judge never discovers GT dates on its own.
  - The judge (llm_council.judge_atoms) emits atomic booleans only — per-GT-date
    match, contradiction, meaning_correct, uncertain — never a numeric score.
  - Every score is computed here (`score_base`, `score_perturbation_explanation`)
    from those atoms plus deterministic matching, so scoring is reproducible.

Rubric versions:
  - "2.0": base rubric. Granularity rule (Decision B): a GT date is satisfied
    iff the model asserts every component the GT specifies, with matching
    values (GT 2019-06 ⊨ model 2019-06-12; GT 2019-06-12 ⊭ model 2019-06).
  - "2.1" (relative_time_qa only): required match granularity follows the
    record's `relative_offset`, overriding Decision B — year offsets need a
    year match; month offsets of 1-3 months need a full exact date; month
    offsets of >3 months need year+month; day/week offsets need a full date.
    Components finer than the requirement are ignored (a wrong day under a
    year offset is neither a mismatch nor a contradiction). An unparsable
    offset falls back to Decision B and is counted (offset_granularity_unparsed).

Calendar policy: Bangla-calendar dates (বৈশাখ…চৈত্র) are parsed and flagged but
NEVER converted (no converter exists in this repo); deterministic matching
refuses cross-calendar comparisons, so those items route to the judge and are
counted via `calendar_mismatch`.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

RUBRIC_VERSION = "2.0"
RUBRIC_VERSION_RELATIVE_TIME = "2.1"

# ════════════════════════════════════════════════════════════════════════════════
# Script / month tables
# ════════════════════════════════════════════════════════════════════════════════

_BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def bengali_digits_to_ascii(text: str) -> str:
    """Map Bengali numerals ০-৯ to ASCII 0-9 (1:1, offsets preserved)."""
    return text.translate(_BENGALI_DIGITS)


# Gregorian month names -> month number. English full + abbreviations, and
# Bangla-script Gregorian months (common spelling variants included).
_GREGORIAN_MONTHS: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    "জানুয়ারি": 1, "জানুয়ারী": 1, "ফেব্রুয়ারি": 2, "ফেব্রুয়ারী": 2, "মার্চ": 3,
    "এপ্রিল": 4, "মে": 5, "জুন": 6, "জুলাই": 7, "আগস্ট": 8, "অগাস্ট": 8,
    "সেপ্টেম্বর": 9, "অক্টোবর": 10, "নভেম্বর": 11, "ডিসেম্বর": 12,
}

# Bangla-calendar months -> month number within the Bangla year (Boishakh = 1).
# Parsed and flagged calendar="bangla"; never converted to Gregorian.
_BANGLA_MONTHS: dict[str, int] = {
    "বৈশাখ": 1, "জ্যৈষ্ঠ": 2, "আষাঢ়": 3, "শ্রাবণ": 4, "ভাদ্র": 5, "আশ্বিন": 6,
    "কার্তিক": 7, "অগ্রহায়ণ": 8, "পৌষ": 9, "মাঘ": 10, "ফাল্গুন": 11, "চৈত্র": 12,
}

_ALL_MONTHS: dict[str, tuple[int, str]] = {
    **{name: (num, "gregorian") for name, num in _GREGORIAN_MONTHS.items()},
    **{name: (num, "bangla") for name, num in _BANGLA_MONTHS.items()},
}

# Longest-first so "জানুয়ারী" wins over any prefix and "sept" over "sep".
_MONTH_ALT = "|".join(
    re.escape(m) for m in sorted(_ALL_MONTHS, key=len, reverse=True))

_GRANULARITY_RANK = {"year": 1, "month": 2, "day": 3}


# ════════════════════════════════════════════════════════════════════════════════
# CanonicalDate
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CanonicalDate:
    """A date at its asserted granularity (never padded to a full date)."""

    year: int
    month: int | None = None
    day: int | None = None
    calendar: str = "gregorian"      # "gregorian" | "bangla"
    range_id: int | None = None      # both endpoints of one range share an id
    range_role: str | None = None    # "start" | "end" | None
    source_span: str = ""            # verbatim text this was parsed from

    @property
    def granularity(self) -> str:
        if self.day is not None:
            return "day"
        if self.month is not None:
            return "month"
        return "year"

    def canonical(self) -> str:
        if self.day is not None:
            return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"
        if self.month is not None:
            return f"{self.year:04d}-{self.month:02d}"
        return f"{self.year:04d}"

    def to_dict(self) -> dict:
        return {
            "canonical": self.canonical(),
            "calendar": self.calendar,
            "granularity": self.granularity,
            "range_id": self.range_id,
            "range_role": self.range_role,
            "source_span": self.source_span,
        }


# ════════════════════════════════════════════════════════════════════════════════
# Date extraction
# ════════════════════════════════════════════════════════════════════════════════

_ORDINAL = r"(?:st|nd|rd|th)?"

# ISO: YYYY-MM-DD or YYYY-MM (year leading, so it never collides with the
# day-first numeric form below).
_ISO_RE = re.compile(r"\b(1[0-9]{3}|2[0-9]{3})-(\d{1,2})(?:-(\d{1,2}))?\b")

# "MonthName D, YYYY" / "MonthName D YYYY" (English order).
_MDY_RE = re.compile(
    rf"\b({_MONTH_ALT})\s+(\d{{1,2}}){_ORDINAL}\s*,?\s+(\d{{4}})\b",
    re.IGNORECASE)

# "D MonthName, YYYY" / "D MonthName YYYY" (Bangla + European order;
# Bengali numerals are ASCII by the time this runs).
_DMY_RE = re.compile(
    rf"\b(\d{{1,2}}){_ORDINAL}\s+({_MONTH_ALT})\s*,?\s+(\d{{4}})\b",
    re.IGNORECASE)

# "MonthName YYYY" (month granularity).
_MY_RE = re.compile(rf"\b({_MONTH_ALT})\s*,?\s+(\d{{4}})\b", re.IGNORECASE)

# Numeric D/M/YYYY — day-first (Bangladeshi convention); also D-M-YYYY and
# D.M.YYYY. The trailing 4-digit year keeps this distinct from ISO.
_NUMERIC_RE = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b")

# Bare 4-digit years (model-answer side only; GT text is full of years inside
# instrument identifiers like "S.R.O. No. 150-AIN/2013/2437/Customs").
_BARE_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")

# Range connector between two adjacent dates. থেকে/হইতে = "from/starting",
# পর্যন্ত = "until"; a bare dash between two dates also reads as a range.
_RANGE_CONNECTOR_RE = re.compile(
    r"^[\s,]*(?:থেকে|হইতে|to|through|until|till|[–—-])[\s,]*$",
    re.IGNORECASE)
_MAX_RANGE_GAP = 12  # chars between the two date spans


def _valid(year: int, month: int | None, day: int | None, calendar: str) -> bool:
    if not (1000 <= year <= 2999):
        return False
    if month is not None and not (1 <= month <= 12):
        return False
    max_day = 32 if calendar == "bangla" else 31
    if day is not None and not (1 <= day <= max_day):
        return False
    return True


def normalize_dates(text: str, *, include_bare_years: bool = False
                    ) -> list[CanonicalDate]:
    """Extract every date expression in `text` as CanonicalDate objects.

    Handles Bengali numerals, Bangla-script Gregorian months, English month
    names/abbreviations, ISO forms, day-first numeric dates, and month-year
    forms; Bangla-calendar months are parsed with calendar="bangla". Two dates
    joined by থেকে/হইতে/to/a dash become a range pair (shared range_id). Bare
    4-digit years are extracted only with include_bare_years=True — GT-side
    extraction must never take them (instrument identifiers are full of years).
    Identical canonicals are deduped (first occurrence wins).
    """
    if not text:
        return []
    ascii_text = bengali_digits_to_ascii(text)

    # (start, end, CanonicalDate-kwargs); earlier patterns consume their spans
    # so later, looser patterns cannot re-extract inside them.
    found: list[tuple[int, int, dict]] = []

    def taken(start: int, end: int) -> bool:
        return any(s < end and start < e for s, e, _ in found)

    def add(m: re.Match, year: int, month: int | None, day: int | None,
            calendar: str) -> None:
        if taken(m.start(), m.end()) or not _valid(year, month, day, calendar):
            return
        found.append((m.start(), m.end(), {
            "year": year, "month": month, "day": day, "calendar": calendar,
            "source_span": text[m.start():m.end()],
        }))

    for m in _ISO_RE.finditer(ascii_text):
        add(m, int(m.group(1)), int(m.group(2)),
            int(m.group(3)) if m.group(3) else None, "gregorian")

    for m in _MDY_RE.finditer(ascii_text):
        num, cal = _ALL_MONTHS[m.group(1).lower()]
        add(m, int(m.group(3)), num, int(m.group(2)), cal)

    for m in _DMY_RE.finditer(ascii_text):
        num, cal = _ALL_MONTHS[m.group(2).lower()]
        add(m, int(m.group(3)), num, int(m.group(1)), cal)

    for m in _MY_RE.finditer(ascii_text):
        num, cal = _ALL_MONTHS[m.group(1).lower()]
        add(m, int(m.group(2)), num, None, cal)

    for m in _NUMERIC_RE.finditer(ascii_text):
        a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if b <= 12:                      # day-first (Bangladeshi convention)
            add(m, year, b, a, "gregorian")
        elif a <= 12:                    # US-style fallback when day-first is impossible
            add(m, year, a, b, "gregorian")

    if include_bare_years:
        for m in _BARE_YEAR_RE.finditer(ascii_text):
            add(m, int(m.group(1)), None, None, "gregorian")

    found.sort(key=lambda t: t[0])

    # Mark range pairs: two adjacent dates whose in-between text is just a
    # range connector. "১ জুলাই ২০১৯ থেকে কার্যকর" has no second date -> no range.
    range_id = 0
    for i in range(len(found) - 1):
        _, end_a, kw_a = found[i]
        start_b, _, kw_b = found[i + 1]
        if kw_a.get("range_id") is not None:
            continue
        between = ascii_text[end_a:start_b]
        if len(between) <= _MAX_RANGE_GAP and _RANGE_CONNECTOR_RE.match(between):
            kw_a.update(range_id=range_id, range_role="start")
            kw_b.update(range_id=range_id, range_role="end")
            range_id += 1

    dates: list[CanonicalDate] = []
    seen: set[tuple[str, str]] = set()
    for _, _, kw in found:
        d = CanonicalDate(**kw)
        key = (d.canonical(), d.calendar)
        if key in seen:
            continue
        seen.add(key)
        dates.append(d)
    return dates


# ════════════════════════════════════════════════════════════════════════════════
# Date matching
# ════════════════════════════════════════════════════════════════════════════════

def date_satisfied(gt: CanonicalDate, model_dates: list[CanonicalDate],
                   required_granularity: str | None = None) -> bool:
    """True iff some model date asserts every required component of `gt` with
    matching values.

    Decision B (required_granularity=None): the required components are exactly
    those `gt` specifies. Rubric v2.1: `required_granularity` caps the
    requirement (components finer than it are ignored on both sides); it can
    never require more than `gt` itself specifies. Cross-calendar comparisons
    are always False (no converter — the judge decides those).
    """
    required_rank = _GRANULARITY_RANK[gt.granularity]
    if required_granularity is not None:
        required_rank = min(required_rank, _GRANULARITY_RANK[required_granularity])

    for md in model_dates:
        if md.calendar != gt.calendar:
            continue
        if _GRANULARITY_RANK[md.granularity] < required_rank:
            continue  # model is coarser than required — cannot satisfy
        if md.year != gt.year:
            continue
        if required_rank >= 2 and md.month != gt.month:
            continue
        if required_rank >= 3 and md.day != gt.day:
            continue
        return True
    return False


# ════════════════════════════════════════════════════════════════════════════════
# Relative-offset granularity (rubric v2.1)
# ════════════════════════════════════════════════════════════════════════════════

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "couple": 2, "few": 3,
    "এক": 1, "দুই": 2, "তিন": 3, "চার": 4, "পাঁচ": 5, "ছয়": 6, "সাত": 7,
    "আট": 8, "নয়": 9, "দশ": 10, "এগারো": 11, "বারো": 12,
}

_DAY_UNIT_RE = re.compile(r"\b(day|days|week|weeks|fortnight)\b|দিন|সপ্তাহ",
                          re.IGNORECASE)
_MONTH_UNIT_RE = re.compile(r"\b(month|months)\b|মাস", re.IGNORECASE)
_YEAR_UNIT_RE = re.compile(r"\b(year|years|yrs?|decade|decades)\b|বছর|দশক",
                           re.IGNORECASE)


def _count_before(text: str, unit_start: int) -> int | None:
    """Numeric count expressed just before a unit word ('three months',
    '2 months', 'a couple of months'); None if no count can be parsed."""
    # \w alone splits Bengali words at combining vowel marks (category Mn),
    # so the Bengali block is included explicitly.
    tokens = re.findall(r"[0-9]+|[\wঀ-৿]+", text[:unit_start], re.UNICODE)
    for tok in reversed(tokens[-3:]):
        if tok.isdigit():
            return int(tok)
        val = _NUMBER_WORDS.get(tok.lower())
        if val is not None:
            return val
    return None


def offset_granularity(relative_offset: str | None) -> str | None:
    """Required match granularity implied by a relative-time offset.

    Year/decade units -> "year". Month units are count-dependent: 1-3 months ->
    "day" (full exact date, same strictness as day/week offsets), >3 months ->
    "month"; a month unit whose count cannot be parsed -> None (Decision B
    fallback, counted as offset_granularity_unparsed). Day/week/fortnight
    units -> "day". No recognizable unit -> None. When several units appear,
    the finest one wins.
    """
    if not relative_offset or not str(relative_offset).strip():
        return None
    text = bengali_digits_to_ascii(str(relative_offset))

    best_rank = 0
    if _DAY_UNIT_RE.search(text):
        best_rank = max(best_rank, _GRANULARITY_RANK["day"])
    for m in _MONTH_UNIT_RE.finditer(text):
        count = _count_before(text, m.start())
        if count is None:
            return None
        rank = (_GRANULARITY_RANK["day"] if count <= 3
                else _GRANULARITY_RANK["month"])
        best_rank = max(best_rank, rank)
    if _YEAR_UNIT_RE.search(text):
        best_rank = max(best_rank, _GRANULARITY_RANK["year"])

    if best_rank == 0:
        return None
    return {v: k for k, v in _GRANULARITY_RANK.items()}[best_rank]


# ════════════════════════════════════════════════════════════════════════════════
# GT preparation
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GTDatePlan:
    """Deterministic GT-side extraction + per-item rubric flags."""

    gt_dates: list[CanonicalDate]
    required_granularity: str | None    # v2.1 override; None = Decision B
    rubric_version: str                 # "2.0" | "2.1"
    calendar_mismatch: bool             # some GT date is Bangla-calendar
    anchor_missing: bool                # relative_time_qa only
    offset_granularity_unparsed: bool   # relative_time_qa only


def prepare_gt(dataset: str, rec: dict) -> GTDatePlan:
    """Extract canonical GT dates from the record's answer text and derive the
    per-item rubric flags. relative_time_qa (rubric 2.1): GT dates still come
    from the ANSWER TEXT — never from seed_time+offset arithmetic (the
    dataset's offsets are approximate framing); seed_time/relative_offset only
    set the required granularity and the anchor context for the judge."""
    gt_dates = normalize_dates(rec.get("answer") or "", include_bare_years=False)
    calendar_mismatch = any(d.calendar == "bangla" for d in gt_dates)

    if dataset == "relative_time_qa":
        offset = rec.get("relative_offset")
        granularity = offset_granularity(offset)
        return GTDatePlan(
            gt_dates=gt_dates,
            required_granularity=granularity,
            rubric_version=RUBRIC_VERSION_RELATIVE_TIME,
            calendar_mismatch=calendar_mismatch,
            anchor_missing=not rec.get("seed_time"),
            offset_granularity_unparsed=granularity is None,
        )
    return GTDatePlan(
        gt_dates=gt_dates,
        required_granularity=None,
        rubric_version=RUBRIC_VERSION,
        calendar_mismatch=calendar_mismatch,
        anchor_missing=False,
        offset_granularity_unparsed=False,
    )


# ════════════════════════════════════════════════════════════════════════════════
# Judge atoms (typed view of the judge's validated JSON)
# ════════════════════════════════════════════════════════════════════════════════

_BASE_ATOM_KEYS = {"date_matches", "contradicting_dates_in_answer",
                   "meaning_correct", "meaning_rationale", "uncertain",
                   "confidence", "reasoning"}


def _coerce_confidence(value) -> float | None:
    """Coerce the judge's self-reported confidence to a float clamped to
    [0, 100]; None when absent or unusable (never a parse failure — confidence
    is logged-only and must not burn repair retries)."""
    try:
        c = float(value)
    except (TypeError, ValueError):
        return None
    if c != c:  # NaN
        return None
    return max(0.0, min(100.0, c))


@dataclass
class JudgeAtoms:
    """Validated atomic judgments. Booleans only — the judge never scores."""

    date_matched: list[bool]            # aligned to the canonical GT date list
    model_expressions: list[str]        # judge-quoted spans ("" if unmatched)
    contradicting_dates_in_answer: bool
    meaning_correct: bool
    meaning_rationale: str
    uncertain: bool
    confidence: float | None = None     # self-reported 0-100; logged only,
                                        # NEVER an input to any score
    extras: dict = field(default_factory=dict)  # per-type extension atoms
    vote_splits: dict | None = None             # set by the majority-vote wrapper

    @classmethod
    def from_json(cls, data: dict, n_gt_dates: int,
                  required_extras: tuple[str, ...] = ()) -> "JudgeAtoms":
        """Build from parsed judge JSON; raises ValueError on structural
        problems (missing/mistyped required booleans), which triggers the
        judge's repair retry. A date_matches entry with a bad/omitted gt_index
        is tolerated (that date counts unmatched) — the schema makes omissions
        rare, and treating them as parse failures would burn retries."""
        if not isinstance(data, dict):
            raise ValueError("judge response is not a JSON object")
        for key in ("contradicting_dates_in_answer", "meaning_correct", "uncertain"):
            if not isinstance(data.get(key), bool):
                raise ValueError(f"missing/non-boolean atom: {key}")
        raw_matches = data.get("date_matches")
        if not isinstance(raw_matches, list):
            raise ValueError("missing/non-list atom: date_matches")

        date_matched = [False] * n_gt_dates
        model_expressions = [""] * n_gt_dates
        for entry in raw_matches:
            if not isinstance(entry, dict):
                logger.warning("judge date_matches entry not an object: %r", entry)
                continue
            idx = entry.get("gt_index")
            if not isinstance(idx, int) or not (0 <= idx < n_gt_dates):
                logger.warning("judge date_matches entry with bad gt_index: %r", entry)
                continue
            date_matched[idx] = bool(entry.get("matched"))
            expr = entry.get("model_expression")
            model_expressions[idx] = expr if isinstance(expr, str) else ""

        extras = {k: v for k, v in data.items() if k not in _BASE_ATOM_KEYS}
        for key in required_extras:
            if not isinstance(extras.get(key), bool):
                raise ValueError(f"missing/non-boolean extension atom: {key}")

        return cls(
            date_matched=date_matched,
            model_expressions=model_expressions,
            contradicting_dates_in_answer=data["contradicting_dates_in_answer"],
            meaning_correct=data["meaning_correct"],
            meaning_rationale=str(data.get("meaning_rationale", "")).strip(),
            uncertain=data["uncertain"],
            confidence=_coerce_confidence(data.get("confidence")),
            extras=extras,
        )


# ════════════════════════════════════════════════════════════════════════════════
# Match combination (deterministic first, judge as fallback matcher)
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class DateMatch:
    """One GT date's match outcome, with provenance."""

    gt: CanonicalDate
    matched: bool
    match_source: str                   # "deterministic" | "judge" | "none"
    model_expression: str | None
    deterministic_matched: bool
    judge_matched: bool | None          # None when no judge atoms were available

    def to_dict(self) -> dict:
        return {
            "gt_date": self.gt.canonical(),
            "gt_calendar": self.gt.calendar,
            "matched": self.matched,
            "match_source": self.match_source,
            "model_expression": self.model_expression,
            "deterministic_matched": self.deterministic_matched,
            "judge_matched": self.judge_matched,
        }


def combine_matches(gt_dates: list[CanonicalDate],
                    model_dates: list[CanonicalDate],
                    atoms: JudgeAtoms | None,
                    required_granularity: str | None = None) -> list[DateMatch]:
    """matched_i = deterministic_i OR judge_i, provenance recorded.

    Deterministic matching runs first; the judge is the fallback matcher for
    expressions code cannot resolve (spelled-out ordinals, mixed script,
    relative phrasing, Bangla-calendar GT dates). atoms=None (judge-error
    path) degrades to deterministic-only."""
    matches: list[DateMatch] = []
    for i, gt in enumerate(gt_dates):
        det = date_satisfied(gt, model_dates, required_granularity)
        jm: bool | None = None
        expr: str | None = None
        if atoms is not None and i < len(atoms.date_matched):
            jm = atoms.date_matched[i]
            expr = atoms.model_expressions[i] or None
        matched = det or bool(jm)
        source = "deterministic" if det else ("judge" if jm else "none")
        matches.append(DateMatch(
            gt=gt, matched=matched, match_source=source,
            model_expression=expr, deterministic_matched=det, judge_matched=jm,
        ))
    return matches


# ════════════════════════════════════════════════════════════════════════════════
# Scoring (all arithmetic lives here; the judge never scores)
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class ScoreResult:
    final_score: float
    date_score: float | None            # None when n_gt_dates == 0
    n_gt_dates: int
    n_matched: int
    meaning_correct_effective: bool     # after Decision C forcing
    contradiction_forced_meaning_false: bool
    hard_gate_applied: str | None       # "single_date_unsatisfied" | "not_localized" | None
    date_matches: list[DateMatch]

    def to_record_fields(self) -> dict:
        """Per-item subscore fields for results.json."""
        return {
            "final_score": self.final_score,
            "date_score": self.date_score,
            "n_gt_dates": self.n_gt_dates,
            "n_matched": self.n_matched,
            "meaning_correct": self.meaning_correct_effective,
            "contradiction_forced_meaning_false":
                self.contradiction_forced_meaning_false,
            "hard_gate_applied": self.hard_gate_applied,
            "date_matches": [m.to_dict() for m in self.date_matches],
        }


def score_base(gt_dates: list[CanonicalDate], matches: list[DateMatch],
               atoms: JudgeAtoms) -> ScoreResult:
    """Base rubric (Decisions A, C, E). n = len(gt_dates), m = matched count.

      n=0 : meaning ? 1.0 : 0.0
      n=1 : matched ? (meaning ? 1.0 : 0.5) : 0.0   -- date is a hard gate
      n>=2: 0.5*(m/n) + (0.5 iff m == n AND meaning) -- all-or-nothing gate (A)

    meaning := meaning_correct AND NOT contradicting_dates_in_answer (C).
    Range endpoints are ordinary list members, so a range weighs 2 and both
    endpoints must match for the meaning half (E holds by construction).
    Extra correct, non-contradicting model dates are never penalized."""
    n = len(gt_dates)
    m = sum(1 for x in matches if x.matched)
    forced = atoms.meaning_correct and atoms.contradicting_dates_in_answer
    meaning = atoms.meaning_correct and not atoms.contradicting_dates_in_answer

    hard_gate = None
    if n == 0:
        final: float = 1.0 if meaning else 0.0
        date_score: float | None = None
    elif n == 1:
        if m == 1:
            final = 1.0 if meaning else 0.5
        else:
            final = 0.0
            hard_gate = "single_date_unsatisfied"
        date_score = 0.5 * m
    else:
        date_score = 0.5 * (m / n)
        final = date_score + (0.5 if (m == n and meaning) else 0.0)

    return ScoreResult(
        final_score=round(final, 4),
        date_score=round(date_score, 4) if date_score is not None else None,
        n_gt_dates=n,
        n_matched=m,
        meaning_correct_effective=meaning,
        contradiction_forced_meaning_false=forced,
        hard_gate_applied=hard_gate,
        date_matches=matches,
    )


def score_perturbation_explanation(gt_dates: list[CanonicalDate],
                                   matches: list[DateMatch],
                                   atoms: JudgeAtoms) -> ScoreResult:
    """Perturbation step 2: base rubric gated on localization. Saying
    "something is off" without identifying the specific perturbed element is a
    0. correction_correct is logged by the caller but NEVER scored in v2."""
    result = score_base(gt_dates, matches, atoms)
    if not atoms.extras.get("perturbation_localized"):
        result.final_score = 0.0
        result.hard_gate_applied = "not_localized"
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Custom BLEU (judge-sanity similarity metric — diagnostic only, never scored)
#
# Standard BLEU (Papineni et al., 2002): clipped modified n-gram precision,
# equal-weight cumulative geometric mean, brevity penalty — implemented here
# without NLTK/sacrebleu so it stays pure and unit-testable.
#
# Primary metric = cumulative BLEU-2. Rationale: answers are 1-4 sentences
# against a SINGLE reference, and legal paraphrase rarely reproduces exact
# 3-/4-grams, so BLEU-4 collapses toward 0 on short answers and loses
# discrimination; BLEU-1 ignores word order entirely. BLEU-2 (unigram content
# overlap + bigram local ordering) is the standard compromise for short-answer
# QA. BLEU-1..4 are all returned for transparency.
# ════════════════════════════════════════════════════════════════════════════════

BLEU_PRIMARY_N = 2
_BLEU_EPSILON = 1e-9  # smoothing: zero precisions get epsilon, not 0
                      # (Chen & Cherry 2014, method 1) so short texts
                      # don't zero the geometric mean outright

_BLEU_TOKEN_RE = re.compile(r"[0-9]+|[\wঀ-৿]+", re.UNICODE)


def bleu_tokenize(text: str) -> list[str]:
    """Lowercased word tokens; the Bengali block is included explicitly
    because \\w splits Bengali words at combining vowel marks. Punctuation is
    dropped, so 'S.R.O. No. 131-AIN/2017' -> ['s','r','o','no','131','ain','2017']."""
    return [t.lower() for t in _BLEU_TOKEN_RE.findall(text or "")]


def _ngram_counts(tokens: list[str], n: int) -> dict[tuple, int]:
    counts: dict[tuple, int] = {}
    for i in range(len(tokens) - n + 1):
        g = tuple(tokens[i:i + n])
        counts[g] = counts.get(g, 0) + 1
    return counts


def bleu_score(reference: str, candidate: str, max_n: int = 4) -> dict:
    """BLEU of `candidate` against a single `reference`.

    Returns {"bleu": <cumulative BLEU-2, the primary metric>,
             "bleu_by_n": {"1"..."4": cumulative BLEU-n},
             "brevity_penalty", "candidate_tokens", "reference_tokens"}.
    Modified precision is clipped per n-gram (a repeated word cannot inflate
    precision beyond its reference count). Empty candidate -> all zeros."""
    ref = bleu_tokenize(reference)
    cand = bleu_tokenize(candidate)
    if not cand or not ref:
        return {"bleu": 0.0,
                "bleu_by_n": {str(n): 0.0 for n in range(1, max_n + 1)},
                "brevity_penalty": 0.0,
                "candidate_tokens": len(cand), "reference_tokens": len(ref)}

    bp = 1.0 if len(cand) >= len(ref) else math.exp(1.0 - len(ref) / len(cand))

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        cand_counts = _ngram_counts(cand, n)
        total = sum(cand_counts.values())
        if total == 0:  # candidate shorter than n
            precisions.append(0.0)
            continue
        ref_counts = _ngram_counts(ref, n)
        clipped = sum(min(c, ref_counts.get(g, 0))
                      for g, c in cand_counts.items())
        precisions.append(clipped / total)

    by_n: dict[str, float] = {}
    for n in range(1, max_n + 1):
        smoothed = [p if p > 0 else _BLEU_EPSILON for p in precisions[:n]]
        geo_mean = math.exp(sum(math.log(p) for p in smoothed) / n)
        by_n[str(n)] = round(bp * geo_mean, 4)

    return {"bleu": by_n[str(min(BLEU_PRIMARY_N, max_n))],
            "bleu_by_n": by_n,
            "brevity_penalty": round(bp, 4),
            "candidate_tokens": len(cand), "reference_tokens": len(ref)}
