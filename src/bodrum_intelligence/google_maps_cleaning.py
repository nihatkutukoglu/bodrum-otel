"""Auditable, minimal cleaning helpers for the Google Travel (Google Maps-aggregated) review corpus."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from difflib import SequenceMatcher

from bodrum_intelligence.sikayetvar_cleaning import (  # reuse: generic, not complaint-specific
    clean_raw_text_minimal,
    determine_very_short_threshold,
    is_missing,
    normalize_for_duplicate,
)

__all__ = [
    "clean_raw_text_minimal",
    "is_missing",
    "normalize_for_duplicate",
    "determine_very_short_threshold",
    "parse_review_rating",
    "rating_group",
    "RELATIVE_DATE_RE",
    "parse_relative_review_date",
    "build_review_id",
    "exact_duplicate_audit",
]

RATING_RAW_RE = re.compile(r"^\s*(?P<num>\d+(?:[.,]\d+)?)\s*/\s*5\s*$")

RELATIVE_DATE_RE = re.compile(
    r"^(?:(?P<platform>Google|Tripadvisor|Trip\.com)\s+üzerinde\s+)?"
    r"(?P<qty>bir|\d+)\s+(?P<unit>dakika|saat|gün|hafta|ay|yıl)\s+önce"
    r"(?P<edited>\s+düzenlendi)?\s*$",
    flags=re.IGNORECASE,
)

_UNIT_DAYS = {"dakika": 0, "saat": 0, "gün": 1, "hafta": 7, "ay": 30, "yıl": 365}
_UNIT_MINUTES = {"dakika": 1, "saat": 60}


def parse_review_rating(value: Any) -> Any:
    """Parse an 'N/5' rating string into a validated 1-5 int; NaN if out of range or unparsable."""

    if is_missing(value):
        return pd.NA
    match = RATING_RAW_RE.match(str(value))
    if not match:
        return pd.NA
    number = float(match.group("num").replace(",", "."))
    if not (1 <= number <= 5):
        return pd.NA
    return int(round(number))


def rating_group(rating: Any) -> Any:
    """Map a validated 1-5 rating to LOW (1-2) / MIXED (3) / HIGH (4-5)."""

    if is_missing(rating):
        return pd.NA
    rating = int(rating)
    if rating <= 2:
        return "LOW"
    if rating == 3:
        return "MIXED"
    return "HIGH"


def parse_relative_review_date(raw: Any, collected_at: Any) -> tuple[Any, bool, bool, str]:
    """Approximate an absolute date from a Turkish relative expression anchored at collection time.

    Returns (review_date, is_approximate, is_edited, parse_status).
    Only relative expressions were observed in this corpus (README: platform aggregates
    the rendered "N <unit> önce" string, never an absolute calendar date), so no absolute-date
    branch is implemented here.
    """

    if is_missing(raw):
        return pd.NaT, False, False, "MISSING"
    reference = pd.to_datetime(collected_at, errors="coerce", utc=True)
    if pd.isna(reference):
        return pd.NaT, False, False, "MISSING_REFERENCE"
    reference = reference.tz_convert(None)

    match = RELATIVE_DATE_RE.match(str(raw).strip())
    if not match:
        return pd.NaT, False, False, "UNRECOGNIZED"

    qty = 1 if match.group("qty").casefold() == "bir" else int(match.group("qty"))
    unit = match.group("unit").casefold()
    is_edited = bool(match.group("edited"))

    if unit in _UNIT_MINUTES:
        delta = pd.Timedelta(minutes=qty) if unit == "dakika" else pd.Timedelta(hours=qty)
    else:
        delta = pd.Timedelta(days=_UNIT_DAYS[unit] * qty)

    parsed = (reference - delta).normalize()
    return parsed, True, is_edited, "RELATIVE_PARSED"


def build_review_id(review_hash: Any) -> Any:
    """Derive a stable, prefixed review_id from the collector's existing content hash."""

    if is_missing(review_hash):
        return pd.NA
    return f"gm_{str(review_hash)[:16]}"


def exact_duplicate_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Row-level evidence for exact duplicates by review_id and by (hotel + normalized text)."""

    records: list[dict[str, Any]] = []
    definitions = {
        "review_id": ["review_id"],
        "hotel_text": ["hotel_name", "_normalized_text"],
    }
    work = frame.copy()
    work["_normalized_text"] = work["review_text"].map(normalize_for_duplicate)
    for duplicate_type, columns in definitions.items():
        valid_mask = work[columns].notna().all(axis=1) & (work["_normalized_text"] != "")
        duplicate_mask = valid_mask & work.duplicated(columns, keep=False)
        subset = work.loc[duplicate_mask].copy()
        if subset.empty:
            continue
        subset["_duplicate_key"] = subset[columns].astype(str).agg(" | ".join, axis=1)
        subset["_occurrence_rank"] = subset.groupby("_duplicate_key").cumcount() + 1
        for row_index, row in subset.iterrows():
            records.append(
                {
                    "duplicate_type": duplicate_type,
                    "duplicate_key": row["_duplicate_key"],
                    "row_index": row_index,
                    "review_id": row.get("review_id"),
                    "hotel_name": row.get("hotel_name"),
                    "occurrence_rank": int(row["_occurrence_rank"]),
                }
            )
    return pd.DataFrame(
        records,
        columns=["duplicate_type", "duplicate_key", "row_index", "review_id", "hotel_name", "occurrence_rank"],
    )


def near_duplicate_candidates(frame: pd.DataFrame, threshold: float = 0.94) -> pd.DataFrame:
    """Find high-similarity same-hotel text pairs without automatically dropping them."""

    work = frame.copy()
    work["_comparison_text"] = work["review_text"].map(normalize_for_duplicate)
    records: list[dict[str, Any]] = []
    for hotel_name, group in work.groupby("hotel_name", dropna=False):
        rows = list(group.iterrows())
        for left in range(len(rows)):
            idx_a, row_a = rows[left]
            text_a = row_a["_comparison_text"]
            if not text_a:
                continue
            for right in range(left + 1, len(rows)):
                idx_b, row_b = rows[right]
                text_b = row_b["_comparison_text"]
                if not text_b:
                    continue
                length_ratio = min(len(text_a), len(text_b)) / max(len(text_a), len(text_b))
                if length_ratio < 0.75:
                    continue
                matcher = SequenceMatcher(None, text_a, text_b, autojunk=False)
                if matcher.quick_ratio() < threshold:
                    continue
                similarity = matcher.ratio()
                if similarity >= threshold:
                    records.append(
                        {
                            "hotel_name": hotel_name,
                            "row_index_a": idx_a,
                            "row_index_b": idx_b,
                            "review_id_a": row_a.get("review_id"),
                            "review_id_b": row_b.get("review_id"),
                            "similarity": round(similarity, 4),
                            "action": "REVIEW_ONLY",
                        }
                    )
    return pd.DataFrame(
        records,
        columns=["hotel_name", "row_index_a", "row_index_b", "review_id_a", "review_id_b", "similarity", "action"],
    )
