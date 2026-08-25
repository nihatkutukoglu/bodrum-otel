"""Auditable, minimal cleaning helpers for the Şikayetvar complaint corpus."""

from __future__ import annotations

import hashlib
import html
import math
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd


TURKISH_MONTHS = {
    "ocak": 1,
    "şubat": 2,
    "mart": 3,
    "nisan": 4,
    "mayıs": 5,
    "haziran": 6,
    "temmuz": 7,
    "ağustos": 8,
    "eylül": 9,
    "ekim": 10,
    "kasım": 11,
    "aralık": 12,
}
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"\s+")
DATE_RE = re.compile(
    r"^(?P<day>\d{1,2})\s+"
    r"(?P<month>Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)"
    r"(?:\s+(?P<year>\d{4}))?"
    r"(?:\s+(?P<hour>\d{1,2}):(?P<minute>\d{2}))?$",
    flags=re.IGNORECASE,
)


def is_missing(value: Any) -> bool:
    """Return True for pandas/NumPy missing values and blank strings."""

    if value is None or value is pd.NA:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and not value.strip()


def clean_raw_text_minimal(value: Any) -> Any:
    """Clean transport artifacts without altering linguistic content."""

    if is_missing(value):
        return pd.NA
    text = unicodedata.normalize("NFKC", html.unescape(str(value)))
    text = ZERO_WIDTH_RE.sub("", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text if text else pd.NA


def normalize_for_duplicate(value: Any) -> str:
    """Create a comparison-only representation; never replace the raw text with it."""

    cleaned = clean_raw_text_minimal(value)
    if is_missing(cleaned):
        return ""
    text = str(cleaned).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return WHITESPACE_RE.sub(" ", text).strip()


def parse_nullable_bool(value: Any) -> Any:
    """Parse common boolean representations while preserving missingness."""

    if is_missing(value):
        return pd.NA
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "evet", "var"}:
        return True
    if normalized in {"false", "0", "no", "hayır", "yok"}:
        return False
    return pd.NA


def parse_numeric_count(value: Any) -> Any:
    """Parse Turkish/international count formats without converting missing to zero."""

    if is_missing(value):
        return pd.NA
    text = unicodedata.normalize("NFKC", str(value)).strip().upper()
    text = text.replace("\u00a0", "").replace(" ", "")
    multiplier = 1.0
    suffix_match = re.search(r"(B|K|M)$", text)
    if suffix_match:
        suffix = suffix_match.group(1)
        multiplier = {"B": 1_000.0, "K": 1_000.0, "M": 1_000_000.0}[suffix]
        text = text[:-1]
    if not text:
        return pd.NA

    if multiplier != 1.0:
        normalized = text.replace(".", "").replace(",", ".") if "," in text else text
    elif re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+", text):
        normalized = text.replace(".", "")
    elif re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", text):
        normalized = text.replace(",", "")
    else:
        normalized = text.replace(",", ".")

    try:
        number = float(normalized) * multiplier
    except ValueError:
        return pd.NA
    if not math.isfinite(number) or number < 0:
        return pd.NA
    return int(round(number))


def classify_date_pattern(value: Any) -> str:
    """Classify the raw date representation before parsing."""

    if is_missing(value):
        return "MISSING"
    text = str(value).strip()
    match = DATE_RE.fullmatch(text)
    if match:
        return "DAY_MONTH_YEAR_TIME" if match.group("year") else "DAY_MONTH_TIME_NO_YEAR"
    relative = text.casefold()
    if re.fullmatch(r"bugün(?:\s+\d{1,2}:\d{2})?", relative):
        return "RELATIVE_TODAY"
    if re.fullmatch(r"dün(?:\s+\d{1,2}:\d{2})?", relative):
        return "RELATIVE_YESTERDAY"
    if re.fullmatch(r"\d+\s+gün\s+önce", relative):
        return "RELATIVE_DAYS_AGO"
    return "UNRECOGNIZED"


def parse_sikayetvar_date(value: Any, reference_date: Any) -> tuple[pd.Timestamp, bool, str]:
    """Deterministically parse Turkish absolute or limited relative dates."""

    pattern = classify_date_pattern(value)
    if pattern == "MISSING":
        return pd.NaT, False, pattern
    reference = pd.to_datetime(reference_date, errors="coerce", utc=True)
    if pd.isna(reference):
        reference = pd.Timestamp.utcnow()
    reference = reference.tz_convert(None) if reference.tzinfo is not None else reference
    text = str(value).strip()

    match = DATE_RE.fullmatch(text)
    if match:
        day = int(match.group("day"))
        month = TURKISH_MONTHS[match.group("month").casefold()]
        hour = int(match.group("hour") or 0)
        minute = int(match.group("minute") or 0)
        approximate = match.group("year") is None
        year = int(match.group("year") or reference.year)
        try:
            parsed = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
        except ValueError:
            return pd.NaT, approximate, "INVALID_CALENDAR_DATE"
        if approximate and parsed > reference + pd.Timedelta(days=1):
            parsed = parsed.replace(year=year - 1)
        return parsed, approximate, pattern

    folded = text.casefold()
    if pattern in {"RELATIVE_TODAY", "RELATIVE_YESTERDAY"}:
        base = reference.normalize()
        if pattern == "RELATIVE_YESTERDAY":
            base -= pd.Timedelta(days=1)
        time_match = re.search(r"(\d{1,2}):(\d{2})", folded)
        if time_match:
            base += pd.Timedelta(hours=int(time_match.group(1)), minutes=int(time_match.group(2)))
        return base, True, pattern
    if pattern == "RELATIVE_DAYS_AGO":
        days = int(re.match(r"\d+", folded).group())
        return reference.normalize() - pd.Timedelta(days=days), True, pattern
    return pd.NaT, False, pattern


def determine_very_short_threshold(word_counts: pd.Series, cap: int = 40) -> int:
    """Use the observed lower tail while keeping the audit flag conservative."""

    valid = pd.to_numeric(word_counts, errors="coerce").dropna()
    if valid.empty:
        return cap
    return max(1, min(cap, int(math.floor(valid.quantile(0.05)))))


def build_reply_id(canonical_url: Any, reply_order: Any, reply_text: Any) -> str:
    """Build a stable reply key from URL, order and minimally normalized text."""

    payload = "|".join(
        [str(canonical_url or "").strip(), str(reply_order or "").strip(), normalize_for_duplicate(reply_text)]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def standardize_author_type(value: Any) -> str:
    """Map reply authors to COMPANY, USER or UNKNOWN."""

    if is_missing(value):
        return "UNKNOWN"
    normalized = str(value).strip().casefold()
    if normalized in {"company", "firma", "marka", "brand"}:
        return "COMPANY"
    if normalized in {"user", "kullanıcı", "customer", "müşteri"}:
        return "USER"
    return "UNKNOWN"


def exact_duplicate_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Return row-level evidence for the three explicit exact-duplicate definitions."""

    records: list[dict[str, Any]] = []
    definitions = {
        "canonical_complaint_url": ["canonical_complaint_url"],
        "complaint_id": ["complaint_id"],
        "title_text_hotel": ["complaint_title", "complaint_text", "hotel_id"],
    }
    for duplicate_type, columns in definitions.items():
        valid_mask = frame[columns].notna().all(axis=1)
        duplicate_mask = valid_mask & frame.duplicated(columns, keep=False)
        subset = frame.loc[duplicate_mask].copy()
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
                    "complaint_id": row.get("complaint_id"),
                    "canonical_complaint_url": row.get("canonical_complaint_url"),
                    "hotel_id": row.get("hotel_id"),
                    "hotel_name": row.get("hotel_name"),
                    "entity_match_status": row.get("entity_match_status"),
                    "occurrence_rank": int(row["_occurrence_rank"]),
                }
            )
    return pd.DataFrame(
        records,
        columns=[
            "duplicate_type",
            "duplicate_key",
            "row_index",
            "complaint_id",
            "canonical_complaint_url",
            "hotel_id",
            "hotel_name",
            "entity_match_status",
            "occurrence_rank",
        ],
    )


def near_duplicate_candidates(frame: pd.DataFrame, threshold: float = 0.94) -> pd.DataFrame:
    """Find high-similarity same-hotel text pairs without automatically dropping them."""

    work = frame.copy()
    work["_comparison_text"] = (
        work["complaint_title"].map(normalize_for_duplicate)
        + " "
        + work["complaint_text"].map(normalize_for_duplicate)
    ).str.strip()
    records: list[dict[str, Any]] = []
    for hotel_id, group in work.groupby("hotel_id", dropna=False):
        rows = list(group.iterrows())
        for left_index in range(len(rows)):
            idx_a, row_a = rows[left_index]
            text_a = row_a["_comparison_text"]
            if not text_a:
                continue
            for right_index in range(left_index + 1, len(rows)):
                idx_b, row_b = rows[right_index]
                if row_a.get("canonical_complaint_url") == row_b.get("canonical_complaint_url"):
                    continue
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
                            "hotel_id": hotel_id,
                            "hotel_name": row_a.get("hotel_name"),
                            "row_index_a": idx_a,
                            "row_index_b": idx_b,
                            "complaint_id_a": row_a.get("complaint_id"),
                            "complaint_id_b": row_b.get("complaint_id"),
                            "canonical_url_a": row_a.get("canonical_complaint_url"),
                            "canonical_url_b": row_b.get("canonical_complaint_url"),
                            "similarity": similarity,
                            "action": "REVIEW_ONLY",
                        }
                    )
    return pd.DataFrame(
        records,
        columns=[
            "hotel_id",
            "hotel_name",
            "row_index_a",
            "row_index_b",
            "complaint_id_a",
            "complaint_id_b",
            "canonical_url_a",
            "canonical_url_b",
            "similarity",
            "action",
        ],
    )


def prepare_complaints(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Add minimal clean fields, parsed values and transparent quality flags."""

    result = frame.copy(deep=True)
    text_columns = [
        "complaint_title",
        "complaint_text",
        "company_response_text",
        "progress_text",
    ]
    for column in text_columns:
        if column not in result:
            result[column] = pd.NA
        result[f"{column}_clean"] = result[column].map(clean_raw_text_minimal)

    length_sources = {
        "complaint": "complaint_text_clean",
        "title": "complaint_title_clean",
        "company_response": "company_response_text_clean",
        "progress": "progress_text_clean",
    }
    for prefix, column in length_sources.items():
        result[f"{prefix}_char_count"] = result[column].map(
            lambda value: len(str(value)) if not is_missing(value) else pd.NA
        ).astype("Int64")
        result[f"{prefix}_word_count"] = result[column].map(
            lambda value: len(str(value).split()) if not is_missing(value) else pd.NA
        ).astype("Int64")

    threshold = determine_very_short_threshold(result["complaint_word_count"])
    result["complaint_text_missing_flag"] = result["complaint_text_clean"].isna()
    result["complaint_title_missing_flag"] = result["complaint_title_clean"].isna()
    result["complaint_text_very_short_flag"] = (
        result["complaint_word_count"].notna() & result["complaint_word_count"].le(threshold)
    )
    result["missing_text_flag"] = result["complaint_text_missing_flag"]
    result["missing_title_flag"] = result["complaint_title_missing_flag"]
    result["very_short_text_flag"] = result["complaint_text_very_short_flag"]

    normalized_text = result["complaint_text_clean"].map(normalize_for_duplicate)
    repeated_text = normalized_text.ne("") & normalized_text.duplicated(keep=False)
    boilerplate_pattern = r"devamını oku|marka profilini gör|şikayetin çözümü"
    result["complaint_text_possible_boilerplate_flag"] = repeated_text | result[
        "complaint_text_clean"
    ].astype("string").str.contains(boilerplate_pattern, case=False, regex=True, na=False)
    result["complaint_text_html_artifact_flag"] = result["complaint_text"].astype("string").str.contains(
        r"<[^>]+>|&(?:nbsp|amp|quot|lt|gt);", case=False, regex=True, na=False
    )
    result["complaint_text_encoding_artifact_flag"] = result["complaint_text"].astype("string").str.contains(
        r"�|Ã.|Â.", regex=True, na=False
    )

    parsed_collected = pd.to_datetime(result.get("collected_at"), errors="coerce", utc=True)
    result["collected_at_parsed"] = parsed_collected.dt.tz_convert(None)
    parsed_rows = [
        parse_sikayetvar_date(value, reference)
        for value, reference in zip(result.get("complaint_date_raw"), result["collected_at_parsed"])
    ]
    result["complaint_date"] = pd.to_datetime([row[0] for row in parsed_rows], errors="coerce")
    result["complaint_date_is_approximate"] = pd.Series([row[1] for row in parsed_rows], dtype="boolean")
    result["complaint_date_pattern"] = [row[2] for row in parsed_rows]
    result["complaint_date_parse_failed_flag"] = result["complaint_date"].isna()
    result["date_parse_failed_flag"] = result["complaint_date_parse_failed_flag"]
    result["date_is_approximate_flag"] = result["complaint_date_is_approximate"]
    result["complaint_date_future_flag"] = (
        result["complaint_date"].notna()
        & result["collected_at_parsed"].notna()
        & result["complaint_date"].gt(result["collected_at_parsed"] + pd.Timedelta(days=1))
    )
    result["complaint_year"] = result["complaint_date"].dt.year.astype("Int64")
    result["complaint_month"] = result["complaint_date"].dt.month.astype("Int64")
    result["complaint_year_month"] = result["complaint_date"].dt.strftime("%Y-%m").astype("string")

    for raw_column, parsed_column in [
        ("company_response_date", "company_response_date_parsed"),
        ("progress_date", "progress_date_parsed"),
    ]:
        parsed = [
            parse_sikayetvar_date(value, reference)[0]
            for value, reference in zip(result.get(raw_column), result["collected_at_parsed"])
        ]
        result[parsed_column] = pd.to_datetime(parsed, errors="coerce")

    result["view_count_numeric"] = result.get("view_count").map(parse_numeric_count).astype("Int64")
    result["support_count_numeric"] = result.get("support_count").map(parse_numeric_count).astype("Int64")
    result["user_reply_count_numeric"] = result.get("user_reply_count").map(parse_numeric_count).astype("Int64")
    result["view_count_parse_failed_flag"] = result.get("view_count").map(lambda value: not is_missing(value)) & result[
        "view_count_numeric"
    ].isna()
    result["company_response_exists_clean"] = result.get("company_response_exists").map(parse_nullable_bool).astype(
        "boolean"
    )
    result["progress_exists_clean"] = result.get("progress_exists").map(parse_nullable_bool).astype("boolean")

    pii_pattern = re.compile(
        r"(?:[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})|(?:\+?\d[\d\s().-]{8,}\d)",
        flags=re.IGNORECASE,
    )
    result["potential_pii_flag"] = result["complaint_text_clean"].map(
        lambda value: bool(pii_pattern.search(str(value))) if not is_missing(value) else False
    )
    return result, threshold


def prepare_replies(frame: pd.DataFrame, complaint_reference_dates: pd.Series) -> pd.DataFrame:
    """Clean reply text, dates, author labels and stable identifiers."""

    result = frame.copy(deep=True)
    result["reply_text_clean"] = result.get("reply_text").map(clean_raw_text_minimal)
    result["reply_author_type_clean"] = result.get("reply_author_type").map(standardize_author_type)
    result["reply_order_numeric"] = result.get("reply_order").map(parse_numeric_count).astype("Int64")
    reference = result.get("canonical_complaint_url").map(complaint_reference_dates)
    parsed = [parse_sikayetvar_date(value, ref) for value, ref in zip(result.get("reply_date_raw"), reference)]
    result["reply_date"] = pd.to_datetime([row[0] for row in parsed], errors="coerce")
    result["reply_date_is_approximate"] = pd.Series([row[1] for row in parsed], dtype="boolean")
    result["reply_date_pattern"] = [row[2] for row in parsed]
    result["reply_date_parse_failed_flag"] = result["reply_date"].isna()
    result["reply_id"] = [
        build_reply_id(url, order, text)
        for url, order, text in zip(
            result.get("canonical_complaint_url"), result["reply_order_numeric"], result.get("reply_text")
        )
    ]
    return result


def derive_reply_metrics(replies: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned replies to complaint-level audit metrics."""

    if replies.empty:
        return pd.DataFrame(
            columns=[
                "canonical_complaint_url",
                "reply_count_total_derived",
                "reply_count_company_derived",
                "reply_count_user_derived",
                "reply_count_unknown_derived",
                "first_reply_date",
                "last_reply_date",
                "company_reply_exists_derived",
            ]
        )
    base = replies.groupby("canonical_complaint_url", dropna=False).agg(
        reply_count_total_derived=("reply_id", "size"),
        first_reply_date=("reply_date", "min"),
        last_reply_date=("reply_date", "max"),
    )
    author_counts = (
        replies.pivot_table(
            index="canonical_complaint_url",
            columns="reply_author_type_clean",
            values="reply_id",
            aggfunc="size",
            fill_value=0,
        )
        .reindex(columns=["COMPANY", "USER", "UNKNOWN"], fill_value=0)
        .rename(
            columns={
                "COMPANY": "reply_count_company_derived",
                "USER": "reply_count_user_derived",
                "UNKNOWN": "reply_count_unknown_derived",
            }
        )
    )
    result = base.join(author_counts).reset_index()
    result["company_reply_exists_derived"] = result["reply_count_company_derived"].gt(0)
    return result
