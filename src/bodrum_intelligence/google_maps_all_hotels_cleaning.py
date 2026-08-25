"""Cleaning and audit helpers isolated from the existing five-hotel case study."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from bodrum_intelligence.google_maps_all_hotels_collection import RAW_FIELDS, rating_group, stable_review_id
from bodrum_intelligence.google_maps_cleaning import parse_relative_review_date

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?90\s*)?(?:0?5\d{2})[\s().-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)")


def minimal_clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\u200b", "")).strip()


def potential_pii(value: Any) -> bool:
    text = "" if value is None or pd.isna(value) else str(value)
    return bool(EMAIL_RE.search(text) or PHONE_RE.search(text))


def clean_reviews(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = raw.copy().reindex(columns=RAW_FIELDS)
    input_n = len(work)
    work["review_rating"] = pd.to_numeric(work["review_rating"], errors="coerce").astype("Int64")
    invalid_rating = work["review_rating"].notna() & ~work["review_rating"].between(1, 5)
    work.loc[invalid_rating, "review_rating"] = pd.NA
    work["rating_group"] = work["review_rating"].map(rating_group)
    work["review_text"] = work["review_text"].fillna("").astype(str)
    work["review_text_clean"] = work["review_text"].map(minimal_clean)
    work["is_rating_only"] = work["review_rating"].notna() & work["review_text_clean"].eq("")
    work["potential_pii_flag"] = work["review_text"].map(potential_pii)
    work["review_char_count"] = work["review_text_clean"].str.len()
    work["review_word_count"] = work["review_text_clean"].str.split().str.len().fillna(0).astype(int)
    parsed_dates = work.apply(
        lambda row: parse_relative_review_date(row.get("review_date_raw"), row.get("collected_at")), axis=1
    )
    work["review_date"] = parsed_dates.map(lambda value: value[0])
    work["review_date_is_approximate"] = parsed_dates.map(lambda value: value[1])
    work["review_is_edited"] = parsed_dates.map(lambda value: value[2])
    work["review_date_parse_status"] = parsed_dates.map(lambda value: value[3])
    missing_id = work["review_id"].isna() | work["review_id"].astype(str).str.strip().eq("")
    work.loc[missing_id, "review_id"] = work.loc[missing_id].apply(
        lambda row: stable_review_id(str(row.hotel_id), row.review_rating, "" if pd.isna(row.review_date_raw) else str(row.review_date_raw), row.review_text_clean), axis=1
    )
    duplicate_mask = work.duplicated("review_id", keep="first")
    clean = work.loc[~duplicate_mask].copy().reset_index(drop=True)
    clean = clean.drop(columns=["reviewer_name_raw"], errors="ignore")
    audit = {
        "input_row_n": input_n, "clean_row_n": len(clean), "exact_duplicate_n": int(duplicate_mask.sum()),
        "invalid_rating_n": int(invalid_rating.sum()), "rating_only_n": int(clean["is_rating_only"].sum()),
        "text_review_n": int(clean["review_text_clean"].ne("").sum()),
        "potential_pii_n": int(clean["potential_pii_flag"].sum()),
        "date_parsed_n": int(clean["review_date"].notna().sum()),
    }
    return clean, audit


def sample_readiness(clean: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hotel in targets.itertuples(index=False):
        group = clean[clean["hotel_id"].astype(str) == str(hotel.hotel_id)]
        text = group[group["review_text_clean"].ne("") & ~group["is_rating_only"]]
        counts = text["rating_group"].value_counts()
        high, low, mixed = int(counts.get("HIGH", 0)), int(counts.get("LOW", 0)), int(counts.get("MIXED", 0))
        rows.append({
            "hotel_id": hotel.hotel_id, "hotel_name": hotel.hotel_name, "area": hotel.area,
            "total_review_n": len(group), "text_review_n": len(text), "low_n": low, "mixed_n": mixed, "high_n": high,
            "general_nlp_ready": len(text) >= 15, "positive_driver_ready": high >= 10,
            "negative_driver_ready": low >= 10, "full_driver_ready": high >= 10 and low >= 10,
        })
    return pd.DataFrame(rows)


def run_cleaning(raw_path: Path, targets_path: Path, processed_dir: Path, reports_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(raw_path, dtype=str) if raw_path.exists() and raw_path.stat().st_size else pd.DataFrame(columns=RAW_FIELDS)
    targets = pd.read_csv(targets_path)
    clean, audit = clean_reviews(raw)
    readiness = sample_readiness(clean, targets)
    processed_dir.mkdir(parents=True, exist_ok=True); reports_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(processed_dir / "google_maps_all_hotels_reviews_clean.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([audit]).to_csv(reports_dir / "google_maps_all_hotels_cleaning_summary.csv", index=False, encoding="utf-8-sig")
    readiness.to_csv(reports_dir / "google_maps_all_hotels_sample_readiness.csv", index=False, encoding="utf-8-sig")
    (reports_dir / "google_maps_all_hotels_cleaning_key_findings.txt").write_text(
        f"Clean reviews: {audit['clean_row_n']}\nText reviews: {audit['text_review_n']}\nExact duplicates removed: {audit['exact_duplicate_n']}\n",
        encoding="utf-8",
    )
    (reports_dir / "google_maps_all_hotels_cleaning_limitations.txt").write_text(
        "Public rendered samples are capped and are not representative probability samples. Relative dates remain raw when no safe deterministic parse is available. Reviewer names are excluded from processed output use.\n",
        encoding="utf-8",
    )
    return clean, readiness, audit
