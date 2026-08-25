#!/usr/bin/env python3
"""Build targets, safely collect public Google Maps reviews, and run clean/NLP stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bodrum_intelligence.google_maps_all_hotels_cleaning import run_cleaning  # noqa: E402
from bodrum_intelligence.google_maps_all_hotels_collection import (  # noqa: E402
    MAPPING_FIELDS, RAW_FIELDS, STATUS_FIELDS, TARGET_FIELDS, MapsPublicCollector,
    build_targets, merge_checkpoint, rating_group, select_smoke_targets, stable_review_id,
    summarize_collection, utc_now, write_csv,
)
from bodrum_intelligence.google_maps_all_hotels_nlp import run_nlp  # noqa: E402

MASTER = PROJECT_ROOT / "data/processed/hotels_clean.csv"
RAW_DIR = PROJECT_ROOT / "data/raw/google_maps_all_hotels"
TARGETS = RAW_DIR / "google_maps_all_hotels_targets.csv"
MAPPING = RAW_DIR / "google_maps_all_hotels_mapping.csv"
REVIEWS = RAW_DIR / "google_maps_all_hotels_reviews_raw.csv"
STATUS = RAW_DIR / "google_maps_all_hotels_collection_status.csv"
RATING_DIST = RAW_DIR / "google_maps_all_hotels_rating_distribution.csv"
FAILURES = RAW_DIR / "google_maps_all_hotels_failures.csv"
BY_HOTEL = RAW_DIR / "by_hotel"
PROCESSED = PROJECT_ROOT / "data/processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures/google_maps_all_hotels"
SMOKE_REPORT = REPORTS / "google_maps_all_hotels_smoke_test.csv"
SUMMARY_TEXT = REPORTS / "google_maps_all_hotels_scraping_summary.txt"
CASE_STUDY_FEATURES = PROCESSED / "google_maps_hotel_nlp_features.csv"
SIK_MAPPING = PROJECT_ROOT / "data/raw/sikayetvar/sikayetvar_hotel_mapping.csv"
CASE_STUDY_RAW_DIR = PROJECT_ROOT / "maps-reviews-main/data/raw/google_travel_reviews"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-hotels", type=int)
    parser.add_argument("--max-reviews-per-hotel", type=int, default=75)
    parser.add_argument("--hotel-id")
    parser.add_argument("--area")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--targets-only", action="store_true")
    parser.add_argument("--process-only", action="store_true")
    parser.add_argument("--delay-min", type=float, default=2.0)
    parser.add_argument("--delay-max", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.max_reviews_per_hotel < 1:
        parser.error("--max-reviews-per-hotel must be positive")
    if args.max_hotels is not None and args.max_hotels < 1:
        parser.error("--max-hotels must be positive")
    if args.force:
        args.resume = False
    if args.smoke_test:
        args.max_reviews_per_hotel = 10
    return args


def _read_ids(path: Path, status_column: str | None = None) -> list[str]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if status_column and status_column in frame:
        frame = frame[frame[status_column].astype(str).str.startswith("FOUND")]
    return frame.get("hotel_id", pd.Series(dtype=str)).dropna().astype(str).tolist()


def generate_targets(cap: int) -> pd.DataFrame:
    master = pd.read_csv(MASTER)
    current_ids = _read_ids(CASE_STUDY_FEATURES)
    sik_ids = _read_ids(SIK_MAPPING, "match_status")
    targets = build_targets(master, current_ids, sik_ids, cap)
    write_csv(TARGETS, targets, TARGET_FIELDS)
    return targets


def _load_or_empty(path: Path, fields) -> pd.DataFrame:
    if path.exists() and path.stat().st_size:
        return pd.read_csv(path, dtype=str).reindex(columns=list(fields))
    return pd.DataFrame(columns=list(fields))


def _ensure_hotel_universe(targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mappings = _load_or_empty(MAPPING, MAPPING_FIELDS)
    statuses = _load_or_empty(STATUS, STATUS_FIELDS)
    mapped_ids = set(mappings.hotel_id.dropna().astype(str))
    status_ids = set(statuses.hotel_id.dropna().astype(str))
    mapping_rows, status_rows = [], []
    for hotel in targets.itertuples(index=False):
        if str(hotel.hotel_id) not in mapped_ids:
            row = {field: "" for field in MAPPING_FIELDS}
            row.update(hotel_id=hotel.hotel_id, place_id=hotel.place_id, hotel_name=hotel.hotel_name, area=hotel.area, source_url=hotel.source_url)
            mapping_rows.append(row)
        if str(hotel.hotel_id) not in status_ids:
            row = {field: "" for field in STATUS_FIELDS}
            row.update(hotel_id=hotel.hotel_id, hotel_name=hotel.hotel_name, area=hotel.area, total_google_review_count_master=hotel.google_review_count, target_cap=hotel.target_review_cap)
            status_rows.append(row)
    if mapping_rows:
        mappings = pd.concat([mappings, pd.DataFrame(mapping_rows)], ignore_index=True).reindex(columns=MAPPING_FIELDS)
    if status_rows:
        statuses = pd.concat([statuses, pd.DataFrame(status_rows)], ignore_index=True).reindex(columns=STATUS_FIELDS)
    cap_by_id = targets.set_index(targets.hotel_id.astype(str))["target_review_cap"]
    count_by_id = targets.set_index(targets.hotel_id.astype(str))["google_review_count"]
    statuses["target_cap"] = statuses.hotel_id.astype(str).map(cap_by_id)
    statuses["total_google_review_count_master"] = statuses.hotel_id.astype(str).map(count_by_id)
    mappings = mappings.set_index("hotel_id").reindex(targets.hotel_id.astype(str)).reset_index().rename(columns={"index":"hotel_id"})
    statuses = statuses.set_index("hotel_id").reindex(targets.hotel_id.astype(str)).reset_index().rename(columns={"index":"hotel_id"})
    write_csv(MAPPING, mappings, MAPPING_FIELDS); write_csv(STATUS, statuses, STATUS_FIELDS)
    return mappings, statuses


def _upsert(path: Path, frame: pd.DataFrame, row: dict, key: str, fields) -> pd.DataFrame:
    frame = frame[frame[key].astype(str) != str(row[key])]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True).reindex(columns=list(fields))
    write_csv(path, frame, fields)
    return frame


def _rebuild_combined() -> pd.DataFrame:
    frames = [pd.read_csv(path, dtype=str) for path in sorted(BY_HOTEL.glob("*.csv")) if path.stat().st_size]
    combined = pd.concat(frames, ignore_index=True).reindex(columns=RAW_FIELDS) if frames else pd.DataFrame(columns=RAW_FIELDS)
    combined = combined.drop_duplicates("review_id", keep="first")
    write_csv(REVIEWS, combined, RAW_FIELDS)
    return combined


def _full_gate_passed() -> bool:
    if not SMOKE_REPORT.exists():
        return False
    smoke = pd.read_csv(SMOKE_REPORT)
    return len(smoke) == 10 and bool(smoke["smoke_pass"].fillna(False).all())


def _case_study_fallback(hotel, cap: int, batch: str) -> list[dict]:
    """Read-only import of the five already-collected Google Travel corpora."""
    candidates = []
    for path in sorted(CASE_STUDY_RAW_DIR.glob("*.csv")):
        frame = pd.read_csv(path, dtype=str)
        if frame.empty:
            continue
        names = frame.get("hotel_name", pd.Series(dtype=str)).dropna().astype(str)
        if names.empty or names.iloc[0].casefold() != str(hotel.hotel_name).casefold():
            continue
        for order, row in enumerate(frame.head(cap).itertuples(index=False), start=1):
            numeric = pd.to_numeric(getattr(row, "review_rating_numeric", None), errors="coerce")
            rating = None if pd.isna(numeric) else int(numeric)
            text = str(getattr(row, "review_text", "") or "")
            date_raw = str(getattr(row, "review_date_raw", "") or "")
            native = str(getattr(row, "review_hash", "") or "")
            source = str(getattr(row, "review_source", "UNKNOWN") or "UNKNOWN")
            candidates.append({
                "review_id": stable_review_id(str(hotel.hotel_id), rating, date_raw, text, native),
                "hotel_id": hotel.hotel_id, "place_id": hotel.place_id, "hotel_name": hotel.hotel_name,
                "area": hotel.area, "review_rating": rating, "review_date_raw": date_raw,
                "review_text": text, "review_language": "", "reviewer_name_raw": "",
                "source_url": hotel.source_url, "review_url": str(getattr(row, "google_travel_url", "") or ""),
                "source_platform": f"Google Travel aggregated ({source})", "collected_at": str(getattr(row, "collected_at", "") or utc_now()),
                "collection_order": order, "rating_group": rating_group(rating),
                "collection_batch": f"{batch}_read_only_case_study_import", "is_rating_only": not bool(text.strip()),
                "extraction_confidence": "HIGH" if rating is not None and native else "MEDIUM",
            })
        break
    return candidates


def _status_with_rows(status: dict, rows: list[dict], cap: int) -> dict:
    ratings = pd.Series([row.get("review_rating") for row in rows], dtype="object")
    groups = pd.Series([row.get("rating_group") for row in rows], dtype="object")
    status.update(
        reviews_collected=len(rows), reviews_with_text=sum(bool(str(row.get("review_text") or "").strip()) for row in rows),
        **{f"rating_{n}_n": int((ratings == n).sum()) for n in range(1, 6)},
        low_n=int((groups == "LOW").sum()), mixed_n=int((groups == "MIXED").sum()), high_n=int((groups == "HIGH").sum()),
        target_reached=len(rows) >= cap,
    )
    if rows and status.get("error_type") == "PUBLIC_REVIEW_PANEL_NOT_ACCESSIBLE":
        status["error_type"] = "PUBLIC_REVIEW_PANEL_NOT_ACCESSIBLE; READ_ONLY_CASE_STUDY_FALLBACK"
    return status


def _smoke_rows(selection: pd.DataFrame, mappings: pd.DataFrame, statuses: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    output = []
    for hotel in selection.itertuples(index=False):
        mapping = mappings[mappings.hotel_id.astype(str) == str(hotel.hotel_id)]
        status = statuses[statuses.hotel_id.astype(str) == str(hotel.hotel_id)]
        sample = reviews[reviews.hotel_id.astype(str) == str(hotel.hotel_id)]
        mapped = not mapping.empty and mapping.iloc[0].mapping_status in {"FOUND_EXACT_PLACE_ID", "FOUND_HIGH_CONFIDENCE"}
        rating_ok = len(sample) > 0 and pd.to_numeric(sample.review_rating, errors="coerce").between(1, 5).all()
        text_ok = sample.review_text.fillna("").str.strip().ne("").any() if len(sample) else False
        no_dupes = sample.review_id.nunique() == len(sample)
        checkpoint_ok = not status.empty and bool(str(status.iloc[0].last_checkpoint).strip())
        blocked = not status.empty and str(status.iloc[0].blocked).casefold() == "true"
        output.append({
            "hotel_id": hotel.hotel_id, "hotel_name": hotel.hotel_name, "area": hotel.area,
            "google_review_count_master": hotel.google_review_count, "identity_ok": mapped,
            "reviews_collected": len(sample), "rating_parse_ok": rating_ok, "text_extraction_ok": text_ok,
            "date_field_present": "review_date_raw" in sample.columns, "no_duplicate": no_dupes,
            "checkpoint_ok": checkpoint_ok, "safe_stop_triggered": blocked,
            "smoke_pass": mapped and len(sample) > 0 and rating_ok and text_ok and no_dupes and checkpoint_ok and not blocked,
        })
    return pd.DataFrame(output)


def _write_collection_reports(targets: pd.DataFrame, mappings: pd.DataFrame, statuses: pd.DataFrame, reviews: pd.DataFrame, smoke: pd.DataFrame | None = None) -> dict:
    REPORTS.mkdir(parents=True, exist_ok=True)
    mappings.to_csv(REPORTS / "google_maps_all_hotels_mapping_summary.csv", index=False, encoding="utf-8-sig")
    statuses.to_csv(REPORTS / "google_maps_all_hotels_collection_summary.csv", index=False, encoding="utf-8-sig")
    manual = mappings[mappings.mapping_status.astype(str).eq("REVIEW_REQUIRED")]
    manual.to_csv(REPORTS / "google_maps_all_hotels_manual_review.csv", index=False, encoding="utf-8-sig")
    failure_status = statuses[statuses.error_type.fillna("").astype(str).str.strip().ne("")].copy()
    failures = failure_status.merge(
        mappings[["hotel_id","place_id","source_url","detected_hotel_name","detected_address","review_panel_found","mapping_note"]],
        on="hotel_id", how="left",
    )
    failures.to_csv(REPORTS / "google_maps_all_hotels_failures.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(FAILURES, index=False, encoding="utf-8-sig")
    rating = reviews.groupby(["hotel_id","rating_group"], dropna=False).size().unstack(fill_value=0).reset_index() if len(reviews) else pd.DataFrame(columns=["hotel_id","LOW","MIXED","HIGH"])
    for col in ["LOW","MIXED","HIGH"]:
        if col not in rating: rating[col] = 0
    rating.to_csv(RATING_DIST, index=False, encoding="utf-8-sig")
    rating.to_csv(REPORTS / "google_maps_all_hotels_rating_coverage.csv", index=False, encoding="utf-8-sig")
    status_numeric = statuses.copy()
    for col in ["reviews_collected","reviews_with_text","low_n","mixed_n","high_n"]:
        status_numeric[col] = pd.to_numeric(status_numeric[col], errors="coerce").fillna(0)
    coverage = targets[["hotel_id","area"]].merge(status_numeric[["hotel_id","mapping_status","reviews_collected","reviews_with_text","low_n","mixed_n","high_n"]], on="hotel_id", how="left")
    area = coverage.groupby("area", dropna=False).agg(
        project_hotel_count=("hotel_id","size"), hotels_attempted=("mapping_status",lambda s:s.fillna("").astype(str).str.strip().ne("").sum()),
        hotels_with_reviews=("reviews_collected",lambda s:(s>0).sum()), hotels_with_text_reviews=("reviews_with_text",lambda s:(s>0).sum()),
        review_n=("reviews_collected","sum"), text_review_n=("reviews_with_text","sum"), low_n=("low_n","sum"), mixed_n=("mixed_n","sum"), high_n=("high_n","sum"),
    ).reset_index()
    area.to_csv(REPORTS / "google_maps_all_hotels_coverage_by_area.csv", index=False, encoding="utf-8-sig")
    summary = summarize_collection(targets, statuses, reviews)
    summary["duplicate_count"] = int(reviews.duplicated("review_id").sum()) if len(reviews) else 0
    summary["collection_success_rate_pct"] = round(summary["hotels_with_reviews"] * 100 / summary["hotels_attempted"], 2) if summary["hotels_attempted"] else 0.0
    summary["areas_with_text_reviews"] = int(area.loc[area.text_review_n > 0, "area"].nunique())
    summary["project_area_count"] = int(area.area.nunique())
    summary["hotels_low_sample"] = int(((status_numeric.reviews_with_text > 0) & (status_numeric.reviews_with_text < 15)).sum())
    gate = bool(smoke is not None and len(smoke) == 10 and smoke.smoke_pass.all()) if smoke is not None else _full_gate_passed()
    summary["full_collection_safe_to_continue"] = "YES" if gate else "NO"
    lines = ["GOOGLE MAPS ALL-HOTELS COLLECTION SUMMARY", ""] + [f"{key}: {value}" for key,value in summary.items()]
    SUMMARY_TEXT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def process_outputs() -> dict:
    clean, readiness, audit = run_cleaning(REVIEWS, TARGETS, PROCESSED, REPORTS)
    nlp = run_nlp(PROCESSED / "google_maps_all_hotels_reviews_clean.csv", TARGETS, PROCESSED, REPORTS, FIGURES)
    return {"cleaning": audit, "nlp": nlp}


def run(argv=None) -> int:
    args = parse_args(argv)
    targets = generate_targets(args.max_reviews_per_hotel)
    RAW_DIR.mkdir(parents=True, exist_ok=True); BY_HOTEL.mkdir(parents=True, exist_ok=True)
    for path, fields in [(MAPPING,MAPPING_FIELDS),(STATUS,STATUS_FIELDS),(REVIEWS,RAW_FIELDS)]:
        if not path.exists(): write_csv(path, [], fields)
    mappings, statuses = _ensure_hotel_universe(targets)
    if args.targets_only:
        print(f"Targets written: {len(targets)}")
        return 0
    if args.process_only:
        current_reviews = _rebuild_combined()
        _write_collection_reports(targets, mappings, statuses, current_reviews)
        print(json.dumps(process_outputs(), ensure_ascii=False, indent=2, default=str))
        return 0

    selection = targets.copy()
    if args.smoke_test:
        selection = select_smoke_targets(selection, 10, _read_ids(CASE_STUDY_FEATURES))
    elif not _full_gate_passed():
        print("Full collection blocked: run a passing --smoke-test first.", file=sys.stderr)
        return 2
    if args.hotel_id:
        selection = selection[selection.hotel_id.astype(str) == args.hotel_id]
    if args.area:
        selection = selection[selection.area.astype(str).str.casefold() == args.area.casefold()]
    if args.max_hotels:
        selection = selection.head(args.max_hotels)
    if selection.empty:
        print("No hotels selected.", file=sys.stderr); return 2
    if args.dry_run:
        plan = selection.copy(); plan["planned_cap"] = args.max_reviews_per_hotel
        plan.to_csv(REPORTS / "google_maps_all_hotels_collection_plan.csv", index=False, encoding="utf-8-sig")
        print(f"Dry run: {len(selection)} hotels planned; no pages opened.")
        return 0

    mappings, statuses = _ensure_hotel_universe(targets)
    batch = ("smoke" if args.smoke_test else "full") + "_" + utc_now().replace(":", "").replace("+", "_")
    with MapsPublicCollector(args.headless, args.delay_min, args.delay_max) as collector:
        for index, hotel in enumerate(selection.itertuples(index=False), start=1):
            path = BY_HOTEL / f"{hotel.hotel_id}.csv"
            existing = pd.read_csv(path, dtype=str) if path.exists() and path.stat().st_size else pd.DataFrame(columns=RAW_FIELDS)
            if args.resume and len(existing) >= args.max_reviews_per_hotel:
                print(f"[{index}/{len(selection)}] {hotel.hotel_id} resume checkpoint already meets cap")
                continue
            print(f"[{index}/{len(selection)}] {hotel.hotel_id} {hotel.hotel_name}", flush=True)
            result = collector.collect_hotel(pd.Series(hotel._asdict()), args.max_reviews_per_hotel, batch)
            if not result.reviews and result.mapping.get("mapping_status") in {"FOUND_EXACT_PLACE_ID", "FOUND_HIGH_CONFIDENCE"}:
                result.reviews = _case_study_fallback(hotel, args.max_reviews_per_hotel, batch)
                result.status = _status_with_rows(result.status, result.reviews, args.max_reviews_per_hotel)
            merge_checkpoint(path, result.reviews, force=args.force)
            mappings = _upsert(MAPPING, mappings, result.mapping, "hotel_id", MAPPING_FIELDS)
            statuses = _upsert(STATUS, statuses, result.status, "hotel_id", STATUS_FIELDS)
            _rebuild_combined()
            if result.safe_stop:
                print("BLOCKED_SAFE_STOP: challenge detected; checkpoint saved and run stopped.", file=sys.stderr)
                break
    reviews = _rebuild_combined()
    smoke = _smoke_rows(selection, mappings, statuses, reviews) if args.smoke_test else None
    if smoke is not None:
        smoke.to_csv(SMOKE_REPORT, index=False, encoding="utf-8-sig")
    summary = _write_collection_reports(targets, mappings, statuses, reviews, smoke)
    processed = process_outputs()
    print(json.dumps({"collection":summary,"processed":processed}, ensure_ascii=False, indent=2, default=str))
    return 0 if summary["hotels_blocked"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(run())
