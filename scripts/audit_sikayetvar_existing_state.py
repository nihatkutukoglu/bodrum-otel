"""Recompute the existing-state audit for the repository's Sikayetvar layer.

This script is deliberately read-only with respect to data/raw and data/processed.
It performs no network access and writes only the six audit CSVs requested under
reports/.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw" / "sikayetvar"
PROCESSED_DIR = ROOT / "data" / "processed"

OUTPUT_NAMES = {
    "sikayetvar_existing_assets_inventory.csv",
    "sikayetvar_existing_entity_mapping_audit.csv",
    "sikayetvar_existing_data_quality.csv",
    "sikayetvar_report_figure_inventory.csv",
    "sikayetvar_master_coverage_audit.csv",
    "sikayetvar_current_state_matrix.csv",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def pct(numerator: int | float, denominator: int | float) -> float | None:
    return round(100 * float(numerator) / float(denominator), 2) if denominator else None


def normalized_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def file_purpose(path: Path) -> str:
    name = path.name.lower()
    posix = rel(path).lower()
    if "mapping_candidates" in name:
        return "Discovery candidate evidence for hotel-to-Sikayetvar mapping"
    if "hotel_mapping" in name or "mapping_summary" in name or "manual_review" in name:
        return "Hotel entity mapping, mapping summary, or manual-review evidence"
    if "complaint_links" in name:
        return "Discovered complaint URL inventory"
    if "complaints_raw" in name:
        return "Raw complaint detail records; one row per hotel assignment and complaint URL"
    if "replies_raw" in name or name == "sikayetvar_replies.csv":
        return "Raw reply records linked to complaint URLs"
    if "scrape_status" in name or "scraping_summary" in name:
        return "Scraping checkpoint/status summary"
    if "complaints_clean" in name:
        return "Canonical matched complaint corpus cleaned for downstream analysis"
    if "replies_clean" in name:
        return "Cleaned reply corpus"
    if "complaints_nlp" in name:
        return "Complaint-level derived NLP and aspect features"
    if "aspect" in name:
        return "Aspect taxonomy, mention, lift, validation, or aspect summary output"
    if "nlp" in name or "ngram" in name or "unigram" in name or "topic" in name:
        return "NLP corpus, term, sample-readiness, or topic-gate output"
    if "response" in name or "reply" in name:
        return "Company/user reply coverage or response-behavior output"
    if "duplicate" in name or "quality" in name or "audit" in name or "cleaning" in name:
        return "Data quality, schema, deduplication, or cleaning evidence"
    if "coverage" in name:
        return "Hotel/area coverage evidence"
    if "eda" in name or "temporal" in name or "correlation" in name or "visibility" in name:
        return "Descriptive EDA or cross-platform visibility output"
    if path.suffix.lower() == ".ipynb":
        return "Executed Sikayetvar analysis notebook"
    if "/tests/" in f"/{posix}":
        return "Automated test for Sikayetvar code"
    if "/src/" in f"/{posix}" or "/scripts/" in f"/{posix}":
        return "Sikayetvar implementation or pipeline script"
    if "/config/" in f"/{posix}":
        return "Sikayetvar configuration or manual alias data"
    return "Other Sikayetvar-related repository asset"


def notebook_metadata(path: Path) -> tuple[str, int | None, int | None, str]:
    if path.suffix.lower() != ".ipynb":
        return "", None, None, ""
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
        executed = sum(cell.get("execution_count") is not None for cell in code_cells)
        output_cells = sum(bool(cell.get("outputs")) for cell in code_cells)
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
        refs = sorted(
            set(re.findall(r"[\w./-]*sikayetvar[\w./-]*\.(?:csv|txt)", source, flags=re.I))
        )
        status = "EXECUTED" if code_cells and executed == len(code_cells) else "PARTIALLY_EXECUTED"
        if not code_cells:
            status = "NO_CODE_CELLS"
        return status, executed, output_cells, "|".join(refs)
    except Exception as exc:  # audit must retain unreadable assets rather than fail inventory
        return f"READ_ERROR: {exc}", None, None, ""


def csv_profile(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "row_count": None,
        "column_count": None,
        "columns": "",
        "hotel_count": None,
        "source_url_coverage_pct": None,
        "detail_coverage_pct": None,
        "reply_coverage_pct": None,
        "date_coverage_pct": None,
        "duplicate_candidate_excess_count": None,
        "empty_complaint_text_count": None,
    }
    if path.suffix.lower() != ".csv":
        return result
    try:
        frame = read_csv(path)
    except Exception as exc:
        result["columns"] = f"READ_ERROR: {exc}"
        return result
    result.update(
        row_count=len(frame),
        column_count=len(frame.columns),
        columns="|".join(map(str, frame.columns)),
    )
    if "hotel_id" in frame:
        result["hotel_count"] = int(frame.loc[nonempty(frame["hotel_id"]), "hotel_id"].nunique())
    source_column = next(
        (column for column in ("source_url", "source_page", "sikayetvar_url") if column in frame), None
    )
    if source_column:
        result["source_url_coverage_pct"] = pct(int(nonempty(frame[source_column]).sum()), len(frame))
    if "complaint_text" in frame:
        filled = int(nonempty(frame["complaint_text"]).sum())
        result["detail_coverage_pct"] = pct(filled, len(frame))
        result["empty_complaint_text_count"] = int(len(frame) - filled)
    if "company_response_exists_clean" in frame:
        result["reply_coverage_pct"] = pct(int(bool_series(frame["company_response_exists_clean"]).sum()), len(frame))
    elif "company_response_exists" in frame:
        result["reply_coverage_pct"] = pct(int(bool_series(frame["company_response_exists"]).sum()), len(frame))
    elif "reply_author_type" in frame:
        result["reply_coverage_pct"] = pct(
            int(frame["reply_author_type"].astype(str).str.upper().eq("COMPANY").sum()), len(frame)
        )
    date_column = next(
        (column for column in ("complaint_date", "complaint_date_raw", "reply_date", "reply_date_raw") if column in frame),
        None,
    )
    if date_column:
        result["date_coverage_pct"] = pct(int(nonempty(frame[date_column]).sum()), len(frame))
    key = next(
        (column for column in ("canonical_complaint_url", "complaint_url", "complaint_id") if column in frame),
        None,
    )
    if key:
        valid = frame.loc[nonempty(frame[key]), key]
        result["duplicate_candidate_excess_count"] = int(valid.duplicated().sum())
    return result


def build_asset_inventory() -> pd.DataFrame:
    scoped_roots = [
        ROOT / "data" / "raw",
        ROOT / "data" / "interim",
        ROOT / "data" / "processed",
        ROOT / "reports",
        ROOT / "notebooks",
        ROOT / "scripts",
        ROOT / "src",
        ROOT / "config",
        ROOT / "tests",
        ROOT / "siikayet-var-scraping",
    ]
    core_pattern = re.compile(r"sikayet|şikayet|complaint|complaints|customer_voice", re.I)
    files: set[Path] = set()
    text_suffixes = {".py", ".md", ".txt", ".json", ".toml", ".ipynb", ".csv"}
    for scoped_root in scoped_roots:
        if not scoped_root.exists():
            continue
        for path in scoped_root.rglob("*"):
            if not path.is_file() or any(part in {".git", ".pytest_cache", "__pycache__"} for part in path.parts):
                continue
            if path.name in OUTPUT_NAMES:
                continue
            if core_pattern.search(rel(path)):
                files.add(path)
                continue
            if path.suffix.lower() in text_suffixes and path.stat().st_size <= 10_000_000:
                try:
                    if core_pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                        files.add(path)
                except OSError:
                    pass

    rows: list[dict[str, Any]] = []
    for path in sorted(files, key=lambda item: rel(item).lower()):
        stat = path.stat()
        profile = csv_profile(path)
        line_count = None
        if path.suffix.lower() in text_suffixes:
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    line_count = sum(1 for _ in handle)
            except OSError:
                pass
        execution_status, executed_cells, output_cells, references = notebook_metadata(path)
        rows.append(
            {
                "file_path": rel(path),
                "file_type": path.suffix.lower().lstrip(".") or "no_extension",
                "size_bytes": stat.st_size,
                "modified_at_local": pd.Timestamp(stat.st_mtime, unit="s").isoformat(),
                "line_count": line_count,
                **profile,
                "purpose": file_purpose(path),
                "notebook_execution_status": execution_status,
                "notebook_executed_code_cells": executed_cells,
                "notebook_output_cells": output_cells,
                "notebook_sikayetvar_references": references,
            }
        )
    return pd.DataFrame(rows)


def report_category(path: Path) -> str:
    text = rel(path).lower()
    name = path.name.lower()
    if "cross_platform" in name or "google" in name:
        return "CROSS_SOURCE"
    if "mapping" in name or "manual_review" in name or "coverage" in name:
        return "MAPPING"
    if any(token in name for token in ("audit", "clean", "duplicate", "missing", "orphan", "validation", "schema")):
        return "QUALITY"
    if "aspect" in name or "severity" in name:
        return "ASPECT"
    if any(token in name for token in ("nlp", "ngram", "unigram", "topic", "term")):
        return "NLP"
    if "response" in name or "reply" in name:
        return "REPLY"
    if "eda" in name or "temporal" in name or "correlation" in name:
        return "EDA"
    if "scrap" in name or "status" in name:
        return "SCRAPING"
    if "figures/sikayetvar_audit" in text:
        return "QUALITY"
    if "figures/sikayetvar_eda" in text:
        return "EDA"
    if "figures/sikayetvar_nlp" in text:
        return "NLP"
    if "figures/sikayetvar_severity" in text:
        return "ASPECT"
    return "OTHER"


def build_report_inventory() -> pd.DataFrame:
    paths = []
    for path in REPORTS.rglob("*"):
        if not path.is_file() or path.name in OUTPUT_NAMES:
            continue
        if re.search(r"sikayet|şikayet|complaint|customer_voice|reply|response", rel(path), re.I):
            paths.append(path)
    rows = []
    for path in sorted(paths, key=lambda item: rel(item).lower()):
        stat = path.stat()
        profile = csv_profile(path)
        rows.append(
            {
                "file_path": rel(path),
                "file_type": path.suffix.lower().lstrip("."),
                "category": report_category(path),
                "size_bytes": stat.st_size,
                "modified_at_local": pd.Timestamp(stat.st_mtime, unit="s").isoformat(),
                "row_count": profile["row_count"],
                "column_count": profile["column_count"],
                "columns": profile["columns"],
                "purpose": file_purpose(path),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    master = read_csv(ROOT / "bodrum_hotels_master_2026-08-24.csv")
    mapping = read_csv(RAW_DIR / "sikayetvar_hotel_mapping.csv")
    raw = read_csv(RAW_DIR / "sikayetvar_all_hotels_complaints_raw.csv")
    links = read_csv(RAW_DIR / "sikayetvar_all_complaint_links.csv")
    replies = read_csv(RAW_DIR / "sikayetvar_all_hotels_replies_raw.csv")
    scrape_status = read_csv(RAW_DIR / "sikayetvar_scrape_status_all_hotels.csv")
    clean = read_csv(PROCESSED_DIR / "sikayetvar_all_hotels_complaints_clean.csv")
    clean_replies = read_csv(PROCESSED_DIR / "sikayetvar_all_hotels_replies_clean.csv")

    # Inventory is intentionally based on pre-existing assets and excludes these audit outputs.
    build_asset_inventory().to_csv(
        REPORTS / "sikayetvar_existing_assets_inventory.csv", index=False, encoding="utf-8-sig"
    )

    raw_by_hotel = raw.groupby("hotel_id").agg(
        complaint_count=("complaint_id", "size"),
        unique_complaint_count=("canonical_complaint_url", "nunique"),
        detail_nonempty_count=("complaint_text", lambda values: int(nonempty(values).sum())),
        company_response_count=("company_response_exists", lambda values: int(bool_series(values).sum())),
    )
    matched_by_hotel = (
        raw.loc[raw["entity_match_status"].eq("COMPLAINT_MATCHED")]
        .groupby("hotel_id")["canonical_complaint_url"]
        .nunique()
        .rename("matched_complaint_count")
    )
    review_by_hotel = (
        raw.loc[raw["entity_match_status"].eq("COMPLAINT_REVIEW_REQUIRED")]
        .groupby("hotel_id")["canonical_complaint_url"]
        .nunique()
        .rename("review_required_complaint_count")
    )
    hotel_stats = raw_by_hotel.join(matched_by_hotel, how="outer").join(review_by_hotel, how="outer").fillna(0)

    entity = mapping.merge(hotel_stats, left_on="hotel_id", right_index=True, how="left")
    count_columns = [
        "complaint_count",
        "unique_complaint_count",
        "detail_nonempty_count",
        "company_response_count",
        "matched_complaint_count",
        "review_required_complaint_count",
    ]
    entity[count_columns] = entity[count_columns].fillna(0).astype(int)
    entity_audit = entity.rename(
        columns={
            "hotel_name": "master_hotel_name",
            "sikayetvar_company_name": "detected_sikayetvar_hotel_name",
            "match_status": "mapping_status",
            "match_score": "confidence",
            "sikayetvar_url": "source_url",
        }
    )[
        [
            "hotel_id",
            "master_hotel_name",
            "area",
            "detected_sikayetvar_hotel_name",
            "mapping_status",
            "confidence",
            "source_url",
            "page_accessible",
            "visible_complaint_count",
            "complaint_count",
            "unique_complaint_count",
            "matched_complaint_count",
            "review_required_complaint_count",
            "detail_nonempty_count",
            "company_response_count",
            "match_method",
            "match_reason",
            "manual_review_required",
            "checked_at",
        ]
    ]
    entity_audit.to_csv(
        REPORTS / "sikayetvar_existing_entity_mapping_audit.csv", index=False, encoding="utf-8-sig"
    )

    mapping_merged = master[["hotel_id", "hotel_name", "area"]].merge(
        mapping[
            [
                "hotel_id",
                "sikayetvar_company_name",
                "sikayetvar_url",
                "match_status",
                "match_score",
                "page_accessible",
                "visible_complaint_count",
                "checked_at",
            ]
        ],
        on="hotel_id",
        how="left",
    ).merge(hotel_stats, left_on="hotel_id", right_index=True, how="left")
    mapping_merged[count_columns] = mapping_merged[count_columns].fillna(0).astype(int)
    mapping_merged["checked_on_sikayetvar"] = mapping_merged["match_status"].notna()
    mapping_merged["valid_page"] = mapping_merged["match_status"].isin(
        {"FOUND_EXACT", "FOUND_HIGH_CONFIDENCE", "PAGE_FOUND_NO_COMPLAINT"}
    )
    mapping_merged["hotels_with_complaints"] = mapping_merged["matched_complaint_count"].gt(0)
    mapping_merged["hotels_with_zero_complaints"] = mapping_merged["match_status"].eq(
        "PAGE_FOUND_NO_COMPLAINT"
    )
    mapping_merged["not_found"] = mapping_merged["match_status"].eq("NOT_FOUND")
    mapping_merged["review_required"] = mapping_merged["match_status"].isin({"REVIEW_REQUIRED", "AMBIGUOUS"})
    mapping_merged["wrong_entity"] = mapping_merged["match_status"].isin({"WRONG_ENTITY", "EXCLUDED"})
    mapping_merged["unprocessed"] = mapping_merged["match_status"].isna()
    coverage = mapping_merged.rename(
        columns={
            "sikayetvar_company_name": "detected_sikayetvar_hotel_name",
            "sikayetvar_url": "source_url",
            "match_status": "mapping_status",
            "match_score": "confidence",
        }
    )
    coverage.to_csv(REPORTS / "sikayetvar_master_coverage_audit.csv", index=False, encoding="utf-8-sig")

    raw_norm = raw["complaint_text"].map(normalized_text)
    hotel_text_pairs = pd.DataFrame({"hotel_id": raw["hotel_id"], "normalized_text": raw_norm})
    valid_text_pairs = hotel_text_pairs.loc[hotel_text_pairs["normalized_text"].ne("")]
    master_ids = set(master["hotel_id"].astype(str))
    invalid_hotel = ~raw["hotel_id"].astype(str).isin(master_ids)
    clean_company = bool_series(clean["company_response_exists_clean"])
    raw_company = bool_series(raw["company_response_exists"])
    response_flag_by_url = (
        pd.DataFrame(
            {
                "canonical_complaint_url": raw["canonical_complaint_url"],
                "raw_company_response_flag": raw_company,
            }
        )
        .groupby("canonical_complaint_url")["raw_company_response_flag"]
        .any()
    )
    company_reply_urls = set(
        replies.loc[
            replies["reply_author_type"].astype(str).str.upper().eq("COMPANY"),
            "canonical_complaint_url",
        ]
    )
    raw_reply_table_mismatch = int(
        sum(bool(flag) != (url in company_reply_urls) for url, flag in response_flag_by_url.items())
    )
    preserved_text = clean[["canonical_complaint_url", "hotel_id", "complaint_text"]].merge(
        raw.loc[
            raw["entity_match_status"].eq("COMPLAINT_MATCHED"),
            ["canonical_complaint_url", "hotel_id", "complaint_text"],
        ],
        on=["canonical_complaint_url", "hotel_id"],
        how="left",
        suffixes=("_clean_file", "_raw_file"),
        validate="one_to_one",
    )
    exact_text_mismatch = preserved_text["complaint_text_clean_file"].fillna("<NA>").ne(
        preserved_text["complaint_text_raw_file"].fillna("<NA>")
    )
    normalize_newlines = lambda value: str(value).replace("\r\n", "\n").replace("\r", "\n")
    newline_normalized_mismatch = preserved_text["complaint_text_clean_file"].fillna("<NA>").map(
        normalize_newlines
    ).ne(preserved_text["complaint_text_raw_file"].fillna("<NA>").map(normalize_newlines))
    quality_rows = [
        ("EXACT_DUPLICATE_ROWS", int(raw.duplicated().sum()), len(raw), "Exact duplicate row excess in raw complaints"),
        ("DUPLICATE_CANONICAL_COMPLAINT_URL", int(raw.duplicated("canonical_complaint_url").sum()), len(raw), "Duplicate excess; raw retains cross-assignment evidence"),
        ("DUPLICATE_COMPLAINT_ID", int(raw.duplicated("complaint_id").sum()), len(raw), "Duplicate excess"),
        ("SAME_HOTEL_NORMALIZED_TEXT", int(valid_text_pairs.duplicated(["hotel_id", "normalized_text"]).sum()), len(valid_text_pairs), "Duplicate candidate excess after NFKC/case/whitespace normalization"),
        ("EMPTY_COMPLAINT_TITLE", int((~nonempty(raw["complaint_title"])).sum()), len(raw), "Raw complaint title missing"),
        ("EMPTY_COMPLAINT_TEXT", int((~nonempty(raw["complaint_text"])).sum()), len(raw), "Raw complaint text missing"),
        ("INVALID_HOTEL_ID", int(invalid_hotel.sum()), len(raw), "hotel_id absent from 192-row master"),
        ("MISSING_SOURCE_URL", int((~nonempty(raw["source_page"])).sum()), len(raw), "Raw complaint source_page missing"),
        ("WRONG_ENTITY_CONTAMINATION", int(raw["entity_match_status"].eq("COMPLAINT_EXCLUDED_OTHER_PROPERTY").sum()), len(raw), "Confident wrong-property rows; review-required is reported separately"),
        ("COMPLAINT_REVIEW_REQUIRED", int(raw["entity_match_status"].eq("COMPLAINT_REVIEW_REQUIRED").sum()), len(raw), "Quarantined from the clean matched corpus"),
        ("PARSER_HTML_LEAKAGE_FLAG", int(bool_series(clean["complaint_text_html_artifact_flag"]).sum()), len(clean), "Existing cleaning flag"),
        ("PARSER_BOILERPLATE_FLAG", int(bool_series(clean["complaint_text_possible_boilerplate_flag"]).sum()), len(clean), "Existing cleaning flag"),
        ("ENCODING_ARTIFACT_FLAG", int(bool_series(clean["complaint_text_encoding_artifact_flag"]).sum()), len(clean), "Existing cleaning flag"),
        ("CLEAN_DUPLICATE_CANONICAL_URL", int(clean.duplicated("canonical_complaint_url").sum()), len(clean), "Canonical duplicate excess in clean corpus"),
        ("ORPHAN_CLEAN_REPLIES", int((~clean_replies["canonical_complaint_url"].isin(set(clean["canonical_complaint_url"]))).sum()), len(clean_replies), "Clean reply URL absent from clean complaint corpus"),
        ("CLEAN_COMPANY_RESPONSE_INTERNAL_MISMATCH", int(bool_series(clean["company_response_mismatch_flag"]).sum()), len(clean), "Clean response flag versus clean-derived COMPANY reply mismatch"),
        ("RAW_RESPONSE_FLAG_VS_REPLY_TABLE_MISMATCH", raw_reply_table_mismatch, len(response_flag_by_url), "Raw complaint response flag versus separate raw COMPANY reply-row visibility"),
        ("RAW_TEXT_EXACT_PRESERVATION_MISMATCH", int(exact_text_mismatch.sum()), len(clean), "All observed mismatches are CSV newline serialization differences (LF versus CRLF); content mismatch after newline normalization is reported separately"),
        ("RAW_TEXT_CONTENT_MISMATCH_AFTER_NEWLINE_NORMALIZATION", int(newline_normalized_mismatch.sum()), len(clean), "Semantic text-preservation check after normalizing LF/CRLF"),
        ("DATE_PARSE_FAILED", int(bool_series(clean["complaint_date_parse_failed_flag"]).sum()), len(clean), "Clean matched complaint dates not parsed"),
        ("CLEAN_REPLY_COVERAGE", int(clean_company.sum()), len(clean), "Matched complaints with visible company response; not resolution"),
        ("RAW_REPLY_COVERAGE", int(raw_company.sum()), len(raw), "Raw complaint rows with visible company response; not resolution"),
        ("MISSING_EXPECTED_COMPLAINT_DATE", int("complaint_date" not in raw.columns), 1, "Raw uses complaint_date_raw; parsed complaint_date exists in clean data"),
        ("MISSING_EXPECTED_STATUS", int("status" not in raw.columns), 1, "Raw has entity_match_status plus response/progress flags, not a generic status field"),
        ("MISSING_EXPECTED_COMPANY_REPLY_DATE", int("company_reply_date" not in raw.columns), 1, "Equivalent field is company_response_date"),
        ("MISSING_EXPECTED_SOURCE_URL", int("source_url" not in raw.columns), 1, "Equivalent field is source_page"),
    ]
    quality = pd.DataFrame(
        quality_rows, columns=["check", "issue_count", "denominator", "notes"]
    )
    quality["issue_rate_pct"] = quality.apply(
        lambda row: pct(row["issue_count"], row["denominator"]), axis=1
    )
    quality["evidence_file"] = "data/raw/sikayetvar/sikayetvar_all_hotels_complaints_raw.csv"
    quality.loc[quality["check"].str.contains("CLEAN|PARSER|ENCODING|DATE_PARSE|MISMATCH"), "evidence_file"] = (
        "data/processed/sikayetvar_all_hotels_complaints_clean.csv"
    )
    quality.to_csv(REPORTS / "sikayetvar_existing_data_quality.csv", index=False, encoding="utf-8-sig")

    build_report_inventory().to_csv(
        REPORTS / "sikayetvar_report_figure_inventory.csv", index=False, encoding="utf-8-sig"
    )

    mapping_counts = mapping["match_status"].value_counts()
    page_found = int(bool_series(mapping["page_accessible"]).sum())
    matched_complaints = int(raw["entity_match_status"].eq("COMPLAINT_MATCHED").sum())
    unique_clean = int(clean["canonical_complaint_url"].nunique())
    details_success = int(pd.to_numeric(scrape_status["details_success"], errors="coerce").fillna(0).sum())
    company_reply_rows = int(replies["reply_author_type"].astype(str).str.upper().eq("COMPANY").sum())
    matrix_rows = [
        ("URL / PAGE DISCOVERY", "COMPLETE", "data/raw/sikayetvar/sikayetvar_hotel_mapping.csv", len(mapping), True, False, f"All {len(master)} master hotels have a discovery status; accessible page records={page_found}."),
        ("ENTITY MAPPING", "PARTIAL", "data/raw/sikayetvar/sikayetvar_hotel_mapping.csv", int(mapping_counts.get("FOUND_EXACT", 0) + mapping_counts.get("FOUND_HIGH_CONFIDENCE", 0)), False, True, f"REVIEW_REQUIRED={mapping_counts.get('REVIEW_REQUIRED', 0)}; AMBIGUOUS={mapping_counts.get('AMBIGUOUS', 0)}; NOT_FOUND={mapping_counts.get('NOT_FOUND', 0)}; PAGE_FOUND_NO_COMPLAINT={mapping_counts.get('PAGE_FOUND_NO_COMPLAINT', 0)}."),
        ("RAW COMPLAINT SCRAPING", "PARTIAL", "data/raw/sikayetvar/sikayetvar_all_hotels_complaints_raw.csv", len(raw), False, True, "All discovered links have rows, but unresolved hotel mappings limit all-hotel coverage."),
        ("DETAIL SCRAPING", "COMPLETE", "data/raw/sikayetvar/sikayetvar_scrape_status_all_hotels.csv", details_success, True, False, f"details_failed={int(pd.to_numeric(scrape_status['details_failed'], errors='coerce').fillna(0).sum())}; raw text non-empty={int(nonempty(raw['complaint_text']).sum())}/{len(raw)}."),
        ("COMPANY REPLIES", "PARTIAL", "data/raw/sikayetvar/sikayetvar_all_hotels_replies_raw.csv", company_reply_rows, False, True, f"Separate COMPANY reply rows={company_reply_rows}; raw complaints with company response flag={int(raw_company.sum())}; clean matched coverage={int(clean_company.sum())}/{len(clean)} ({pct(int(clean_company.sum()), len(clean))}%)."),
        ("DEDUPLICATION", "COMPLETE", "data/processed/sikayetvar_all_hotels_complaints_clean.csv", unique_clean, True, False, f"Clean canonical URL duplicate excess={int(clean.duplicated('canonical_complaint_url').sum())}; raw duplicate excess=3."),
        ("AUDIT/CLEANING", "COMPLETE", "notebooks/12_sikayetvar_all_hotels_audit_cleaning.ipynb", len(clean), True, False, "Notebook fully executed; matched corpus and review-required quarantine exist."),
        ("EDA", "COMPLETE", "notebooks/13_sikayetvar_all_hotels_eda.ipynb", len(clean), True, False, "Notebook fully executed with hotel, area, temporal, reply and visibility outputs."),
        ("NLP", "COMPLETE", "data/processed/sikayetvar_all_hotels_complaints_nlp.csv", len(clean), True, False, "Rule-based/token/TF-IDF outputs exist; no overall sentiment model; topic-model gate says not reliable."),
        ("ASPECT ANALYSIS", "COMPLETE", "data/processed/sikayetvar_complaint_aspects_long.csv", len(read_csv(PROCESSED_DIR / "sikayetvar_complaint_aspects_long.csv")), True, False, "18-aspect explainable multi-label taxonomy with validation outputs."),
        ("HOTEL-LEVEL SUMMARY", "COMPLETE", "reports/sikayetvar_hotel_eda_summary.csv", len(read_csv(REPORTS / "sikayetvar_hotel_eda_summary.csv")), True, False, "Complaint-bearing hotels only; small-n and denominator flags retained."),
        ("AREA-LEVEL SUMMARY", "COMPLETE", "reports/sikayetvar_area_eda_summary.csv", len(read_csv(REPORTS / "sikayetvar_area_eda_summary.csv")), True, False, "All 14 master areas retained with coverage fields."),
        ("NORMALIZED COMPLAINT VISIBILITY", "COMPLETE", "reports/sikayetvar_cross_platform_visibility.csv", len(read_csv(REPORTS / "sikayetvar_cross_platform_visibility.csv")), True, False, "Uses master google_review_count; non-positive/missing denominators remain NaN. This is not a complaint rate."),
        ("GOOGLE TRAVEL CROSS-SOURCE", "NOT_STARTED", "", 0, False, True, "No evidence of a Sikayetvar-to-Google-Travel dataset join. Existing visibility metric uses master Google review count context."),
        ("FINAL CUSTOMER VOICE SUMMARY", "PARTIAL", "reports/sikayetvar_eda_key_findings.txt|reports/sikayetvar_nlp_key_findings.txt", 2, False, True, "Layer-specific findings exist, but Notebook 11 explicitly says Sikayetvar is not included in the project summary."),
        ("TESTS", "PARTIAL_WITH_FAILURE", "tests/test_sikayetvar_*.py|siikayet-var-scraping/tests/test_sikayetvar_scraper.py", 58, False, True, "58 test functions in 5 Sikayetvar test files. Safe runnable subset: 16 passed, 1 failed (LF/CRLF-only raw-text preservation assertion). Full collection blocked by missing requests and sklearn; no packages installed and no scraping run."),
    ]
    matrix = pd.DataFrame(
        matrix_rows,
        columns=["layer", "status", "evidence_file", "row_count_or_count", "complete", "needs_work", "notes"],
    )
    matrix.to_csv(REPORTS / "sikayetvar_current_state_matrix.csv", index=False, encoding="utf-8-sig")

    print(
        json.dumps(
            {
                "master_hotels": len(master),
                "mapping_statuses": {str(key): int(value) for key, value in mapping_counts.items()},
                "accessible_pages": page_found,
                "raw_rows": len(raw),
                "unique_raw_complaints": int(raw["canonical_complaint_url"].nunique()),
                "raw_hotel_count": int(raw["hotel_id"].nunique()),
                "matched_complaints": matched_complaints,
                "clean_unique_complaints": unique_clean,
                "clean_hotel_count": int(clean["hotel_id"].nunique()),
                "clean_company_response_count": int(clean_company.sum()),
                "clean_company_response_coverage_pct": pct(int(clean_company.sum()), len(clean)),
                "detail_success": details_success,
                "company_reply_rows": company_reply_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
