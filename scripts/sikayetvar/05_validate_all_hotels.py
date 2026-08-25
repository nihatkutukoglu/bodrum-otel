#!/usr/bin/env python3
"""Validate the all-hotel Sikayetvar dataset and produce final reports.

Checks (section 46 of the spec): hotel master row count, mapping row
count/uniqueness, complaint URL uniqueness, complaint_id uniqueness (within
a hotel -- duplicated *across* hotels is expected on shared/chain pages),
orphan complaints/replies, entity status distributions, missing
text/date/company fields, company-response consistency, reply-count
consistency.

Writes:
  reports/sikayetvar_all_hotels_validation_summary.csv
  reports/sikayetvar_orphan_complaints.csv
  reports/sikayetvar_orphan_replies.csv
  reports/sikayetvar_all_hotels_coverage.csv
  reports/sikayetvar_coverage_by_area.csv
  reports/sikayetvar_all_hotels_scraping_summary.txt

Usage:
    python3 scripts/sikayetvar/05_validate_all_hotels.py
"""
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from bodrum_intelligence.sikayetvar_matching import (  # noqa: E402
    AMBIGUOUS, AUTO_ACCEPTED_STATUSES, NOT_FOUND, PAGE_FOUND_NO_COMPLAINT,
)
from bodrum_intelligence.sikayetvar_matching import REVIEW_REQUIRED as MAPPING_REVIEW_REQUIRED  # noqa: E402
from bodrum_intelligence.sikayetvar_scraper import (  # noqa: E402
    EXCLUDED, MATCHED, parse_display_date, read_csv_rows, write_csv,
)
from bodrum_intelligence.sikayetvar_scraper import REVIEW_REQUIRED as COMPLAINT_REVIEW_REQUIRED  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
HOTEL_MASTER = os.path.join(REPO_ROOT, "data", "processed", "hotels_enriched.csv")
MAPPING_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_hotel_mapping.csv")
LINKS_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_all_complaint_links.csv")
COMPLAINTS_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_all_hotels_complaints_raw.csv")
REPLIES_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_all_hotels_replies_raw.csv")

VALIDATION_CSV = os.path.join(REPO_ROOT, "reports", "sikayetvar_all_hotels_validation_summary.csv")
ORPHAN_COMPLAINTS_CSV = os.path.join(REPO_ROOT, "reports", "sikayetvar_orphan_complaints.csv")
ORPHAN_REPLIES_CSV = os.path.join(REPO_ROOT, "reports", "sikayetvar_orphan_replies.csv")
COVERAGE_CSV = os.path.join(REPO_ROOT, "reports", "sikayetvar_all_hotels_coverage.csv")
AREA_COVERAGE_CSV = os.path.join(REPO_ROOT, "reports", "sikayetvar_coverage_by_area.csv")
SUMMARY_TXT = os.path.join(REPO_ROOT, "reports", "sikayetvar_all_hotels_scraping_summary.txt")


def check(name, value, note=""):
    return {"check": name, "value": value, "note": note}


def main():
    hotels = read_csv_rows(HOTEL_MASTER)
    mapping = read_csv_rows(MAPPING_CSV)
    links = read_csv_rows(LINKS_CSV)
    complaints = read_csv_rows(COMPLAINTS_CSV)
    replies = read_csv_rows(REPLIES_CSV)

    checks = []
    checks.append(check("hotel_master_row_count", len(hotels)))
    checks.append(check("mapping_row_count", len(mapping)))
    mapping_hotel_ids = [m["hotel_id"] for m in mapping]
    checks.append(check("mapping_unique_hotel_ids", len(set(mapping_hotel_ids)),
                         "should equal mapping_row_count" if len(set(mapping_hotel_ids)) == len(mapping_hotel_ids)
                         else "DUPLICATE hotel_id ROWS IN MAPPING"))

    hotel_master_ids = {h["hotel_id"] for h in hotels}
    orphan_complaints = [c for c in complaints if c["hotel_id"] not in hotel_master_ids]
    write_csv(ORPHAN_COMPLAINTS_CSV, list(complaints[0].keys()) if complaints else ["hotel_id"], orphan_complaints)
    checks.append(check("orphan_complaints_no_hotel_master_row", len(orphan_complaints)))

    complaint_keys = {(c["hotel_id"], c["canonical_complaint_url"]) for c in complaints}
    orphan_replies = [r for r in replies if (r["hotel_id"], r["canonical_complaint_url"]) not in complaint_keys]
    write_csv(ORPHAN_REPLIES_CSV, list(replies[0].keys()) if replies else ["hotel_id"], orphan_replies)
    checks.append(check("orphan_replies_no_matching_complaint_row", len(orphan_replies)))

    within_hotel_pairs = [(c["hotel_id"], c["complaint_id"]) for c in complaints]
    dup_within_hotel = len(within_hotel_pairs) - len(set(within_hotel_pairs))
    checks.append(check("duplicate_complaint_id_within_same_hotel", dup_within_hotel))

    global_urls = [c["canonical_complaint_url"] for c in complaints]
    checks.append(check("total_complaint_rows", len(complaints)))
    checks.append(check("unique_canonical_complaint_urls_across_all_hotels", len(set(global_urls))))

    # Section 68: links discovered vs. complaint detail rows actually
    # collected -- a gap is expected while a run is still in progress
    # (resumable, stage-by-stage) but should shrink to ~0 once 04 finishes.
    checks.append(check("links_discovered_hotel_complaint_pairs", len(links)))
    checks.append(check("links_minus_complaint_rows_gap", len(links) - len(complaints)))

    checks.append(check("missing_complaint_title", sum(1 for c in complaints if not c.get("complaint_title"))))
    checks.append(check("missing_complaint_text", sum(1 for c in complaints if not c.get("complaint_text"))))
    checks.append(check("missing_complaint_date", sum(1 for c in complaints if not c.get("complaint_date_raw"))))
    checks.append(check("missing_sikayetvar_company_name",
                         sum(1 for c in complaints if not c.get("sikayetvar_company_name"))))

    resp_true = [c for c in complaints if str(c.get("company_response_exists")).lower() == "true"]
    resp_inconsistent = sum(1 for c in resp_true if not c.get("company_response_text"))
    checks.append(check("company_response_exists_true_but_text_empty", resp_inconsistent))

    status_counts = Counter(c["entity_match_status"] for c in complaints)
    for s in (MATCHED, COMPLAINT_REVIEW_REQUIRED, EXCLUDED):
        checks.append(check(f"entity_status_{s}", status_counts.get(s, 0)))

    mapping_status_counts = Counter(m["match_status"] for m in mapping)
    for s in (
        "FOUND_EXACT", "FOUND_HIGH_CONFIDENCE", MAPPING_REVIEW_REQUIRED, AMBIGUOUS,
        NOT_FOUND, PAGE_FOUND_NO_COMPLAINT, "EXCLUDED_WRONG_ENTITY", "ERROR",
    ):
        checks.append(check(f"mapping_status_{s}", mapping_status_counts.get(s, 0)))

    write_csv(VALIDATION_CSV, ["check", "value", "note"], checks)
    print(f"Wrote {VALIDATION_CSV}")

    # --- coverage per hotel (section 42) ---
    complaints_by_hotel = defaultdict(list)
    for c in complaints:
        complaints_by_hotel[c["hotel_id"]].append(c)

    coverage_rows = []
    for m in mapping:
        hc = complaints_by_hotel.get(m["hotel_id"], [])
        matched = [c for c in hc if c["entity_match_status"] == MATCHED]
        review = [c for c in hc if c["entity_match_status"] == COMPLAINT_REVIEW_REQUIRED]
        excl = [c for c in hc if c["entity_match_status"] == EXCLUDED]
        resp = [c for c in hc if str(c.get("company_response_exists")).lower() == "true"]
        dated = sorted(
            (c["complaint_date_raw"] for c in hc if c.get("complaint_date_raw")),
            key=lambda raw: parse_display_date(raw) or (0, 0, 0, 0, 0),
        )
        coverage_rows.append({
            "hotel_id": m["hotel_id"], "hotel_name": m["hotel_name"], "area": m["area"],
            "mapping_status": m["match_status"], "sikayetvar_url": m["sikayetvar_url"],
            "page_found": bool(m["sikayetvar_url"]), "page_accessible": m.get("page_accessible", ""),
            "complaint_count_matched": len(matched), "complaint_count_review_required": len(review),
            "complaint_count_excluded": len(excl), "company_response_count": len(resp),
            "company_response_rate_in_scraped_corpus": f"{len(resp) / len(matched):.1%}" if matched else "0.0%",
            "first_complaint_date": dated[0] if dated else "", "last_complaint_date": dated[-1] if dated else "",
            "last_checked": m.get("checked_at", ""),
        })
    write_csv(COVERAGE_CSV, list(coverage_rows[0].keys()) if coverage_rows else ["hotel_id"], coverage_rows)
    print(f"Wrote {COVERAGE_CSV}")

    # --- suspicious brand-page volume outliers (section 67) ---
    outliers = [r for r in coverage_rows if r["complaint_count_matched"] >= 100]
    if outliers:
        outlier_csv = os.path.join(REPO_ROOT, "reports", "sikayetvar_suspicious_brand_volume.csv")
        write_csv(outlier_csv, list(outliers[0].keys()), outliers)
        print(f"Wrote {outlier_csv} ({len(outliers)} hotel(s) flagged SUSPICIOUS_BRAND_PAGE_VOLUME)")

    # --- area coverage (section 43) ---
    by_area = defaultdict(lambda: {"project_hotel_count": 0, "mapped_hotel_count": 0,
                                    "hotels_with_complaints": 0, "hotels_with_zero_complaints": 0,
                                    "not_found_hotels": 0, "matched_complaint_count": 0})
    coverage_by_id = {r["hotel_id"]: r for r in coverage_rows}
    for h in hotels:
        area = h.get("area", "")
        by_area[area]["project_hotel_count"] += 1
        cov = coverage_by_id.get(h["hotel_id"])
        if not cov:
            continue
        if cov["mapping_status"] in AUTO_ACCEPTED_STATUSES:
            by_area[area]["mapped_hotel_count"] += 1
        if cov["mapping_status"] == NOT_FOUND:
            by_area[area]["not_found_hotels"] += 1
        if cov["complaint_count_matched"] > 0:
            by_area[area]["hotels_with_complaints"] += 1
        elif cov["mapping_status"] in AUTO_ACCEPTED_STATUSES:
            by_area[area]["hotels_with_zero_complaints"] += 1
        by_area[area]["matched_complaint_count"] += cov["complaint_count_matched"]

    area_rows = []
    for area, d in sorted(by_area.items()):
        d["area"] = area
        d["mapped_hotel_rate_pct"] = round(100 * d["mapped_hotel_count"] / d["project_hotel_count"], 1) \
            if d["project_hotel_count"] else 0.0
        area_rows.append(d)
    write_csv(AREA_COVERAGE_CSV, ["area", "project_hotel_count", "mapped_hotel_count", "mapped_hotel_rate_pct",
                                   "hotels_with_complaints", "hotels_with_zero_complaints", "not_found_hotels",
                                   "matched_complaint_count"], area_rows)
    print(f"Wrote {AREA_COVERAGE_CSV}")

    # --- scraping summary (section 48) ---
    top10 = sorted(coverage_rows, key=lambda r: r["complaint_count_matched"], reverse=True)[:10]
    total_replies = len(replies)
    status_rows = read_csv_rows(os.path.join(REPO_ROOT, "data", "raw", "sikayetvar",
                                               "sikayetvar_scrape_status_all_hotels.csv"))
    total_details_failed = sum(int(r.get("details_failed") or 0) for r in status_rows)
    lines = [
        "SIKAYETVAR ALL-HOTELS SCRAPING SUMMARY",
        "=" * 55,
        "",
        f"Project hotels: {len(hotels)}",
        f"Exact matches: {mapping_status_counts.get('FOUND_EXACT', 0)}",
        f"High confidence matches: {mapping_status_counts.get('FOUND_HIGH_CONFIDENCE', 0)}",
        f"Review required: {mapping_status_counts.get(MAPPING_REVIEW_REQUIRED, 0)}",
        f"Ambiguous: {mapping_status_counts.get(AMBIGUOUS, 0)}",
        f"Not found: {mapping_status_counts.get(NOT_FOUND, 0)}",
        f"Page found no complaint: {mapping_status_counts.get(PAGE_FOUND_NO_COMPLAINT, 0)}",
        f"Hotels with >=1 matched complaint: {sum(1 for r in coverage_rows if r['complaint_count_matched'] > 0)}",
        f"Total unique complaints (matched, across hotels): {status_counts.get(MATCHED, 0)}",
        f"Complaint review required: {status_counts.get(COMPLAINT_REVIEW_REQUIRED, 0)}",
        f"Complaint excluded wrong property: {status_counts.get(EXCLUDED, 0)}",
        f"Company responses: {sum(1 for c in complaints if str(c.get('company_response_exists')).lower() == 'true')}",
        f"Replies: {total_replies}",
        f"Detail rows collected: {len(complaints)}",
        f"Failed details: {total_details_failed}",
        "",
        "Top 10 complaint-volume hotels:",
    ]
    for r in top10:
        if r["complaint_count_matched"] > 0:
            lines.append(f"  {r['hotel_name']} ({r['area']}) -> {r['complaint_count_matched']} matched complaint(s)")
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
