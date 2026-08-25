#!/usr/bin/env python3
"""Turn the discovery output into human-reviewable reports.

Reads sikayetvar_hotel_mapping.csv + sikayetvar_mapping_candidates.csv
(produced by 01_discover_hotels.py) and writes:

  reports/sikayetvar_all_hotels_manual_review.csv
      One row per non-auto-accepted hotel (REVIEW_REQUIRED / AMBIGUOUS),
      showing its top-3 candidates side by side for a human to judge.

  reports/sikayetvar_all_hotels_mapping_summary.csv
      match_status distribution (overall and per area) -- section 36 of
      the spec. Pure reporting; does not change any mapping decision.

Re-run any time after 01 (or after editing config/sikayetvar_manual_aliases.json
and re-running 01) to refresh these reports -- it never re-fetches anything.

Usage:
    python3 scripts/sikayetvar/02_review_discovery_matches.py
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from bodrum_intelligence.sikayetvar_matching import (  # noqa: E402
    AMBIGUOUS, AUTO_ACCEPTED_STATUSES, NOT_FOUND, PAGE_FOUND_NO_COMPLAINT, REVIEW_REQUIRED,
)
from bodrum_intelligence.sikayetvar_scraper import read_csv_rows, write_csv  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MAPPING_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_hotel_mapping.csv")
CANDIDATES_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_mapping_candidates.csv")
REVIEW_CSV = os.path.join(REPO_ROOT, "reports", "sikayetvar_all_hotels_manual_review.csv")
SUMMARY_CSV = os.path.join(REPO_ROOT, "reports", "sikayetvar_all_hotels_mapping_summary.csv")

REVIEW_FIELDS = [
    "hotel_id", "hotel_name", "area", "candidate_rank", "candidate_company_name",
    "candidate_url", "name_similarity", "candidate_score", "match_reason",
    "negative_conflict", "suggested_action",
]
ALL_STATUSES = [
    "FOUND_EXACT", "FOUND_HIGH_CONFIDENCE", REVIEW_REQUIRED, AMBIGUOUS,
    NOT_FOUND, PAGE_FOUND_NO_COMPLAINT, "EXCLUDED_WRONG_ENTITY", "ERROR",
]


def suggest_action(row: dict) -> str:
    if row["negative_brand_conflict"] in ("True", True):
        return "Confirm this isn't a sibling/other property before accepting"
    try:
        sim = float(row["name_similarity"])
    except (TypeError, ValueError):
        sim = 0.0
    if sim >= 0.8:
        return "Likely correct -- spot-check then promote to accepted"
    if sim >= 0.5:
        return "Uncertain -- open the URL and compare manually"
    return "Weak signal -- likely wrong, consider NOT_FOUND"


def main():
    mapping = read_csv_rows(MAPPING_CSV)
    candidates = read_csv_rows(CANDIDATES_CSV)
    if not mapping:
        print(f"No mapping found at {MAPPING_CSV}. Run 01_discover_hotels.py first.")
        return

    candidates_by_hotel = defaultdict(list)
    for c in candidates:
        candidates_by_hotel[c["hotel_id"]].append(c)

    review_rows = []
    for m in mapping:
        if m["match_status"] in AUTO_ACCEPTED_STATUSES | {NOT_FOUND, PAGE_FOUND_NO_COMPLAINT}:
            continue
        top3 = sorted(
            candidates_by_hotel.get(m["hotel_id"], []),
            key=lambda c: float(c["candidate_score"] or 0), reverse=True,
        )[:3]
        for c in top3:
            review_rows.append({
                "hotel_id": m["hotel_id"], "hotel_name": m["hotel_name"], "area": m["area"],
                "candidate_rank": c["candidate_rank"], "candidate_company_name": c["candidate_company_name"],
                "candidate_url": c["candidate_url"], "name_similarity": c["name_similarity"],
                "candidate_score": c["candidate_score"], "match_reason": m["match_reason"],
                "negative_conflict": c["negative_brand_conflict"], "suggested_action": suggest_action(c),
            })
    write_csv(REVIEW_CSV, REVIEW_FIELDS, review_rows)
    print(f"Wrote {REVIEW_CSV} ({len(review_rows)} candidate row(s) "
          f"for {len({r['hotel_id'] for r in review_rows})} hotel(s) needing review)")

    total = len(mapping)
    status_counts = defaultdict(int)
    for m in mapping:
        status_counts[m["match_status"]] += 1
    summary_rows = [
        {"match_status": s, "hotel_count": status_counts.get(s, 0),
         "share_pct": round(100 * status_counts.get(s, 0) / total, 1) if total else 0.0}
        for s in ALL_STATUSES if status_counts.get(s, 0) or True
    ]

    by_area = defaultdict(lambda: defaultdict(int))
    for m in mapping:
        by_area[m["area"]]["hotel_count"] += 1
        by_area[m["area"]][m["match_status"]] += 1
    area_rows = [
        {"area": area, "hotel_count": counts["hotel_count"],
         "found_exact": counts.get("FOUND_EXACT", 0),
         "found_high_confidence": counts.get("FOUND_HIGH_CONFIDENCE", 0),
         "review_required": counts.get(REVIEW_REQUIRED, 0),
         "ambiguous": counts.get(AMBIGUOUS, 0),
         "not_found": counts.get(NOT_FOUND, 0),
         "page_found_no_complaint": counts.get(PAGE_FOUND_NO_COMPLAINT, 0)}
        for area, counts in sorted(by_area.items())
    ]

    write_csv(SUMMARY_CSV, ["match_status", "hotel_count", "share_pct"], summary_rows)
    area_csv = SUMMARY_CSV.replace(".csv", "_by_area.csv")
    write_csv(area_csv, ["area", "hotel_count", "found_exact", "found_high_confidence",
                          "review_required", "ambiguous", "not_found", "page_found_no_complaint"], area_rows)
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {area_csv}")

    accepted = sum(status_counts.get(s, 0) for s in AUTO_ACCEPTED_STATUSES)
    print(f"\nTotal project hotels: {total}")
    print(f"Auto-accepted for scraping (FOUND_EXACT + FOUND_HIGH_CONFIDENCE): {accepted}")
    print(f"Needing manual review (REVIEW_REQUIRED + AMBIGUOUS): "
          f"{status_counts.get(REVIEW_REQUIRED, 0) + status_counts.get(AMBIGUOUS, 0)}")


if __name__ == "__main__":
    main()
