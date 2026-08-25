#!/usr/bin/env python3
"""
Validate the collected Sikayetvar dataset and produce:
  - a console report (counts, duplicates, missing fields, match status distribution)
  - reports/sikayetvar_top3_coverage.csv   (per-hotel coverage)
  - reports/sikayetvar_scraping_summary.txt (plain-text summary, real numbers only)

Usage:
    python scripts/sikayetvar/sikayetvar_validate.py
"""
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from bodrum_intelligence.sikayetvar_scraper import (  # noqa: E402
    EXCLUDED, MATCHED, REVIEW_REQUIRED, load_targets, parse_display_date,
)

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CONFIG_PATH = os.path.join(REPO_ROOT, "config", "sikayetvar_targets.json")
LINKS_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_complaint_links.csv")
COMPLAINTS_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_top3_complaints_raw.csv")
COVERAGE_CSV = os.path.join(REPO_ROOT, "reports", "sikayetvar_top3_coverage.csv")
SUMMARY_TXT = os.path.join(REPO_ROOT, "reports", "sikayetvar_scraping_summary.txt")


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    os.makedirs(os.path.dirname(COVERAGE_CSV), exist_ok=True)
    targets = load_targets(CONFIG_PATH)
    links = read_csv(LINKS_CSV)
    complaints = read_csv(COMPLAINTS_CSV)

    print("=" * 70)
    print("SIKAYETVAR DATASET VALIDATION")
    print("=" * 70)

    total = len(complaints)
    urls = [c["complaint_url"] for c in complaints]
    unique_urls = set(urls)
    dup_count = len(urls) - len(unique_urls)

    print(f"\nTotal complaint rows:        {total}")
    print(f"Unique complaint URLs:       {len(unique_urls)}")
    print(f"Duplicate URL rows:          {dup_count}")

    missing_title = sum(1 for c in complaints if not c.get("complaint_title"))
    missing_text = sum(1 for c in complaints if not c.get("complaint_text"))
    missing_date = sum(1 for c in complaints if not c.get("complaint_date_raw"))
    print(f"Missing complaint_title:     {missing_title}")
    print(f"Missing complaint_text:      {missing_text}")
    print(f"Missing complaint_date_raw:  {missing_date}")

    hotel_counts = Counter(c["hotel_name"] for c in complaints)
    print("\nComplaints per hotel:")
    for hotel, count in hotel_counts.most_common():
        print(f"  {hotel}: {count}")

    resp_count = sum(1 for c in complaints if str(c.get("company_response_exists")).lower() == "true")
    prog_count = sum(1 for c in complaints if str(c.get("progress_exists")).lower() == "true")
    print(f"\nCompany response rate:       {resp_count}/{total}"
          f" ({resp_count / total:.1%})" if total else "\nCompany response rate:       0/0")
    print(f"Progress-update rate:        {prog_count}/{total}"
          f" ({prog_count / total:.1%})" if total else "Progress-update rate:        0/0")

    status_counts = Counter(c["entity_match_status"] for c in complaints)
    print("\nEntity match status distribution:")
    for status in (MATCHED, REVIEW_REQUIRED, EXCLUDED):
        print(f"  {status}: {status_counts.get(status, 0)}")

    selectum_target = next(
        (t for t in targets if t.requires_entity_validation), None
    )
    if selectum_target:
        sel_complaints = [c for c in complaints if c["hotel_name"] == selectum_target.canonical_hotel_name]
        sel_excluded = sum(1 for c in sel_complaints if c["entity_match_status"] == EXCLUDED)
        sel_review = sum(1 for c in sel_complaints if c["entity_match_status"] == REVIEW_REQUIRED)
        print(f"\nSelectum excluded (other property): {sel_excluded}")
        print(f"Selectum review-required:           {sel_review}")

    # --- reports/sikayetvar_top3_coverage.csv ---
    coverage_rows = []
    for target in targets:
        hotel = target.canonical_hotel_name
        hotel_complaints = [c for c in complaints if c["hotel_name"] == hotel]
        matched = [c for c in hotel_complaints if c["entity_match_status"] == MATCHED]
        review = [c for c in hotel_complaints if c["entity_match_status"] == REVIEW_REQUIRED]
        excl = [c for c in hotel_complaints if c["entity_match_status"] == EXCLUDED]
        resp = [c for c in hotel_complaints if str(c.get("company_response_exists")).lower() == "true"]
        dated = sorted(
            (c["complaint_date_raw"] for c in hotel_complaints if c.get("complaint_date_raw")),
            key=lambda raw: parse_display_date(raw) or (0, 0, 0, 0, 0),
        )
        coverage_rows.append({
            "hotel_name": hotel,
            "source_pages": "; ".join(target.source_pages),
            "complaints_collected": len(hotel_complaints),
            "matched_complaints": len(matched),
            "review_required": len(review),
            "excluded": len(excl),
            "company_response_count": len(resp),
            "company_response_rate": f"{len(resp) / len(hotel_complaints):.1%}" if hotel_complaints else "0.0%",
            "date_min": dated[0] if dated else "",
            "date_max": dated[-1] if dated else "",
        })

    with open(COVERAGE_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "hotel_name", "source_pages", "complaints_collected", "matched_complaints",
            "review_required", "excluded", "company_response_count",
            "company_response_rate", "date_min", "date_max",
        ])
        writer.writeheader()
        writer.writerows(coverage_rows)
    print(f"\nWrote {COVERAGE_CSV}")

    # --- reports/sikayetvar_scraping_summary.txt ---
    status_rows = read_csv(os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_scrape_status.csv"))
    failed_total = sum(int(r.get("details_failed") or 0) for r in status_rows)
    lines = [
        "SIKAYETVAR TOP-3 BODRUM HOTELS — SCRAPING SUMMARY",
        "=" * 55,
        "",
    ]
    for row in coverage_rows:
        lines.append(f"{row['hotel_name']} -> {row['complaints_collected']} complaint(s) "
                      f"(matched: {row['matched_complaints']}, review: {row['review_required']}, "
                      f"excluded: {row['excluded']})")
    lines += [
        "",
        f"Toplam benzersiz complaint -> {len(unique_urls)}",
        f"Duplicate URL kaldirilan kayit -> {dup_count}",
        f"Company response bulunan -> {resp_count}",
        f"Review required -> {status_counts.get(REVIEW_REQUIRED, 0)}",
        f"Selectum'dan excluded (baska tesis) -> {status_counts.get(EXCLUDED, 0)}",
        f"Links discovered (raw, all source pages) -> {len(links)}",
        f"Failed detail pages -> {failed_total}",
    ]
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
