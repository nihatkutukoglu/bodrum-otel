#!/usr/bin/env python3
"""
Read data/raw/sikayetvar/sikayetvar_complaint_links.csv, fetch each
not-yet-collected complaint detail page, run Selectum entity-match
validation, and append results to:

  data/raw/sikayetvar/sikayetvar_top3_complaints_raw.csv  (one row per complaint)
  data/raw/sikayetvar/sikayetvar_replies.csv               (one row per reply/answer)

Resumable: complaint_url values already present in the raw complaints CSV
are skipped. Writes/flushes after every complaint so a partial run keeps
its progress.

Usage:
    python scripts/sikayetvar/sikayetvar_collect_details.py
    python scripts/sikayetvar/sikayetvar_collect_details.py --limit 5   # smoke test
"""
import argparse
import os
import sys
from collections import defaultdict
from urllib.parse import urlsplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from bodrum_intelligence.sikayetvar_scraper import (  # noqa: E402
    AntiBotBlock, COMPLAINTS_FIELDS, EXCLUDED, MATCHED, REPLIES_FIELDS,
    REQUEST_DELAY_SECONDS, REVIEW_REQUIRED, append_rows, build_session,
    entity_match, fetch_soup, load_targets, parse_complaint_detail,
    polite_sleep, read_existing_values, update_status,
)
import csv

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CONFIG_PATH = os.path.join(REPO_ROOT, "config", "sikayetvar_targets.json")
LINKS_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_complaint_links.csv")
COMPLAINTS_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_top3_complaints_raw.csv")
REPLIES_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_replies.csv")
STATUS_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_scrape_status.csv")


def read_link_rows(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY_SECONDS)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N not-yet-collected complaints (smoke test).")
    parser.add_argument("--config", default=CONFIG_PATH)
    args = parser.parse_args()

    targets = {t.canonical_hotel_name: t for t in load_targets(args.config)}
    link_rows = read_link_rows(LINKS_CSV)
    if not link_rows:
        print(f"No links found in {LINKS_CSV}. Run sikayetvar_collect_links.py first.")
        return

    already_done = read_existing_values(COMPLAINTS_CSV, "complaint_url")
    print(f"Resume: {len(already_done)} complaint(s) already collected.")

    pending = [r for r in link_rows if r["complaint_url"] not in already_done]
    # De-dup by complaint_url in case the same URL appears under >1 source page.
    seen = set()
    unique_pending = []
    for row in pending:
        if row["complaint_url"] in seen:
            continue
        seen.add(row["complaint_url"])
        unique_pending.append(row)
    pending = unique_pending
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(pending)} complaint(s) to fetch this run.")

    session = build_session()
    failed_this_run = defaultdict(int)  # (hotel, source_page) -> count

    for i, row in enumerate(pending, 1):
        hotel = row["canonical_hotel_name"]
        source_page = row["source_page"]
        url = row["complaint_url"]
        target = targets.get(hotel)
        company_hrefs = {urlsplit(sp).path for sp in target.source_pages} if target else set()

        print(f"[{i}/{len(pending)}] {url}")
        try:
            soup = fetch_soup(session, url)
        except AntiBotBlock as exc:
            print(f"  [BLOCKED] {exc} -- stopping run, partial results saved.")
            update_status(STATUS_CSV, {(hotel, source_page): {"status": "BLOCKED"}})
            break
        except Exception as exc:
            print(f"  [FAILED] {exc}")
            failed_this_run[(hotel, source_page)] += 1
            polite_sleep(args.delay)
            continue

        try:
            fields, replies = parse_complaint_detail(soup, url, company_hrefs=company_hrefs)
        except Exception as exc:
            print(f"  [FAILED to parse] {exc}")
            failed_this_run[(hotel, source_page)] += 1
            polite_sleep(args.delay)
            continue

        fields["hotel_name"] = hotel
        fields["area"] = row["area"]
        fields["source_page"] = source_page
        fields["collected_at"] = row["collected_at"]

        status, reason = entity_match(
            target, url, fields["complaint_title"], fields["complaint_text"],
            fields["category"], fields["product_name"],
        ) if target else (REVIEW_REQUIRED, "Unknown target (not in config)")
        fields["entity_match_status"] = status
        fields["entity_match_reason"] = reason

        append_rows(COMPLAINTS_CSV, COMPLAINTS_FIELDS, [fields])
        append_rows(REPLIES_CSV, REPLIES_FIELDS, replies)

        print(f"  -> {status} | title: {fields['complaint_title'][:60]!r}")
        polite_sleep(args.delay)

    # Recompute success/match counts from the raw CSV itself (ground truth,
    # correct across resumed runs) rather than accumulating per-run counters.
    all_complaints = read_link_rows(COMPLAINTS_CSV)
    by_key = defaultdict(lambda: {
        "details_success": 0, "matched_links": 0, "review_required": 0, "excluded": 0,
    })
    for c in all_complaints:
        key = (c["hotel_name"], c["source_page"])
        by_key[key]["details_success"] += 1
        if c["entity_match_status"] == MATCHED:
            by_key[key]["matched_links"] += 1
        elif c["entity_match_status"] == REVIEW_REQUIRED:
            by_key[key]["review_required"] += 1
        elif c["entity_match_status"] == EXCLUDED:
            by_key[key]["excluded"] += 1

    prior_status = {}
    if os.path.exists(STATUS_CSV):
        with open(STATUS_CSV, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                prior_status[(row["hotel_name"], row["source_page"])] = row

    all_keys = set(by_key) | set(failed_this_run) | set(prior_status)
    updates = {}
    for key in all_keys:
        prior_failed = int((prior_status.get(key) or {}).get("details_failed") or 0)
        updates[key] = dict(by_key.get(key, {}))
        updates[key]["details_failed"] = prior_failed + failed_this_run.get(key, 0)
        updates[key].setdefault("status", "DETAILS_COLLECTED")
    if updates:
        update_status(STATUS_CSV, updates)

    print(f"\nDone. Raw complaints CSV: {COMPLAINTS_CSV}")
    print(f"Replies CSV: {REPLIES_CSV}")


if __name__ == "__main__":
    main()
