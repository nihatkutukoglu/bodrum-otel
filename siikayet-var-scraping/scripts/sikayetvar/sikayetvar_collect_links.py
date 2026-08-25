#!/usr/bin/env python3
"""
Visit each target hotel's Sikayetvar source page(s), paginate, and collect
canonical complaint detail URLs into data/raw/sikayetvar/sikayetvar_complaint_links.csv.

Resumable: complaint URLs already present in the output CSV are not
re-added. Stops paginating a source page as soon as a page yields zero new
unique complaint URLs (handles both "ran out of pages" and the observed
Sikayetvar behavior of a past-the-end ?page=N clamping back to page 1).

Usage:
    python scripts/sikayetvar/sikayetvar_collect_links.py
    python scripts/sikayetvar/sikayetvar_collect_links.py --max-pages 1   # smoke test
    python scripts/sikayetvar/sikayetvar_collect_links.py --hotel "La Blanche"
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from bodrum_intelligence.sikayetvar_scraper import (  # noqa: E402
    LINKS_FIELDS, REQUEST_DELAY_SECONDS,
    append_rows, build_session, collect_links_for_source_page,
    load_targets, read_existing_values, update_status,
)

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CONFIG_PATH = os.path.join(REPO_ROOT, "config", "sikayetvar_targets.json")
LINKS_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_complaint_links.csv")
STATUS_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_scrape_status.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pages", type=int, default=None,
                         help="Max pages per source page (e.g. 1 for a smoke test).")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY_SECONDS)
    parser.add_argument("--hotel", default=None, help="Only process hotels whose name contains this substring.")
    parser.add_argument("--config", default=CONFIG_PATH)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(LINKS_CSV), exist_ok=True)
    targets = load_targets(args.config)
    if args.hotel:
        targets = [t for t in targets if args.hotel.lower() in t.canonical_hotel_name.lower()]
        if not targets:
            print(f"No target matches --hotel {args.hotel!r}")
            return

    session = build_session()
    global_seen = read_existing_values(LINKS_CSV, "complaint_url")
    print(f"Resume: {len(global_seen)} complaint URL(s) already in {LINKS_CSV}")

    any_block = False
    for target in targets:
        print(f"\n=== {target.canonical_hotel_name} ({target.area}) ===")
        for source_page in target.source_pages:
            print(f"  source page: {source_page}")
            rows, last_page, blocked = collect_links_for_source_page(
                session, target, source_page, max_pages=args.max_pages, delay=args.delay,
            )
            new_rows = [r for r in rows if r["complaint_url"] not in global_seen]
            for r in new_rows:
                global_seen.add(r["complaint_url"])
            append_rows(LINKS_CSV, LINKS_FIELDS, new_rows)
            print(f"  -> {len(rows)} link(s) found this run, {len(new_rows)} new, "
                  f"pages visited: {last_page}")

            update_status(STATUS_CSV, {
                (target.canonical_hotel_name, source_page): {
                    "links_found": len(rows),
                    "last_page": last_page,
                    "status": "BLOCKED" if blocked else "LINKS_COLLECTED",
                }
            })
            if blocked:
                any_block = True

    if any_block:
        print("\n[!] Anti-bot block detected on at least one source page. "
              "Partial results were saved. Stopping run safely.")
    print(f"\nDone. Links CSV: {LINKS_CSV}")


if __name__ == "__main__":
    main()
