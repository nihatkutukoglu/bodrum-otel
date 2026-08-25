#!/usr/bin/env python3
"""Orchestrator: discovery -> review report -> link collection -> detail
collection -> validation, for the full 192-hotel Sikayetvar pipeline.

Usage:
    python3 scripts/sikayetvar/run_all_hotels_pipeline.py --smoke-test
        10 hotels, MAX_PAGES=1, 20-complaint detail cap -- a quick
        end-to-end check (section 32 of the spec).

    python3 scripts/sikayetvar/run_all_hotels_pipeline.py --dry-run
        Discovery + review report only, no complaint scraping (section 55).

    python3 scripts/sikayetvar/run_all_hotels_pipeline.py
        Full run: all hotels, all pages, all accepted-hotel complaints.

    python3 scripts/sikayetvar/run_all_hotels_pipeline.py --allow-review-required
        Also scrape REVIEW_REQUIRED hotels (off by default -- section 37 gate).
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def run(args):
    print(f"\n$ {' '.join(args)}")
    result = subprocess.run([sys.executable] + args)
    if result.returncode != 0:
        print(f"[!] {args[1]} exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Discovery + review report only.")
    parser.add_argument("--max-hotels", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap on complaint_urls fetched in stage 04.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-review-required", action="store_true")
    args = parser.parse_args()

    max_hotels = 10 if args.smoke_test else args.max_hotels
    max_pages = 1 if args.smoke_test else args.max_pages
    limit = 20 if args.smoke_test else args.limit

    discover_args = [str(SCRIPT_DIR / "01_discover_hotels.py")]
    if max_hotels is not None:
        discover_args += ["--max-hotels", str(max_hotels)]
    if args.resume:
        discover_args += ["--resume"]
    run(discover_args)

    run([str(SCRIPT_DIR / "02_review_discovery_matches.py")])

    if args.dry_run:
        print("\n--dry-run: stopping before complaint link/detail collection.")
        return

    links_args = [str(SCRIPT_DIR / "03_collect_complaint_links.py")]
    if max_pages is not None:
        links_args += ["--max-pages", str(max_pages)]
    if args.allow_review_required:
        links_args += ["--allow-review-required"]
    run(links_args)

    details_args = [str(SCRIPT_DIR / "04_collect_complaint_details.py")]
    if limit is not None:
        details_args += ["--limit", str(limit)]
    run(details_args)

    run([str(SCRIPT_DIR / "05_validate_all_hotels.py")])


if __name__ == "__main__":
    main()
