#!/usr/bin/env python3
"""
Orchestrator: link collection -> detail collection -> validation.

Usage:
    python scripts/sikayetvar/run_sikayetvar_scraper.py               # full run
    python scripts/sikayetvar/run_sikayetvar_scraper.py --smoke-test  # MAX_PAGES=1, 5 details
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                         help="MAX_PAGES=1 and a 5-complaint detail limit, for a quick end-to-end check.")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap on complaints fetched in the detail stage.")
    args = parser.parse_args()

    max_pages = 1 if args.smoke_test else args.max_pages
    limit = 5 if args.smoke_test else args.limit

    links_args = [str(SCRIPT_DIR / "sikayetvar_collect_links.py")]
    if max_pages is not None:
        links_args += ["--max-pages", str(max_pages)]
    run(links_args)

    details_args = [str(SCRIPT_DIR / "sikayetvar_collect_details.py")]
    if limit is not None:
        details_args += ["--limit", str(limit)]
    run(details_args)

    run([str(SCRIPT_DIR / "sikayetvar_validate.py")])


if __name__ == "__main__":
    main()
