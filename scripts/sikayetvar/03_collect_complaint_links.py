#!/usr/bin/env python3
"""Collect complaint detail-page links for every accepted hotel mapping.

Only FOUND_EXACT / FOUND_HIGH_CONFIDENCE hotels are crawled by default
(the full-scraping gate, section 37 of the spec); pass
--allow-review-required to also include REVIEW_REQUIRED hotels (never
recommended for an unattended full run -- section 34: fewer reliable
matches beats many wrong ones).

Hotels that share a Sikayetvar URL (chain/umbrella accounts, e.g. multiple
Selectum properties under /selectum-hotels/...) are crawled ONCE per unique
URL; the resulting complaint links are then written out once per sharing
hotel_id, each tagged with a *preliminary* entity_match_status computed
from just the listing-card title (cheap, early signal only -- the
authoritative per-complaint decision happens in 04 using the full detail
page text).

Usage:
    python3 scripts/sikayetvar/03_collect_complaint_links.py --max-pages 1   # smoke test
    python3 scripts/sikayetvar/03_collect_complaint_links.py                 # full run
"""
import argparse
import json
import logging
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from bodrum_intelligence.sikayetvar_matching import (  # noqa: E402
    AUTO_ACCEPTED_STATUSES, REVIEW_REQUIRED, build_complaint_validation_terms,
)
from bodrum_intelligence.sikayetvar_scraper import (  # noqa: E402
    AntiBotBlock, append_rows, build_session, canonicalize_url, entity_match,
    paginate_listing, read_csv_rows, utc_now_iso, write_csv,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("sikayetvar.links")

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MAPPING_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_hotel_mapping.csv")
LINKS_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_all_complaint_links.csv")
STATUS_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_scrape_status_all_hotels.csv")
MANUAL_ALIASES = os.path.join(REPO_ROOT, "config", "sikayetvar_manual_aliases.json")
DISCOVERY_CONFIG = os.path.join(REPO_ROOT, "config", "sikayetvar_discovery_config.json")

LINK_FIELDS = [
    "hotel_id", "hotel_name", "area", "sikayetvar_company_name", "source_page",
    "complaint_url", "canonical_complaint_url", "discovered_page", "discovered_at",
    "entity_match_status", "entity_match_score",
]
STATUS_FIELDS = [
    "hotel_id", "hotel_name", "area", "mapping_status", "mapping_score", "source_page",
    "source_page_accessible", "complaint_links_found", "unique_complaints_found",
    "details_success", "details_failed", "company_response_count", "last_page_checked",
    "scrape_status", "last_error", "last_updated",
]


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--allow-review-required", action="store_true")
    parser.add_argument("--hotel-id", default=None)
    parser.add_argument("--force", action="store_true", help="Re-crawl source pages already present in the links CSV.")
    args = parser.parse_args()

    config = load_json(DISCOVERY_CONFIG)
    delay = config.get("request_delay_seconds", 1.5)
    manual_aliases = load_json(MANUAL_ALIASES)

    mapping = read_csv_rows(MAPPING_CSV)
    if not mapping:
        log.info(f"No mapping found at {MAPPING_CSV}. Run 01_discover_hotels.py first.")
        return

    allowed_statuses = set(AUTO_ACCEPTED_STATUSES)
    if args.allow_review_required:
        allowed_statuses.add(REVIEW_REQUIRED)
    accepted = [m for m in mapping if m["match_status"] in allowed_statuses and m["sikayetvar_url"]]
    if args.hotel_id:
        accepted = [m for m in accepted if m["hotel_id"] == args.hotel_id]
    log.info(f"{len(accepted)}/{len(mapping)} hotel(s) accepted for link collection "
             f"(statuses: {sorted(allowed_statuses)})")

    by_url = defaultdict(list)
    for m in accepted:
        by_url[m["sikayetvar_url"]].append(m)

    already_crawled = set()
    if not args.force:
        already_crawled = {r["source_page"] for r in read_csv_rows(LINKS_CSV)}

    session = build_session()
    status_updates = {}

    pending = {u: h for u, h in by_url.items() if u not in already_crawled}
    log.info(f"Resume: {len(already_crawled & set(by_url))} source page(s) already crawled, "
             f"{len(pending)} remaining.")

    for i, (source_page, hotels_here) in enumerate(sorted(pending.items()), 1):
        names = ", ".join(h["hotel_name"] for h in hotels_here)
        log.info(f"[{i}/{len(pending)}] {source_page}  ({names})")
        try:
            links, last_page, blocked = paginate_listing(session, source_page, max_pages=args.max_pages,
                                                           delay=delay, log=log.info)
        except AntiBotBlock as exc:
            log.warning(f"  [BLOCKED_SAFE_STOP] {exc}")
            for h in hotels_here:
                status_updates[h["hotel_id"]] = {"scrape_status": "BLOCKED_SAFE_STOP", "last_error": str(exc)}
            break

        now = utc_now_iso()
        for hotel in hotels_here:
            terms = build_complaint_validation_terms(
                hotel["hotel_id"], hotel["hotel_name"], source_page, accepted, manual_aliases,
            )
            rows = []
            for complaint_url, title, page_num in links:
                status, _ = entity_match(
                    complaint_url, title, "", "", "",
                    terms.match_patterns, terms.exclude_patterns, terms.ambiguous_terms,
                    requires_validation=terms.requires_validation,
                )
                rows.append({
                    "hotel_id": hotel["hotel_id"], "hotel_name": hotel["hotel_name"],
                    "area": hotel["area"], "sikayetvar_company_name": hotel["sikayetvar_company_name"],
                    "source_page": source_page, "complaint_url": complaint_url,
                    "canonical_complaint_url": canonicalize_url(complaint_url),
                    "discovered_page": page_num, "discovered_at": now,
                    "entity_match_status": status, "entity_match_score": "",
                })
            append_rows(LINKS_CSV, LINK_FIELDS, rows)
            status_updates[hotel["hotel_id"]] = {
                "hotel_name": hotel["hotel_name"], "area": hotel["area"],
                "mapping_status": hotel["match_status"], "mapping_score": hotel["match_score"],
                "source_page": source_page, "source_page_accessible": True,
                "complaint_links_found": len(links),
                "unique_complaints_found": len({r["canonical_complaint_url"] for r in rows}),
                "last_page_checked": last_page, "scrape_status": "LINKS_COLLECTED", "last_error": "",
            }
            log.info(f"    {hotel['hotel_name']}: {len(rows)} link(s)")

        if blocked:
            break

    prior_status = {r["hotel_id"]: r for r in read_csv_rows(STATUS_CSV)}
    for hotel_id, updates in status_updates.items():
        row = prior_status.get(hotel_id, {f: "" for f in STATUS_FIELDS})
        row["hotel_id"] = hotel_id
        row.update(updates)
        row["last_updated"] = utc_now_iso()
        prior_status[hotel_id] = row
    write_csv(STATUS_CSV, STATUS_FIELDS, list(prior_status.values()))

    log.info(f"\nWrote {LINKS_CSV}")
    log.info(f"Wrote {STATUS_CSV}")


if __name__ == "__main__":
    main()
