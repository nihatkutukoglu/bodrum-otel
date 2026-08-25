#!/usr/bin/env python3
"""Fetch complaint detail pages and run per-complaint entity validation.

Reads sikayetvar_all_complaint_links.csv. Each distinct complaint_url is
fetched at most ONCE even if several hotels share it (chain/umbrella
pages) -- the fetched page is then validated separately against each
sharing hotel's own match/exclude terms, so the same complaint can end up
COMPLAINT_MATCHED for one hotel and COMPLAINT_EXCLUDED_OTHER_PROPERTY for
another (the Selectum Colours Bodrum / Selectum Collection Bodrum case).

Writes:
  sikayetvar_all_hotels_complaints_raw.csv  -- one row per (hotel, complaint)
  sikayetvar_all_hotels_replies_raw.csv     -- one row per reply, only for
                                                COMPLAINT_MATCHED complaints

Resumable: a (hotel_id, canonical_complaint_url) pair already present in
the raw complaints CSV is not re-evaluated; a complaint_url is not
re-fetched from the network once every hotel referencing it is resolved.

Usage:
    python3 scripts/sikayetvar/04_collect_complaint_details.py --limit 20   # smoke test
    python3 scripts/sikayetvar/04_collect_complaint_details.py              # full run
"""
import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from urllib.parse import urlsplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from bodrum_intelligence.sikayetvar_matching import (  # noqa: E402
    build_complaint_validation_terms,
)
from bodrum_intelligence.sikayetvar_scraper import (  # noqa: E402
    AntiBotBlock, COMPLAINT_DETAIL_FIELDS, EXCLUDED, MATCHED, REVIEW_REQUIRED,
    append_rows, build_session, entity_match, fetch_soup, parse_complaint_detail,
    polite_sleep, read_csv_rows, utc_now_iso, write_csv,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("sikayetvar.details")

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MAPPING_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_hotel_mapping.csv")
LINKS_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_all_complaint_links.csv")
COMPLAINTS_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_all_hotels_complaints_raw.csv")
REPLIES_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_all_hotels_replies_raw.csv")
STATUS_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_scrape_status_all_hotels.csv")
MANUAL_ALIASES = os.path.join(REPO_ROOT, "config", "sikayetvar_manual_aliases.json")
DISCOVERY_CONFIG = os.path.join(REPO_ROOT, "config", "sikayetvar_discovery_config.json")

COMPLAINT_FIELDS = [
    "complaint_id", "complaint_url", "canonical_complaint_url", "hotel_id", "hotel_name", "area",
    "sikayetvar_company_name",
] + COMPLAINT_DETAIL_FIELDS + [
    "entity_match_status", "entity_match_score", "entity_match_reason",
    "source_page", "collected_at",
]
REPLY_FIELDS = ["complaint_id", "canonical_complaint_url", "hotel_id", "hotel_name",
                 "reply_order", "reply_author_type", "reply_date_raw", "reply_text"]


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Only fetch the first N not-yet-collected complaint_urls.")
    args = parser.parse_args()

    config = load_json(DISCOVERY_CONFIG)
    delay = config.get("request_delay_seconds", 1.5)
    manual_aliases = load_json(MANUAL_ALIASES)
    mapping_rows = read_csv_rows(MAPPING_CSV)

    links = read_csv_rows(LINKS_CSV)
    if not links:
        log.info(f"No links found in {LINKS_CSV}. Run 03_collect_complaint_links.py first.")
        return

    hotels_by_complaint = defaultdict(list)
    seen_pairs = set()
    for r in links:
        key = (r["hotel_id"], r["canonical_complaint_url"])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        hotels_by_complaint[r["canonical_complaint_url"]].append({
            "hotel_id": r["hotel_id"], "hotel_name": r["hotel_name"], "area": r["area"],
            "sikayetvar_company_name": r["sikayetvar_company_name"], "source_page": r["source_page"],
            "url": r["canonical_complaint_url"], "complaint_url": r["complaint_url"],
        })

    already_done = {(r["hotel_id"], r["canonical_complaint_url"]) for r in read_csv_rows(COMPLAINTS_CSV)}
    log.info(f"Resume: {len(already_done)} (hotel, complaint) pair(s) already collected.")

    terms_cache = {}

    def terms_for(hotel_id, hotel_name, source_page):
        key = (hotel_id, source_page)
        if key not in terms_cache:
            terms_cache[key] = build_complaint_validation_terms(
                hotel_id, hotel_name, source_page, mapping_rows, manual_aliases,
            )
        return terms_cache[key]

    company_hrefs_cache = {}

    def company_hrefs_for(source_page):
        if source_page not in company_hrefs_cache:
            company_hrefs_cache[source_page] = {urlsplit(source_page).path}
        return company_hrefs_cache[source_page]

    pending_urls = [
        url for url, hotels in hotels_by_complaint.items()
        if any((h["hotel_id"], url) not in already_done for h in hotels)
    ]
    if args.limit:
        pending_urls = pending_urls[: args.limit]
    log.info(f"{len(pending_urls)} distinct complaint_url(s) to fetch this run "
             f"(covering {sum(len(hotels_by_complaint[u]) for u in pending_urls)} hotel-complaint pair(s)).")

    session = build_session()
    counts = defaultdict(lambda: defaultdict(int))
    now = utc_now_iso()

    for i, url in enumerate(pending_urls, 1):
        hotels_here = hotels_by_complaint[url]
        fetch_url = hotels_here[0]["complaint_url"]
        log.info(f"[{i}/{len(pending_urls)}] {url}  ({len(hotels_here)} candidate hotel(s))")
        try:
            soup = fetch_soup(session, fetch_url)
        except AntiBotBlock as exc:
            log.warning(f"  [BLOCKED_SAFE_STOP] {exc}")
            break
        except Exception as exc:
            log.warning(f"  [FETCH FAILED] {exc}")
            for h in hotels_here:
                counts[h["hotel_id"]]["details_failed"] += 1
            polite_sleep(delay)
            continue

        for h in hotels_here:
            if (h["hotel_id"], url) in already_done:
                continue
            try:
                fields, replies = parse_complaint_detail(
                    soup, url, company_hrefs=company_hrefs_for(h["source_page"]),
                )
            except Exception as exc:
                log.warning(f"  [PARSE FAILED for {h['hotel_name']}] {exc}")
                counts[h["hotel_id"]]["details_failed"] += 1
                continue

            terms = terms_for(h["hotel_id"], h["hotel_name"], h["source_page"])
            status, reason = entity_match(
                url, fields["complaint_title"], fields["complaint_text"],
                fields["category"], fields["product_name"],
                terms.match_patterns, terms.exclude_patterns, terms.ambiguous_terms,
                requires_validation=terms.requires_validation,
            )

            row = {
                "complaint_id": fields["complaint_id"], "complaint_url": url,
                "canonical_complaint_url": url, "hotel_id": h["hotel_id"], "hotel_name": h["hotel_name"],
                "area": h["area"], "sikayetvar_company_name": h["sikayetvar_company_name"],
                **fields, "entity_match_status": status, "entity_match_score": "",
                "entity_match_reason": reason, "source_page": h["source_page"], "collected_at": now,
            }
            append_rows(COMPLAINTS_CSV, COMPLAINT_FIELDS, [row])
            counts[h["hotel_id"]]["details_success"] += 1
            if fields["company_response_exists"]:
                counts[h["hotel_id"]]["company_response_count"] += 1

            if status == MATCHED:
                reply_rows = [
                    {"complaint_id": fields["complaint_id"], "canonical_complaint_url": url,
                     "hotel_id": h["hotel_id"], "hotel_name": h["hotel_name"], **r}
                    for r in replies
                ]
                append_rows(REPLIES_CSV, REPLY_FIELDS, reply_rows)

            log.info(f"    {h['hotel_name']}: {status}"
                     + (f" ({reason})" if status in (EXCLUDED, REVIEW_REQUIRED) else ""))

        polite_sleep(delay)

    prior_status = {r["hotel_id"]: r for r in read_csv_rows(STATUS_CSV)}
    for hotel_id, c in counts.items():
        row = prior_status.get(hotel_id, {})
        row["hotel_id"] = hotel_id
        row["details_success"] = int(row.get("details_success") or 0) + c["details_success"]
        row["details_failed"] = int(row.get("details_failed") or 0) + c["details_failed"]
        row["company_response_count"] = int(row.get("company_response_count") or 0) + c["company_response_count"]
        row["scrape_status"] = "DETAILS_COLLECTED"
        row["last_updated"] = utc_now_iso()
        prior_status[hotel_id] = row
    if prior_status:
        fieldnames = sorted({k for r in prior_status.values() for k in r})
        write_csv(STATUS_CSV, fieldnames, list(prior_status.values()))

    log.info(f"\nWrote {COMPLAINTS_CSV}")
    log.info(f"Wrote {REPLIES_CSV}")


if __name__ == "__main__":
    main()
