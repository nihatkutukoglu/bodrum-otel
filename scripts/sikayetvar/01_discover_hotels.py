#!/usr/bin/env python3
"""Discover each project hotel's Sikayetvar company page.

For every hotel_id in the project hotel master, tries (in order):
  1. Any `known_urls` seeded in config/sikayetvar_manual_aliases.json
     (required for chain umbrella accounts a slug guess can't find, e.g.
     Selectum Colours Bodrum under /selectum-hotels/selectum-colours).
  2. Direct slug guesses derived from the hotel name (see
     bodrum_intelligence.sikayetvar_discovery for the slugify rules and why
     this is the primary method, not the site's own search).
  3. The /sikayetler?k=... fallback page, filtered to slugs sharing a brand
     token with the hotel (only tried if 1-2 found nothing, since it's a
     noisier, weaker signal).

Every candidate tried is logged to sikayetvar_mapping_candidates.csv (never
overwritten -- append-only, resumable). The best candidate per hotel is
scored/classified and written to sikayetvar_hotel_mapping.csv (rewritten
each run from the accumulated candidates, so re-running after a threshold
tune reclassifies without re-fetching).

Usage:
    python3 scripts/sikayetvar/01_discover_hotels.py --max-hotels 10   # smoke test
    python3 scripts/sikayetvar/01_discover_hotels.py                   # full run
    python3 scripts/sikayetvar/01_discover_hotels.py --resume          # skip already-discovered hotels
    python3 scripts/sikayetvar/01_discover_hotels.py --hotel-id BOD068
    python3 scripts/sikayetvar/01_discover_hotels.py --area Gümbet
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from bodrum_intelligence.sikayetvar_discovery import (  # noqa: E402
    check_slug, generate_search_queries, generate_slug_candidates, search_fallback_candidates,
)
from bodrum_intelligence.sikayetvar_matching import (  # noqa: E402
    AUTO_ACCEPTED_STATUSES, MatchThresholds, NOT_FOUND, PAGE_FOUND_NO_COMPLAINT,
    classify_candidate, detect_negative_conflict, score_candidate,
)
from bodrum_intelligence.sikayetvar_scraper import (  # noqa: E402
    AntiBotBlock, append_rows, build_session, polite_sleep, read_csv_rows, utc_now_iso, write_csv,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("sikayetvar.discover")

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
HOTEL_MASTER_CANDIDATES = [
    os.path.join(REPO_ROOT, "data", "processed", "hotels_enriched.csv"),
    os.path.join(REPO_ROOT, "data", "processed", "hotels_features.csv"),
    os.path.join(REPO_ROOT, "data", "processed", "hotels_clean.csv"),
]
DISCOVERY_CONFIG = os.path.join(REPO_ROOT, "config", "sikayetvar_discovery_config.json")
MANUAL_ALIASES = os.path.join(REPO_ROOT, "config", "sikayetvar_manual_aliases.json")
CANDIDATES_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_mapping_candidates.csv")
MAPPING_CSV = os.path.join(REPO_ROOT, "data", "raw", "sikayetvar", "sikayetvar_hotel_mapping.csv")

CANDIDATE_FIELDS = [
    "hotel_id", "hotel_name", "area", "search_query", "candidate_company_name",
    "candidate_url", "candidate_slug", "candidate_page_title", "name_similarity",
    "area_evidence", "bodrum_evidence", "address_evidence", "phone_evidence",
    "brand_evidence", "negative_brand_conflict", "candidate_score", "match_method",
    "candidate_rank", "visible_complaint_count", "discovered_at",
]
MAPPING_FIELDS = [
    "hotel_id", "place_id", "hotel_name", "area", "sikayetvar_company_name",
    "sikayetvar_url", "sikayetvar_slug", "match_status", "match_score", "match_method",
    "match_reason", "candidate_count", "manual_review_required", "page_accessible",
    "visible_complaint_count", "checked_at",
]


def find_hotel_master() -> str:
    for path in HOTEL_MASTER_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No hotel master found. Looked for: " + ", ".join(HOTEL_MASTER_CANDIDATES)
    )


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def discover_one_hotel(session, hotel: dict, all_hotel_names: list, manual_aliases: dict,
                        max_slug_candidates: int, delay: float) -> list:
    """Returns a list of candidate rows (dicts matching CANDIDATE_FIELDS)."""
    hotel_id, hotel_name, area = hotel["hotel_id"], hotel["hotel_name"], hotel.get("area", "")
    rows = []
    rank = 0
    now = utc_now_iso()

    tried_slugs = set()
    this_brand = set(hotel_name.lower().split())

    def try_slug(slug: str, search_query: str, match_method: str):
        nonlocal rank
        if slug in tried_slugs:
            return None
        tried_slugs.add(slug)
        try:
            info = check_slug(session, slug)
        except AntiBotBlock:
            raise
        except Exception as exc:
            log.warning(f"    [error checking {slug}] {exc}")
            return None
        polite_sleep(delay)
        if not info["exists"]:
            return None

        # This candidate has zero complaints of its own -- the chain may
        # post under a slightly different active slug instead (see
        # check_slug's docstring: la-blanche-resort-bodrum vs the real
        # la-blanche-resort-**spa**-bodrum). Try whatever related slugs
        # the (near-)empty page itself links to, filtered to ones sharing
        # a brand word with this hotel.
        if info["visible_complaint_count"] == "0":
            for related_slug in info.get("related_slugs", []):
                if set(related_slug.split("-")) & this_brand:
                    log.info(f"    [0 complaints on {slug}, trying related slug {related_slug}]")
                    related_row = try_slug(related_slug, search_query, "related_slug")
                    if related_row:
                        return related_row

        rank += 1
        title = info["page_title"]
        company_name = info["company_name"]
        bodrum_evidence = "bodrum" in title.lower()
        area_evidence = bool(area) and area.lower() in title.lower()
        name_similarity, score = score_candidate(hotel_name, company_name, bodrum_evidence, area_evidence)
        conflict, conflict_reason = detect_negative_conflict(hotel_name, company_name, all_hotel_names)
        row = {
            "hotel_id": hotel_id, "hotel_name": hotel_name, "area": area,
            "search_query": search_query, "candidate_company_name": company_name,
            "candidate_url": info["final_url"], "candidate_slug": slug,
            "candidate_page_title": title, "name_similarity": name_similarity,
            "area_evidence": area_evidence, "bodrum_evidence": bodrum_evidence,
            "address_evidence": False, "phone_evidence": False,
            "brand_evidence": name_similarity >= 0.5, "negative_brand_conflict": conflict,
            "candidate_score": score, "match_method": match_method,
            "candidate_rank": rank, "discovered_at": now,
            "visible_complaint_count": info["visible_complaint_count"],
        }
        if conflict:
            row["candidate_page_title"] += f" [CONFLICT: {conflict_reason}]"
        rows.append(row)
        return row

    # 1) Manual alias seed (known_urls).
    alias_cfg = manual_aliases.get(hotel_id, {})
    for url in alias_cfg.get("known_urls", []):
        slug = url.rstrip("/").split("sikayetvar.com/")[-1]
        try_slug(slug, "manual_alias", "manual_alias")

    # 2) Direct slug guesses.
    if not rows:
        for slug in generate_slug_candidates(hotel_name, area, max_slug_candidates):
            try_slug(slug, hotel_name, "slug_guess")

    # 3) Fallback search page, only if nothing found yet.
    if not rows:
        core_tokens = set((hotel_name or "").lower().split())
        for query in generate_search_queries(hotel_name, area, max_queries=2):
            try:
                candidates = search_fallback_candidates(session, query, core_tokens)
            except AntiBotBlock:
                raise
            except Exception as exc:
                log.warning(f"    [error on search fallback for {query!r}] {exc}")
                continue
            polite_sleep(delay)
            for c in candidates:
                try_slug(c["slug"], query, "search_fallback")
            if rows:
                break

    return rows


def build_mapping_row(hotel: dict, candidate_rows: list, thresholds: MatchThresholds) -> dict:
    now = utc_now_iso()
    base = {
        "hotel_id": hotel["hotel_id"], "place_id": hotel.get("place_id", ""),
        "hotel_name": hotel["hotel_name"], "area": hotel.get("area", ""),
        "checked_at": now, "candidate_count": len(candidate_rows),
    }
    if not candidate_rows:
        return {**base, "sikayetvar_company_name": "", "sikayetvar_url": "", "sikayetvar_slug": "",
                "match_status": NOT_FOUND, "match_score": 0.0, "match_method": "no_candidate",
                "match_reason": "No slug guess or alias resolved to a real page",
                "manual_review_required": False, "page_accessible": False, "visible_complaint_count": ""}

    ranked = sorted(candidate_rows, key=lambda r: r["candidate_score"], reverse=True)
    best = ranked[0]
    runner_up_score = ranked[1]["candidate_score"] if len(ranked) > 1 else 0.0
    reliable_method = best["match_method"] in ("slug_guess", "manual_alias", "related_slug")

    status = classify_candidate(
        best["name_similarity"], best["candidate_score"], reliable_method,
        best["negative_brand_conflict"], runner_up_score, thresholds,
    )
    if status in AUTO_ACCEPTED_STATUSES and best["visible_complaint_count"] == "0":
        status = PAGE_FOUND_NO_COMPLAINT

    reason = (
        f"score={best['candidate_score']} name_sim={best['name_similarity']} "
        f"bodrum={best['bodrum_evidence']} area={best['area_evidence']} "
        f"method={best['match_method']}"
    )
    return {
        **base,
        "sikayetvar_company_name": best["candidate_company_name"],
        "sikayetvar_url": best["candidate_url"], "sikayetvar_slug": best["candidate_slug"],
        "match_status": status, "match_score": best["candidate_score"],
        "match_method": best["match_method"], "match_reason": reason,
        "manual_review_required": status not in AUTO_ACCEPTED_STATUSES,
        "page_accessible": True, "visible_complaint_count": best["visible_complaint_count"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-hotels", type=int, default=None)
    parser.add_argument("--hotel-id", default=None)
    parser.add_argument("--area", default=None)
    parser.add_argument("--resume", action="store_true", help="Skip hotels already in the mapping table.")
    parser.add_argument("--force", action="store_true", help="Re-discover even hotels already mapped.")
    args = parser.parse_args()

    config = load_json(DISCOVERY_CONFIG)
    manual_aliases = load_json(MANUAL_ALIASES)
    thresholds = MatchThresholds(
        high_confidence_min_score=config.get("auto_match_threshold", 0.75),
        review_required_min_score=config.get("review_required_threshold", 0.45),
    )
    delay = config.get("request_delay_seconds", 1.5)
    max_slug_candidates = config.get("max_slug_candidates_per_hotel", 5)

    master_path = find_hotel_master()
    log.info(f"Hotel master: {master_path}")
    hotels = read_csv_rows(master_path)
    all_hotel_names = [h["hotel_name"] for h in hotels]

    if args.hotel_id:
        hotels = [h for h in hotels if h["hotel_id"] == args.hotel_id]
    if args.area:
        hotels = [h for h in hotels if h.get("area", "").lower() == args.area.lower()]

    already_mapped = set()
    if args.resume and not args.force:
        already_mapped = {r["hotel_id"] for r in read_csv_rows(MAPPING_CSV)}
        hotels = [h for h in hotels if h["hotel_id"] not in already_mapped]
        log.info(f"Resume: {len(already_mapped)} hotel(s) already mapped, {len(hotels)} remaining.")

    if args.max_hotels:
        hotels = hotels[: args.max_hotels]

    session = build_session()
    all_candidates = read_csv_rows(CANDIDATES_CSV)
    all_mapping = [r for r in read_csv_rows(MAPPING_CSV) if r["hotel_id"] not in {h["hotel_id"] for h in hotels}]

    for i, hotel in enumerate(hotels, 1):
        log.info(f"[{i}/{len(hotels)}] {hotel['hotel_name']} ({hotel.get('area', '')})")
        try:
            candidate_rows = discover_one_hotel(
                session, hotel, all_hotel_names, manual_aliases, max_slug_candidates, delay,
            )
        except AntiBotBlock as exc:
            log.warning(f"  [BLOCKED_SAFE_STOP] {exc}")
            write_csv(MAPPING_CSV, MAPPING_FIELDS, all_mapping)
            log.warning("Partial results saved. Stopping discovery run.")
            return

        append_rows(CANDIDATES_CSV, CANDIDATE_FIELDS, candidate_rows)
        all_candidates.extend(candidate_rows)
        mapping_row = build_mapping_row(hotel, candidate_rows, thresholds)
        all_mapping.append(mapping_row)

        count_note = mapping_row["visible_complaint_count"] or "?"
        log.info(f"  -> {mapping_row['match_status']} -> {mapping_row['sikayetvar_url'] or '(none)'} "
                  f"(~{count_note} complaints visible)")

    write_csv(MAPPING_CSV, MAPPING_FIELDS, all_mapping)
    log.info(f"\nWrote {CANDIDATES_CSV}")
    log.info(f"Wrote {MAPPING_CSV}")


if __name__ == "__main__":
    main()
