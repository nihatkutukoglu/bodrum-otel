#!/usr/bin/env python3
"""Minimal assert-based self-check for the pure-logic parts of the
Sikayetvar scraper: URL canonicalization, source-page prefix filtering,
Selectum entity matching, and Turkish display-date parsing.

Usage: python tests/test_sikayetvar_scraper.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bodrum_intelligence.sikayetvar_scraper import (  # noqa: E402
    EXCLUDED, MATCHED, REVIEW_REQUIRED, Target,
    canonicalize_url, entity_match, parse_display_date, source_page_prefix,
)


def test_canonicalize_url():
    assert canonicalize_url("https://www.sikayetvar.com/x/y?utm=1#frag") == \
        "https://www.sikayetvar.com/x/y"
    assert canonicalize_url("/x/y/") == "https://www.sikayetvar.com/x/y"
    assert canonicalize_url("https://www.sikayetvar.com/x/y") == \
        canonicalize_url("https://www.sikayetvar.com/x/y?page=2")


def test_source_page_prefix():
    assert source_page_prefix("https://www.sikayetvar.com/green-bay-resort") == "/green-bay-resort/"
    assert source_page_prefix("https://www.sikayetvar.com/selectum-hotels/selectum-bodrum") == "/selectum-hotels/"


SELECTUM = Target(
    canonical_hotel_name="Selectum Colours Bodrum",
    area="Gümbet",
    source_pages=["https://www.sikayetvar.com/selectum-hotels/selectum-bodrum"],
    requires_entity_validation=True,
    match_patterns=["selectum colours bodrum", "selectum colors bodrum"],
    exclude_patterns=["selectum collection", "belek", "selectum family"],
    ambiguous_terms=["selectum bodrum", "selectum hotels"],
)

DEDICATED = Target(
    canonical_hotel_name="La Blanche Island Bodrum",
    area="Güvercinlik",
    source_pages=["https://www.sikayetvar.com/la-blanche-island-bodrum"],
    requires_entity_validation=False,
)


def test_entity_match_matched():
    status, _ = entity_match(SELECTUM, "u", "t", "Selectum Colours Bodrum otelinde kaldık", "", "")
    assert status == MATCHED


def test_entity_match_excluded_beats_ambiguous():
    status, _ = entity_match(SELECTUM, "u", "t", "Selectum Collection Bodrum berbattı", "", "")
    assert status == EXCLUDED


def test_entity_match_review_required_on_generic_mention():
    status, _ = entity_match(SELECTUM, "u", "t", "Selectum Bodrum otelinde sorun yaşadık", "", "")
    assert status == REVIEW_REQUIRED


def test_entity_match_dedicated_page_always_matched():
    status, reason = entity_match(DEDICATED, "u", "t", "anything at all", "", "")
    assert status == MATCHED and "single-property" in reason


def test_parse_display_date():
    assert parse_display_date("22 Ağustos 13:45", assume_year=2026) == (2026, 8, 22, 13, 45)
    assert parse_display_date("21 Aralık 2023 11:40") == (2023, 12, 21, 11, 40)
    assert parse_display_date("") is None
    assert parse_display_date("not a date") is None


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok: {t.__name__}")
    print(f"\n{len(tests)} check(s) passed.")


if __name__ == "__main__":
    main()
