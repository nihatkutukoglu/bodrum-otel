import unittest

from bodrum_intelligence.sikayetvar_discovery import generate_slug_candidates, slugify
from bodrum_intelligence.sikayetvar_matching import (
    classify_candidate, detect_negative_conflict, score_candidate,
    build_complaint_validation_terms,
)
from bodrum_intelligence.sikayetvar_scraper import (
    canonicalize_url, entity_match, parse_display_date, source_page_prefix,
    MATCHED, REVIEW_REQUIRED, EXCLUDED,
)


class SlugifyTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(slugify("Rixos Premium Bodrum"), "rixos-premium-bodrum")

    def test_turkish_characters(self):
        self.assertEqual(slugify("LİV OTEL"), "liv-otel")

    def test_ampersand(self):
        self.assertEqual(slugify("Green Bay Resort & Spa"), "green-bay-resort-and-spa")

    def test_slug_candidates_include_verbatim_slug(self):
        candidates = generate_slug_candidates("Rixos Premium Bodrum", "Torba")
        self.assertIn("rixos-premium-bodrum", candidates)


class MatchingTests(unittest.TestCase):
    def test_score_rewards_name_and_bodrum_evidence(self):
        _, score_with = score_candidate("Kefaluka Resort", "Kefaluka Resort", True, False)
        _, score_without = score_candidate("Kefaluka Resort", "Kefaluka Resort", False, False)
        self.assertGreater(score_with, score_without)

    def test_negative_conflict_detects_sibling_collision(self):
        others = ["Selectum Collection Bodrum"]
        conflict, reason = detect_negative_conflict(
            "Selectum Colours Bodrum", "Selectum Collection Bodrum", others
        )
        self.assertTrue(conflict)
        self.assertIn("Selectum Collection Bodrum", reason)

    def test_no_conflict_for_unrelated_hotel(self):
        others = ["Kefaluka Resort"]
        conflict, _ = detect_negative_conflict("La Blanche Island Bodrum", "La Blanche Island Bodrum", others)
        self.assertFalse(conflict)

    def test_conflict_when_own_brand_word_missing_from_candidate(self):
        # Regression: "Mira Beach Resort Bodrum" false-matched to "Amilla
        # Beach Resort Bodrum" purely off shared filler words.
        conflict, reason = detect_negative_conflict(
            "Mira Beach Resort Bodrum", "Amilla Beach Resort Bodrum", []
        )
        self.assertTrue(conflict)
        self.assertIn("mira", reason)

    def test_search_fallback_never_auto_accepts_even_at_high_score(self):
        status = classify_candidate(0.95, 0.95, reliable_method=False, negative_conflict=False)
        self.assertNotIn(status, ("FOUND_EXACT", "FOUND_HIGH_CONFIDENCE"))

    def test_classify_candidate_exact_needs_reliable_method(self):
        status_reliable = classify_candidate(0.95, 0.7, True, False)
        status_unreliable = classify_candidate(0.95, 0.7, False, False)
        self.assertEqual(status_reliable, "FOUND_EXACT")
        self.assertNotEqual(status_unreliable, "FOUND_EXACT")

    def test_classify_candidate_conflict_forces_review(self):
        status = classify_candidate(0.99, 0.95, True, negative_conflict=True)
        self.assertEqual(status, "REVIEW_REQUIRED")

    def test_build_complaint_validation_terms_shared_page_requires_validation(self):
        mapping_rows = [
            {"hotel_id": "BOD068", "hotel_name": "Selectum Colours Bodrum",
             "sikayetvar_url": "https://www.sikayetvar.com/selectum-hotels"},
            {"hotel_id": "BOD012", "hotel_name": "Selectum Collection Bodrum",
             "sikayetvar_url": "https://www.sikayetvar.com/selectum-hotels"},
        ]
        terms = build_complaint_validation_terms(
            "BOD068", "Selectum Colours Bodrum", "https://www.sikayetvar.com/selectum-hotels", mapping_rows,
        )
        self.assertTrue(terms.requires_validation)
        self.assertIn("Selectum Collection Bodrum", terms.exclude_patterns)

    def test_build_complaint_validation_terms_umbrella_slug_without_sibling_still_requires_validation(self):
        # Regression: Rixos Premium Bodrum has no sibling Rixos hotel in
        # this dataset, but its complaints live under the generic
        # multi-city /rixos-hotels/ umbrella account -- validation must
        # still run so a complaint has to actually name this property.
        terms = build_complaint_validation_terms(
            "BOD148", "Rixos Premium Bodrum", "https://www.sikayetvar.com/rixos-hotels", [],
        )
        self.assertTrue(terms.requires_validation)

    def test_build_complaint_validation_terms_dedicated_page_skips_validation(self):
        mapping_rows = [
            {"hotel_id": "BOD106", "hotel_name": "La Blanche Island Bodrum",
             "sikayetvar_url": "https://www.sikayetvar.com/la-blanche-island-bodrum"},
        ]
        terms = build_complaint_validation_terms(
            "BOD106", "La Blanche Island Bodrum",
            "https://www.sikayetvar.com/la-blanche-island-bodrum", mapping_rows,
        )
        self.assertFalse(terms.requires_validation)


class ScraperCoreTests(unittest.TestCase):
    def test_canonicalize_url_strips_query_and_fragment(self):
        self.assertEqual(
            canonicalize_url("https://www.sikayetvar.com/x/y?utm=1#frag"),
            "https://www.sikayetvar.com/x/y",
        )

    def test_canonicalize_url_dedupes_pagination(self):
        self.assertEqual(
            canonicalize_url("https://www.sikayetvar.com/x/y"),
            canonicalize_url("https://www.sikayetvar.com/x/y?page=2"),
        )

    def test_source_page_prefix(self):
        self.assertEqual(source_page_prefix("https://www.sikayetvar.com/green-bay-resort"), "/green-bay-resort/")
        self.assertEqual(
            source_page_prefix("https://www.sikayetvar.com/selectum-hotels/selectum-bodrum"),
            "/selectum-hotels/",
        )

    def test_entity_match_matched_excluded_review_required(self):
        matched, _ = entity_match(
            "u", "t", "Selectum Colours Bodrum otelinde kaldik", "", "",
            match_patterns=["selectum colours bodrum"], exclude_patterns=["selectum collection"],
            ambiguous_terms=["selectum"],
        )
        excluded, _ = entity_match(
            "u", "t", "Selectum Collection Bodrum berbatti", "", "",
            match_patterns=["selectum colours bodrum"], exclude_patterns=["selectum collection"],
            ambiguous_terms=["selectum"],
        )
        review, _ = entity_match(
            "u", "t", "Selectum otelinde sorun yasadik", "", "",
            match_patterns=["selectum colours bodrum"], exclude_patterns=["selectum collection"],
            ambiguous_terms=["selectum"],
        )
        self.assertEqual(matched, MATCHED)
        self.assertEqual(excluded, EXCLUDED)
        self.assertEqual(review, REVIEW_REQUIRED)

    def test_entity_match_dedicated_page_always_matched(self):
        status, reason = entity_match(
            "u", "t", "anything at all", "", "",
            match_patterns=[], exclude_patterns=[], ambiguous_terms=[], requires_validation=False,
        )
        self.assertEqual(status, MATCHED)
        self.assertIn("single-property", reason)

    def test_parse_display_date(self):
        self.assertEqual(parse_display_date("22 Ağustos 13:45", assume_year=2026), (2026, 8, 22, 13, 45))
        self.assertEqual(parse_display_date("21 Aralık 2023 11:40"), (2023, 12, 21, 11, 40))
        self.assertIsNone(parse_display_date(""))
        self.assertIsNone(parse_display_date("not a date"))


if __name__ == "__main__":
    unittest.main()
