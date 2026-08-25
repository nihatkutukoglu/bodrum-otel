import unittest

import pandas as pd

from bodrum_intelligence.google_maps_cleaning import (
    build_review_id,
    exact_duplicate_audit,
    near_duplicate_candidates,
    parse_relative_review_date,
    parse_review_rating,
    rating_group,
)


class GoogleMapsCleaningHelpersTest(unittest.TestCase):
    def test_rating_parses_valid_and_rejects_out_of_range(self):
        self.assertEqual(parse_review_rating("5/5"), 5)
        self.assertEqual(parse_review_rating("1/5"), 1)
        self.assertIs(parse_review_rating("7/5"), pd.NA)
        self.assertIs(parse_review_rating(None), pd.NA)

    def test_rating_group_boundaries(self):
        self.assertEqual(rating_group(1), "LOW")
        self.assertEqual(rating_group(2), "LOW")
        self.assertEqual(rating_group(3), "MIXED")
        self.assertEqual(rating_group(4), "HIGH")
        self.assertEqual(rating_group(5), "HIGH")

    def test_relative_date_bir_and_numeric_units(self):
        reference = "2026-08-25T12:00:00+00:00"
        parsed, approx, edited, status = parse_relative_review_date("Google üzerinde bir ay önce", reference)
        self.assertEqual(parsed, pd.Timestamp("2026-07-26"))
        self.assertTrue(approx)
        self.assertFalse(edited)
        self.assertEqual(status, "RELATIVE_PARSED")

        parsed2, _, edited2, _ = parse_relative_review_date("Google üzerinde 2 hafta önce düzenlendi", reference)
        self.assertEqual(parsed2, pd.Timestamp("2026-08-11"))
        self.assertTrue(edited2)

    def test_relative_date_unrecognized_flagged(self):
        parsed, _, _, status = parse_relative_review_date("5 Ağustos 2025", "2026-08-25T12:00:00+00:00")
        self.assertTrue(pd.isna(parsed))
        self.assertEqual(status, "UNRECOGNIZED")

    def test_review_id_is_stable_and_prefixed(self):
        self.assertEqual(build_review_id("abcdef1234567890" * 4), "gm_abcdef1234567890")
        self.assertIs(build_review_id(None), pd.NA)

    def test_exact_duplicate_audit_flags_repeated_review_id(self):
        frame = pd.DataFrame(
            {
                "review_id": ["a", "a", "b"],
                "hotel_name": ["H1", "H1", "H2"],
                "review_text": ["metin bir", "metin iki", "metin üç"],
            }
        )
        result = exact_duplicate_audit(frame)
        self.assertEqual(set(result["review_id"]), {"a"})

    def test_near_duplicate_candidates_finds_similar_same_hotel_text(self):
        frame = pd.DataFrame(
            {
                "review_id": ["a", "b", "c"],
                "hotel_name": ["H1", "H1", "H2"],
                "review_text": [
                    "Otel çok temizdi ve personel güler yüzlüydü",
                    "Otel çok temizdi ve personel guler yuzluydu!",
                    "Tamamen farklı bir yorum burada",
                ],
            }
        )
        result = near_duplicate_candidates(frame, threshold=0.9)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["hotel_name"], "H1")


if __name__ == "__main__":
    unittest.main()
