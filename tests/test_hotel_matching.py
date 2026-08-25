from pathlib import Path
import unittest

import pandas as pd

from bodrum_intelligence.hotel_matching import (
    build_best_match_table,
    build_enriched_dataset,
    detect_official_duplicates,
    generate_candidates,
    normalize_phone,
    prepare_official_normalization,
    prepare_project_normalization,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "hotels_features.csv"
OFFICIAL_PATH = PROJECT_ROOT / "data" / "external" / "hotel" / "hotel_attributes_official_bodrum.csv"


class HotelMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_df = pd.read_csv(FEATURES_PATH, dtype={"phone": "string"})
        cls.official_df = pd.read_csv(OFFICIAL_PATH)

        project_norm = prepare_project_normalization(cls.project_df)
        official_norm = detect_official_duplicates(prepare_official_normalization(cls.official_df))
        candidates = generate_candidates(project_norm, official_norm)
        cls.best_matches = build_best_match_table(candidates, project_norm)
        cls.enriched = build_enriched_dataset(cls.project_df, cls.best_matches)

    def test_project_hotel_count_and_keys_preserved(self):
        self.assertEqual(len(self.enriched), len(self.project_df))
        self.assertEqual(self.enriched["hotel_id"].nunique(), len(self.project_df))
        self.assertEqual(self.enriched["place_id"].nunique(), len(self.project_df))
        self.assertEqual(self.best_matches["hotel_id"].nunique(), len(self.project_df))

    def test_no_official_facility_linked_to_more_than_one_hotel(self):
        high_confidence = self.best_matches.loc[
            self.best_matches["match_status"].eq("MATCHED_HIGH_CONFIDENCE")
        ]
        self.assertFalse(high_confidence["official_facility_id"].duplicated().any())

    def test_verified_star_rating_in_range(self):
        stars = self.enriched["official_star_rating_verified"].dropna()
        self.assertTrue(stars.between(1, 5).all())

    def test_no_negative_room_or_bed_counts(self):
        self.assertTrue((self.enriched["official_room_count"].dropna() > 0).all())
        self.assertTrue((self.enriched["official_bed_count"].dropna() > 0).all())

    def test_unmatched_hotels_have_no_fabricated_official_data(self):
        unmatched = self.enriched.loc[self.enriched["official_match_status"].eq("UNMATCHED")]
        for column in [
            "official_star_rating_verified", "official_room_count", "official_bed_count",
            "official_facility_id", "official_name",
        ]:
            self.assertTrue(unmatched[column].isna().all(), column)

    def test_existing_official_star_rating_column_not_overwritten(self):
        self.assertIn("official_star_rating", self.enriched.columns)
        self.assertIn("official_star_rating_verified", self.enriched.columns)
        self.assertTrue(self.enriched["official_star_rating"].equals(self.project_df["official_star_rating"]))

    def test_normalize_phone_strips_country_code_and_leading_zero(self):
        self.assertEqual(normalize_phone("+90 252 319 7171"), normalize_phone("0252 319 7171"))
        self.assertIsNone(normalize_phone("123"))
        self.assertIsNone(normalize_phone(None))


if __name__ == "__main__":
    unittest.main()
