from pathlib import Path
import unittest

from bodrum_intelligence.cleaning import clean_hotels, load_raw_hotels


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "bodrum_hotels_master_2026-08-24.csv"


class CleaningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_raw_hotels(RAW_PATH)
        cls.result = clean_hotels(cls.raw)

    def test_cleaning_preserves_rows_and_keys(self):
        self.assertEqual(len(self.result.hotels), len(self.raw))
        self.assertEqual(len(self.raw), 192)
        self.assertTrue(self.result.hotels["hotel_id"].is_unique)
        self.assertTrue(self.result.hotels["place_id"].is_unique)
        self.assertTrue(self.result.hotels["place_id"].equals(self.raw["place_id"].str.strip()))

    def test_cleaning_does_not_impute_intentionally_missing_fields(self):
        self.assertTrue(self.result.hotels["official_star_rating"].isna().all())
        self.assertEqual(self.result.hotels["business_status"].isna().sum(), 191)
        self.assertEqual(self.result.hotels["search_price_usd_snapshot"].isna().sum(), 24)

    def test_cleaned_schema_and_validation(self):
        self.assertEqual(str(self.result.hotels["phone"].dtype), "string")
        self.assertEqual(str(self.result.hotels["google_review_count"].dtype), "Int64")
        self.assertEqual(str(self.result.hotels["google_rating"].dtype), "Float64")
        self.assertEqual(self.result.hotels.shape, (192, 20))
        expected = self.result.hotels.groupby("area")["hotel_id"].transform("size")
        self.assertTrue(self.result.hotels["area_hotel_count"].eq(expected).all())
        self.assertEqual(set(self.result.validation_report["status"]), {"PASS"})


if __name__ == "__main__":
    unittest.main()
