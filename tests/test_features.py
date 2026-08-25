from pathlib import Path
import unittest

import pandas as pd

from bodrum_intelligence.features import build_basic_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "hotels_clean.csv"


class FeatureEngineeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean_df = pd.read_csv(CLEAN_PATH, dtype={"phone": "string"})
        cls.result = build_basic_features(cls.clean_df)

    def test_rows_keys_and_source_columns_are_preserved(self):
        self.assertEqual(len(self.result.df), len(self.clean_df))
        self.assertTrue(
            self.result.df[["hotel_id", "place_id"]].equals(
                self.clean_df[["hotel_id", "place_id"]]
            )
        )
        for column in self.clean_df.columns:
            self.assertTrue(self.result.df[column].equals(self.clean_df[column]), column)

    def test_missing_prices_are_not_imputed(self):
        missing_price = self.clean_df["search_price_usd_snapshot"].isna()
        price_features = [
            "price_gap_from_area_median",
            "price_ratio_to_area_median",
            "price_percentile_within_area",
        ]
        self.assertTrue(self.result.df.loc[missing_price, price_features].isna().all().all())
        self.assertEqual(int(self.result.df["has_price_snapshot"].sum()), 168)

    def test_derived_feature_ranges_and_validation(self):
        self.assertTrue(self.result.df["review_confidence_weight"].between(0, 1).all())
        self.assertTrue(self.result.df["weighted_google_rating"].between(0, 5).all())
        self.assertTrue(self.result.df["price_percentile_within_area"].dropna().between(0, 1).all())
        self.assertEqual(set(self.result.validation_report["status"]), {"PASS"})
        self.assertEqual(len(self.result.feature_dictionary), 14)


if __name__ == "__main__":
    unittest.main()
