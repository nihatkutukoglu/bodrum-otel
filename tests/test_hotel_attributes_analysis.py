import unittest

import numpy as np
import pandas as pd

from bodrum_intelligence.hotel_attributes_analysis import (
    assign_size_groups,
    correlation_results,
    coverage_summary,
    destination_capacity,
    official_analysis_sample,
)


class HotelAttributesAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "hotel_id": ["A", "B", "C", "D"],
                "area": ["X", "X", "Y", "Y"],
                "official_match_status": [
                    "MATCHED_HIGH_CONFIDENCE",
                    "MATCHED_HIGH_CONFIDENCE",
                    "UNMATCHED",
                    "MATCHED_HIGH_CONFIDENCE",
                ],
                "official_star_rating_verified": [5, 4, np.nan, 3],
                "official_room_count": [100, 50, np.nan, 20],
                "official_bed_count": [200, 100, np.nan, 40],
                "official_type": ["5 YILDIZLI OTEL", "4 YILDIZLI OTEL", np.nan, "3 YILDIZLI OTEL"],
                "google_rating": [4.6, 4.4, 4.2, 4.1],
                "google_review_count": [1000, 500, 100, 50],
                "search_price_usd_snapshot": [500, 300, np.nan, 100],
                "weighted_google_rating": [4.5, 4.35, 4.2, 4.15],
                "review_confidence_weight": [0.9, 0.8, 0.5, 0.4],
                "rating_gap_from_area_median": [0.1, -0.1, 0.0, -0.1],
                "price_ratio_to_area_median": [1.25, 0.75, np.nan, 1.0],
                "price_percentile_within_area": [1.0, 0.5, np.nan, 1.0],
            }
        )

    def test_analysis_sample_only_keeps_high_confidence(self):
        sample = official_analysis_sample(self.frame)
        self.assertEqual(len(sample), 3)
        self.assertNotIn("C", sample["hotel_id"].tolist())

    def test_coverage_uses_full_hotel_universe(self):
        coverage = coverage_summary(self.frame).set_index("metric")
        self.assertEqual(coverage.loc["total_hotels", "hotel_count"], 4)
        self.assertEqual(coverage.loc["high_confidence_official_match", "hotel_count"], 3)
        self.assertEqual(coverage.loc["room_count_available", "coverage_pct_of_192"], 75.0)

    def test_size_groups_follow_distribution(self):
        groups, thresholds = assign_size_groups(pd.Series([10, 20, 30, 40, 50, 60]))
        self.assertEqual(set(groups.dropna()), {"Small", "Medium", "Large"})
        self.assertLess(thresholds["q33"], thresholds["q67"])

    def test_destination_capacity_respects_sample_grain(self):
        sample = official_analysis_sample(self.frame)
        profile = destination_capacity(sample).set_index("area")
        self.assertEqual(profile.loc["X", "matched_hotel_count"], 2)
        self.assertEqual(profile.loc["X", "total_official_rooms"], 150)
        self.assertEqual(profile.loc["Y", "verified_5star_count"], 0)

    def test_correlation_returns_sample_size(self):
        sample = official_analysis_sample(self.frame)
        results = correlation_results(sample, "official_room_count", "google_rating", "test")
        self.assertEqual({result.n for result in results}, {3})
        self.assertTrue(all(np.isfinite(result.coefficient) for result in results))


if __name__ == "__main__":
    unittest.main()
