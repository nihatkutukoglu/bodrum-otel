import unittest

import numpy as np
import pandas as pd

from bodrum_intelligence.destination_intelligence import (
    EXPECTED_AREAS,
    add_quadrants,
    add_subindices,
    aggregate_hotel_metrics,
    build_destination_master,
    value_sensitivity,
)


class DestinationIntelligenceTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for index, area in enumerate(EXPECTED_AREAS):
            for hotel_no in range(2):
                matched = hotel_no == 0 and index < 12
                rows.append(
                    {
                        "hotel_id": f"{index}-{hotel_no}",
                        "area": area,
                        "google_rating": 4.0 + index / 100 + hotel_no / 100,
                        "weighted_google_rating": 4.1 + index / 100,
                        "google_review_count": 100 + index * 10 + hotel_no,
                        "search_price_usd_snapshot": 100 + index * 5,
                        "rating_gap_from_area_median": hotel_no - 0.5,
                        "official_match_status": "MATCHED_HIGH_CONFIDENCE" if matched else "UNMATCHED",
                        "official_star_rating_verified": 5 if matched else np.nan,
                        "official_room_count": 50 + index if matched else np.nan,
                        "official_bed_count": 100 + index if matched else np.nan,
                        "official_type": "5 YILDIZLI OTEL" if matched else np.nan,
                    }
                )
        self.hotels = pd.DataFrame(rows)
        self.context = pd.DataFrame(
            {
                "area": EXPECTED_AREAS,
                "has_marina_official_context": [False] * 14,
                "has_weekly_market_official_context": [True] * 14,
                "weekly_market_days_official_context": ["Pazar"] * 14,
            }
        )

    def test_hotel_aggregation_preserves_fourteen_areas(self):
        result = aggregate_hotel_metrics(self.hotels)
        self.assertEqual(result["area"].tolist(), EXPECTED_AREAS)
        self.assertEqual(result["sample_hotel_count"].sum(), 28)
        self.assertAlmostEqual(result["sample_supply_share_pct"].sum(), 100.0, places=1)

    def test_zero_match_area_keeps_capacity_unknown(self):
        result = build_destination_master(self.hotels, self.context)
        zero_match = result.loc[result["area"].eq(EXPECTED_AREAS[-1])].iloc[0]
        self.assertEqual(zero_match["official_matched_hotel_count"], 0)
        self.assertTrue(np.isnan(zero_match["total_official_rooms"]))
        self.assertTrue(zero_match["low_coverage_flag"])

    def test_subindices_do_not_impute_missing_official_capacity(self):
        master = add_subindices(build_destination_master(self.hotels, self.context))
        zero_match = master.loc[master["area"].eq(EXPECTED_AREAS[-1])].iloc[0]
        self.assertTrue(np.isnan(zero_match["supply_capacity_index"]))
        self.assertTrue(np.isnan(zero_match["luxury_index"]))

    def test_value_sensitivity_has_three_scenarios_per_area(self):
        master = add_subindices(build_destination_master(self.hotels, self.context))
        sensitivity = value_sensitivity(master)
        self.assertEqual(len(sensitivity), 42)
        self.assertEqual(sensitivity.groupby("area")["scenario"].nunique().min(), 3)

    def test_quadrant_marks_low_coverage_price_luxury_ineligible(self):
        master = add_quadrants(add_subindices(build_destination_master(self.hotels, self.context)))
        self.assertFalse(master["price_luxury_quadrant_eligible"].any())
        self.assertTrue(master["price_luxury_quadrant"].isna().all())


if __name__ == "__main__":
    unittest.main()
