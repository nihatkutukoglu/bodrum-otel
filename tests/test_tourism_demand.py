import unittest

import numpy as np
import pandas as pd

from bodrum_intelligence.tourism_demand import (
    add_annual_features,
    add_monthly_features,
    bodrum_profile,
    safe_ratio,
    seasonality_metrics,
)


class TourismDemandTests(unittest.TestCase):
    def test_safe_ratio_does_not_create_infinity(self):
        result = safe_ratio(pd.Series([10, 10]), pd.Series([2, 0]))
        self.assertEqual(result.iloc[0], 5)
        self.assertTrue(np.isnan(result.iloc[1]))

    def test_annual_features_distinguish_pct_and_percentage_points(self):
        annual = pd.DataFrame(
            {
                "year": [2019, 2020],
                "domestic_arrivals": [40, 30],
                "foreign_arrivals": [60, 20],
                "total_arrivals": [100, 50],
                "total_overnights": [300, 100],
                "avg_stay_nights": [3.0, 2.0],
                "occupancy_rate_pct": [40.0, 25.0],
                "foreign_arrival_share_pct": [60.0, 40.0],
            }
        )
        result = add_annual_features(annual).set_index("year")
        self.assertEqual(result.loc[2020, "total_arrivals_yoy_pct"], -50.0)
        self.assertEqual(result.loc[2020, "occupancy_yoy_change_pp"], -15.0)
        self.assertEqual(result.loc[2020, "arrival_total_difference"], 0)

    def test_monthly_features_keep_chronological_order_and_match_annual(self):
        monthly = pd.DataFrame(
            {
                "period": ["2025-02", "2025-01", "2025-03"],
                "year": [2025] * 3,
                "month_name_tr": ["Şubat", "Ocak", "Mart"],
                "domestic_arrivals": [20, 10, 30],
                "foreign_arrivals": [10, 5, 15],
                "total_arrivals": [30, 15, 45],
                "total_overnights": [60, 30, 90],
                "occupancy_rate_pct": [20, 10, 30],
                "derived_avg_stay_nights": [2.0] * 3,
                "derived_foreign_arrival_share_pct": [33.3333] * 3,
            }
        )
        annual = pd.Series(
            {
                "domestic_arrivals": 60,
                "foreign_arrivals": 30,
                "total_arrivals": 90,
                "total_overnights": 180,
            }
        )
        result, metadata = add_monthly_features(monthly, annual)
        self.assertEqual(result["period"].tolist(), ["2025-01", "2025-02", "2025-03"])
        self.assertTrue(metadata["monthly_totals_match_annual_2025"])
        self.assertAlmostEqual(result["monthly_arrival_share_pct"].sum(), 100.0)

    def test_seasonality_hhi_uses_decimal_shares(self):
        monthly = pd.DataFrame(
            {
                "total_arrivals": [25, 25, 25, 25],
                "total_overnights": [50, 50, 50, 50],
                "occupancy_rate_pct": [10, 10, 10, 10],
            }
        )
        result = seasonality_metrics(monthly).iloc[0]
        self.assertAlmostEqual(result["hhi_monthly_arrival_concentration"], 0.25)

    def test_bodrum_profile_recalculates_shares_without_overwrite(self):
        source = pd.DataFrame(
            {
                "year": [2025],
                "geography": ["Bodrum"],
                "domestic_arrivals": [60],
                "foreign_arrivals": [40],
                "total_arrivals": [100],
                "total_overnights": [250],
                "derived_avg_stay_nights": [2.5],
                "derived_foreign_arrival_share_pct": [40.0],
            }
        )
        result = bodrum_profile(source).iloc[0]
        self.assertEqual(result["domestic_share_pct"], 60.0)
        self.assertEqual(result["foreign_share_pct"], 40.0)
        self.assertEqual(result["avg_stay_nights_recalculated"], 2.5)


if __name__ == "__main__":
    unittest.main()
