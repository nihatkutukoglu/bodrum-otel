import unittest

import numpy as np
import pandas as pd

from bodrum_intelligence.airport_tourism import (
    airport_quality_audit,
    build_joint_monthly,
    lag_correlations,
)


class AirportTourismTests(unittest.TestCase):
    def setUp(self):
        periods = [f"2025-{month:02d}" for month in range(1, 13)]
        airport_monthly = np.arange(1, 13) * 10
        self.airport = pd.DataFrame(
            {
                "period": periods,
                "year": [2025] * 12,
                "month_name_tr": [str(month) for month in range(1, 13)],
                "monthly_domestic_passengers": airport_monthly * 6,
                "monthly_international_passengers": airport_monthly * 4,
                "monthly_total_passengers": airport_monthly * 10,
                "cumulative_domestic_passengers": np.cumsum(airport_monthly * 6),
                "cumulative_international_passengers": np.cumsum(airport_monthly * 4),
                "cumulative_total_passengers": np.cumsum(airport_monthly * 10),
                "monthly_international_share_pct": [40.0] * 12,
            }
        )
        self.tourism = pd.DataFrame(
            {
                "period": periods,
                "year": [2025] * 12,
                "month_name_tr": [str(month) for month in range(1, 13)],
                "domestic_arrivals": np.arange(1, 13) * 20,
                "foreign_arrivals": np.arange(1, 13) * 30,
                "total_arrivals": np.arange(1, 13) * 50,
                "total_overnights": np.arange(1, 13) * 100,
                "occupancy_rate_pct": np.arange(1, 13) * 5,
                "derived_avg_stay_nights_recalculated": [2.0] * 12,
                "derived_foreign_arrival_share_pct_recalculated": [60.0] * 12,
                "season_group": ["LOW"] * 4 + ["SHOULDER"] * 4 + ["PEAK"] * 4,
            }
        )

    def test_airport_cumulative_differences_reconcile(self):
        checks, audited = airport_quality_audit(self.airport)
        self.assertTrue(audited["cumulative_total_monthly_difference"].eq(0).all())
        self.assertFalse((checks["status"] == "FAIL").any())

    def test_joint_merge_preserves_twelve_periods_and_no_infinity(self):
        joint = build_joint_monthly(self.airport, self.tourism)
        self.assertEqual(len(joint), 12)
        self.assertTrue(joint["period"].is_unique)
        numeric = joint.select_dtypes(include="number")
        self.assertTrue(np.isfinite(numeric).all().all())

    def test_lag_one_uses_airport_t_vs_tourism_t_plus_one(self):
        joint = build_joint_monthly(self.airport, self.tourism)
        lag = lag_correlations(joint)
        lag_one = lag.loc[
            lag["metric_pair"].eq("total airport vs total tourism")
            & lag["lag_months"].eq(1)
        ].iloc[0]
        self.assertEqual(lag_one["n"], 11)
        self.assertEqual(lag_one["alignment"], "airport_t vs tourism_t+1")

    def test_proxy_ratios_are_not_named_conversion_rate(self):
        joint = build_joint_monthly(self.airport, self.tourism)
        self.assertTrue(any("proxy_ratio" in column for column in joint.columns))
        self.assertFalse(any("conversion" in column for column in joint.columns))


if __name__ == "__main__":
    unittest.main()
