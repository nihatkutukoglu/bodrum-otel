import unittest
from pathlib import Path

import pandas as pd

from bodrum_intelligence.sikayetvar_eda import (
    build_area_eda_summary,
    build_hotel_eda_summary,
    concentration_metrics,
    nlp_sample_tier,
    safe_ratio,
)


class SikayetvarEdaHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.complaints = pd.read_csv(
            root / "data/processed/sikayetvar_all_hotels_complaints_clean.csv",
            parse_dates=["complaint_date", "company_response_date_parsed"],
        )
        cls.complaints["response_time_days"] = (
            cls.complaints["company_response_date_parsed"] - cls.complaints["complaint_date"]
        ).dt.total_seconds() / 86400
        cls.mapping = pd.read_csv(root / "data/raw/sikayetvar/sikayetvar_hotel_mapping.csv")
        cls.master = pd.read_csv(root / "data/processed/hotels_enriched.csv")
        cls.replies = pd.read_csv(root / "data/processed/sikayetvar_all_hotels_replies_clean.csv")
        cls.root = root

    def test_safe_ratio_is_zero_safe(self):
        result = safe_ratio(pd.Series([10, 10, 10]), pd.Series([2, 0, None]), scale=1000)
        self.assertEqual(result.iloc[0], 5000)
        self.assertTrue(pd.isna(result.iloc[1]))
        self.assertTrue(pd.isna(result.iloc[2]))

    def test_hotel_and_area_aggregations_reconcile_to_corpus(self):
        hotel, _ = build_hotel_eda_summary(self.complaints)
        area = build_area_eda_summary(self.complaints, self.mapping, self.master)
        self.assertEqual(hotel["matched_complaint_count"].sum(), len(self.complaints))
        self.assertEqual(area["matched_complaint_count"].sum(), len(self.complaints))
        self.assertEqual(hotel["company_response_count"].sum(), int(self.complaints["company_response_exists_clean"].sum()))

    def test_cross_platform_visibility_only_has_positive_denominators(self):
        hotel, _ = build_hotel_eda_summary(self.complaints)
        metric = "cross_platform_complaint_visibility_per_1000_google_reviews"
        self.assertTrue(hotel.loc[hotel[metric].notna(), "google_review_count"].gt(0).all())

    def test_concentration_and_nlp_tiers(self):
        hotel, _ = build_hotel_eda_summary(self.complaints)
        metrics = concentration_metrics(hotel)
        self.assertGreater(metrics["top5_hotel_complaint_share_pct"], 0)
        self.assertLessEqual(metrics["top5_hotel_complaint_share_pct"], 100)
        self.assertEqual(nlp_sample_tier(15), "HIGH_SAMPLE")
        self.assertEqual(nlp_sample_tier(5), "MEDIUM_SAMPLE")
        self.assertEqual(nlp_sample_tier(4), "LOW_SAMPLE")

    def test_reply_aggregation_integrity(self):
        self.assertEqual(int(self.complaints["reply_count_total_derived"].sum()), len(self.replies))
        self.assertEqual(self.replies["reply_id"].nunique(), len(self.replies))
        self.assertTrue(
            self.replies["canonical_complaint_url"].isin(self.complaints["canonical_complaint_url"]).all()
        )

    def test_notebook_outputs_exist_and_validate(self):
        required = [
            "sikayetvar_hotel_eda_summary.csv",
            "sikayetvar_area_eda_summary.csv",
            "sikayetvar_company_response_time_summary.csv",
            "sikayetvar_hotel_level_correlations.csv",
            "sikayetvar_eda_notable_cases.csv",
            "sikayetvar_eda_key_findings.txt",
            "sikayetvar_eda_limitations.txt",
            "sikayetvar_nlp_sample_readiness.csv",
        ]
        self.assertTrue(all((self.root / "reports" / name).exists() for name in required))
        validation = pd.read_csv(self.root / "reports/sikayetvar_eda_output_validation.csv")
        self.assertTrue(validation["passed"].all())


if __name__ == "__main__":
    unittest.main()
