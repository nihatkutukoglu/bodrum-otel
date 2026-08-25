import unittest
from pathlib import Path

import pandas as pd

from bodrum_intelligence.sikayetvar_nlp import (
    CANONICAL_ASPECTS,
    add_aspect_columns,
    aspect_cooccurrence_table,
    aspects_long_table,
    detect_aspects,
    detect_severity,
    normalize_for_nlp,
    tokenize,
)


class SikayetvarNlpHelpersTest(unittest.TestCase):
    def test_turkish_normalization_and_negation_are_preserved(self):
        text = normalize_for_nlp("KLİMA çalışmıyor; çözüm YOK! test@example.com")
        self.assertIn("klima", text)
        self.assertIn("çalışmıyor", tokenize(text))
        self.assertIn("yok", tokenize(text))
        self.assertNotIn("test@example.com", text)

    def test_dirty_room_is_multilabel(self):
        aspects, _ = detect_aspects("oda çok kirliydi")
        self.assertIn("ROOM", aspects)
        self.assertIn("CLEANLINESS_HYGIENE", aspects)

    def test_food_and_staff_are_multilabel(self):
        aspects, _ = detect_aspects("yemekler soğuktu ve garson ilgilenmedi")
        self.assertIn("FOOD_BEVERAGE", aspects)
        self.assertIn("STAFF_SERVICE", aspects)

    def test_reservation_and_refund_are_multilabel(self):
        aspects, _ = detect_aspects("rezervasyonu iptal ettim ama ücret iadesi yapılmadı")
        self.assertIn("RESERVATION", aspects)
        self.assertIn("PAYMENT_REFUND", aspects)

    def test_severity_high_when_legal_or_safety_language_present(self):
        tier, high, medium = detect_severity("avukatımıza durumu ilettik, yasal işlem başlatacağız")
        self.assertEqual(tier, "HIGH")
        self.assertIn("avukat", high)

    def test_severity_medium_when_only_strong_dissatisfaction_present(self):
        tier, high, medium = detect_severity("hayal kırıklığına uğradık, bir daha gelmeyiz")
        self.assertEqual(tier, "MEDIUM")
        self.assertEqual(high, [])
        self.assertTrue(medium)

    def test_severity_baseline_when_no_escalation_language(self):
        tier, high, medium = detect_severity("oda küçüktü ve klima soğutmuyordu")
        self.assertEqual(tier, "BASELINE")
        self.assertEqual(high, [])
        self.assertEqual(medium, [])

    def test_beach_example(self):
        aspects, _ = detect_aspects("deniz güzeldi ama şezlong yoktu")
        self.assertIn("BEACH_SEA", aspects)

    def test_service_vehicle_context_disambiguation(self):
        transfer, transfer_hits = detect_aspects("servis aracı gelmedi")
        general, _ = detect_aspects("servis çok kötüydü")
        self.assertIn("TRANSPORT_TRANSFER", transfer)
        self.assertNotIn("STAFF_SERVICE", transfer_hits)
        self.assertIn("STAFF_SERVICE", general)

    def test_aspect_columns_and_long_relation_reconcile(self):
        frame = pd.DataFrame({
            "complaint_id": ["c1", "c2"], "hotel_id": [1, 1], "hotel_name": ["A", "A"],
            "area": ["X", "X"], "company_response_exists_clean": [True, False],
            "complaint_date": ["2025-01-01", "2025-01-02"], "google_rating": [4.0, 4.0],
            "google_review_count": [100, 100],
            "nlp_text_normalized": ["oda kirli", "deniz ve şezlong"],
        })
        enriched = add_aspect_columns(frame)
        aspect_columns = [f"aspect_{aspect.lower()}" for aspect in CANONICAL_ASPECTS]
        self.assertTrue(all(enriched[column].dtype == bool for column in aspect_columns))
        self.assertTrue((enriched[aspect_columns].sum(axis=1) == enriched["aspect_count"]).all())
        long = aspects_long_table(enriched)
        self.assertEqual(len(long), len(frame) * len(CANONICAL_ASPECTS))
        self.assertEqual(int(long["matched"].sum()), int(enriched["aspect_count"].sum()))
        cooccurrence = aspect_cooccurrence_table(enriched, minimum_support=1)
        self.assertFalse(cooccurrence.empty)


class SikayetvarNlpProcessedOutputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.clean = pd.read_csv(cls.root / "data/processed/sikayetvar_all_hotels_complaints_clean.csv")
        cls.nlp = pd.read_csv(cls.root / "data/processed/sikayetvar_all_hotels_complaints_nlp.csv")
        cls.long = pd.read_csv(cls.root / "data/processed/sikayetvar_complaint_aspects_long.csv")
        cls.hotel = pd.read_csv(cls.root / "data/processed/sikayetvar_hotel_nlp_features.csv")
        cls.area = pd.read_csv(cls.root / "data/processed/sikayetvar_area_nlp_features.csv")

    def test_clean_rows_keys_and_raw_text_are_preserved(self):
        self.assertEqual(len(self.nlp), len(self.clean))
        self.assertTrue(self.nlp["complaint_id"].is_unique)
        self.assertEqual(self.nlp["complaint_id"].tolist(), self.clean["complaint_id"].tolist())
        pd.testing.assert_series_equal(self.nlp["complaint_text"], self.clean["complaint_text"], check_names=False)
        pd.testing.assert_series_equal(self.nlp["complaint_title"], self.clean["complaint_title"], check_names=False)

    def test_aspect_columns_counts_and_long_relation_reconcile(self):
        aspect_columns = [f"aspect_{aspect.lower()}" for aspect in CANONICAL_ASPECTS]
        self.assertTrue(all(self.nlp[column].dtype == bool for column in aspect_columns))
        self.assertTrue((self.nlp[aspect_columns].sum(axis=1) == self.nlp["aspect_count"]).all())
        self.assertEqual(len(self.long), len(self.nlp) * len(CANONICAL_ASPECTS))
        self.assertEqual(int(self.long["matched"].sum()), int(self.nlp["aspect_count"].sum()))
        self.assertEqual(int(self.nlp["no_aspect_detected_flag"].sum()), int(self.nlp["aspect_count"].eq(0).sum()))

    def test_feature_keys_rates_and_reliability_are_valid(self):
        self.assertTrue(self.hotel["hotel_id"].is_unique)
        self.assertTrue(self.area["area"].is_unique)
        rate_columns = [column for column in self.hotel if column.endswith("_mention_rate_pct")]
        self.assertTrue(self.hotel[rate_columns].apply(lambda column: column.dropna().between(0, 100).all()).all())
        self.assertEqual(
            self.hotel.loc[self.hotel["complaint_n"].lt(5), "small_n_flag"].eq(True).all(), True
        )
        self.assertEqual(set(self.hotel["nlp_feature_reliability"]), {"HIGH", "MEDIUM", "LOW"})

    def test_notebook_output_validation_passed(self):
        validation = pd.read_csv(self.root / "reports/sikayetvar_nlp_output_validation.csv")
        self.assertTrue(validation["passed"].all())


if __name__ == "__main__":
    unittest.main()
