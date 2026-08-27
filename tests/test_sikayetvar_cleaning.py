import unittest
from pathlib import Path

import pandas as pd

from bodrum_intelligence.sikayetvar_cleaning import (
    build_reply_id,
    clean_raw_text_minimal,
    parse_nullable_bool,
    parse_numeric_count,
    parse_sikayetvar_date,
    standardize_author_type,
)


class SikayetvarCleaningHelpersTest(unittest.TestCase):
    def test_minimal_text_cleaning_preserves_language(self):
        raw = "  Hiç iyi değil!\u200b\n🙂  <b>Çözüm</b> &amp; destek yok.  "
        cleaned = clean_raw_text_minimal(raw)
        self.assertEqual(cleaned, "Hiç iyi değil! 🙂 Çözüm & destek yok.")

    def test_numeric_formats_and_missing(self):
        self.assertEqual(parse_numeric_count("1.234"), 1234)
        self.assertEqual(parse_numeric_count("1,2 B"), 1200)
        self.assertEqual(parse_numeric_count("1.2K"), 1200)
        self.assertEqual(parse_numeric_count("1200"), 1200)
        self.assertIs(parse_numeric_count(None), pd.NA)

    def test_date_with_year_is_exact(self):
        parsed, approximate, pattern = parse_sikayetvar_date(
            "5 Ağustos 2025 09:58", "2026-08-25T12:00:00+00:00"
        )
        self.assertEqual(parsed, pd.Timestamp("2025-08-05 09:58:00"))
        self.assertFalse(approximate)
        self.assertEqual(pattern, "DAY_MONTH_YEAR_TIME")

    def test_date_without_year_uses_reference_and_rolls_back_future_month(self):
        parsed, approximate, _ = parse_sikayetvar_date(
            "30 Eylül 17:12", "2026-08-25T12:00:00+00:00"
        )
        self.assertEqual(parsed, pd.Timestamp("2025-09-30 17:12:00"))
        self.assertTrue(approximate)

    def test_invalid_date_is_not_invented(self):
        parsed, approximate, pattern = parse_sikayetvar_date(
            "Teşekkür mesajı", "2026-08-25T12:00:00+00:00"
        )
        self.assertTrue(pd.isna(parsed))
        self.assertFalse(approximate)
        self.assertEqual(pattern, "UNRECOGNIZED")

    def test_reply_id_and_categories_are_stable(self):
        first = build_reply_id("https://example.test/a", 1, " Yanıt ")
        second = build_reply_id("https://example.test/a", 1, "Yanıt")
        self.assertEqual(first, second)
        self.assertEqual(standardize_author_type("firma"), "COMPANY")
        self.assertEqual(standardize_author_type("other"), "UNKNOWN")
        self.assertIs(parse_nullable_bool(""), pd.NA)


class SikayetvarProcessedCorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.raw = pd.read_csv(
            cls.project_root / "data/raw/sikayetvar/sikayetvar_all_hotels_complaints_raw.csv",
            dtype=str,
        )
        cls.raw_replies = pd.read_csv(
            cls.project_root / "data/raw/sikayetvar/sikayetvar_all_hotels_replies_raw.csv",
            dtype=str,
        )
        cls.clean = pd.read_csv(
            cls.project_root / "data/processed/sikayetvar_all_hotels_complaints_clean.csv",
            dtype=str,
        )
        cls.clean_replies = pd.read_csv(
            cls.project_root / "data/processed/sikayetvar_all_hotels_replies_clean.csv",
            dtype=str,
        )

    def test_clean_corpus_only_contains_matched_unique_complaints(self):
        self.assertEqual(set(self.clean["entity_match_status"]), {"COMPLAINT_MATCHED"})
        self.assertTrue(self.clean["canonical_complaint_url"].is_unique)
        self.assertFalse(self.clean.duplicated(["complaint_title", "complaint_text", "hotel_id"]).any())

    def test_clean_replies_have_valid_unique_parent_keys(self):
        self.assertTrue(self.clean_replies["reply_id"].is_unique)
        self.assertTrue(
            self.clean_replies["canonical_complaint_url"].isin(
                set(self.clean["canonical_complaint_url"])
            ).all()
        )

    def test_raw_complaint_and_reply_text_are_preserved(self):
        def normalize_line_endings(series):
            return series.fillna("<NA>").str.replace("\r\n", "\n", regex=False).str.replace(
                "\r", "\n", regex=False
            )

        raw_matched = self.raw.loc[
            self.raw["entity_match_status"].eq("COMPLAINT_MATCHED"),
            ["canonical_complaint_url", "hotel_id", "complaint_text"],
        ]
        complaint_check = self.clean[["canonical_complaint_url", "hotel_id", "complaint_text"]].merge(
            raw_matched,
            on=["canonical_complaint_url", "hotel_id"],
            how="left",
            suffixes=("_clean", "_raw"),
            validate="one_to_one",
        )
        self.assertTrue(
            normalize_line_endings(complaint_check["complaint_text_clean"]).eq(
                normalize_line_endings(complaint_check["complaint_text_raw"])
            ).all()
        )

        reply_check = self.clean_replies[
            ["canonical_complaint_url", "reply_order", "reply_text"]
        ].merge(
            self.raw_replies[["canonical_complaint_url", "reply_order", "reply_text"]],
            on=["canonical_complaint_url", "reply_order"],
            how="left",
            suffixes=("_clean", "_raw"),
            validate="one_to_one",
        )
        self.assertTrue(
            normalize_line_endings(reply_check["reply_text_clean"]).eq(
                normalize_line_endings(reply_check["reply_text_raw"])
            ).all()
        )

    def test_missing_support_is_not_converted_to_zero(self):
        missing_raw = self.clean["support_count"].isna()
        self.assertTrue(self.clean.loc[missing_raw, "support_count_numeric"].isna().all())

    def test_boolean_fields_are_valid_and_raw_hash_check_passed(self):
        for column in ["company_response_exists_clean", "progress_exists_clean"]:
            values = set(self.clean[column].dropna().str.casefold())
            self.assertTrue(values.issubset({"true", "false"}))
        validation = pd.read_csv(
            self.project_root / "reports/sikayetvar_output_validation.csv"
        ).set_index("check")
        self.assertEqual(validation.loc["raw_files_unchanged", "status"], "PASS")


if __name__ == "__main__":
    unittest.main()
