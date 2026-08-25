import unittest

from bodrum_intelligence.project_summary import (
    consistency_row,
    format_number_tr,
    format_pct_tr,
    interpret_spearman,
)


class ProjectSummaryHelpersTest(unittest.TestCase):
    def test_turkish_number_format(self):
        self.assertEqual(format_number_tr(4412884), "4.412.884")
        self.assertEqual(format_number_tr(42.48, 1), "42,5")
        self.assertEqual(format_pct_tr(42.48, 1), "%42,5")

    def test_spearman_interpretation_is_cautious(self):
        result = interpret_spearman(0.986)
        self.assertIn("Çok güçlü", result)
        self.assertIn("nedensellik göstermez", result)

    def test_consistency_status(self):
        self.assertEqual(consistency_row("x", 10, 10)["status"], "PASS")
        self.assertEqual(consistency_row("x", 10, 11)["status"], "REVIEW")


if __name__ == "__main__":
    unittest.main()
