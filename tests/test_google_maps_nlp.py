import unittest

import pandas as pd

from bodrum_intelligence.google_maps_nlp import (
    CANONICAL_ASPECTS,
    ASPECT_KEYWORDS,
    classify_driver,
    detect_aspects,
    distinctive_terms,
    normalize_for_nlp,
    sample_tier,
    tokenize,
)


class GoogleMapsNlpHelpersTest(unittest.TestCase):
    def test_aspect_dictionary_uses_only_canonical_names(self):
        self.assertEqual(set(ASPECT_KEYWORDS.keys()), set(CANONICAL_ASPECTS))

    def test_detect_aspects_multi_label(self):
        text = normalize_for_nlp("Oda çok temizdi ama personel ilgisizdi ve deniz güzeldi")
        aspects, matched = detect_aspects(text)
        self.assertIn("ROOM", aspects)
        self.assertIn("CLEANLINESS_HYGIENE", aspects)
        self.assertIn("STAFF_SERVICE", aspects)
        self.assertIn("BEACH_SEA", aspects)
        self.assertIn("ilgisiz", matched["STAFF_SERVICE"])

    def test_tokenize_drops_stopwords_and_short_tokens(self):
        tokens = tokenize(normalize_for_nlp("Bu otel ve bu oda çok iyi"))
        self.assertNotIn("bu", tokens)
        self.assertNotIn("ve", tokens)
        self.assertIn("oda", tokens)

    def test_classify_driver_thresholds(self):
        self.assertEqual(classify_driver(10, 40, low_n=10, high_n=40)[0], "POSITIVE_DRIVER_CANDIDATE")
        self.assertEqual(classify_driver(40, 10, low_n=10, high_n=40)[0], "NEGATIVE_DRIVER_CANDIDATE")
        self.assertEqual(classify_driver(35, 35, low_n=10, high_n=40)[0], "EXPERIENCE_DEFINING")
        self.assertEqual(classify_driver(5, 5, low_n=10, high_n=40)[0], "LOW_SIGNAL")
        cls, support_ok = classify_driver(50, 10, low_n=2, high_n=40)
        self.assertEqual(cls, "INSUFFICIENT_SAMPLE")
        self.assertFalse(support_ok)

    def test_sample_tier_boundaries(self):
        self.assertEqual(sample_tier(30), "HIGH_SAMPLE")
        self.assertEqual(sample_tier(10), "MEDIUM_SAMPLE")
        self.assertEqual(sample_tier(9), "LOW_SAMPLE")

    def test_distinctive_terms_ranks_group_specific_words_first(self):
        group = pd.Series(["oda kirliydi berbat", "oda yine kirliydi"])
        other = pd.Series(["her şey harikaydı", "personel çok iyiydi"])
        result = distinctive_terms(group, other, n=1, min_doc_count=1)
        self.assertEqual(result.iloc[0]["term"], "kirliydi")


if __name__ == "__main__":
    unittest.main()
