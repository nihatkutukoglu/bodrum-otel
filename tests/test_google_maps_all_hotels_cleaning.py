import pandas as pd

from bodrum_intelligence.google_maps_all_hotels_cleaning import clean_reviews, potential_pii, sample_readiness
from bodrum_intelligence.google_maps_all_hotels_collection import RAW_FIELDS
from bodrum_intelligence.google_maps_all_hotels_nlp import build_review_nlp


def raw_row(**kwargs):
    row = {field: "" for field in RAW_FIELDS}
    row.update(hotel_id="BOD001", hotel_name="Hotel", area="Area", review_id="r1", review_rating="5", review_text="Temiz oda")
    row.update(kwargs)
    return row


def test_rating_only_is_preserved_but_excluded_from_nlp():
    raw = pd.DataFrame([raw_row(), raw_row(review_id="r2", review_rating="2", review_text="")])
    clean, audit = clean_reviews(raw)
    nlp, _ = build_review_nlp(clean)
    assert len(clean) == 2
    assert audit["rating_only_n"] == 1
    assert set(nlp.review_id) == {"r1"}


def test_pii_flag_does_not_mutate_raw_text():
    text = "Bana 0555 111 22 33 veya a@example.com üzerinden ulaşın"
    clean, _ = clean_reviews(pd.DataFrame([raw_row(review_text=text)]))
    assert potential_pii(text)
    assert clean.iloc[0].review_text == text
    assert bool(clean.iloc[0].potential_pii_flag)
    assert "reviewer_name_raw" not in clean.columns


def test_full_driver_ready_requires_ten_low_and_ten_high():
    rows = [raw_row(review_id=f"h{i}", review_rating="5") for i in range(10)]
    rows += [raw_row(review_id=f"l{i}", review_rating="1", review_text="Kirli oda") for i in range(9)]
    clean, _ = clean_reviews(pd.DataFrame(rows))
    targets = pd.DataFrame([{"hotel_id":"BOD001","hotel_name":"Hotel","area":"Area"}])
    ready = sample_readiness(clean, targets).iloc[0]
    assert bool(ready.positive_driver_ready)
    assert not bool(ready.negative_driver_ready)
    assert not bool(ready.full_driver_ready)
