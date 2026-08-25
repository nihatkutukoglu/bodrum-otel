from pathlib import Path

import numpy as np
import pandas as pd

from bodrum_intelligence.google_maps_all_hotels_nlp import build_hotel_features, build_review_nlp
from bodrum_intelligence.google_maps_nlp import CANONICAL_ASPECTS


EXPECTED = [
    "STAFF_SERVICE", "CLEANLINESS_HYGIENE", "FOOD_BEVERAGE", "ROOM", "BEACH_SEA", "POOL",
    "FACILITIES_MAINTENANCE", "RESERVATION", "PAYMENT_REFUND", "PRICE_VALUE", "CHECKIN_CHECKOUT",
    "AIR_CONDITIONING", "NOISE", "FAMILY_CHILDREN", "TRANSPORT_TRANSFER",
    "MANAGEMENT_COMMUNICATION", "SPA_WELLNESS", "SAFETY_SECURITY",
]


def test_canonical_aspect_names_are_exact():
    assert CANONICAL_ASPECTS == EXPECTED


def test_all_192_master_rows_and_missing_is_not_zero():
    root = Path(__file__).resolve().parents[1]
    targets = pd.read_csv(root / "data/processed/hotels_clean.csv")
    review_nlp = pd.DataFrame(columns=["hotel_id","rating_group"] + [f"aspect_{a}" for a in EXPECTED])
    features = build_hotel_features(targets, review_nlp)
    assert len(features) == 192
    assert not features.review_sample_available.any()
    assert features[f"overall_aspect_{EXPECTED[0]}_rate_pct"].isna().all()
    assert features[f"driver_gap_{EXPECTED[0]}_pp"].isna().all()


def test_driver_gap_only_exists_when_low_and_high_are_ready():
    clean_rows = []
    for i in range(10):
        clean_rows.append({"review_id":f"h{i}","hotel_id":"BOD001","hotel_name":"H","area":"A","review_rating":5,"rating_group":"HIGH","review_date":pd.NaT,"review_text_clean":"personel temiz","review_word_count":2,"potential_pii_flag":False,"is_rating_only":False})
        clean_rows.append({"review_id":f"l{i}","hotel_id":"BOD001","hotel_name":"H","area":"A","review_rating":1,"rating_group":"LOW","review_date":pd.NaT,"review_text_clean":"kirli oda","review_word_count":2,"potential_pii_flag":False,"is_rating_only":False})
    nlp, _ = build_review_nlp(pd.DataFrame(clean_rows))
    targets = pd.DataFrame([{"hotel_id":"BOD001","hotel_name":"H","area":"A","google_review_count":100}])
    feature = build_hotel_features(targets, nlp).iloc[0]
    assert bool(feature.full_driver_ready)
    assert not np.isnan(feature.driver_gap_STAFF_SERVICE_pp)
