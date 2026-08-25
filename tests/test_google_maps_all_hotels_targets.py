from pathlib import Path

import pandas as pd

from bodrum_intelligence.google_maps_all_hotels_collection import TARGET_FIELDS, build_targets, select_smoke_targets


ROOT = Path(__file__).resolve().parents[1]


def test_master_builds_all_192_place_id_targets():
    master = pd.read_csv(ROOT / "data/processed/hotels_clean.csv")
    targets = build_targets(master, cap=75)
    assert len(targets) == 192
    assert tuple(targets.columns) == TARGET_FIELDS
    assert targets.hotel_id.nunique() == targets.place_id.nunique() == 192
    assert targets.target_review_cap.eq(75).all()
    assert targets.collection_priority.tolist() == list(range(1, 193))


def test_existing_five_case_study_mappings_remain_present():
    master = pd.read_csv(ROOT / "data/processed/hotels_clean.csv")
    case = pd.read_csv(ROOT / "data/processed/google_maps_hotel_nlp_features.csv")
    targets = build_targets(master, current_nlp_hotel_ids=case.hotel_id)
    assert set(case.hotel_id).issubset(set(targets.hotel_id))


def test_smoke_selection_is_diverse_and_capped():
    master = pd.read_csv(ROOT / "data/processed/hotels_clean.csv")
    targets = build_targets(master)
    selected = select_smoke_targets(targets, 10, ["BOD012", "BOD068"])
    assert len(selected) == 10
    assert {"BOD012", "BOD068"}.issubset(set(selected.hotel_id))
    assert selected.area.nunique() >= 4
