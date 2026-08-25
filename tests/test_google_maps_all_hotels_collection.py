from pathlib import Path

import pandas as pd

from bodrum_intelligence.google_maps_all_hotels_collection import (
    RAW_FIELDS, identity_status, merge_checkpoint, rating_group, stable_review_id,
)


def test_rating_groups_are_exact():
    assert [rating_group(value) for value in [1, 2, 3, 4, 5]] == ["LOW", "LOW", "MIXED", "HIGH", "HIGH"]
    assert rating_group(None) is None
    assert rating_group(6) is None


def test_place_id_and_name_collision_fails_closed():
    url = "https://www.google.com/maps/search/?api=1&query=x&query_place_id=abc"
    assert identity_status("Selectum Collection Bodrum", "Selectum Collection Bodrum", "abc", url)[0] == "FOUND_EXACT_PLACE_ID"
    assert identity_status("Selectum Collection Bodrum", "Selectum Colours Bodrum", "abc", url)[0] == "REVIEW_REQUIRED"


def test_checkpoint_dedupes_and_cap_can_be_applied(tmp_path: Path):
    path = tmp_path / "hotel.csv"
    row = {field: "" for field in RAW_FIELDS}
    row.update(review_id=stable_review_id("BOD001", 5, "today", "great"), hotel_id="BOD001", review_rating=5, review_text="great")
    merged = merge_checkpoint(path, [row, row])
    assert len(merged) == 1
    assert merged.iloc[0].hotel_id == "BOD001"
