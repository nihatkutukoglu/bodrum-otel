from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"


def test_sikayetvar_clean_v3_has_no_duplicates_or_wrong_entities():
    clean = pd.read_csv(PROCESSED / "sikayetvar_complaints_clean_v3.csv", low_memory=False)
    assert len(clean) == 353
    assert clean["canonical_complaint_url"].is_unique
    assert clean["complaint_id"].is_unique
    assert set(clean["entity_match_status"]) == {"COMPLAINT_MATCHED"}
    assert not clean["complaint_mapping_closure_status"].isin(
        {"REJECTED_WRONG_ENTITY", "AMBIGUOUS_REMAINS"}
    ).any()
    # v3 must be a strict superset of v2's matched complaints (targeted scrape only adds).
    clean_v2 = pd.read_csv(PROCESSED / "sikayetvar_complaints_clean_v2.csv", low_memory=False)
    assert set(clean_v2["canonical_complaint_url"]).issubset(set(clean["canonical_complaint_url"]))


def test_sikayetvar_v3_promoted_hotels_have_no_validated_uncollected_pages_left():
    master = pd.read_csv(REPORTS / "sikayetvar_final_customer_voice_master_v3.csv")
    assert len(master) == 192
    assert master["hotel_id"].is_unique
    assert (master["page_status"] == "VALIDATED_PAGE_COMPLAINTS_NOT_COLLECTED").sum() == 0
    assert set(master["page_status"]) == {
        "NO_VALIDATED_SIKAYETVAR_PAGE", "VISIBLE_COMPLAINTS", "REJECTED_CANDIDATE_PAGE",
        "NO_VISIBLE_COMPLAINT_ON_FOUND_PAGE", "MAPPING_AMBIGUOUS",
        "VALIDATED_PAGE_ONLY_REVIEW_REQUIRED_COMPLAINTS",
    }


def test_sikayetvar_v3_remaining_ambiguous_excluded_from_clean_set():
    master = pd.read_csv(REPORTS / "sikayetvar_final_customer_voice_master_v3.csv")
    clean = pd.read_csv(PROCESSED / "sikayetvar_complaints_clean_v3.csv", low_memory=False)
    ambiguous_ids = set(master.loc[master["mapping_status"].eq("AMBIGUOUS_REMAINS"), "hotel_id"])
    rejected_ids = set(master.loc[master["mapping_status"].eq("REJECTED_WRONG_ENTITY"), "hotel_id"])
    assert len(ambiguous_ids) == 9
    assert len(rejected_ids) == 17
    assert not (set(clean["hotel_id"]) & (ambiguous_ids | rejected_ids))


def test_sikayetvar_v3_selectum_collision_still_protected():
    resolution = pd.read_csv(REPORTS / "sikayetvar_selectum_colours_mapping_resolution.csv")
    assert len(resolution) == 6
    collection = resolution.loc[resolution["complaint_text_signals"].str.contains("Collection", case=False, na=False)]
    assert collection["decision"].eq("REJECTED_WRONG_ENTITY").all()
    clean = pd.read_csv(PROCESSED / "sikayetvar_complaints_clean_v3.csv", low_memory=False)
    selectum_complaints = clean.loc[clean["hotel_id"].eq("BOD068")]
    assert len(selectum_complaints) >= 1


def test_sikayetvar_v3_visibility_denominator_reconciles():
    master = pd.read_csv(REPORTS / "sikayetvar_final_customer_voice_master_v3.csv")
    valid = master["complaint_visibility_per_1000_google_reviews"].notna()
    assert master.loc[valid, "google_review_count"].gt(0).all()
    expected = master.loc[valid, "complaint_n"] / master.loc[valid, "google_review_count"] * 1000
    assert np.allclose(master.loc[valid, "complaint_visibility_per_1000_google_reviews"], expected)


def test_sikayetvar_v3_google_cross_source_readiness_improved():
    readiness = pd.read_csv(REPORTS / "sikayetvar_google_cross_source_readiness_v3.csv")
    intersection_row = readiness.loc[readiness["check"].eq("HOTEL_COVERAGE_INTERSECTION")]
    if not intersection_row.empty and int(intersection_row["value"].iloc[0]) > 0:
        old_readiness = pd.read_csv(REPORTS / "sikayetvar_google_cross_source_readiness.csv")
        old_row = old_readiness.loc[old_readiness["check"].eq("HOTEL_COVERAGE_INTERSECTION")]
        assert int(intersection_row["value"].iloc[0]) >= int(old_row["value"].iloc[0])
