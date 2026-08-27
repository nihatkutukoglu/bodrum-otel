from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"


def test_sikayetvar_mapping_closure_has_one_decision_per_target():
    # NOTE: this does NOT re-derive "target" from the live mapping.csv's match_status
    # column. The v3 build (scripts/build_sikayetvar_clean_v3_final_customer_voice.py)
    # deliberately promotes closure-confirmed hotels away from REVIEW_REQUIRED/AMBIGUOUS
    # in that file (to unlock their targeted complaint scrape), so match_status no longer
    # reflects the historical review population. decisions["previous_status"] is the
    # frozen record of what each hotel's status WAS at closure-review time.
    decisions = pd.read_csv(REPORTS / "sikayetvar_mapping_closure_decisions.csv")
    assert len(decisions) == 44
    assert decisions["hotel_id"].is_unique
    assert set(decisions["previous_status"]) == {"REVIEW_REQUIRED", "AMBIGUOUS"}
    assert set(decisions["final_status"]).issubset(
        {"MATCHED_EXACT", "MATCHED_HIGH_CONFIDENCE", "REJECTED_WRONG_ENTITY", "AMBIGUOUS_REMAINS"}
    )


def test_sikayetvar_matched_high_confidence_has_evidence():
    decisions = pd.read_csv(REPORTS / "sikayetvar_mapping_closure_decisions.csv")
    matched = decisions.loc[decisions["final_status"].eq("MATCHED_HIGH_CONFIDENCE")]
    assert len(matched) > 0
    assert matched["decision_reason"].fillna("").str.len().ge(20).all()
    assert matched["name_evidence"].fillna("").str.len().ge(20).all()
    assert matched["confidence"].ge(0.80).all()


def test_sikayetvar_selectum_collision_is_preserved():
    resolution = pd.read_csv(REPORTS / "sikayetvar_selectum_colours_mapping_resolution.csv")
    assert len(resolution) == 6
    assert resolution["decision"].value_counts().to_dict() == {
        "REJECTED_WRONG_ENTITY": 3,
        "AMBIGUOUS_REMAINS": 2,
        "MATCHED_HIGH_CONFIDENCE": 1,
    }
    collection = resolution.loc[resolution["complaint_text_signals"].str.contains("Collection", case=False, na=False)]
    assert collection["decision"].eq("REJECTED_WRONG_ENTITY").all()


def test_sikayetvar_page_found_no_complaint_semantics():
    master = pd.read_csv(REPORTS / "sikayetvar_final_customer_voice_master.csv")
    page_zero = master.loc[master["mapping_status"].eq("PAGE_FOUND_NO_COMPLAINT")]
    assert len(page_zero) == 12
    assert page_zero["page_status"].eq("NO_VISIBLE_COMPLAINT_ON_FOUND_PAGE").all()
    assert page_zero["complaint_n"].eq(0).all()


def test_sikayetvar_not_found_is_not_zero_complaint_semantics():
    master = pd.read_csv(REPORTS / "sikayetvar_final_customer_voice_master.csv")
    not_found = master.loc[master["mapping_status"].eq("NOT_FOUND")]
    assert len(not_found) == 104
    assert not_found["page_status"].eq("NO_VALIDATED_SIKAYETVAR_PAGE").all()
    assert not_found["complaint_visibility_per_1000_google_reviews"].isna().all()


def test_sikayetvar_clean_v2_has_no_duplicates_or_wrong_entities():
    clean = pd.read_csv(PROCESSED / "sikayetvar_complaints_clean_v2.csv", low_memory=False)
    assert len(clean) == 237
    assert clean["canonical_complaint_url"].is_unique
    assert clean["complaint_id"].is_unique
    assert set(clean["entity_match_status"]) == {"COMPLAINT_MATCHED"}
    assert not clean["complaint_mapping_closure_status"].isin(
        {"REJECTED_WRONG_ENTITY", "AMBIGUOUS_REMAINS"}
    ).any()


def test_sikayetvar_company_reply_visibility_reconciles():
    clean = pd.read_csv(PROCESSED / "sikayetvar_complaints_clean_v2.csv", low_memory=False)
    hotel = pd.read_csv(REPORTS / "sikayetvar_company_reply_visibility_by_hotel.csv")
    assert int(clean["has_company_reply"].fillna(False).sum()) == 39
    assert hotel["complaint_n"].sum() == len(clean)
    assert hotel["company_reply_n"].sum() == 39
    expected = 100 * hotel["company_reply_n"] / hotel["complaint_n"]
    assert np.allclose(hotel["company_reply_visibility_pct"], expected)


def test_sikayetvar_visibility_denominator_and_final_master_primary_key():
    master = pd.read_csv(REPORTS / "sikayetvar_final_customer_voice_master.csv")
    assert len(master) == 192
    assert master["hotel_id"].is_unique
    valid = master["complaint_visibility_per_1000_google_reviews"].notna()
    assert master.loc[valid, "google_review_count"].gt(0).all()
    expected = master.loc[valid, "complaint_n"] / master.loc[valid, "google_review_count"] * 1000
    assert np.allclose(master.loc[valid, "complaint_visibility_per_1000_google_reviews"], expected)
    assert set(master["visibility_support_flag"]).issubset(
        {"LOW_DENOMINATOR", "MODERATE_DENOMINATOR", "STRONGER_DENOMINATOR", "MISSING_OR_ZERO_DENOMINATOR"}
    )
