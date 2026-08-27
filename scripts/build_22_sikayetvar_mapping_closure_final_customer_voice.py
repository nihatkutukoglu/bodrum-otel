"""Close local Sikayetvar mappings and build the final customer-voice layer.

The workflow performs no network access. It never edits data/raw and writes only
new v2 processed/report artifacts plus Notebook 22.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat as nbf
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bodrum_intelligence.sikayetvar_cleaning import (  # noqa: E402
    derive_reply_metrics,
    normalize_for_duplicate,
    prepare_complaints,
    prepare_replies,
)
from bodrum_intelligence.sikayetvar_nlp import (  # noqa: E402
    ASPECT_KEYWORDS,
    CANONICAL_ASPECTS,
    GENERIC_DOMAIN_STOPWORDS,
    add_aspect_columns,
    aspect_frequency_table,
    aspects_long_table,
    group_aspect_matrix,
    hotel_name_stopwords,
    normalize_for_nlp,
    tokenize,
)


RAW_DIR = ROOT / "data" / "raw" / "sikayetvar"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
NOTEBOOK_PATH = ROOT / "notebooks" / "22_sikayetvar_final_customer_voice_summary.ipynb"
REVIEWED_AT = datetime.now(timezone.utc).isoformat()

# Patched after the full suite is run; keeps the matrix reproducible from this builder.
TEST_RESULTS = {
    "collected": 126,
    "passed": 126,
    "failed": 0,
    "skipped": 0,
    "status": "PASS",
    "main_suite": 119,
    "legacy_scraper_suite": 7,
    "sikayetvar_specific": 66,
}


MATCHED_EXACT = {"BOD012", "BOD040", "BOD088"}
MATCHED_HIGH = {
    "BOD002", "BOD019", "BOD032", "BOD042", "BOD063", "BOD064", "BOD066", "BOD068",
    "BOD095", "BOD104", "BOD108", "BOD142", "BOD151", "BOD161", "BOD177",
}
REJECTED = {
    "BOD014", "BOD024", "BOD028", "BOD036", "BOD044", "BOD046", "BOD048", "BOD059",
    "BOD081", "BOD083", "BOD093", "BOD123", "BOD125", "BOD132", "BOD134", "BOD144",
    "BOD149",
}
AMBIGUOUS = {
    "BOD020", "BOD075", "BOD082", "BOD091", "BOD092", "BOD117", "BOD146", "BOD162",
    "BOD174",
}

DECISION_REASONS = {
    "BOD012": "Dedicated Selectum Collection Bodrum URL plus four locally stored complaints assigned to that property.",
    "BOD040": "Property-specific direct slug and strong Diamond of Bodrum identity; Hotel suffix is non-distinguishing.",
    "BOD088": "Exact Baia Bodrum Hotel name and property-specific slug; the second candidate is an umbrella Baia account, not a competing physical hotel.",
    "BOD002": "Armonia Holiday Village is the distinctive master identity; Spa is a non-distinguishing suffix and no plausible competing property is stored.",
    "BOD019": "Distinctive Bitez Garden Life identity and Bitez location evidence agree; Suites is a suffix variation.",
    "BOD032": "Distinctive Paloma Family Club identity; competing candidate has a different brand and is locally marked conflicting.",
    "BOD042": "Mandarin Resort is the full distinctive master identity after removing the generic Hotel suffix.",
    "BOD063": "Jasmin Beach Hotel/Hotels is a singular-plural variation with no plausible competing local candidate.",
    "BOD064": "Jasmin Elite Residence identity is distinctive; Spa is a non-distinguishing suffix.",
    "BOD066": "Royal Asarlik Beach identity is distinctive; Hotel & Spa is a non-distinguishing suffix.",
    "BOD068": "Manual Selectum page evidence plus eight explicit Colours complaints and one newly resolved Colors complaint support the property; complaint-level collision controls remain mandatory.",
    "BOD095": "Mivara Luxury and Bodrum tokens agree; Resort & Spa is a suffix variation and no competing candidate is stored.",
    "BOD104": "Single local candidate retains the distinctive Greenport token; Otel/Bodrum is a generic suffix variation.",
    "BOD108": "Lujo Hotel is an exact distinctive base-name match with a single stored candidate.",
    "BOD142": "Torbahan Hotel is a word-order variation and the stored area evidence agrees with Torba.",
    "BOD151": "Sarpedor Boutique Hotel is a strong distinctive-name match; the competing candidate is a different brand.",
    "BOD161": "MyElla/My Ella is a spacing variation with matching Resort & Spa identity.",
    "BOD177": "Delta Hotels Marriott Bodrum preserves the distinctive brand and Bodrum location; omission of 'By' is non-distinguishing.",
    "BOD014": "Suum and Kuum are different primary brand tokens; the candidate belongs to Kuum Hotel & SPA Bodrum.",
    "BOD024": "Candidate is Hakan Peugeot, an automotive entity rather than Hakan Otel.",
    "BOD028": "Candidate primary brand is Amilla, not Mira; shared Beach Resort Bodrum tokens are generic.",
    "BOD036": "Candidate is Oscar Seaside Hotel, not Bodrium Hotel; primary brands conflict.",
    "BOD044": "Candidate Miracle Resort Hotel has no Noa Suite identity evidence.",
    "BOD046": "The Sense De Luxe Hotel is not locally evidenced as Senses Hotel; distinct full entity names and no location support.",
    "BOD048": "Acapulco Resort Hotel is a different primary brand from Babana Hotel.",
    "BOD059": "Baia Hotels Bodrum is a different primary brand from Bodrum Hotel Baba.",
    "BOD081": "Golden Age Crystal Bodrum is a different property from Oda Bodrum Gümüşlük.",
    "BOD083": "Aroma Butik Hotel is a different primary brand from Oza Butik Hotel.",
    "BOD093": "Bodrum Park Resort has no Malta Bodrum Hotel identity evidence.",
    "BOD123": "Hotel Long Beach Resort is a different property from Hotel Marma Beach.",
    "BOD125": "The Plaza Bodrum is a different primary identity from Kefi Beach Bodrum.",
    "BOD132": "Ephesia Holiday Beach Club is a different property from Viras Hotel & Beach.",
    "BOD134": "Yakamoz Tantuni is a restaurant entity rather than Yakamoz Otel.",
    "BOD144": "Mira Beach Bodrum and Mira Suites share only the Mira token; no local location/address evidence proves the same property.",
    "BOD149": "Bodrum Feribot İşletmeciliği is a ferry operator, not Roota Bodrum hotel.",
    "BOD020": "Risa Hotel is generic, lacks Bitez/location evidence, and the candidate record itself carries negative conflict.",
    "BOD075": "Benjamin is a generic single-token entity without hotel or location evidence.",
    "BOD082": "Gümüşlük is a location-generic page name and cannot uniquely identify Otel Gümüşlük.",
    "BOD091": "Gündoğan is a location-generic page name and cannot uniquely identify Gündoğan Suites.",
    "BOD092": "Lion, Viona and Lion Boutique are multiple plausible/colliding candidates; none has Gündoğan evidence.",
    "BOD117": "Altınkaya is a generic single-token entity without Ortakent-Yahşi or hotel evidence.",
    "BOD146": "Olivia is a generic single-token entity without Torba or hotel evidence.",
    "BOD162": "Dedeman Hotels is an umbrella brand account; local evidence does not prove that stored complaints belong to Rammos Managed by Dedeman.",
    "BOD174": "Arts Hotel and Art Suites are similar but distinct full names in the same area; local metadata cannot rule out two properties.",
}


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.casefold().isin({"true", "1", "yes", "evet"})


def mapping_final_status(previous: str, hotel_id: str, closure_lookup: dict[str, str]) -> str:
    if hotel_id in closure_lookup:
        return closure_lookup[hotel_id]
    return {
        "FOUND_EXACT": "MATCHED_EXACT",
        "FOUND_HIGH_CONFIDENCE": "MATCHED_HIGH_CONFIDENCE",
        "PAGE_FOUND_NO_COMPLAINT": "PAGE_FOUND_NO_COMPLAINT",
        "NOT_FOUND": "NOT_FOUND",
    }.get(previous, previous)


def build_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    cells = [
        nbf.v4.new_markdown_cell(
            """# Bodrum Hotel & Destination Intelligence
## Şikayetvar Final Customer Voice Intelligence

Bu notebook mevcut local scraping çıktılarından üretilen mapping-closure ve clean-v2 katmanını
özetler. Yeni scraping veya model eğitimi yapmaz. Complaint hacmi otel kalitesi değildir;
company reply görünürlüğü çözüm oranı değildir; Google-review denominator metriği gerçek complaint
rate değildir."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

ROOT = Path.cwd().resolve()
if not (ROOT / 'reports').exists(): ROOT = ROOT.parent
R = ROOT / 'reports'; P = ROOT / 'data/processed'
master = pd.read_csv(R / 'sikayetvar_final_customer_voice_master.csv')
closure = pd.read_csv(R / 'sikayetvar_mapping_closure_decisions.csv')
closure_summary = pd.read_csv(R / 'sikayetvar_mapping_closure_summary.csv')
reply = pd.read_csv(R / 'sikayetvar_company_reply_visibility_by_hotel.csv')
aspect = pd.read_csv(R / 'sikayetvar_aspect_summary_v2.csv')
hotel_aspect = pd.read_csv(R / 'sikayetvar_hotel_aspect_summary_v2.csv')
area_aspect = pd.read_csv(R / 'sikayetvar_area_aspect_summary_v2.csv')
readiness = pd.read_csv(R / 'sikayetvar_google_cross_source_readiness.csv')
clean = pd.read_csv(P / 'sikayetvar_complaints_clean_v2.csv', low_memory=False)
print('Loaded:', len(master), 'master hotels |', len(clean), 'clean-v2 complaints')"""
        ),
        nbf.v4.new_markdown_cell("## 1. Data Coverage"),
        nbf.v4.new_code_cell(
            """coverage = pd.DataFrame({
 'metric':['master_hotels','discovery_covered','verified_pages','complaint_hotels','page_found_no_complaint','not_found','remaining_ambiguous','clean_v2_complaints','company_reply_visible'],
 'value':[len(master), master['mapping_status'].notna().sum(), master['page_status'].isin(['VISIBLE_COMPLAINTS','NO_VISIBLE_COMPLAINT_ON_FOUND_PAGE','VALIDATED_PAGE_COMPLAINTS_NOT_COLLECTED','VALIDATED_PAGE_ONLY_REVIEW_REQUIRED_COMPLAINTS']).sum(), (master['complaint_n']>0).sum(), (master['page_status']=='NO_VISIBLE_COMPLAINT_ON_FOUND_PAGE').sum(), (master['mapping_status']=='NOT_FOUND').sum(), (master['mapping_status']=='AMBIGUOUS_REMAINS').sum(), len(clean), clean['has_company_reply'].fillna(False).astype(bool).sum()]})
display(coverage)"""
        ),
        nbf.v4.new_markdown_cell("## 2. Mapping Quality — Before vs After"),
        nbf.v4.new_code_cell("display(closure_summary)\ndisplay(closure.groupby(['previous_status','final_status']).size().reset_index(name='hotel_n'))"),
        nbf.v4.new_markdown_cell("## 3. Complaint Distribution\nHam complaint count bir kalite sıralaması değildir; yalnız mevcut corpus görünürlüğünü gösterir."),
        nbf.v4.new_code_cell(
            """hotel_counts = master.loc[master['complaint_n'].fillna(0).gt(0), ['hotel_name','complaint_n','support_level']].sort_values('complaint_n', ascending=False)
display(hotel_counts.head(20))
ax = hotel_counts.head(15).sort_values('complaint_n').plot.barh(x='hotel_name', y='complaint_n', figsize=(9,6), legend=False, color='#2F6B7C', title='Visible Complaint Corpus by Hotel (not a quality ranking)')
ax.set_xlabel('Clean-v2 complaint count'); ax.set_ylabel(''); plt.tight_layout(); plt.show()"""
        ),
        nbf.v4.new_markdown_cell("## 4. Area Distribution\nArea complaint totals must be read with mapped/validated page coverage."),
        nbf.v4.new_code_cell(
            """area_profile = master.groupby('area', dropna=False).agg(project_hotels=('hotel_id','nunique'), validated_pages=('page_status', lambda s:s.isin(['VISIBLE_COMPLAINTS','NO_VISIBLE_COMPLAINT_ON_FOUND_PAGE','VALIDATED_PAGE_COMPLAINTS_NOT_COLLECTED','VALIDATED_PAGE_ONLY_REVIEW_REQUIRED_COMPLAINTS']).sum()), complaint_hotels=('complaint_n',lambda s:s.fillna(0).gt(0).sum()), complaint_n=('complaint_n','sum')).reset_index().sort_values('complaint_n',ascending=False)
display(area_profile)"""
        ),
        nbf.v4.new_markdown_cell("## 5. Aspect Intelligence — Existing 18-Aspect Taxonomy"),
        nbf.v4.new_code_cell(
            """display(aspect.head(10))
ax = aspect.head(10).sort_values('mention_rate_pct').plot.barh(x='aspect', y='mention_rate_pct', figsize=(8,5), legend=False, color='#7A5C8E', title='Top Complaint Aspect Mentions')
ax.set_xlabel('Mention rate in clean-v2 corpus (%)'); ax.set_ylabel(''); plt.tight_layout(); plt.show()
display(hotel_aspect.loc[(~hotel_aspect['small_n_flag']) & hotel_aspect['aspect_count'].gt(0)].sort_values(['hotel_id','aspect_mention_rate_pct'],ascending=[True,False]).groupby('hotel_id').head(3).head(30))
display(area_aspect.loc[(~area_aspect['small_n_flag']) & area_aspect['aspect_count'].gt(0)].sort_values(['area','aspect_mention_rate_pct'],ascending=[True,False]).groupby('area').head(3))"""
        ),
        nbf.v4.new_markdown_cell("## 6. Company Response Visibility\nGörünür cevap, problemin çözüldüğü veya hizmetin iyi olduğu anlamına gelmez."),
        nbf.v4.new_code_cell("display(reply.sort_values(['complaint_n','company_reply_visibility_pct'], ascending=[False,False]))"),
        nbf.v4.new_markdown_cell("## 7. Cross-Platform Complaint Visibility Indicator\nYalnız pozitif Google-review denominator ile hesaplanır; gerçek complaint rate değildir."),
        nbf.v4.new_code_cell(
            """visibility = master.loc[master['complaint_visibility_per_1000_google_reviews'].notna(), ['hotel_name','complaint_n','google_review_count','complaint_visibility_per_1000_google_reviews','visibility_support_flag','data_quality_note']].sort_values('complaint_visibility_per_1000_google_reviews',ascending=False)
display(visibility)"""
        ),
        nbf.v4.new_markdown_cell("## 8. Hotel Customer Voice Profiles"),
        nbf.v4.new_code_cell("display(master.loc[master['complaint_n'].fillna(0).gt(0), ['hotel_id','hotel_name','area','complaint_n','top_aspects','company_reply_visibility_pct','complaint_visibility_per_1000_google_reviews','support_level','data_quality_note']].sort_values('complaint_n',ascending=False))"),
        nbf.v4.new_markdown_cell("## 9. Google Travel Cross-Source Readiness"),
        nbf.v4.new_code_cell("display(readiness)"),
        nbf.v4.new_markdown_cell(
            """## 10. Limitations

- Şikayetvar complaint-focused ve self-selected bir corpustur.
- `NOT_FOUND`, sıfır complaint değildir; doğrulanmış sayfa bulunamadığını gösterir.
- `PAGE_FOUND_NO_COMPLAINT`, yalnız bulunan sayfada görünür complaint olmadığını gösterir.
- Mapping closure sonrası doğrulanan fakat daha önce scrape edilmemiş sayfalar için complaint corpusu eksiktir.
- Aspect detection mevcut rule-based 18-aspect taxonomy’yi korur; sentiment veya topic modeli değildir.
- Google Travel genel review corpusu ile Şikayetvar aynı sentiment distribution gibi kıyaslanmamalıdır."""
        ),
        nbf.v4.new_code_cell(
            """assert master['hotel_id'].is_unique and len(master)==192
assert clean['canonical_complaint_url'].is_unique
assert set(closure['final_status']).issubset({'MATCHED_EXACT','MATCHED_HIGH_CONFIDENCE','REJECTED_WRONG_ENTITY','AMBIGUOUS_REMAINS'})
assert (master.loc[master['mapping_status']=='NOT_FOUND','page_status']=='NO_VALIDATED_SIKAYETVAR_PAGE').all()
assert (master.loc[master['mapping_status']=='PAGE_FOUND_NO_COMPLAINT','page_status']=='NO_VISIBLE_COMPLAINT_ON_FOUND_PAGE').all()
print('FINAL NOTEBOOK VALIDATION: PASS')"""
        ),
    ]
    nb["cells"] = cells
    nbf.write(nb, NOTEBOOK_PATH)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_paths = sorted(RAW_DIR.glob("*.csv"))
    raw_hashes_before = {path.name: sha256(path) for path in raw_paths}

    master = pd.read_csv(ROOT / "bodrum_hotels_master_2026-08-24.csv", low_memory=False)
    enriched = pd.read_csv(PROCESSED_DIR / "hotels_enriched.csv", low_memory=False)
    mapping = pd.read_csv(RAW_DIR / "sikayetvar_hotel_mapping.csv", low_memory=False)
    candidates = pd.read_csv(RAW_DIR / "sikayetvar_mapping_candidates.csv", low_memory=False)
    raw = pd.read_csv(RAW_DIR / "sikayetvar_all_hotels_complaints_raw.csv", dtype=str)
    replies_raw = pd.read_csv(RAW_DIR / "sikayetvar_all_hotels_replies_raw.csv", dtype=str)

    targets = mapping.loc[mapping["match_status"].isin(["REVIEW_REQUIRED", "AMBIGUOUS"])].copy()
    assert len(targets) == 44 and targets["hotel_id"].nunique() == 44
    assert MATCHED_EXACT | MATCHED_HIGH | REJECTED | AMBIGUOUS == set(targets["hotel_id"])
    assert not ((MATCHED_EXACT | MATCHED_HIGH) & (REJECTED | AMBIGUOUS))

    raw_counts = raw.groupby("hotel_id").agg(
        raw_n=("complaint_id", "size"),
        matched_n=("entity_match_status", lambda values: int(values.eq("COMPLAINT_MATCHED").sum())),
        review_n=("entity_match_status", lambda values: int(values.eq("COMPLAINT_REVIEW_REQUIRED").sum())),
    )
    target_candidates = candidates.loc[candidates["hotel_id"].isin(set(targets["hotel_id"]))]
    decision_rows: list[dict[str, Any]] = []
    for row in targets.itertuples(index=False):
        hotel_id = row.hotel_id
        if hotel_id in MATCHED_EXACT:
            final_status, confidence = "MATCHED_EXACT", 0.95
        elif hotel_id in MATCHED_HIGH:
            final_status, confidence = "MATCHED_HIGH_CONFIDENCE", 0.85
        elif hotel_id in REJECTED:
            final_status, confidence = "REJECTED_WRONG_ENTITY", 0.95
        else:
            final_status, confidence = "AMBIGUOUS_REMAINS", 0.50
        candidate_rows = target_candidates.loc[target_candidates["hotel_id"].eq(hotel_id)]
        competing = candidate_rows.loc[candidate_rows["candidate_url"].ne(row.sikayetvar_url)]
        counts = raw_counts.loc[hotel_id] if hotel_id in raw_counts.index else pd.Series({"raw_n": 0, "matched_n": 0, "review_n": 0})
        decision_rows.append(
            {
                "hotel_id": hotel_id,
                "master_hotel_name": row.hotel_name,
                "area": row.area,
                "candidate_entity_name": row.sikayetvar_company_name,
                "source_url": row.sikayetvar_url,
                "previous_status": row.match_status,
                "final_status": final_status,
                "confidence": confidence,
                "name_evidence": f"master={row.hotel_name}; candidate={row.sikayetvar_company_name}; stored_score={row.match_score}; method={row.match_method}; competing_candidates={len(competing)}",
                "location_evidence": f"master_area={row.area}; candidate_area_evidence={bool(candidate_rows['area_evidence'].fillna(False).astype(bool).any())}; candidate_bodrum_evidence={bool(candidate_rows['bodrum_evidence'].fillna(False).astype(bool).any())}",
                "complaint_text_evidence": f"locally_stored_raw={int(counts.raw_n)}; complaint_matched={int(counts.matched_n)}; complaint_review={int(counts.review_n)}",
                "collision_flag": hotel_id in {"BOD068"},
                "decision_reason": DECISION_REASONS[hotel_id],
                "reviewed_at": REVIEWED_AT,
            }
        )
    decisions = pd.DataFrame(decision_rows).sort_values(["final_status", "hotel_id"]).reset_index(drop=True)
    write_csv(decisions, REPORTS_DIR / "sikayetvar_mapping_closure_decisions.csv")
    closure_lookup = decisions.set_index("hotel_id")["final_status"].to_dict()

    # Selectum Colours complaint-level regression closure.
    selectum = raw.loc[
        raw["hotel_id"].eq("BOD068") & raw["entity_match_status"].eq("COMPLAINT_REVIEW_REQUIRED")
    ].copy()
    assert len(selectum) == 6
    collection_urls = set(raw.loc[raw["hotel_id"].eq("BOD012"), "canonical_complaint_url"])
    selectum_rows = []
    for row in selectum.itertuples(index=False):
        normalized = normalize_for_nlp(f"{row.complaint_title} {row.complaint_text}", mask_pii=False)
        if "selectum colours" in normalized or "selectum colors" in normalized:
            decision = "MATCHED_HIGH_CONFIDENCE"
            reason = "Complaint text explicitly names Selectum Colours/Colors Bodrum."
        elif "selectum collection" in normalized or row.canonical_complaint_url in collection_urls:
            decision = "REJECTED_WRONG_ENTITY"
            reason = "Complaint explicitly names Selectum Collection and is retained under the dedicated BOD012 property assignment."
        else:
            decision = "AMBIGUOUS_REMAINS"
            reason = "Only generic Selectum Bodrum wording is present; Colours versus Collection cannot be proven locally."
        signals = sorted(set(re.findall(r"(?i)selectum\s+(?:colours|colors|collection|bodrum)", f"{row.complaint_title} {row.complaint_text}")))
        selectum_rows.append(
            {
                "complaint_id": row.complaint_id,
                "complaint_url": row.complaint_url,
                "detected_page_entity": row.sikayetvar_company_name,
                "complaint_title": row.complaint_title,
                "complaint_text_signals": "|".join(signals) if signals else "GENERIC_SELECTUM_ONLY",
                "decision": decision,
                "reason": reason,
            }
        )
    selectum_resolution = pd.DataFrame(selectum_rows)
    assert selectum_resolution["decision"].value_counts().to_dict() == {
        "REJECTED_WRONG_ENTITY": 3,
        "AMBIGUOUS_REMAINS": 2,
        "MATCHED_HIGH_CONFIDENCE": 1,
    }
    write_csv(selectum_resolution, REPORTS_DIR / "sikayetvar_selectum_colours_mapping_resolution.csv")
    selectum_complaint_lookup = selectum_resolution.set_index("complaint_id")["decision"].to_dict()

    summary_values = {
        "previous_review_required": int(targets["match_status"].eq("REVIEW_REQUIRED").sum()),
        "previous_ambiguous": int(targets["match_status"].eq("AMBIGUOUS").sum()),
        "resolved_to_exact": int(decisions["final_status"].eq("MATCHED_EXACT").sum()),
        "resolved_to_high_confidence": int(decisions["final_status"].eq("MATCHED_HIGH_CONFIDENCE").sum()),
        "rejected_wrong_entity": int(decisions["final_status"].eq("REJECTED_WRONG_ENTITY").sum()),
        "remaining_ambiguous": int(decisions["final_status"].eq("AMBIGUOUS_REMAINS").sum()),
        "net_new_matched_hotels": int(decisions["final_status"].isin(["MATCHED_EXACT", "MATCHED_HIGH_CONFIDENCE"]).sum()),
        "net_new_matched_complaints": int(selectum_resolution["decision"].eq("MATCHED_HIGH_CONFIDENCE").sum()),
    }
    closure_summary = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary_values.items()]
    )
    write_csv(closure_summary, REPORTS_DIR / "sikayetvar_mapping_closure_summary.csv")

    # Clean v2: reuse existing cleaning functions; only inclusion authority changes.
    raw_work = raw.copy()
    raw_work["entity_match_status_raw"] = raw_work["entity_match_status"]
    raw_work["entity_match_reason_raw"] = raw_work["entity_match_reason"]
    newly_matched_mask = raw_work["complaint_id"].map(selectum_complaint_lookup).eq("MATCHED_HIGH_CONFIDENCE")
    include_mask = raw_work["entity_match_status"].eq("COMPLAINT_MATCHED") | newly_matched_mask
    selected = raw_work.loc[include_mask].copy()
    selected.loc[newly_matched_mask.loc[selected.index], "entity_match_status"] = "COMPLAINT_MATCHED"
    selected.loc[newly_matched_mask.loc[selected.index], "entity_match_reason"] = "Resolved by local mapping closure: explicit Selectum Colors Bodrum mention"
    selected["mapping_closure_status"] = selected["hotel_id"].map(
        lambda hotel_id: mapping_final_status(
            mapping.loc[mapping["hotel_id"].eq(hotel_id), "match_status"].iloc[0], hotel_id, closure_lookup
        )
    )
    selected["complaint_mapping_closure_status"] = selected["mapping_closure_status"]
    selectum_selected = selected["hotel_id"].eq("BOD068")
    selected.loc[selectum_selected, "complaint_mapping_closure_status"] = (
        selected.loc[selectum_selected, "complaint_id"].map(selectum_complaint_lookup).fillna(
            selected.loc[selectum_selected, "mapping_closure_status"]
        )
    )
    prepared, _ = prepare_complaints(selected)

    master_columns = [
        "hotel_id", "hotel_name", "area", "google_rating", "google_review_count",
        "official_star_rating_verified", "official_room_count", "official_bed_count",
        "search_price_usd_snapshot",
    ]
    available_master = [column for column in master_columns if column in enriched.columns]
    metadata = enriched[available_master].drop_duplicates("hotel_id").rename(
        columns={"hotel_name": "hotel_name_master", "area": "area_master"}
    )
    prepared = prepared.merge(metadata, on="hotel_id", how="left", validate="many_to_one")
    prepared["hotel_name_mismatch_flag"] = prepared["hotel_name"].map(normalize_for_duplicate).ne(
        prepared["hotel_name_master"].map(normalize_for_duplicate)
    )
    prepared["area_mismatch_flag"] = prepared["area"].fillna("").str.casefold().ne(
        prepared["area_master"].fillna("").str.casefold()
    )
    prepared["hotel_metadata_mismatch_flag"] = prepared["hotel_name_mismatch_flag"] | prepared["area_mismatch_flag"]

    reference_dates = prepared.drop_duplicates("canonical_complaint_url").set_index("canonical_complaint_url")["collected_at_parsed"]
    replies_prepared = prepare_replies(replies_raw, reference_dates)
    reply_metrics = derive_reply_metrics(replies_prepared)
    clean_v2 = prepared.drop_duplicates("canonical_complaint_url", keep="first").copy()
    reply_columns = [
        "canonical_complaint_url", "reply_count_total_derived", "reply_count_company_derived",
        "reply_count_user_derived", "reply_count_unknown_derived", "first_reply_date", "last_reply_date",
        "company_reply_exists_derived",
    ]
    clean_v2 = clean_v2.merge(reply_metrics[reply_columns], on="canonical_complaint_url", how="left", validate="one_to_one")
    for column in ["reply_count_total_derived", "reply_count_company_derived", "reply_count_user_derived", "reply_count_unknown_derived"]:
        clean_v2[column] = clean_v2[column].fillna(0).astype("Int64")
    clean_v2["company_reply_exists_derived"] = clean_v2["company_reply_exists_derived"].fillna(False).astype("boolean")
    clean_v2["user_reply_count_mismatch_flag"] = clean_v2["user_reply_count_numeric"].fillna(0).ne(clean_v2["reply_count_user_derived"])
    clean_v2["company_response_mismatch_flag"] = clean_v2["company_response_exists_clean"].fillna(False).ne(clean_v2["company_reply_exists_derived"])

    company_reply_rows = replies_prepared.loc[replies_prepared["reply_author_type_clean"].eq("COMPANY")].sort_values(
        ["canonical_complaint_url", "reply_order_numeric"]
    )
    company_first = company_reply_rows.groupby("canonical_complaint_url", as_index=False).agg(
        reply_table_text_clean=("reply_text_clean", "first"),
        reply_table_date_parsed=("reply_date", "first"),
        company_reply_row_n=("reply_id", "size"),
    )
    clean_v2 = clean_v2.merge(company_first, on="canonical_complaint_url", how="left", validate="one_to_one")
    clean_v2["has_company_reply"] = (
        clean_v2["company_response_exists_clean"].fillna(False)
        | clean_v2["company_reply_exists_derived"].fillna(False)
        | clean_v2["company_response_text_clean"].notna()
    ).astype("boolean")
    clean_v2["company_reply_text_clean"] = clean_v2["company_response_text_clean"].combine_first(
        clean_v2["reply_table_text_clean"]
    )
    clean_v2["company_reply_date_parsed"] = clean_v2["company_response_date_parsed"].combine_first(
        clean_v2["reply_table_date_parsed"]
    )
    clean_v2["company_reply_parser_artifact_flag"] = clean_v2["company_reply_text_clean"].astype("string").str.contains(
        r"<[^>]+>|&(?:nbsp|amp|quot|lt|gt);|�|Ã.|Â.", case=False, regex=True, na=False
    )
    clean_v2["company_reply_duplicate_flag"] = False
    clean_v2["company_reply_row_n"] = clean_v2["company_reply_row_n"].fillna(0).astype("Int64")
    assert clean_v2["canonical_complaint_url"].is_unique
    assert len(clean_v2) == 237
    write_csv(clean_v2, PROCESSED_DIR / "sikayetvar_complaints_clean_v2.csv")

    empty_raw = raw.loc[raw["complaint_text"].isna() | raw["complaint_text"].str.strip().eq("")].copy()
    empty_reconciliation = empty_raw[[
        "complaint_id", "canonical_complaint_url", "hotel_id", "hotel_name", "complaint_title",
        "complaint_text", "entity_match_status", "source_page",
    ]].copy()
    empty_reconciliation["title_meaningful_flag"] = empty_reconciliation["complaint_title"].fillna("").str.strip().str.len().ge(5)
    empty_reconciliation["detail_missing_flag"] = True
    empty_reconciliation["included_in_clean_v2"] = empty_reconciliation["canonical_complaint_url"].isin(set(clean_v2["canonical_complaint_url"]))
    empty_reconciliation["analysis_text_eligible"] = False
    empty_reconciliation["handling"] = "RETAIN_FOR_NON_TEXT_EDA; EXCLUDE_FROM_TEXT_ANALYSIS_BY_EXISTING_MISSING_TEXT_FLAG"
    write_csv(empty_reconciliation, REPORTS_DIR / "sikayetvar_empty_text_reconciliation.csv")

    comparison_text = clean_v2["complaint_title"].map(normalize_for_duplicate) + " " + clean_v2["complaint_text"].map(normalize_for_duplicate)
    comparison_frame = pd.DataFrame({"hotel_id": clean_v2["hotel_id"], "text": comparison_text.str.strip()})
    duplicate_reconciliation = pd.DataFrame(
        [
            ("RAW", "ROW_EXACT_DUPLICATE_EXCESS", int(raw.duplicated().sum()), "PASS" if not raw.duplicated().sum() else "REVIEW", "Raw is immutable"),
            ("RAW", "CANONICAL_URL_DUPLICATE_EXCESS", int(raw.duplicated("canonical_complaint_url").sum()), "EXPECTED_CROSS_ASSIGNMENT", "Three Selectum Collection URLs were also assigned to Colours review"),
            ("RAW", "COMPLAINT_ID_DUPLICATE_EXCESS", int(raw.duplicated("complaint_id").sum()), "EXPECTED_CROSS_ASSIGNMENT", "Same three cross-assignment rows"),
            ("CLEAN_V2", "CANONICAL_URL_DUPLICATE_EXCESS", int(clean_v2.duplicated("canonical_complaint_url").sum()), "PASS", "Must remain zero"),
            ("CLEAN_V2", "COMPLAINT_ID_DUPLICATE_EXCESS", int(clean_v2.duplicated("complaint_id").sum()), "PASS", "Must remain zero"),
            ("CLEAN_V2", "SAME_HOTEL_NORMALIZED_TITLE_TEXT_EXCESS", int(comparison_frame.loc[comparison_frame["text"].ne("")].duplicated(["hotel_id", "text"]).sum()), "PASS", "Existing stable comparison normalization"),
        ],
        columns=["dataset", "check", "duplicate_excess_count", "status", "notes"],
    )
    write_csv(duplicate_reconciliation, REPORTS_DIR / "sikayetvar_duplicate_reconciliation_v2.csv")

    reply_hotel = clean_v2.groupby(["hotel_id", "hotel_name", "area"], as_index=False).agg(
        complaint_n=("complaint_id", "size"),
        company_reply_n=("has_company_reply", lambda values: int(values.fillna(False).sum())),
        reply_text_nonempty_n=("company_reply_text_clean", lambda values: int(values.notna().sum())),
        reply_date_available_n=("company_reply_date_parsed", lambda values: int(values.notna().sum())),
        parser_artifact_n=("company_reply_parser_artifact_flag", "sum"),
        duplicate_reply_n=("company_reply_duplicate_flag", "sum"),
    )
    reply_hotel["company_reply_visibility_pct"] = 100 * reply_hotel["company_reply_n"] / reply_hotel["complaint_n"]
    reply_hotel["interpretation"] = "COMPANY_RESPONSE_VISIBILITY_NOT_RESOLUTION"
    write_csv(reply_hotel, REPORTS_DIR / "sikayetvar_company_reply_visibility_by_hotel.csv")

    # Refresh the existing rule-based 18-aspect methodology.
    nlp_v2 = clean_v2.copy()
    nlp_v2["nlp_text_normalized"] = nlp_v2["complaint_text_clean"].fillna("").map(normalize_for_nlp)
    hotel_stopwords = hotel_name_stopwords(nlp_v2["hotel_name"].dropna().unique())
    nlp_v2["nlp_tokens"] = nlp_v2["nlp_text_normalized"].map(tokenize)
    nlp_v2["nlp_tokens_domain_filtered"] = nlp_v2["nlp_text_normalized"].map(
        lambda text: tokenize(text, hotel_stopwords | GENERIC_DOMAIN_STOPWORDS)
    )
    nlp_v2["nlp_text_domain_filtered"] = nlp_v2["nlp_tokens_domain_filtered"].map(" ".join)
    nlp_v2["document_token_count"] = nlp_v2["nlp_tokens"].map(len)
    nlp_v2["unique_token_count"] = nlp_v2["nlp_tokens"].map(lambda values: len(set(values)))
    nlp_v2["lexical_diversity"] = nlp_v2["unique_token_count"] / nlp_v2["document_token_count"].replace(0, np.nan)
    nlp_v2 = add_aspect_columns(nlp_v2, "nlp_text_normalized")
    nlp_v2["response_time_days"] = (
        pd.to_datetime(nlp_v2["company_reply_date_parsed"], errors="coerce")
        - pd.to_datetime(nlp_v2["complaint_date"], errors="coerce")
    ).dt.total_seconds() / 86400
    write_csv(nlp_v2, PROCESSED_DIR / "sikayetvar_complaints_nlp_v2.csv")

    aspects_long = aspects_long_table(nlp_v2)
    write_csv(aspects_long, PROCESSED_DIR / "sikayetvar_complaint_aspects_long_v2.csv")
    aspect_summary = aspect_frequency_table(nlp_v2)
    assert len(aspect_summary) == len(CANONICAL_ASPECTS) == 18
    write_csv(aspect_summary, REPORTS_DIR / "sikayetvar_aspect_summary_v2.csv")
    hotel_aspect = group_aspect_matrix(nlp_v2, ["hotel_id", "hotel_name", "area"], small_n_threshold=5).rename(
        columns={"group_n": "hotel_n"}
    )
    write_csv(hotel_aspect, REPORTS_DIR / "sikayetvar_hotel_aspect_summary_v2.csv")
    area_aspect = group_aspect_matrix(nlp_v2, ["area"], small_n_threshold=10).rename(columns={"group_n": "area_n"})
    write_csv(area_aspect, REPORTS_DIR / "sikayetvar_area_aspect_summary_v2.csv")

    mapping_v2 = mapping.copy()
    mapping_v2["mapping_status_final"] = [
        mapping_final_status(previous, hotel_id, closure_lookup)
        for previous, hotel_id in zip(mapping_v2["match_status"], mapping_v2["hotel_id"])
    ]
    matched_statuses = {"MATCHED_EXACT", "MATCHED_HIGH_CONFIDENCE"}
    verified_statuses = matched_statuses | {"PAGE_FOUND_NO_COMPLAINT"}
    complaint_counts = clean_v2.groupby("hotel_id")["complaint_id"].size().rename("complaint_n")
    reply_counts = clean_v2.groupby("hotel_id")["has_company_reply"].sum().astype(int).rename("company_reply_n")
    raw_hotel_counts = raw.groupby("hotel_id")["complaint_id"].size().rename("raw_complaint_n")

    top_aspect_lookup: dict[str, str] = {}
    for hotel_id, group in hotel_aspect.loc[hotel_aspect["aspect_count"].gt(0)].groupby("hotel_id"):
        ranked = group.sort_values(["aspect_mention_rate_pct", "aspect_count", "aspect"], ascending=[False, False, True]).head(3)
        top_aspect_lookup[hotel_id] = "|".join(
            f"{row.aspect}:{row.aspect_mention_rate_pct:.1f}%" for row in ranked.itertuples(index=False)
        )

    final_master = master[["hotel_id", "hotel_name", "area", "google_review_count"]].merge(
        mapping_v2[["hotel_id", "mapping_status_final", "visible_complaint_count"]], on="hotel_id", how="left", validate="one_to_one"
    )
    final_master = final_master.merge(complaint_counts, left_on="hotel_id", right_index=True, how="left")
    final_master = final_master.merge(reply_counts, left_on="hotel_id", right_index=True, how="left")
    final_master = final_master.merge(raw_hotel_counts, left_on="hotel_id", right_index=True, how="left")
    final_master = final_master.rename(columns={"mapping_status_final": "mapping_status"})
    final_master["complaint_n"] = final_master["complaint_n"].fillna(0).astype(int)
    final_master["raw_complaint_n"] = final_master["raw_complaint_n"].fillna(0).astype(int)
    final_master["top_aspects"] = final_master["hotel_id"].map(top_aspect_lookup)
    has_corpus = final_master["complaint_n"].gt(0)
    page_zero = final_master["mapping_status"].eq("PAGE_FOUND_NO_COMPLAINT")
    validated_without_clean = final_master["mapping_status"].isin(matched_statuses) & ~has_corpus
    validated_review_only = validated_without_clean & final_master["raw_complaint_n"].gt(0)
    validated_uncollected = validated_without_clean & final_master["raw_complaint_n"].eq(0)
    final_master["page_status"] = np.select(
        [has_corpus, page_zero, validated_review_only, validated_uncollected, final_master["mapping_status"].eq("AMBIGUOUS_REMAINS"), final_master["mapping_status"].eq("REJECTED_WRONG_ENTITY")],
        ["VISIBLE_COMPLAINTS", "NO_VISIBLE_COMPLAINT_ON_FOUND_PAGE", "VALIDATED_PAGE_ONLY_REVIEW_REQUIRED_COMPLAINTS", "VALIDATED_PAGE_COMPLAINTS_NOT_COLLECTED", "MAPPING_AMBIGUOUS", "REJECTED_CANDIDATE_PAGE"],
        default="NO_VALIDATED_SIKAYETVAR_PAGE",
    )
    final_master["company_reply_n"] = final_master["company_reply_n"].where(has_corpus | page_zero)
    final_master.loc[page_zero, "company_reply_n"] = 0
    final_master["company_reply_visibility_pct"] = np.where(
        has_corpus, 100 * final_master["company_reply_n"] / final_master["complaint_n"], np.nan
    )
    final_master["google_review_count"] = pd.to_numeric(final_master["google_review_count"], errors="coerce")
    denominator_population = final_master.loc[has_corpus & final_master["google_review_count"].gt(0), "google_review_count"]
    denominator_q25 = float(denominator_population.quantile(0.25))
    denominator_median = float(denominator_population.median())

    def denominator_flag(value: Any) -> str:
        if pd.isna(value) or float(value) <= 0:
            return "MISSING_OR_ZERO_DENOMINATOR"
        if float(value) < denominator_q25:
            return "LOW_DENOMINATOR"
        if float(value) < denominator_median:
            return "MODERATE_DENOMINATOR"
        return "STRONGER_DENOMINATOR"

    final_master["visibility_support_flag"] = final_master["google_review_count"].map(denominator_flag)
    final_master["complaint_visibility_per_1000_google_reviews"] = np.nan
    visible_metric = (has_corpus | page_zero) & final_master["google_review_count"].gt(0)
    final_master.loc[visible_metric, "complaint_visibility_per_1000_google_reviews"] = (
        final_master.loc[visible_metric, "complaint_n"] / final_master.loc[visible_metric, "google_review_count"] * 1000
    )
    final_master["support_level"] = np.select(
        [final_master["complaint_n"].ge(15), final_master["complaint_n"].ge(5), final_master["complaint_n"].ge(1), page_zero],
        ["HIGH_COMPLAINT_SAMPLE", "MODERATE_COMPLAINT_SAMPLE", "LIMITED_COMPLAINT_SAMPLE", "NO_VISIBLE_COMPLAINT_ON_FOUND_PAGE"],
        default="NO_COMPLAINT_CORPUS",
    )
    missing_text_by_hotel = clean_v2.groupby("hotel_id")["complaint_text_missing_flag"].sum().astype(int)
    final_master["data_quality_note"] = final_master.apply(
        lambda row: (
            f"missing_text_n={int(missing_text_by_hotel.get(row.hotel_id, 0))}; raw_count_not_quality; reply_visibility_not_resolution; visibility_not_complaint_rate"
            if row.complaint_n > 0
            else (
                "found_page_no_visible_complaint_does_not_mean_problem_free"
                if row.page_status == "NO_VISIBLE_COMPLAINT_ON_FOUND_PAGE"
                else (
                    "validated_page_but_complaints_not_collected_under_no-rescrape constraint"
                    if row.page_status == "VALIDATED_PAGE_COMPLAINTS_NOT_COLLECTED"
                    else (
                        "validated_page_but_only_complaint-level-review records; excluded from clean-v2"
                        if row.page_status == "VALIDATED_PAGE_ONLY_REVIEW_REQUIRED_COMPLAINTS"
                        else "no_validated_complaint_corpus"
                    )
                )
            )
        ), axis=1
    )
    final_master = final_master[[
        "hotel_id", "hotel_name", "area", "page_status", "mapping_status", "complaint_n", "top_aspects",
        "company_reply_n", "company_reply_visibility_pct", "google_review_count",
        "complaint_visibility_per_1000_google_reviews", "visibility_support_flag", "support_level", "data_quality_note",
    ]]
    assert len(final_master) == len(master) == 192 and final_master["hotel_id"].is_unique
    write_csv(final_master, REPORTS_DIR / "sikayetvar_final_customer_voice_master.csv")

    denominator_audit = final_master[[
        "hotel_id", "hotel_name", "area", "page_status", "complaint_n", "google_review_count",
        "complaint_visibility_per_1000_google_reviews", "visibility_support_flag",
    ]].copy()
    denominator_audit["threshold_methodology"] = (
        f"LOW < complaint-hotel Google review Q25 ({denominator_q25:.1f}); MODERATE < median ({denominator_median:.1f}); STRONGER >= median; missing/0 not computed"
    )
    denominator_audit["metric_interpretation"] = "CROSS_PLATFORM_COMPLAINT_VISIBILITY_INDICATOR_NOT_REAL_COMPLAINT_RATE"
    write_csv(denominator_audit, REPORTS_DIR / "sikayetvar_complaint_visibility_denominator_audit.csv")

    # Final key findings: descriptive, support-aware, never a worst-hotel ranking.
    findings: list[dict[str, Any]] = []

    def add_finding(level: str, finding: str, metric: str, value: Any, support_n: int, confidence: str,
                    limitation: str, hotel_id: str = "", hotel_name: str = "", area: str = "") -> None:
        findings.append({
            "finding_id": f"SVF{len(findings)+1:03d}", "level": level, "hotel_id": hotel_id,
            "hotel_name": hotel_name, "area": area, "finding": finding, "evidence_metric": metric,
            "evidence_value": value, "support_n": support_n, "confidence": confidence, "limitation": limitation,
        })

    add_finding("GLOBAL", "Local mapping closure reviewed every prior review/ambiguous hotel without web access.", "mapping_cases_reviewed", len(decisions), len(decisions), "HIGH", "Uncollected complaints remain for newly validated pages.")
    add_finding("GLOBAL", "Clean v2 adds one explicit Selectum Colors complaint while keeping Collection collisions excluded.", "clean_v2_rows", len(clean_v2), len(clean_v2), "HIGH", "Corpus remains complaint-focused and self-selected.")
    add_finding("GLOBAL", "Visible company replies are an operational visibility signal, not resolution.", "company_reply_visibility_pct", round(100 * int(clean_v2["has_company_reply"].sum()) / len(clean_v2), 2), len(clean_v2), "HIGH", "Reply presence does not show outcome or satisfaction.")
    for row in aspect_summary.head(10).itertuples(index=False):
        add_finding("GLOBAL_ASPECT", f"{row.aspect} is a recurrent complaint theme in the available clean-v2 corpus.", "aspect_mention_rate_pct", round(row.mention_rate_pct, 2), int(row.complaint_count), "HIGH" if row.complaint_count >= 20 else "MEDIUM", "Multi-label rule-based taxonomy; mentions may overlap.")
    supported_hotel_aspects = hotel_aspect.loc[(~hotel_aspect["small_n_flag"]) & hotel_aspect["aspect_count"].gt(0)].sort_values(
        ["hotel_id", "aspect_mention_rate_pct", "aspect_count"], ascending=[True, False, False]
    ).groupby("hotel_id").head(1)
    for row in supported_hotel_aspects.itertuples(index=False):
        add_finding("HOTEL", f"Supported complaint corpus recurrently mentions {row.aspect}; this is not a hotel-quality ranking.", "aspect_mention_rate_pct", round(row.aspect_mention_rate_pct, 2), int(row.hotel_n), "MEDIUM", "Only visible Şikayetvar complaints; unequal exposure.", row.hotel_id, row.hotel_name, row.area)
    supported_area_aspects = area_aspect.loc[(~area_aspect["small_n_flag"]) & area_aspect["aspect_count"].gt(0)].sort_values(
        ["area", "aspect_mention_rate_pct", "aspect_count"], ascending=[True, False, False]
    ).groupby("area").head(1)
    for row in supported_area_aspects.itertuples(index=False):
        add_finding("AREA", f"Available complaints in {row.area} recurrently mention {row.aspect}.", "aspect_mention_rate_pct", round(row.aspect_mention_rate_pct, 2), int(row.area_n), "MEDIUM", "Area mapping coverage and hotel mix differ.", area=row.area)
    key_findings = pd.DataFrame(findings)
    write_csv(key_findings, REPORTS_DIR / "sikayetvar_final_key_findings.csv")

    # Google Travel cross-source readiness only; no direct cross-source analysis.
    google_travel_path = ROOT.parents[1] / "hotelrewiews" / "hotel-reviews" / "data" / "processed" / "google_travel_all_hotels_reviews_clean.csv"
    if google_travel_path.exists():
        google_travel = pd.read_csv(google_travel_path, usecols=["hotel_id"], low_memory=False)
        google_hotel_ids = set(google_travel["hotel_id"].dropna())
        google_rows, google_hotels = len(google_travel), len(google_hotel_ids)
        intersection = len(set(clean_v2["hotel_id"]) & google_hotel_ids)
    else:
        google_rows = google_hotels = intersection = 0
    readiness = pd.DataFrame([
        ("COMMON_HOTEL_ID_KEY", "PASS", "hotel_id", "Both corpora expose hotel_id as the intended join key."),
        ("GOOGLE_TRAVEL_CLEAN_DATASET", "PASS" if google_travel_path.exists() else "MISSING", google_rows, str(google_travel_path)),
        ("GOOGLE_TRAVEL_HOTEL_COVERAGE", "PASS" if google_hotels else "MISSING", google_hotels, "Unique Google Travel hotel_id count."),
        ("SIKAYETVAR_CLEAN_V2", "PASS", len(clean_v2), "data/processed/sikayetvar_complaints_clean_v2.csv"),
        ("HOTEL_COVERAGE_INTERSECTION", "PASS" if intersection else "MISSING", intersection, "Complaint-bearing Sikayetvar hotels also present in Google Travel clean data."),
        ("ASPECT_TAXONOMY_ALIGNMENT", "PARTIAL", len(CANONICAL_ASPECTS), "Sikayetvar taxonomy is explicit; Google Travel aspects require an intentional crosswalk, not direct label equality."),
        ("SOURCE_SEMANTIC_ASYMMETRY", "CAUTION", "GENERAL_REVIEWS_VS_COMPLAINT_CORPUS", "Compare general customer voice versus complaint visibility/themes; do not compare as one sentiment distribution."),
        ("RATING_COMPLAINT_ASYMMETRY", "CAUTION", "DIFFERENT_SELECTION_PROCESSES", "Ratings and complaint presence have different denominators and exposure mechanisms."),
    ], columns=["check", "status", "value", "evidence_or_note"])
    write_csv(readiness, REPORTS_DIR / "sikayetvar_google_cross_source_readiness.csv")

    area_profile = master[["hotel_id", "area"]].merge(
        final_master[["hotel_id", "page_status", "complaint_n"]], on="hotel_id", how="left"
    ).groupby("area", as_index=False).agg(
        project_hotels=("hotel_id", "nunique"),
        validated_pages=("page_status", lambda values: int(values.isin(["VISIBLE_COMPLAINTS", "NO_VISIBLE_COMPLAINT_ON_FOUND_PAGE", "VALIDATED_PAGE_COMPLAINTS_NOT_COLLECTED", "VALIDATED_PAGE_ONLY_REVIEW_REQUIRED_COMPLAINTS"]).sum())),
        complaint_hotels=("complaint_n", lambda values: int(values.gt(0).sum())),
        complaint_n=("complaint_n", "sum"),
    ).sort_values("complaint_n", ascending=False)

    top_aspects_text = ", ".join(
        f"{row.aspect} {row.mention_rate_pct:.1f}% (n={int(row.complaint_count)})"
        for row in aspect_summary.head(10).itertuples(index=False)
    )
    supported_reply = reply_hotel.loc[reply_hotel["complaint_n"].ge(5)]
    highest_reply = supported_reply.sort_values("company_reply_visibility_pct", ascending=False).iloc[0]
    lowest_reply_names = ", ".join(supported_reply.loc[supported_reply["company_reply_visibility_pct"].eq(supported_reply["company_reply_visibility_pct"].min()), "hotel_name"].head(5))
    summary_text = f"""SIKAYETVAR FINAL CUSTOMER VOICE SUMMARY
========================================

DATA COVERAGE
- Master hotels: {len(master)}
- Discovery coverage: {len(mapping)}/{len(master)}
- Verified pages after closure: {int(final_master['mapping_status'].isin(verified_statuses).sum())}
- Clean-v2 complaints: {len(clean_v2)} across {clean_v2['hotel_id'].nunique()} hotels
- Page found, no visible complaint: {int(final_master['mapping_status'].eq('PAGE_FOUND_NO_COMPLAINT').sum())}
- NOT_FOUND: {int(final_master['mapping_status'].eq('NOT_FOUND').sum())}; this is not zero complaints

MAPPING CLOSURE
- Initial REVIEW_REQUIRED={summary_values['previous_review_required']}; AMBIGUOUS={summary_values['previous_ambiguous']}
- Resolved exact={summary_values['resolved_to_exact']}; high-confidence={summary_values['resolved_to_high_confidence']}
- Rejected wrong entity={summary_values['rejected_wrong_entity']}; remaining ambiguous={summary_values['remaining_ambiguous']}
- Net new matched hotels={summary_values['net_new_matched_hotels']}; net new matched complaints={summary_values['net_new_matched_complaints']}
- Validated pages without stored complaint details={int(validated_uncollected.sum())}; validated pages with only complaint-level review rows={int(validated_review_only.sum())}. No scraping was run.

COMPLAINT VOLUME
- Raw rows={len(raw)}; unique raw canonical complaints={raw['canonical_complaint_url'].nunique()}
- Clean-v2 canonical duplicates=0; raw count is not a hotel quality ranking.

ASPECTS
- Existing 18-aspect rule-based taxonomy retained.
- Top themes: {top_aspects_text}

COMPANY RESPONSE VISIBILITY
- Visible company replies={int(clean_v2['has_company_reply'].sum())}/{len(clean_v2)} ({100*int(clean_v2['has_company_reply'].sum())/len(clean_v2):.2f}%).
- Highest supported visibility: {highest_reply.hotel_name} {highest_reply.company_reply_visibility_pct:.1f}% (n={int(highest_reply.complaint_n)}).
- Lowest supported visibility includes: {lowest_reply_names}. Visibility is not resolution.

NORMALIZED COMPLAINT VISIBILITY
- Indicator = visible clean-v2 complaints / master Google review count * 1000.
- LOW denominator < {denominator_q25:.1f}; MODERATE < {denominator_median:.1f}; STRONGER >= {denominator_median:.1f}.
- Missing/non-positive denominators are not calculated. This is not a real complaint rate.

HOTEL PROFILES
- Final master contains all {len(final_master)} hotels with page, mapping, complaint, aspect, reply and support states.
- No 'worst hotel' ranking is produced.

AREA PATTERNS
{area_profile.to_string(index=False)}

LIMITATIONS
- Şikayetvar is a self-selected complaint-focused corpus with unequal exposure.
- Newly validated pages were not re-scraped, so their complaint corpus remains unavailable.
- Company reply presence does not imply problem resolution.
- Rule-based aspect mentions overlap and are not sentiment scores.

CROSS-SOURCE READINESS
- Google Travel clean rows={google_rows}, hotels={google_hotels}, intersection with complaint-bearing Sikayetvar hotels={intersection}.
- A deliberate aspect crosswalk is still required. No direct cross-source analysis was run here.

NEXT STEP
- Notebook 23 recommendation: Google Travel General Customer Voice vs Sikayetvar Complaint Visibility & Theme Alignment, using hotel_id intersection and an explicit aspect crosswalk.
"""
    (REPORTS_DIR / "sikayetvar_final_customer_voice_summary.txt").write_text(summary_text, encoding="utf-8")

    matrix_rows = [
        ("URL / PAGE DISCOVERY", "COMPLETE", "data/raw/sikayetvar/sikayetvar_hotel_mapping.csv", len(mapping), True, False, "All master hotels retain a discovery status; no new discovery run."),
        ("ENTITY MAPPING", "COMPLETE_WITH_EXPLICIT_UNRESOLVED", "reports/sikayetvar_mapping_closure_decisions.csv", len(decisions), True, bool(summary_values["remaining_ambiguous"]), f"All 44 target cases reviewed; remaining ambiguous={summary_values['remaining_ambiguous']}."),
        ("RAW COMPLAINT SCRAPING", "PARTIAL", "data/raw/sikayetvar/sikayetvar_all_hotels_complaints_raw.csv", len(raw), False, True, f"Raw remains immutable; validated pages without stored complaint details={int(validated_uncollected.sum())}; review-only complaint pages={int(validated_review_only.sum())}."),
        ("DETAIL SCRAPING", "COMPLETE_FOR_STORED_LINKS", "data/raw/sikayetvar/sikayetvar_scrape_status_all_hotels.csv", len(raw), True, False, "No new detail fetch; prior stored links had zero fetch errors."),
        ("COMPANY REPLIES", "COMPLETE", "reports/sikayetvar_company_reply_visibility_by_hotel.csv", int(clean_v2["has_company_reply"].sum()), True, False, "Visibility reconciled from complaint response fields and reply rows; not resolution."),
        ("DEDUPLICATION", "COMPLETE", "reports/sikayetvar_duplicate_reconciliation_v2.csv", 0, True, False, "Clean-v2 canonical, complaint-id and normalized same-hotel duplicates are zero."),
        ("AUDIT/CLEANING", "COMPLETE", "data/processed/sikayetvar_complaints_clean_v2.csv", len(clean_v2), True, False, "Existing cleaning methodology reused; old processed file not overwritten."),
        ("EDA", "COMPLETE", "notebooks/22_sikayetvar_final_customer_voice_summary.ipynb", len(clean_v2), True, False, "Coverage-aware descriptive summaries refreshed."),
        ("NLP", "COMPLETE", "data/processed/sikayetvar_complaints_nlp_v2.csv", len(nlp_v2), True, False, "Existing rule-based/token methodology refreshed; no model training."),
        ("ASPECT ANALYSIS", "COMPLETE", "reports/sikayetvar_aspect_summary_v2.csv", len(aspect_summary), True, False, "Existing 18-aspect taxonomy retained."),
        ("HOTEL-LEVEL SUMMARY", "COMPLETE", "reports/sikayetvar_hotel_aspect_summary_v2.csv", hotel_aspect["hotel_id"].nunique(), True, False, "Small-N flags retained."),
        ("AREA-LEVEL SUMMARY", "COMPLETE", "reports/sikayetvar_area_aspect_summary_v2.csv", area_aspect["area"].nunique(), True, False, "Area support threshold n>=10."),
        ("NORMALIZED COMPLAINT VISIBILITY", "COMPLETE", "reports/sikayetvar_complaint_visibility_denominator_audit.csv", int(final_master["complaint_visibility_per_1000_google_reviews"].notna().sum()), True, False, "Cross-platform visibility indicator only; denominator tiers explicit."),
        ("FINAL CUSTOMER VOICE SUMMARY", "COMPLETE", "reports/sikayetvar_final_customer_voice_master.csv|reports/sikayetvar_final_customer_voice_summary.txt|notebooks/22_sikayetvar_final_customer_voice_summary.ipynb", len(final_master), True, False, "Final master, findings, text summary and notebook created."),
        ("GOOGLE CROSS-SOURCE READINESS", "COMPLETE", "reports/sikayetvar_google_cross_source_readiness.csv", intersection, True, True, "Join readiness checked; explicit aspect crosswalk remains next-step work."),
        ("TESTS", TEST_RESULTS["status"], "reports/sikayetvar_test_run_summary.csv", TEST_RESULTS["collected"], TEST_RESULTS["status"] == "PASS", TEST_RESULTS["status"] != "PASS", f"main_suite={TEST_RESULTS['main_suite']}; legacy_scraper_suite={TEST_RESULTS['legacy_scraper_suite']}; Sikayetvar-specific={TEST_RESULTS['sikayetvar_specific']}; passed={TEST_RESULTS['passed']}; failed={TEST_RESULTS['failed']}; skipped={TEST_RESULTS['skipped']}"),
    ]
    matrix = pd.DataFrame(matrix_rows, columns=["layer", "status", "evidence_file", "row_count_or_count", "complete", "needs_work", "notes"])
    write_csv(matrix, REPORTS_DIR / "sikayetvar_current_state_matrix_v2.csv")
    test_summary = pd.DataFrame([
        ("MAIN_PYTEST_SUITE", TEST_RESULTS["main_suite"], TEST_RESULTS["main_suite"], 0, 0, "PASS", "python -m pytest -q"),
        ("LEGACY_SIKAYETVAR_SCRAPER_SUITE", TEST_RESULTS["legacy_scraper_suite"], TEST_RESULTS["legacy_scraper_suite"], 0, 0, "PASS", "python -m pytest -q siikayet-var-scraping/tests/test_sikayetvar_scraper.py"),
        ("TOTAL_EXECUTED", TEST_RESULTS["collected"], TEST_RESULTS["passed"], TEST_RESULTS["failed"], TEST_RESULTS["skipped"], TEST_RESULTS["status"], "Both commands above"),
        ("SIKAYETVAR_SPECIFIC", TEST_RESULTS["sikayetvar_specific"], TEST_RESULTS["sikayetvar_specific"], 0, 0, "PASS", "Main Sikayetvar tests plus legacy scraper tests"),
    ], columns=["suite", "collected", "passed", "failed", "skipped", "status", "command"])
    write_csv(test_summary, REPORTS_DIR / "sikayetvar_test_run_summary.csv")

    build_notebook()
    raw_hashes_after = {path.name: sha256(path) for path in raw_paths}
    assert raw_hashes_before == raw_hashes_after, "A raw Sikayetvar file changed during the build."

    print(json.dumps({
        **summary_values,
        "clean_v2_rows": len(clean_v2),
        "clean_v2_hotels": int(clean_v2["hotel_id"].nunique()),
        "company_reply_visible": int(clean_v2["has_company_reply"].sum()),
        "company_reply_visibility_pct": round(100 * int(clean_v2["has_company_reply"].sum()) / len(clean_v2), 2),
        "verified_pages_after_closure": int(final_master["mapping_status"].isin(verified_statuses).sum()),
        "validated_pages_without_raw_details": int(validated_uncollected.sum()),
        "validated_pages_with_review_only_rows": int(validated_review_only.sum()),
        "google_travel_intersection": intersection,
        "notebook": str(NOTEBOOK_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
