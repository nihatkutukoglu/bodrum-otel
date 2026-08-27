"""Task D: Google Travel (general customer-voice) x Sikayetvar (complaint-focused)
source-aware cross-platform alignment.

Reads ONLY already-existing processed outputs from both sibling repos - it never
re-scrapes Google Travel (explicitly forbidden) and never re-scrapes Sikayetvar.
Every number below is recomputed from the current on-disk files; nothing is a
copied/hardcoded result from a prior run.

Google Travel = general guest review corpus (any sentiment, unprompted).
Sikayetvar   = complaint-focused, self-selected corpus (only unhappy guests post).
These are NOT the same population and must never be compared as one sentiment
distribution - only as complementary, differently-biased signals.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
GOOGLE_ROOT = ROOT.parents[1] / "hotelrewiews" / "hotel-reviews"
REPORTS_DIR = ROOT / "reports"
PROCESSED_DIR = ROOT / "data" / "processed"
CONFIG_DIR = ROOT / "config"
NOTEBOOK_PATH = ROOT / "notebooks" / "23_sikayetvar_google_travel_cross_source_customer_voice_alignment.ipynb"

GOOGLE_MENTION_THRESHOLD_PCT = 15.0
GOOGLE_CONCERN_LOW_CONTEXT_SHARE = 50.0
GOOGLE_STRENGTH_HIGH_CONTEXT_SHARE = 70.0
GOOGLE_MIN_SUPPORT_N = 5
SIKAYETVAR_THEME_THRESHOLD_PCT = 25.0
MIN_HOTEL_REVIEW_N_GOOGLE = 10
MIN_HOTEL_COMPLAINT_N_SIKAYETVAR = 5


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def load_inputs() -> dict[str, pd.DataFrame]:
    if not GOOGLE_ROOT.exists():
        raise FileNotFoundError(f"Sibling Google Travel repo not found at {GOOGLE_ROOT}")
    g_rating = pd.read_csv(GOOGLE_ROOT / "reports" / "google_travel_all_hotels_rating_summary.csv")
    g_mentions = pd.read_csv(GOOGLE_ROOT / "reports" / "google_travel_all_hotels_hotel_aspect_mentions.csv")
    g_context = pd.read_csv(GOOGLE_ROOT / "reports" / "google_travel_all_hotels_hotel_aspect_rating_context.csv")
    s_master = pd.read_csv(REPORTS_DIR / "sikayetvar_final_customer_voice_master_v3.csv")
    s_hotel_aspect = pd.read_csv(REPORTS_DIR / "sikayetvar_hotel_aspect_summary_v3.csv")
    crosswalk = pd.read_csv(CONFIG_DIR / "google_sikayetvar_aspect_crosswalk.csv")
    return {
        "g_rating": g_rating, "g_mentions": g_mentions, "g_context": g_context,
        "s_master": s_master, "s_hotel_aspect": s_hotel_aspect, "crosswalk": crosswalk,
    }


def build_hotel_coverage(g_rating: pd.DataFrame, s_master: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    google_covered = set(g_rating["hotel_id"])
    sikayetvar_verified = set(
        s_master.loc[s_master["mapping_status"].isin(
            ["MATCHED_EXACT", "MATCHED_HIGH_CONFIDENCE", "PAGE_FOUND_NO_COMPLAINT"]
        ), "hotel_id"]
    )
    complaint_bearing = set(s_master.loc[s_master["complaint_n"].fillna(0).gt(0), "hotel_id"])
    page_no_complaint = set(s_master.loc[s_master["page_status"].eq("NO_VISIBLE_COMPLAINT_ON_FOUND_PAGE"), "hotel_id"])

    g_supported = set(g_rating.loc[g_rating["n"].ge(MIN_HOTEL_REVIEW_N_GOOGLE), "hotel_id"])
    s_supported = set(s_master.loc[s_master["complaint_n"].fillna(0).ge(MIN_HOTEL_COMPLAINT_N_SIKAYETVAR), "hotel_id"])
    supported_common = google_covered & complaint_bearing & g_supported & s_supported

    sets = {
        "google_covered": google_covered,
        "sikayetvar_verified": sikayetvar_verified,
        "complaint_bearing": complaint_bearing,
        "page_no_complaint": page_no_complaint,
        "common_verified": google_covered & sikayetvar_verified,
        "common_complaint_bearing": google_covered & complaint_bearing,
        "common_page_no_complaint": google_covered & page_no_complaint,
        "supported_common": supported_common,
    }
    coverage = pd.DataFrame(
        [(k, len(v)) for k, v in sets.items()],
        columns=["hotel_set", "hotel_count"],
    )
    return coverage, sets


def build_hotel_alignment(inputs: dict[str, pd.DataFrame], sets: dict[str, set[str]]) -> pd.DataFrame:
    g_rating = inputs["g_rating"]
    s_master = inputs["s_master"]
    common = sorted(sets["common_verified"])

    g = g_rating.set_index("hotel_id")
    s = s_master.set_index("hotel_id")

    rows = []
    for hotel_id in common:
        g_row = g.loc[hotel_id] if hotel_id in g.index else None
        s_row = s.loc[hotel_id]
        rows.append({
            "hotel_id": hotel_id,
            "hotel_name": s_row["hotel_name"],
            "area": s_row["area"],
            "google_review_n": int(g_row["n"]) if g_row is not None else 0,
            "google_mean_rating": float(g_row["mean_rating"]) if g_row is not None else np.nan,
            "google_low_share": float(g_row["low_share"]) if g_row is not None else np.nan,
            "google_high_share": float(g_row["high_share"]) if g_row is not None else np.nan,
            "google_sample_support": g_row["sample_support"] if g_row is not None else "NO_GOOGLE_DATA",
            "sikayetvar_complaint_n": int(s_row["complaint_n"]),
            "sikayetvar_page_status": s_row["page_status"],
            "sikayetvar_top_aspects": s_row.get("top_aspects", ""),
            "sikayetvar_company_reply_visibility_pct": s_row.get("company_reply_visibility_pct", np.nan),
            "complaint_visibility_per_1000_google_reviews": s_row.get("complaint_visibility_per_1000_google_reviews", np.nan),
            "visibility_support_flag": s_row.get("visibility_support_flag", ""),
            "hotel_is_supported_common": hotel_id in sets["supported_common"],
        })
    alignment = pd.DataFrame(rows).sort_values(["sikayetvar_complaint_n", "google_review_n"], ascending=False)
    return alignment


def build_aspect_alignment(inputs: dict[str, pd.DataFrame], sets: dict[str, set[str]]) -> pd.DataFrame:
    crosswalk = inputs["crosswalk"]
    g_mentions = inputs["g_mentions"]
    g_context = inputs["g_context"]
    s_hotel_aspect = inputs["s_hotel_aspect"]
    common = sorted(sets["common_verified"])

    g_mentions_idx = g_mentions.set_index(["hotel_id", "aspect"])
    g_context_idx = g_context.set_index(["hotel_id", "aspect"])
    s_idx = s_hotel_aspect.set_index(["hotel_id", "aspect"])

    rows = []
    for _, cw in crosswalk.iterrows():
        canonical = cw["canonical_aspect"]
        g_aspects = [a for a in str(cw["google_aspects"]).split("|") if a and a != "nan"]
        s_aspects = [a for a in str(cw["sikayetvar_aspects"]).split("|") if a and a != "nan"]
        for hotel_id in common:
            # aggregate across any sub-aspects mapped to this canonical bucket
            g_mention_pct = np.nan
            g_support_n = 0
            g_low_share = np.nan
            g_high_share = np.nan
            g_signals = []
            for ga in g_aspects:
                key = (hotel_id, ga)
                if key in g_mentions_idx.index:
                    m = g_mentions_idx.loc[key]
                    g_signals.append((float(m["mention_rate_pct"]), int(m["support_n"])))
                if key in g_context_idx.index:
                    c = g_context_idx.loc[key]
                    g_low_share = np.nanmax([g_low_share, c["low_context_share"]]) if not pd.isna(g_low_share) else c["low_context_share"]
                    g_high_share = np.nanmax([g_high_share, c["high_context_share"]]) if not pd.isna(g_high_share) else c["high_context_share"]
            if g_signals:
                g_mention_pct = max(v for v, _ in g_signals)
                g_support_n = max(n for _, n in g_signals)

            s_mention_pct = np.nan
            s_small_n = True
            s_hotel_n = 0
            for sa in s_aspects:
                key = (hotel_id, sa)
                if key in s_idx.index:
                    row = s_idx.loc[key]
                    if pd.isna(s_mention_pct) or row["aspect_mention_rate_pct"] > s_mention_pct:
                        s_mention_pct = row["aspect_mention_rate_pct"]
                        s_small_n = bool(row["small_n_flag"])
                        s_hotel_n = int(row["hotel_n"])

            has_google_data = not pd.isna(g_mention_pct)
            has_sikayetvar_data = not pd.isna(s_mention_pct)
            google_supported = has_google_data and g_support_n >= GOOGLE_MIN_SUPPORT_N
            sikayetvar_supported = has_sikayetvar_data and not s_small_n

            google_mentioned = google_supported and g_mention_pct >= GOOGLE_MENTION_THRESHOLD_PCT
            google_concern = google_mentioned and not pd.isna(g_low_share) and g_low_share >= GOOGLE_CONCERN_LOW_CONTEXT_SHARE
            google_strength = google_mentioned and not pd.isna(g_high_share) and g_high_share >= GOOGLE_STRENGTH_HIGH_CONTEXT_SHARE
            sikayetvar_theme = sikayetvar_supported and s_mention_pct >= SIKAYETVAR_THEME_THRESHOLD_PCT

            if not (g_aspects or s_aspects):
                continue
            if g_aspects and not s_aspects:
                label = "GOOGLE_GENERAL_ONLY" if google_mentioned else "NO_SIGNAL"
            elif s_aspects and not g_aspects:
                label = "SIKAYETVAR_ONLY" if sikayetvar_theme else "NO_SIGNAL"
            elif (has_google_data and not google_supported) or (has_sikayetvar_data and not sikayetvar_supported):
                label = "LOW_SUPPORT"
            elif google_concern and sikayetvar_theme:
                label = "BOTH_SOURCE_CONCERN"
            elif google_strength and sikayetvar_theme:
                label = "GOOGLE_STRENGTH_VS_COMPLAINT"
            elif sikayetvar_theme and not google_mentioned:
                label = "SIKAYETVAR_ONLY"
            elif google_mentioned and not sikayetvar_theme:
                label = "GOOGLE_GENERAL_ONLY"
            else:
                label = "NO_SIGNAL"

            rows.append({
                "hotel_id": hotel_id, "canonical_aspect": canonical,
                "google_mention_rate_pct": g_mention_pct, "google_support_n": g_support_n,
                "google_low_context_share": g_low_share, "google_high_context_share": g_high_share,
                "sikayetvar_mention_rate_pct": s_mention_pct, "sikayetvar_hotel_n": s_hotel_n,
                "sikayetvar_small_n_flag": s_small_n if has_sikayetvar_data else None,
                "alignment_label": label,
            })
    return pd.DataFrame(rows)


def main() -> None:
    inputs = load_inputs()
    coverage, sets = build_hotel_coverage(inputs["g_rating"], inputs["s_master"])
    write_csv(coverage, REPORTS_DIR / "google_sikayetvar_cross_source_coverage.csv")

    hotel_alignment = build_hotel_alignment(inputs, sets)
    write_csv(hotel_alignment, REPORTS_DIR / "google_sikayetvar_hotel_alignment.csv")

    aspect_alignment = build_aspect_alignment(inputs, sets)
    write_csv(aspect_alignment, REPORTS_DIR / "google_sikayetvar_aspect_alignment.csv")

    divergence = aspect_alignment.loc[
        aspect_alignment["alignment_label"].isin(["BOTH_SOURCE_CONCERN", "GOOGLE_STRENGTH_VS_COMPLAINT"])
    ].merge(
        hotel_alignment[["hotel_id", "hotel_name", "area"]], on="hotel_id", how="left"
    ).sort_values(["alignment_label", "sikayetvar_mention_rate_pct"], ascending=[True, False])
    write_csv(divergence, REPORTS_DIR / "google_sikayetvar_source_divergence.csv")

    label_counts = aspect_alignment["alignment_label"].value_counts()
    findings: list[dict[str, Any]] = []

    def add_finding(level: str, finding: str, metric: str, value: Any, support_n: int, confidence: str,
                    limitation: str, hotel_id: str = "", hotel_name: str = "") -> None:
        findings.append({
            "finding_id": f"GSF{len(findings) + 1:03d}", "level": level, "hotel_id": hotel_id,
            "hotel_name": hotel_name, "finding": finding, "evidence_metric": metric,
            "evidence_value": value, "support_n": support_n, "confidence": confidence, "limitation": limitation,
        })

    add_finding(
        "GLOBAL", f"{len(sets['common_complaint_bearing'])} hotels have both Google Travel review data and at least one visible Sikayetvar complaint.",
        "common_complaint_bearing_hotel_n", len(sets["common_complaint_bearing"]), len(sets["common_complaint_bearing"]), "HIGH",
        "Google and Sikayetvar are different populations with different exposure; this is a coverage fact, not a sentiment comparison.",
    )
    add_finding(
        "GLOBAL", f"{len(sets['supported_common'])} hotels have enough sample size on both sides (Google n>={MIN_HOTEL_REVIEW_N_GOOGLE}, Sikayetvar complaint_n>={MIN_HOTEL_COMPLAINT_N_SIKAYETVAR}) for a genuinely supported comparison.",
        "supported_common_hotel_n", len(sets["supported_common"]), len(sets["supported_common"]), "HIGH",
        "Comparisons outside this supported set should be read as directional only.",
    )
    both_concern = aspect_alignment.loc[aspect_alignment["alignment_label"].eq("BOTH_SOURCE_CONCERN")]
    for row in both_concern.sort_values("sikayetvar_mention_rate_pct", ascending=False).head(8).itertuples(index=False):
        hotel_name = hotel_alignment.loc[hotel_alignment["hotel_id"].eq(row.hotel_id), "hotel_name"]
        add_finding(
            "HOTEL", f"{row.canonical_aspect} shows up as a concern in BOTH Google Travel (low-rating context) and Sikayetvar (recurrent complaint theme) for this hotel.",
            "sikayetvar_mention_rate_pct", round(row.sikayetvar_mention_rate_pct, 1), row.sikayetvar_hotel_n, "MEDIUM",
            "Two independently-biased corpora agreeing increases confidence, but neither is a random sample.",
            row.hotel_id, hotel_name.iloc[0] if len(hotel_name) else "",
        )
    strength_vs_complaint = aspect_alignment.loc[aspect_alignment["alignment_label"].eq("GOOGLE_STRENGTH_VS_COMPLAINT")]
    for row in strength_vs_complaint.sort_values("sikayetvar_mention_rate_pct", ascending=False).head(8).itertuples(index=False):
        hotel_name = hotel_alignment.loc[hotel_alignment["hotel_id"].eq(row.hotel_id), "hotel_name"]
        add_finding(
            "HOTEL", f"{row.canonical_aspect} is a Google Travel strength signal (high-rating context) yet still appears as a Sikayetvar complaint theme for this hotel - a genuine source divergence, not a contradiction.",
            "sikayetvar_mention_rate_pct", round(row.sikayetvar_mention_rate_pct, 1), row.sikayetvar_hotel_n, "MEDIUM",
            "Complaint corpora over-represent unhappy guests by design; this does not mean the Google signal is wrong.",
            row.hotel_id, hotel_name.iloc[0] if len(hotel_name) else "",
        )
    key_findings = pd.DataFrame(findings)
    write_csv(key_findings, REPORTS_DIR / "google_sikayetvar_key_findings.csv")

    top_concern_text = ", ".join(
        f"{r.hotel_id} {r.canonical_aspect} (Sikayetvar {r.sikayetvar_mention_rate_pct:.0f}%)"
        for r in both_concern.sort_values("sikayetvar_mention_rate_pct", ascending=False).head(5).itertuples(index=False)
    ) or "none at current support thresholds"
    top_divergence_text = ", ".join(
        f"{r.hotel_id} {r.canonical_aspect} (Sikayetvar {r.sikayetvar_mention_rate_pct:.0f}%, Google high-context {r.google_high_context_share:.0f}%)"
        for r in strength_vs_complaint.sort_values("sikayetvar_mention_rate_pct", ascending=False).head(5).itertuples(index=False)
    ) or "none at current support thresholds"
    sikayetvar_only = aspect_alignment.loc[aspect_alignment["alignment_label"].eq("SIKAYETVAR_ONLY")]
    google_only = aspect_alignment.loc[aspect_alignment["alignment_label"].eq("GOOGLE_GENERAL_ONLY")]

    summary_text = f"""GOOGLE TRAVEL x SIKAYETVAR CROSS-SOURCE CUSTOMER VOICE ALIGNMENT
=================================================================

SCOPE
- Google Travel = general guest review corpus (any sentiment, self-reported at will).
- Sikayetvar = complaint-focused, self-selected corpus (only unhappy guests post).
- These are NOT compared as one sentiment distribution; only as complementary signals.
- No row-level merge was performed; comparison is hotel_id / canonical-aspect summary level only.
- Google Travel was NOT re-scraped for this task.

COVERAGE
{coverage.to_string(index=False)}

ASPECT ALIGNMENT LABEL DISTRIBUTION
{label_counts.to_string()}

TOP BOTH_SOURCE_CONCERN SIGNALS
- {top_concern_text}

TOP GOOGLE_STRENGTH_VS_COMPLAINT DIVERGENCES
- {top_divergence_text}

SIKAYETVAR-ONLY THEMES (aspect-hotel pairs): {len(sikayetvar_only)}
GOOGLE-GENERAL-ONLY THEMES (aspect-hotel pairs): {len(google_only)}

COMPLAINT VISIBILITY INSIGHT
- complaint_visibility_per_1000_google_reviews is a cross-platform VISIBILITY INDICATOR, not a real complaint rate:
  it only accounts for complaints that are both visible on Sikayetvar and have a positive Google review-count denominator.

SEMANTIC LIMITATIONS
- Aspect crosswalk (config/google_sikayetvar_aspect_crosswalk.csv) required judgment calls (e.g. Google's
  STAFF/SERVICE split vs Sikayetvar's single STAFF_SERVICE bucket); it is documented but not the only valid mapping.
- Rule-based aspect detection on both sides; mentions may overlap and are not sentiment scores.
- Support thresholds (Google n>={MIN_HOTEL_REVIEW_N_GOOGLE}, Sikayetvar complaint_n>={MIN_HOTEL_COMPLAINT_N_SIKAYETVAR}, per-aspect
  Google support_n>={GOOGLE_MIN_SUPPORT_N}) exclude many hotel-aspect pairs from LOW_SUPPORT into a real label; this is a
  deliberate precision-over-coverage choice.
- No hotel ranking ("best"/"worst") is produced anywhere in this analysis.

NOTEBOOK
- notebooks/23_sikayetvar_google_travel_cross_source_customer_voice_alignment.ipynb
"""
    (REPORTS_DIR / "google_sikayetvar_cross_source_summary.txt").write_text(summary_text, encoding="utf-8")

    build_notebook(coverage, label_counts)

    print(coverage.to_string(index=False))
    print()
    print(label_counts.to_string())


def build_notebook(coverage: pd.DataFrame, label_counts: pd.Series) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    cells = [
        nbf.v4.new_markdown_cell(
            """# Bodrum Hotel & Destination Intelligence
## Google Travel x Sikayetvar Cross-Source Customer Voice Alignment

Google Travel is a general guest review corpus (any sentiment); Sikayetvar is a complaint-focused,
self-selected corpus (only unhappy guests post). This notebook compares them at the hotel_id /
canonical-aspect summary level only - never row-level - and never as one sentiment distribution.
Google Travel was **not** re-scraped for this task; Sikayetvar was not re-scraped beyond the
targeted verified-page collection already reflected in the v3 clean dataset."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd().resolve()
if not (ROOT / 'reports').exists(): ROOT = ROOT.parent
R = ROOT / 'reports'

coverage = pd.read_csv(R / 'google_sikayetvar_cross_source_coverage.csv')
hotel_alignment = pd.read_csv(R / 'google_sikayetvar_hotel_alignment.csv')
aspect_alignment = pd.read_csv(R / 'google_sikayetvar_aspect_alignment.csv')
divergence = pd.read_csv(R / 'google_sikayetvar_source_divergence.csv')
key_findings = pd.read_csv(R / 'google_sikayetvar_key_findings.csv')
crosswalk = pd.read_csv(ROOT / 'config' / 'google_sikayetvar_aspect_crosswalk.csv')
print('Loaded:', len(hotel_alignment), 'common hotels |', len(aspect_alignment), 'hotel-aspect rows')"""
        ),
        nbf.v4.new_markdown_cell("## 1. Coverage"),
        nbf.v4.new_code_cell(
            """display(coverage)
ax = coverage.set_index('hotel_set')['hotel_count'].plot.barh(figsize=(8,5), color='#2F6B7C')
ax.set_xlabel('Hotel count'); ax.set_ylabel(''); ax.set_title('Google x Sikayetvar Hotel-Set Coverage')
plt.tight_layout(); plt.show()"""
        ),
        nbf.v4.new_markdown_cell("## 2. Aspect Crosswalk"),
        nbf.v4.new_code_cell("display(crosswalk)"),
        nbf.v4.new_markdown_cell("## 3. Aspect Alignment Label Distribution\nMulti-label rule-based signals, not a model output."),
        nbf.v4.new_code_cell(
            """label_counts = aspect_alignment['alignment_label'].value_counts()
display(label_counts)
ax = label_counts.sort_values().plot.barh(figsize=(8,5), color='#7A5C8E')
ax.set_xlabel('Hotel x canonical-aspect pairs'); ax.set_ylabel(''); ax.set_title('Alignment Label Distribution')
plt.tight_layout(); plt.show()"""
        ),
        nbf.v4.new_markdown_cell("## 4. Aspect Alignment Heatmap (supported hotels)"),
        nbf.v4.new_code_cell(
            """supported_ids = hotel_alignment.loc[hotel_alignment['hotel_is_supported_common'], 'hotel_id']
pivot_source = aspect_alignment.loc[aspect_alignment['hotel_id'].isin(supported_ids)]
label_rank = {'NO_SIGNAL':0,'LOW_SUPPORT':1,'GOOGLE_GENERAL_ONLY':2,'SIKAYETVAR_ONLY':3,'GOOGLE_STRENGTH_VS_COMPLAINT':4,'BOTH_SOURCE_CONCERN':5}
pivot_source = pivot_source.assign(label_rank=pivot_source['alignment_label'].map(label_rank))
pivot = pivot_source.pivot_table(index='hotel_id', columns='canonical_aspect', values='label_rank', aggfunc='max')
if not pivot.empty:
    fig, ax = plt.subplots(figsize=(12, max(4, 0.35*len(pivot))))
    im = ax.imshow(pivot.fillna(-1).values, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=90)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_title('Hotel x Canonical-Aspect Alignment (supported hotels only; darker=stronger signal)')
    plt.tight_layout(); plt.show()
else:
    print('No supported hotels at current thresholds.')"""
        ),
        nbf.v4.new_markdown_cell("## 5. Hotel-Level Alignment Table"),
        nbf.v4.new_code_cell(
            """display(hotel_alignment.loc[hotel_alignment['hotel_is_supported_common']].sort_values('sikayetvar_complaint_n', ascending=False))"""
        ),
        nbf.v4.new_markdown_cell("## 6. Google Low-Context Share vs Sikayetvar Complaint Mention Rate"),
        nbf.v4.new_code_cell(
            """scatter_df = aspect_alignment.dropna(subset=['google_low_context_share','sikayetvar_mention_rate_pct'])
fig, ax = plt.subplots(figsize=(7,6))
colors = scatter_df['alignment_label'].map({'BOTH_SOURCE_CONCERN':'#C0392B','GOOGLE_STRENGTH_VS_COMPLAINT':'#E67E22','SIKAYETVAR_ONLY':'#8E44AD','GOOGLE_GENERAL_ONLY':'#2980B9','LOW_SUPPORT':'#BDC3C7','NO_SIGNAL':'#95A5A6'})
ax.scatter(scatter_df['google_low_context_share'], scatter_df['sikayetvar_mention_rate_pct'], c=colors, alpha=0.6)
ax.set_xlabel('Google low-rating-context share (%)'); ax.set_ylabel('Sikayetvar aspect mention rate (%)')
ax.set_title('Google Low-Context Share vs Sikayetvar Complaint Mention Rate')
plt.tight_layout(); plt.show()"""
        ),
        nbf.v4.new_markdown_cell("## 7. Complaint Visibility Indicator vs Google Rating Profile\nNot a real complaint rate - see limitations."),
        nbf.v4.new_code_cell(
            """vis_df = hotel_alignment.dropna(subset=['complaint_visibility_per_1000_google_reviews'])
fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(vis_df['google_mean_rating'], vis_df['complaint_visibility_per_1000_google_reviews'], alpha=0.6, color='#2F6B7C')
ax.set_xlabel('Google Travel mean rating'); ax.set_ylabel('Complaint visibility per 1000 Google reviews')
ax.set_title('Complaint Visibility Indicator vs Google Rating Profile')
plt.tight_layout(); plt.show()"""
        ),
        nbf.v4.new_markdown_cell("## 8. Selected Divergence Examples\nDescriptive only - no hotel ranking."),
        nbf.v4.new_code_cell("display(divergence.head(20))"),
        nbf.v4.new_markdown_cell("## 9. Key Findings"),
        nbf.v4.new_code_cell("display(key_findings)"),
        nbf.v4.new_markdown_cell(
            """## 10. Limitations

- Google Travel (general reviews) and Sikayetvar (complaints) are different, differently-biased populations;
  they are never compared as one sentiment distribution.
- No row-level merge; all comparison is at hotel_id / canonical-aspect summary level.
- The aspect crosswalk required judgment calls and is not the only valid mapping (see config file).
- Support thresholds are deliberately conservative (precision over coverage); many hotel-aspect pairs are LOW_SUPPORT.
- complaint_visibility_per_1000_google_reviews is a visibility indicator, not a real complaint rate.
- No 'best'/'worst' hotel ranking is produced anywhere in this notebook."""
        ),
        nbf.v4.new_code_cell(
            """assert hotel_alignment['hotel_id'].is_unique
assert set(aspect_alignment['alignment_label']).issubset({
    'BOTH_SOURCE_CONCERN','GOOGLE_GENERAL_ONLY','SIKAYETVAR_ONLY','GOOGLE_STRENGTH_VS_COMPLAINT','NO_SIGNAL','LOW_SUPPORT'
})
assert not aspect_alignment.duplicated(['hotel_id','canonical_aspect']).any()
print('FINAL NOTEBOOK VALIDATION: PASS')"""
        ),
    ]
    nb["cells"] = cells
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}})
    client.execute()
    nbf.write(nb, NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
