import json
import pandas as pd
import numpy as np
from pathlib import Path

HR = Path(r"C:\Users\bilin\OneDrive\Masaüstü\bodrum\hotelrewiews\hotel-reviews")
BO = Path(r"C:\Users\bilin\OneDrive\Masaüstü\bodrum\bodrum otel\bodrum-otel")

out = {}

# ---------------- MASTER TARGETS (coverage flags per hotel) ----------------
targets = pd.read_csv(HR / "config/multiplatform_hotel_targets.csv")
out["master_n"] = int(len(targets))
out["trip_verified_n"] = int((targets["trip_status"] == "verified_direct").sum())
out["google_verified_n"] = int((targets["google_travel_status"] == "verified_direct").sum())

# ---------------- GOOGLE TRAVEL ----------------
g_clean = pd.read_csv(HR / "data/processed/google_travel_all_hotels_reviews_clean.csv", low_memory=False)
g_rating = pd.read_csv(HR / "reports/google_travel_all_hotels_rating_summary.csv")
g_aspect = pd.read_csv(HR / "reports/google_travel_all_hotels_aspect_rating_summary.csv")

out["google"] = {
    "clean_reviews": int(len(g_clean)),
    "hotels": int(g_clean["hotel_id"].nunique()),
    "mean_rating": round(float(g_clean["review_rating_numeric"].dropna().astype(float).mean()), 2),
    "rating_group_dist": g_clean["rating_group"].value_counts().to_dict() if "rating_group" in g_clean.columns else None,
}
# rating group pct
rg = g_clean["rating_group"].value_counts(normalize=True).mul(100).round(1).to_dict()
out["google"]["rating_group_pct"] = rg

g_aspect["driver_score"] = g_aspect["high_rating_share_when_mentioned"] - g_aspect["low_rating_share_when_mentioned"]
out["google"]["aspects"] = g_aspect.sort_values("driver_score", ascending=False).to_dict("records")

# top reviewed hotels among Google-covered set (already in DATA.hotels via master reviews field - separate: top by google clean review_n)
top_g = g_rating.sort_values("n", ascending=False).head(10)[["hotel_id", "hotel_name", "n", "mean_rating"]]
out["google"]["top_reviewed"] = top_g.to_dict("records")

# breakdown of the "no analyzable Google Travel data" hotels: never matched/scraped
# at all vs. a page was found but it genuinely had zero reviews (never "few but
# below a threshold" - there is no such middle bucket in this dataset)
g_raw_files = list((HR / "data/raw/reviews/google_travel").glob("*.csv"))
g_raw_ids = {f.name.split("_")[0] for f in g_raw_files}
all_hotel_ids = set(targets["hotel_id"]) if "hotel_id" in targets.columns else set()
g_zero_review_ids = set()
for f in g_raw_files:
    hid = f.name.split("_")[0]
    if hid not in g_clean["hotel_id"].unique():
        try:
            if len(pd.read_csv(f, low_memory=False)) == 0:
                g_zero_review_ids.add(hid)
        except Exception:
            pass
out["google"]["never_matched_n"] = int(len(all_hotel_ids - g_raw_ids)) if all_hotel_ids else int(192 - len(g_raw_ids))
out["google"]["zero_review_n"] = int(len(g_zero_review_ids))

# rating distribution buckets (of the 104 hotels' mean rating, mirroring old "rating band of hotels")
bins = [0, 3.0, 3.5, 4.0, 4.5, 5.01]
labels = ["< 3.0", "3.0-3.5", "3.5-4.0", "4.0-4.5", "4.5-5.0"]
g_rating["band"] = pd.cut(g_rating["mean_rating"], bins=bins, labels=labels, right=False)
out["google"]["hotel_rating_band_dist"] = g_rating["band"].value_counts().reindex(labels).fillna(0).astype(int).to_dict()

# ---------------- TRIP.COM ----------------
t_clean = pd.read_csv(HR / "data/processed/tripcom_reviews_clean.csv", low_memory=False)
t_prof = pd.read_csv(HR / "reports/tripcom_hotel_profile_summary.csv")
t_trav_rating = pd.read_csv(HR / "reports/tripcom_traveler_type_rating.csv")
t_trav_cov = pd.read_csv(HR / "reports/tripcom_traveler_type_coverage.csv")
t_room_cov = pd.read_csv(HR / "reports/tripcom_room_type_coverage.csv")
t_loc_cov = pd.read_csv(HR / "reports/tripcom_reviewer_location_coverage.csv")
t_amenity = pd.read_csv(HR / "reports/tripcom_amenity_frequency.csv")
policies = pd.read_csv(HR / "data/processed/hotel_policies_features.csv")
policy_detail = pd.read_csv(HR / "reports/tripcom_policy_detail.csv")

def status_dist(col):
    vc = policy_detail[col].value_counts(dropna=True).to_dict()
    stated_n = int(sum(vc.values()))
    return {"stated_n": stated_n, "total_n": int(len(policy_detail)), "counts": {k: int(v) for k, v in vc.items()}}

out["trip"] = {
    "clean_reviews": int(len(t_clean)),
    "hotels": int(t_clean["hotel_id"].nunique()),
    "mean_rating_5scale": round(float(t_clean["rating_5_scale"].dropna().astype(float).mean()), 2) if "rating_5_scale" in t_clean.columns else None,
    "traveler_type_coverage_pct": t_trav_cov.to_dict("records"),
    "traveler_type_rating": t_trav_rating.to_dict("records"),
    "room_type_coverage": t_room_cov.to_dict("records"),
    "reviewer_location_coverage": t_loc_cov.to_dict("records"),
    "top_amenities": t_amenity.sort_values(t_amenity.columns[-1], ascending=False).head(10).to_dict("records") if len(t_amenity.columns) else None,
    "policy_hotels": int(len(policies)),
    "pet_policy_dist": status_dist("pet_status"),
    "crib_policy_dist": status_dist("crib_status"),
    "breakfast_policy_dist": status_dist("breakfast_status"),
}
top_t = t_prof.sort_values("review_n", ascending=False).head(10) if "review_n" in t_prof.columns else t_prof.head(10)
out["trip"]["top_reviewed"] = top_t.to_dict("records")
out["trip"]["profile_columns"] = list(t_prof.columns)
out["trip"]["amenity_columns"] = list(t_amenity.columns)
out["trip"]["traveler_rating_columns"] = list(t_trav_rating.columns)

# ---------------- HOTEL 360 ----------------
h360 = pd.read_csv(HR / "data/processed/hotel_360_intelligence.csv")
out["hotel360"] = {
    "rows": int(len(h360)),
    "confidence_dist": h360["customer_voice_support"].value_counts().to_dict(),
    "columns": list(h360.columns),
}

# ---------------- CROSS PLATFORM (Google x Trip) ----------------
cp_cov = pd.read_csv(HR / "reports/cross_platform_hotel_coverage.csv")
cp_rating = pd.read_csv(HR / "reports/cross_platform_rating_comparison.csv")
out["cross_gt"] = {
    "coverage_columns": list(cp_cov.columns),
    "coverage": cp_cov.to_dict("records") if len(cp_cov) < 20 else None,
    "rating_columns": list(cp_rating.columns),
    "agreement_dist": cp_rating["agreement"].value_counts().to_dict() if "agreement" in cp_rating.columns else None,
    "common_hotels": int(cp_rating["hotel_id"].nunique()) if "hotel_id" in cp_rating.columns else None,
    "supported_n": int((cp_rating["comparison_support"] == "SUPPORTED_COMPARISON").sum()) if "comparison_support" in cp_rating.columns else None,
}

with open(r"C:\Users\bilin\AppData\Local\Temp\claude\c--Users-bilin-OneDrive-Masa-st--bodrum\0149f59c-87c6-4ebf-9dd6-9849209eba4b\scratchpad\site_data_hotelreviews.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)

print("DONE hotel-reviews part")
print(json.dumps({k: (v if not isinstance(v, (list, dict)) else "...") for k, v in out.items()}, ensure_ascii=False, indent=2))
