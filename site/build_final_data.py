import json
import math
import re
import pandas as pd
import numpy as np

HR = r"C:\Users\bilin\OneDrive\Masaüstü\bodrum\hotelrewiews\hotel-reviews"
BO = r"C:\Users\bilin\OneDrive\Masaüstü\bodrum\bodrum otel\bodrum-otel"
SCRATCH = r"C:\Users\bilin\AppData\Local\Temp\claude\c--Users-bilin-OneDrive-Masa-st--bodrum\0149f59c-87c6-4ebf-9dd6-9849209eba4b\scratchpad"

# ---- old DATA blob (destinations / tourism_annual / airport_tourism_monthly / hotels base) ----
with open(SCRATCH + r"\site_data_line733.txt", encoding="utf-8") as f:
    text = f.read()
m = re.match(r"^\s*const DATA\s*=\s*", text)
old_data = json.loads(text[m.end():].rstrip().rstrip(";"))

def clean_nan(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: clean_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean_nan(v) for v in o]
    return o

# ---- hotel_360 (already joins master + google + trip + policy) ----
h360 = pd.read_csv(HR + r"\data\processed\hotel_360_intelligence.csv")
h360 = h360.replace({np.nan: None})

# ---- sikayetvar per-hotel (v3) ----
sk_master = pd.read_csv(BO + r"\reports\sikayetvar_final_customer_voice_master_v3.csv")
sk_master = sk_master.replace({np.nan: None})
sk_by_id = {row["hotel_id"]: row for _, row in sk_master.iterrows()}

# ---- per-hotel Trip.com reviewer-country breakdown (guest origin) ----
# a handful of reviewer_country values are scraping artifacts, not real countries - excluded
JUNK_COUNTRIES = {"aLper B", "Solo traveller", "Couple"}
t_reviews = pd.read_csv(HR + r"\data\processed\tripcom_reviews_clean.csv", low_memory=False)
t_reviews = t_reviews.dropna(subset=["reviewer_country"])
t_reviews = t_reviews[~t_reviews["reviewer_country"].isin(JUNK_COUNTRIES)]
country_by_hotel = {}
for hid, grp in t_reviews.groupby("hotel_id"):
    counts = grp["reviewer_country"].value_counts()
    country_by_hotel[hid] = {
        "total": int(counts.sum()),
        "top": [{"country": c, "n": int(n)} for c, n in counts.head(5).items()],
    }

old_hotels_by_id = {h["id"]: h for h in old_data["hotels"]}

new_hotels = []
for _, row in h360.iterrows():
    hid = row["hotel_id"]
    old = old_hotels_by_id.get(hid, {})
    sk = sk_by_id.get(hid)
    has_sk = sk is not None
    rec = {
        "id": hid,
        "name": row["hotel_name"],
        "area": row["area"],
        "rating": old.get("rating"),
        "reviews": old.get("reviews"),
        "price": old.get("price"),
        "star": old.get("star"),
        "rooms": old.get("rooms"),
        "beds": old.get("beds"),
        "g_n": row["google_travel_review_n"],
        "g_mean": row["google_travel_mean_rating"],
        "g_low": row["google_travel_low_share"],
        "g_high": row["google_travel_high_share"],
        "g_top": row["google_top_aspects"],
        "g_strength": row["google_strength_signals"],
        "g_concern": row["google_concern_signals"],
        "t_n": row["trip_review_n"],
        "t_mean": row["trip_mean_rating_5"],
        "t_low": row["trip_low_share"],
        "t_high": row["trip_high_share"],
        "t_top_traveler": row["trip_top_traveler_type"],
        "t_family_pct": row["trip_family_share"],
        "t_couple_pct": row["trip_couple_share"],
        "t_country_n": row["trip_country_coverage"],
        "policy_status": row["policy_status"],
        "amenity_n": row["amenity_count"],
        "family_feature_n": row["family_feature_count"],
        "wellness_feature_n": row["wellness_feature_count"],
        "water_feature_n": row["water_feature_count"],
        "has_policy": bool(row["has_policy_data"]) if row["has_policy_data"] is not None else False,
        "source_n": row["source_count"],
        "confidence": row["customer_voice_support"],
        "rating_gap": row["cross_platform_rating_gap"],
        "gap_support": row["cross_platform_consistency"],
        "archetypes": row["archetypes"],
        "sk_n": sk.get("complaint_n") if has_sk else 0,
        "sk_top": sk.get("top_aspects") if has_sk else None,
        "sk_reply_pct": sk.get("company_reply_visibility_pct") if has_sk else None,
        "sk_page_status": sk.get("page_status") if has_sk else None,
        "sk_visibility_per1000": sk.get("complaint_visibility_per_1000_google_reviews") if has_sk else None,
        "t_countries": country_by_hotel.get(hid, {"total": 0, "top": []}),
    }
    new_hotels.append(clean_nan(rec))

final_data = {
    "hotels": new_hotels,
    "destinations": old_data["destinations"],
    "tourism_annual": old_data["tourism_annual"],
    "airport_tourism_monthly": old_data["airport_tourism_monthly"],
}

# ---- Google Travel aggregate ----
with open(SCRATCH + r"\site_data_hotelreviews.json", encoding="utf-8") as f:
    hr = json.load(f)
final_data["google"] = hr["google"]
final_data["trip"] = hr["trip"]
final_data["hotel360_meta"] = hr["hotel360"]
final_data["cross_gt"] = hr["cross_gt"]

with open(SCRATCH + r"\site_data_bodrumotel.json", encoding="utf-8") as f:
    bo = json.load(f)
final_data["sikayetvar"] = bo["sikayetvar"]
final_data["cross_gs"] = bo["cross_gs"]

final_data = clean_nan(final_data)

with open(SCRATCH + r"\FINAL_DATA.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False)

print("hotels:", len(final_data["hotels"]))
print("sample hotel BOD007:", json.dumps([h for h in final_data["hotels"] if h["id"]=="BOD007"][0], ensure_ascii=False, indent=2))
print("total size chars:", len(json.dumps(final_data, ensure_ascii=False)))
