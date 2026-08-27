import json
import sys
import pandas as pd

sys.path.insert(0, r"C:\Users\bilin\OneDrive\Masaüstü\bodrum\hotelrewiews\hotel-reviews\src")
from bodrum_intelligence.analysis.aspect_dictionary import detect_aspects

HR = r"C:\Users\bilin\OneDrive\Masaüstü\bodrum\hotelrewiews\hotel-reviews"

g = pd.read_csv(HR + r"\data\processed\google_travel_all_hotels_reviews_clean.csv", low_memory=False)
g = g.dropna(subset=["review_text_clean"])
g = g[g["review_text_clean"].str.len().between(40, 260)]

TARGET_ASPECTS = ["STAFF", "CLEANLINESS", "HYGIENE", "FOOD", "ROOM", "BED_COMFORT", "BEACH_SEA", "POOL",
                  "SERVICE", "PRICE_VALUE", "LOCATION", "FACILITIES", "NOISE", "FAMILY_KIDS",
                  "ANIMATION_ENTERTAINMENT", "AIR_CONDITIONING", "CHECKIN_CHECKOUT", "RESERVATION",
                  "REFUND_PAYMENT", "TRANSPORT_TRANSFER", "MANAGEMENT", "COMMUNICATION", "BAR_DRINKS", "WIFI"]

# widen the pool for rare aspects: also allow the full un-length-filtered set as a fallback
g_wide = pd.read_csv(HR + r"\data\processed\google_travel_all_hotels_reviews_clean.csv", low_memory=False)
g_wide = g_wide.dropna(subset=["review_text_clean"])
g_wide = g_wide[g_wide["review_text_clean"].str.len().between(20, 400)]

def sample_quotes(df, aspect, rating_group, n=2):
    rows = []
    seen = set()
    sub = df[df["rating_group"] == rating_group]
    for _, row in sub.sample(frac=1, random_state=42).iterrows():
        aspects = detect_aspects(row["review_text_clean"])
        if aspect in aspects and row["review_text_clean"] not in seen:
            rows.append({"hotel": row["hotel_name"], "date": str(row.get("review_date_raw", "") or ""), "text": row["review_text_clean"]})
            seen.add(row["review_text_clean"])
        if len(rows) >= n:
            break
    return rows

quotes = {"high": {}, "low": {}}
for a in TARGET_ASPECTS:
    hi = sample_quotes(g, a, "HIGH", 2)
    lo = sample_quotes(g, a, "LOW", 2)
    if len(hi) < 2:
        hi = sample_quotes(g_wide, a, "HIGH", 2)
    if len(lo) < 2:
        lo = sample_quotes(g_wide, a, "LOW", 2)
    quotes["high"][a] = hi
    quotes["low"][a] = lo

# Trip.com traveler-segment quotes
t = pd.read_csv(HR + r"\data\processed\tripcom_reviews_clean.csv", low_memory=False)
t = t.dropna(subset=["review_text_clean"]) if "review_text_clean" in t.columns else t.dropna(subset=["yorum"])
text_col = "review_text_clean" if "review_text_clean" in t.columns else "yorum"
print("trip cols", t.columns.tolist())

with open(r"C:\Users\bilin\AppData\Local\Temp\claude\c--Users-bilin-OneDrive-Masa-st--bodrum\0149f59c-87c6-4ebf-9dd6-9849209eba4b\scratchpad\site_quotes_google.json", "w", encoding="utf-8") as f:
    json.dump(quotes, f, ensure_ascii=False, indent=2)

print(json.dumps({k: {a: len(v2) for a, v2 in v.items()} for k, v in quotes.items()}, indent=2))
