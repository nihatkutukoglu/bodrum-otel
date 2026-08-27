import json
import pandas as pd
from pathlib import Path

BO = Path(r"C:\Users\bilin\OneDrive\Masaüstü\bodrum\bodrum otel\bodrum-otel")

out = {}

clean_v3 = pd.read_csv(BO / "data/processed/sikayetvar_complaints_clean_v3.csv", low_memory=False)
aspect_v3 = pd.read_csv(BO / "reports/sikayetvar_aspect_summary_v3.csv")
master_v3 = pd.read_csv(BO / "reports/sikayetvar_final_customer_voice_master_v3.csv")
reply_v3 = pd.read_csv(BO / "reports/sikayetvar_company_reply_visibility_by_hotel_v3.csv")

out["sikayetvar"] = {
    "clean_rows": int(len(clean_v3)),
    "complaint_hotels": int((master_v3["complaint_n"] > 0).sum()),
    "company_reply_visible": int(clean_v3["has_company_reply"].fillna(False).astype(bool).sum()),
    "reply_visibility_pct": round(100 * int(clean_v3["has_company_reply"].fillna(False).astype(bool).sum()) / len(clean_v3), 1),
    "verified_pages": int(master_v3["mapping_status"].isin(["MATCHED_EXACT", "MATCHED_HIGH_CONFIDENCE", "PAGE_FOUND_NO_COMPLAINT"]).sum()),
    "ambiguous_n": int((master_v3["mapping_status"] == "AMBIGUOUS_REMAINS").sum()),
    "not_found_n": int((master_v3["mapping_status"] == "NOT_FOUND").sum()),
    "top_aspects": aspect_v3.sort_values("mention_rate_pct", ascending=False).to_dict("records"),
    "reply_top": reply_v3.sort_values("complaint_n", ascending=False).head(10).to_dict("records"),
}

# top complaint-bearing hotels
top_hotels = master_v3.loc[master_v3["complaint_n"] > 0].sort_values("complaint_n", ascending=False).head(10)
out["sikayetvar"]["top_hotels"] = top_hotels[["hotel_id", "hotel_name", "area", "complaint_n", "top_aspects", "company_reply_visibility_pct"]].to_dict("records")

# ---------------- CROSS SOURCE (Google x Sikayetvar) ----------------
coverage = pd.read_csv(BO / "reports/google_sikayetvar_cross_source_coverage.csv")
aspect_align = pd.read_csv(BO / "reports/google_sikayetvar_aspect_alignment.csv")
hotel_align = pd.read_csv(BO / "reports/google_sikayetvar_hotel_alignment.csv")
divergence = pd.read_csv(BO / "reports/google_sikayetvar_source_divergence.csv")
key_findings = pd.read_csv(BO / "reports/google_sikayetvar_key_findings.csv")
crosswalk = pd.read_csv(BO / "config/google_sikayetvar_aspect_crosswalk.csv")

out["cross_gs"] = {
    "coverage": coverage.to_dict("records"),
    "label_dist": aspect_align["alignment_label"].value_counts().to_dict(),
    "canonical_aspect_n": int(len(crosswalk)),
    "both_concern_top": aspect_align.loc[aspect_align["alignment_label"] == "BOTH_SOURCE_CONCERN"].merge(
        hotel_align[["hotel_id", "hotel_name"]], on="hotel_id", how="left"
    ).sort_values("sikayetvar_mention_rate_pct", ascending=False).head(8).to_dict("records"),
    "strength_vs_complaint_top": aspect_align.loc[aspect_align["alignment_label"] == "GOOGLE_STRENGTH_VS_COMPLAINT"].merge(
        hotel_align[["hotel_id", "hotel_name"]], on="hotel_id", how="left"
    ).sort_values("sikayetvar_mention_rate_pct", ascending=False).head(8).to_dict("records"),
    "key_findings": key_findings.to_dict("records"),
}

with open(r"C:\Users\bilin\AppData\Local\Temp\claude\c--Users-bilin-OneDrive-Masa-st--bodrum\0149f59c-87c6-4ebf-9dd6-9849209eba4b\scratchpad\site_data_bodrumotel.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)

print("DONE bodrum-otel part")
print("sikayetvar clean_rows", out["sikayetvar"]["clean_rows"], "complaint_hotels", out["sikayetvar"]["complaint_hotels"])
print("reply_visible", out["sikayetvar"]["company_reply_visible"], out["sikayetvar"]["reply_visibility_pct"])
print("cross coverage", out["cross_gs"]["coverage"])
print("cross label dist", out["cross_gs"]["label_dist"])
