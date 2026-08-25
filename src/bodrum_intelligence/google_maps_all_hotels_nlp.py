"""All-hotels EDA and aspect features using the existing canonical taxonomy read-only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bodrum_intelligence.google_maps_nlp import ASPECT_KEYWORDS, CANONICAL_ASPECTS, detect_aspects, normalize_for_nlp
from bodrum_intelligence.google_maps_all_hotels_cleaning import EMAIL_RE, PHONE_RE


def _rate(frame: pd.DataFrame, aspect: str) -> float:
    if frame.empty:
        return np.nan
    return round(float(frame[f"aspect_{aspect}"].mean() * 100), 2)


def _top_rates(rates: dict[str, float], n: int = 3, positive: bool = True) -> list[str]:
    valid = [(aspect, value) for aspect, value in rates.items() if pd.notna(value)]
    valid.sort(key=lambda item: (item[1], item[0]), reverse=positive)
    return [aspect for aspect, value in valid[:n] if value > 0]


def _masked_text(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    return PHONE_RE.sub("[PHONE]", EMAIL_RE.sub("[EMAIL]", text))


def build_review_nlp(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    text = clean[clean["review_text_clean"].fillna("").str.strip().ne("") & ~clean["is_rating_only"].fillna(False)].copy()
    text["review_text_normalized"] = text["review_text_clean"].map(normalize_for_nlp)
    detected = text["review_text_normalized"].map(detect_aspects)
    text["aspects"] = detected.map(lambda item: item[0])
    text["matched_keywords"] = detected.map(lambda item: item[1])
    for aspect in CANONICAL_ASPECTS:
        text[f"aspect_{aspect}"] = text["aspects"].map(lambda values, a=aspect: a in values)
    text["aspect_count"] = text["aspects"].str.len()
    long_rows: list[dict[str, Any]] = []
    for row in text.itertuples(index=False):
        matched = row.matched_keywords
        for aspect in row.aspects:
            long_rows.append({
                "review_id": row.review_id, "hotel_id": row.hotel_id, "hotel_name": row.hotel_name,
                "area": row.area, "review_rating": row.review_rating, "rating_group": row.rating_group,
                "aspect": aspect, "matched_keywords": " | ".join(matched.get(aspect, [])),
                "review_date": row.review_date,
            })
    long = pd.DataFrame(long_rows, columns=[
        "review_id", "hotel_id", "hotel_name", "area", "review_rating", "rating_group",
        "aspect", "matched_keywords", "review_date",
    ])
    keep = [
        "review_id", "hotel_id", "hotel_name", "area", "review_rating", "rating_group",
        "review_date", "review_text_clean", "review_word_count", "potential_pii_flag", "aspect_count",
    ] + [f"aspect_{aspect}" for aspect in CANONICAL_ASPECTS]
    return text.reindex(columns=keep), long


def build_hotel_features(targets: pd.DataFrame, review_nlp: pd.DataFrame, clean: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for hotel in targets.itertuples(index=False):
        sample = review_nlp[review_nlp["hotel_id"].astype(str) == str(hotel.hotel_id)]
        all_sample = sample if clean is None else clean[clean["hotel_id"].astype(str) == str(hotel.hotel_id)]
        low, mixed, high = (sample[sample["rating_group"] == label] for label in ("LOW", "MIXED", "HIGH"))
        general_ready, positive_ready, negative_ready = len(sample) >= 15, len(high) >= 10, len(low) >= 10
        full_ready = positive_ready and negative_ready
        if full_ready:
            reliability = "HIGH"
        elif general_ready or positive_ready or negative_ready:
            reliability = "MEDIUM"
        elif len(sample):
            reliability = "LOW"
        else:
            reliability = "NO_DATA"
        row: dict[str, Any] = {
            "hotel_id": hotel.hotel_id, "hotel_name": hotel.hotel_name, "area": hotel.area,
            "google_review_count_master": hotel.google_review_count,
            "review_sample_available": bool(len(all_sample)), "sample_review_n": len(all_sample),
            "sample_text_review_n": len(sample), "low_n": len(low), "mixed_n": len(mixed), "high_n": len(high),
            "general_nlp_ready": general_ready, "positive_driver_ready": positive_ready,
            "negative_driver_ready": negative_ready, "full_driver_ready": full_ready,
            "sample_reliability": reliability,
        }
        overall_rates, low_rates, high_rates, gaps = {}, {}, {}, {}
        for aspect in CANONICAL_ASPECTS:
            overall_rates[aspect] = _rate(sample, aspect)
            low_rates[aspect] = _rate(low, aspect)
            high_rates[aspect] = _rate(high, aspect)
            gaps[aspect] = round(high_rates[aspect] - low_rates[aspect], 2) if full_ready else np.nan
            row[f"overall_aspect_{aspect}_rate_pct"] = overall_rates[aspect]
            row[f"low_aspect_{aspect}_rate_pct"] = low_rates[aspect]
            row[f"high_aspect_{aspect}_rate_pct"] = high_rates[aspect]
            row[f"driver_gap_{aspect}_pp"] = gaps[aspect]
        row["top_overall_aspects"] = " | ".join(_top_rates(overall_rates)) if len(sample) else ""
        row["top_positive_drivers"] = " | ".join(_top_rates(gaps, 3, True)) if full_ready else "INSUFFICIENT_SAMPLE"
        row["top_negative_drivers"] = " | ".join(_top_rates(gaps, 3, False)) if full_ready else "INSUFFICIENT_SAMPLE"
        row["data_availability_reason"] = "AVAILABLE" if len(sample) else "RATING_ONLY_NO_TEXT" if len(all_sample) else "DATA_NOT_AVAILABLE"
        rows.append(row)
    return pd.DataFrame(rows)


def build_area_features(targets: pd.DataFrame, review_nlp: pd.DataFrame, hotel_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for area, hotels in targets.groupby("area", dropna=False):
        sample = review_nlp[review_nlp["area"] == area]
        hotel = hotel_features[hotel_features["area"] == area]
        row = {
            "area": area, "project_hotel_count": len(hotels),
            "hotels_with_text_reviews": int(hotel["review_sample_available"].sum()),
            "hotels_general_nlp_ready": int(hotel["general_nlp_ready"].sum()),
            "hotels_full_driver_ready": int(hotel["full_driver_ready"].sum()),
            "text_review_n": len(sample), "low_n": int((sample["rating_group"] == "LOW").sum()),
            "mixed_n": int((sample["rating_group"] == "MIXED").sum()), "high_n": int((sample["rating_group"] == "HIGH").sum()),
        }
        for aspect in CANONICAL_ASPECTS:
            row[f"aspect_{aspect}_rate_pct"] = _rate(sample, aspect)
        rows.append(row)
    return pd.DataFrame(rows)


def build_profiles(hotel_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in hotel_features.itertuples(index=False):
        overall = str(row.top_overall_aspects or "").split(" | ") if row.review_sample_available else []
        positive = str(row.top_positive_drivers or "").split(" | ") if row.full_driver_ready else []
        negative = str(row.top_negative_drivers or "").split(" | ") if row.full_driver_ready else []
        note = (
            "Bu otel için Google review metin örneklemi mevcut değil."
            if not row.review_sample_available else
            "Düşük puanlı yorum sayısı karşılaştırma için yetersiz."
            if not row.negative_driver_ready else
            "Sınırlı, herkese açık yorum örneklemi; olasılıklı temsiliyet iddiası taşımaz."
        )
        rows.append({
            "hotel_id": row.hotel_id, "hotel_name": row.hotel_name, "area": row.area,
            "sample_review_n": row.sample_review_n, "sample_text_review_n": row.sample_text_review_n,
            "low_n": row.low_n, "high_n": row.high_n, "sample_reliability": row.sample_reliability,
            **{f"top_overall_aspect_{i}": overall[i - 1] if len(overall) >= i and overall[i - 1] else "" for i in range(1, 4)},
            **{f"top_positive_driver_{i}": positive[i - 1] if len(positive) >= i and positive[i - 1] != "INSUFFICIENT_SAMPLE" else "" for i in range(1, 3)},
            **{f"top_negative_driver_{i}": negative[i - 1] if len(negative) >= i and negative[i - 1] != "INSUFFICIENT_SAMPLE" else "" for i in range(1, 3)},
            "profile_note": note,
        })
    return pd.DataFrame(rows)


def build_eda_reports(targets: pd.DataFrame, clean: pd.DataFrame, hotel_features: pd.DataFrame, reports_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_counts = clean.groupby("hotel_id").size().rename("sample_review_n") if not clean.empty else pd.Series(dtype=int, name="sample_review_n")
    hotel = hotel_features[[
        "hotel_id", "hotel_name", "area", "google_review_count_master", "sample_text_review_n",
        "low_n", "mixed_n", "high_n", "general_nlp_ready", "positive_driver_ready",
        "negative_driver_ready", "full_driver_ready",
    ]].copy().rename(columns={"sample_text_review_n": "sample_text_review_n"})
    hotel = hotel.merge(raw_counts, left_on="hotel_id", right_index=True, how="left")
    hotel["sample_review_n"] = hotel["sample_review_n"].fillna(0).astype(int)
    hotel["sample_fraction_of_master_reviews"] = np.where(
        pd.to_numeric(hotel["google_review_count_master"], errors="coerce") > 0,
        hotel["sample_review_n"] / pd.to_numeric(hotel["google_review_count_master"], errors="coerce"), np.nan,
    )
    hotel = hotel.rename(columns={"google_review_count_master": "master_google_review_count"})
    area_rows = []
    for area, group in hotel.groupby("area", dropna=False):
        area_rows.append({
            "area": area, "project_hotel_count": len(group),
            "hotels_with_google_review_sample": int((group["sample_review_n"] > 0).sum()),
            "hotels_general_nlp_ready": int(group["general_nlp_ready"].sum()),
            "hotels_full_driver_ready": int(group["full_driver_ready"].sum()),
            "review_n": int(group["sample_review_n"].sum()), "text_review_n": int(group["sample_text_review_n"].sum()),
            "low_n": int(group["low_n"].sum()), "high_n": int(group["high_n"].sum()),
        })
    area = pd.DataFrame(area_rows)
    hotel.to_csv(reports_dir / "google_maps_all_hotels_hotel_eda_summary.csv", index=False, encoding="utf-8-sig")
    area.to_csv(reports_dir / "google_maps_all_hotels_area_eda_summary.csv", index=False, encoding="utf-8-sig")
    return hotel, area


def _write_figures(review_nlp: pd.DataFrame, hotel_features: pd.DataFrame, area_features: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    def save(name: str, title: str, plotter) -> None:
        fig, ax = plt.subplots(figsize=(10, 6)); plotter(ax); ax.set_title(title); fig.tight_layout(); fig.savefig(figure_dir / name, dpi=140); plt.close(fig)
    save("01_collection_coverage.png", "Hotel sample coverage", lambda ax: ax.bar(["Available", "No data"], [int(hotel_features.review_sample_available.sum()), int((~hotel_features.review_sample_available).sum())]))
    save("02_reviews_per_hotel_distribution.png", "Text reviews per hotel", lambda ax: ax.hist(hotel_features.sample_text_review_n, bins=15))
    save("03_rating_distribution.png", "Review rating distribution", lambda ax: review_nlp.review_rating.value_counts().sort_index().plot.bar(ax=ax) if len(review_nlp) else ax.text(.5,.5,"No reviews",ha="center"))
    save("04_rating_group_distribution.png", "Rating groups", lambda ax: review_nlp.rating_group.value_counts().reindex(["LOW","MIXED","HIGH"]).fillna(0).plot.bar(ax=ax) if len(review_nlp) else ax.text(.5,.5,"No reviews",ha="center"))
    readiness_counts = pd.Series({"General":hotel_features.general_nlp_ready.sum(),"Positive":hotel_features.positive_driver_ready.sum(),"Negative":hotel_features.negative_driver_ready.sum(),"Full":hotel_features.full_driver_ready.sum()})
    def plot_readiness(ax):
        readiness_counts.plot.bar(ax=ax)
        ax.set_ylim(0, max(1, float(readiness_counts.max()) * 1.15))
        for index, value in enumerate(readiness_counts):
            ax.text(index, value + 0.03, str(int(value)), ha="center")
    save("05_sample_readiness.png", "Hotel sample readiness", plot_readiness)
    aspect_rates = pd.Series({a:_rate(review_nlp,a) for a in CANONICAL_ASPECTS}).dropna().sort_values()
    save("06_overall_aspect_frequency.png", "Overall aspect mention rate (%)", lambda ax: aspect_rates.plot.barh(ax=ax) if len(aspect_rates) else ax.text(.5,.5,"No reviews",ha="center"))
    rating_matrix = pd.DataFrame({g:{a:_rate(review_nlp[review_nlp.rating_group==g],a) for a in CANONICAL_ASPECTS} for g in ["LOW","MIXED","HIGH"]})
    save("07_rating_aspect_heatmap.png", "Rating-group aspect rates", lambda ax: (ax.imshow(rating_matrix.fillna(0).values, aspect="auto"), ax.set_yticks(range(len(CANONICAL_ASPECTS)), CANONICAL_ASPECTS), ax.set_xticks(range(3), rating_matrix.columns)))
    gaps = hotel_features[[f"driver_gap_{a}_pp" for a in CANONICAL_ASPECTS]].mean().dropna().sort_values()
    save("08_positive_negative_driver_gap.png", "Mean driver gap (pp), ready hotels", lambda ax: gaps.plot.barh(ax=ax) if len(gaps) else ax.text(.5,.5,"Insufficient sample",ha="center"))
    high_sample = hotel_features[hotel_features.general_nlp_ready].sort_values("sample_text_review_n", ascending=False).head(20)
    matrix = high_sample[[f"overall_aspect_{a}_rate_pct" for a in CANONICAL_ASPECTS]].fillna(0)
    save("09_hotel_aspect_heatmap_high_sample.png", "Top-sample hotel aspect rates", lambda ax: (ax.imshow(matrix.values, aspect="auto"), ax.set_yticks(range(len(high_sample)), high_sample.hotel_name.str.slice(0,30)), ax.set_xticks(range(len(CANONICAL_ASPECTS)), CANONICAL_ASPECTS, rotation=90)) if len(high_sample) else ax.text(.5,.5,"No GENERAL_NLP_READY hotels",ha="center"))
    covered_areas = area_features[area_features.text_review_n > 0]
    am = covered_areas[[f"aspect_{a}_rate_pct" for a in CANONICAL_ASPECTS]]
    save("10_area_aspect_heatmap.png", "Area aspect rates", lambda ax: (ax.imshow(am.values, aspect="auto"), ax.set_yticks(range(len(covered_areas)), covered_areas.area), ax.set_xticks(range(len(CANONICAL_ASPECTS)), CANONICAL_ASPECTS, rotation=90)) if len(covered_areas) else ax.text(.5,.5,"No covered areas",ha="center"))


def run_nlp(clean_path: Path, targets_path: Path, processed_dir: Path, reports_dir: Path, figure_dir: Path) -> dict[str, Any]:
    clean = pd.read_csv(clean_path) if clean_path.exists() and clean_path.stat().st_size else pd.DataFrame()
    targets = pd.read_csv(targets_path)
    if clean.empty:
        clean = pd.DataFrame(columns=["review_id","hotel_id","hotel_name","area","review_rating","rating_group","review_date","review_text_clean","review_word_count","potential_pii_flag","is_rating_only"])
    review_nlp, long = build_review_nlp(clean)
    hotel = build_hotel_features(targets, review_nlp, clean)
    area = build_area_features(targets, review_nlp, hotel)
    profiles = build_profiles(hotel)
    processed_dir.mkdir(parents=True, exist_ok=True); reports_dir.mkdir(parents=True, exist_ok=True)
    review_nlp.to_csv(processed_dir / "google_maps_all_hotels_reviews_nlp.csv", index=False, encoding="utf-8-sig")
    long.to_csv(processed_dir / "google_maps_all_hotels_review_aspects_long.csv", index=False, encoding="utf-8-sig")
    hotel.to_csv(processed_dir / "google_maps_all_hotels_hotel_nlp_features.csv", index=False, encoding="utf-8-sig")
    area.to_csv(processed_dir / "google_maps_all_hotels_area_nlp_features.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(reports_dir / "google_maps_all_hotels_customer_voice_profiles.csv", index=False, encoding="utf-8-sig")
    build_eda_reports(targets, clean, hotel, reports_dir)
    aspect_frequency = pd.DataFrame([{"aspect":a,"review_n":len(review_nlp),"mention_n":int(review_nlp.get(f"aspect_{a}",pd.Series(dtype=bool)).sum()),"mention_rate_pct":_rate(review_nlp,a)} for a in CANONICAL_ASPECTS])
    aspect_frequency.to_csv(reports_dir / "google_maps_all_hotels_aspect_frequency.csv", index=False, encoding="utf-8-sig")
    hotel_matrix = hotel[["hotel_id","hotel_name","area"]+[f"overall_aspect_{a}_rate_pct" for a in CANONICAL_ASPECTS]]
    hotel_matrix.to_csv(reports_dir / "google_maps_all_hotels_hotel_aspect_matrix.csv", index=False, encoding="utf-8-sig")
    area.to_csv(reports_dir / "google_maps_all_hotels_area_aspect_matrix.csv", index=False, encoding="utf-8-sig")
    rating_matrix = pd.DataFrame([{"rating_group":g,"review_n":int((review_nlp.rating_group==g).sum()),**{f"aspect_{a}_rate_pct":_rate(review_nlp[review_nlp.rating_group==g],a) for a in CANONICAL_ASPECTS}} for g in ["LOW","MIXED","HIGH"]])
    rating_matrix.to_csv(reports_dir / "google_maps_all_hotels_rating_aspect_matrix.csv", index=False, encoding="utf-8-sig")
    driver_rows=[]
    for row in hotel.itertuples(index=False):
        for aspect in CANONICAL_ASPECTS:
            gap=getattr(row,f"driver_gap_{aspect}_pp")
            driver_rows.append({"hotel_id":row.hotel_id,"hotel_name":row.hotel_name,"area":row.area,"aspect":aspect,"low_n":row.low_n,"high_n":row.high_n,"full_driver_ready":row.full_driver_ready,"driver_gap_pp":gap,"driver_class":"INSUFFICIENT_SAMPLE" if not row.full_driver_ready else "POSITIVE" if gap>=10 else "NEGATIVE" if gap<=-10 else "NO_STRONG_SIGNAL"})
    drivers=pd.DataFrame(driver_rows)
    drivers[drivers.driver_class=="POSITIVE"].to_csv(reports_dir / "google_maps_all_hotels_positive_drivers.csv",index=False,encoding="utf-8-sig")
    drivers[drivers.driver_class=="NEGATIVE"].to_csv(reports_dir / "google_maps_all_hotels_negative_drivers.csv",index=False,encoding="utf-8-sig")
    drivers.to_csv(reports_dir / "google_maps_all_hotels_hotel_driver_profiles.csv",index=False,encoding="utf-8-sig")
    hotel[["hotel_id","hotel_name","area","sample_text_review_n","low_n","mixed_n","high_n","sample_reliability","general_nlp_ready","positive_driver_ready","negative_driver_ready","full_driver_ready"]].to_csv(reports_dir / "google_maps_all_hotels_sample_reliability.csv",index=False,encoding="utf-8-sig")
    if len(review_nlp):
        pieces = [group.sample(min(len(group), 7), random_state=42) for _, group in review_nlp.groupby(["rating_group", "hotel_id"], dropna=False)]
        validation = pd.concat(pieces, ignore_index=True)
        desired = min(100, len(review_nlp))
        if len(validation) < desired:
            remainder = review_nlp[~review_nlp.review_id.isin(validation.review_id)].sample(desired - len(validation), random_state=42)
            validation = pd.concat([validation, remainder], ignore_index=True)
        validation = validation.head(100)
        validation["review_text_clean"] = validation["review_text_clean"].map(_masked_text)
    else:
        validation = review_nlp
    validation.reindex(columns=["review_id","hotel_id","hotel_name","area","review_rating","rating_group","review_text_clean","potential_pii_flag"]+[f"aspect_{a}" for a in CANONICAL_ASPECTS]).to_csv(reports_dir / "google_maps_all_hotels_aspect_manual_validation_sample.csv",index=False,encoding="utf-8-sig")
    findings = {"clean_review_count":len(clean),"text_review_count":len(review_nlp),"hotel_coverage":int(hotel.review_sample_available.sum()),"area_coverage":int(area[area.text_review_n>0].area.nunique()),"positive_driver_ready_hotels":int(hotel.positive_driver_ready.sum()),"negative_driver_ready_hotels":int(hotel.negative_driver_ready.sum()),"full_driver_ready_hotels":int(hotel.full_driver_ready.sum()),"low_sample_hotels":int((hotel.sample_reliability=="LOW").sum())}
    top_aspects=aspect_frequency.sort_values("mention_rate_pct",ascending=False).head(5).aspect.tolist()
    (reports_dir / "google_maps_all_hotels_nlp_key_findings.txt").write_text(json.dumps({**findings,"top_overall_aspects":top_aspects},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (reports_dir / "google_maps_all_hotels_nlp_limitations.txt").write_text("Keyword aspect detection is descriptive and requires manual validation. Public capped samples are not probability samples. Missing or insufficient-sample values remain blank/NaN, never synthetic zero. Hotel Explorer integration is intentionally outside this task.\n",encoding="utf-8")
    _write_figures(review_nlp, hotel, area, figure_dir)
    return {**findings,"hotel_rows":len(hotel),"canonical_aspects":len(CANONICAL_ASPECTS)}
