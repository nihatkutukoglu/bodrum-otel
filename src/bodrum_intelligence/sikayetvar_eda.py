"""Coverage-aware descriptive helpers for the Şikayetvar EDA layer."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def safe_ratio(numerator: pd.Series, denominator: pd.Series, scale: float = 1.0) -> pd.Series:
    """Return a zero-safe ratio; non-positive/missing denominators remain missing."""

    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    result = pd.Series(np.nan, index=num.index, dtype="float64")
    valid = num.notna() & den.notna() & den.gt(0)
    result.loc[valid] = num.loc[valid] / den.loc[valid] * scale
    return result


def nlp_sample_tier(count: Any, high_threshold: int = 15, medium_threshold: int = 5) -> str:
    """Assign an explainable document-count tier for downstream NLP scoping."""

    value = int(count)
    if value >= high_threshold:
        return "HIGH_SAMPLE"
    if value >= medium_threshold:
        return "MEDIUM_SAMPLE"
    return "LOW_SAMPLE"


def build_hotel_eda_summary(
    complaints: pd.DataFrame,
    min_complaints_for_rate: int = 5,
    denominator_quantile: float = 0.25,
) -> tuple[pd.DataFrame, float]:
    """Aggregate one row per hotel without interpreting complaint volume as quality."""

    required = {"hotel_id", "hotel_name", "area", "complaint_id"}
    missing = required - set(complaints.columns)
    if missing:
        raise ValueError(f"Missing complaint columns: {sorted(missing)}")

    metadata_candidates = [
        "google_rating",
        "google_review_count",
        "search_price_usd_snapshot",
        "official_star_rating_verified",
        "official_room_count",
        "official_bed_count",
    ]
    aggregations: dict[str, tuple[str, Any]] = {
        "matched_complaint_count": ("complaint_id", "size"),
        "first_complaint_date": ("complaint_date", "min"),
        "last_complaint_date": ("complaint_date", "max"),
        "median_complaint_word_count": ("complaint_word_count", "median"),
        "median_view_count": ("view_count_numeric", "median"),
        "complaints_with_replies": ("reply_count_total_derived", lambda values: int(values.gt(0).sum())),
        "total_reply_count": ("reply_count_total_derived", "sum"),
        "median_reply_count": ("reply_count_total_derived", "median"),
        "max_reply_count": ("reply_count_total_derived", "max"),
        "company_response_count": (
            "company_response_exists_clean",
            lambda values: int(values.fillna(False).astype(bool).sum()),
        ),
        "response_date_available_n": ("response_time_days", "count"),
        "median_response_time_days": ("response_time_days", "median"),
        "median_company_response_word_count": ("company_response_word_count", "median"),
    }
    for column in metadata_candidates:
        if column in complaints.columns:
            aggregations[column] = (column, "first")

    summary = (
        complaints.groupby(["hotel_id", "hotel_name", "area"], dropna=False)
        .agg(**aggregations)
        .reset_index()
    )
    total = len(complaints)
    summary["share_of_corpus_pct"] = 100 * summary["matched_complaint_count"] / total
    summary["company_response_rate_in_corpus"] = safe_ratio(
        summary["company_response_count"], summary["matched_complaint_count"], scale=100
    )
    summary["reply_coverage_pct"] = safe_ratio(
        summary["complaints_with_replies"], summary["matched_complaint_count"], scale=100
    )
    summary["cross_platform_complaint_visibility_per_1000_google_reviews"] = safe_ratio(
        summary.get("matched_complaint_count"), summary.get("google_review_count"), scale=1000
    )
    denominator_threshold = float(
        pd.to_numeric(summary.get("google_review_count"), errors="coerce").dropna().quantile(denominator_quantile)
    )
    summary["low_google_review_denominator_flag"] = pd.to_numeric(
        summary.get("google_review_count"), errors="coerce"
    ).lt(denominator_threshold)
    summary["small_n_flag"] = summary["matched_complaint_count"].lt(min_complaints_for_rate)
    summary["nlp_sample_tier"] = summary["matched_complaint_count"].map(nlp_sample_tier)
    return summary.sort_values("matched_complaint_count", ascending=False).reset_index(drop=True), denominator_threshold


def build_area_eda_summary(
    complaints: pd.DataFrame,
    hotel_mapping: pd.DataFrame,
    hotel_master: pd.DataFrame,
) -> pd.DataFrame:
    """Build a 14-area summary that keeps mapping coverage distinct from complaint volume."""

    project = (
        hotel_master.groupby("area", dropna=False)["hotel_id"]
        .nunique()
        .rename("project_hotel_count")
        .reset_index()
    )
    trusted_statuses = {"FOUND_EXACT", "FOUND_HIGH_CONFIDENCE"}
    mapped = (
        hotel_mapping.loc[hotel_mapping["match_status"].isin(trusted_statuses)]
        .groupby("area")["hotel_id"]
        .nunique()
        .rename("mapped_hotel_count")
        .reset_index()
    )
    status_counts = (
        hotel_mapping.pivot_table(
            index="area", columns="match_status", values="hotel_id", aggfunc="nunique", fill_value=0
        )
        .add_prefix("mapping_status_")
        .reset_index()
    )
    complaint_agg = (
        complaints.groupby("area", dropna=False)
        .agg(
            hotels_with_complaints=("hotel_id", "nunique"),
            matched_complaint_count=("complaint_id", "size"),
            company_response_count=(
                "company_response_exists_clean",
                lambda values: int(values.fillna(False).astype(bool).sum()),
            ),
            median_word_count=("complaint_word_count", "median"),
            median_view_count=("view_count_numeric", "median"),
        )
        .reset_index()
    )
    result = project.merge(mapped, on="area", how="left").merge(
        complaint_agg, on="area", how="left"
    ).merge(status_counts, on="area", how="left")
    count_columns = [
        "mapped_hotel_count",
        "hotels_with_complaints",
        "matched_complaint_count",
        "company_response_count",
    ] + [column for column in result.columns if column.startswith("mapping_status_")]
    for column in count_columns:
        result[column] = result[column].fillna(0).astype(int)
    result["complaint_share_pct"] = 100 * result["matched_complaint_count"] / len(complaints)
    result["complaints_per_mapped_hotel"] = safe_ratio(
        result["matched_complaint_count"], result["mapped_hotel_count"]
    )
    result["complaints_per_hotel_with_complaints"] = safe_ratio(
        result["matched_complaint_count"], result["hotels_with_complaints"]
    )
    result["company_response_rate_in_corpus"] = safe_ratio(
        result["company_response_count"], result["matched_complaint_count"], scale=100
    )
    result["mapping_coverage_pct"] = safe_ratio(
        result["mapped_hotel_count"], result["project_hotel_count"], scale=100
    )
    result["coverage_flag"] = np.where(result["mapping_coverage_pct"].ge(25), "ADEQUATE_EXPLORATORY", "LOW")
    return result.sort_values("matched_complaint_count", ascending=False).reset_index(drop=True)


def concentration_metrics(hotel_summary: pd.DataFrame) -> dict[str, float]:
    """Return transparent corpus-concentration metrics."""

    counts = hotel_summary["matched_complaint_count"].sort_values(ascending=False)
    total = counts.sum()
    cumulative = counts.cumsum() / total
    return {
        "top3_hotel_complaint_share_pct": 100 * counts.head(3).sum() / total,
        "top5_hotel_complaint_share_pct": 100 * counts.head(5).sum() / total,
        "top10_hotel_complaint_share_pct": 100 * counts.head(10).sum() / total,
        "hotels_to_reach_80pct": int((cumulative.lt(0.8)).sum() + 1),
        "hhi_hotel_complaint_concentration": float(((counts / total) ** 2).sum()),
    }


def response_time_summary(complaints: pd.DataFrame) -> pd.DataFrame:
    """Summarize non-negative calculable response lags by hotel."""

    work = complaints.copy()
    work["response_time_days"] = pd.to_numeric(work["response_time_days"], errors="coerce")
    valid = work.loc[work["response_time_days"].ge(0)].copy()
    base = (
        work.groupby(["hotel_id", "hotel_name", "area"], dropna=False)
        .agg(complaint_n=("complaint_id", "size"), response_n=("company_response_exists_clean", lambda x: int(x.fillna(False).sum())))
        .reset_index()
    )
    lag = (
        valid.groupby(["hotel_id", "hotel_name", "area"], dropna=False)
        .agg(
            response_date_available_n=("response_time_days", "count"),
            median_response_time_days=("response_time_days", "median"),
            q25_response_time_days=("response_time_days", lambda x: x.quantile(0.25)),
            q75_response_time_days=("response_time_days", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )
    return base.merge(lag, on=["hotel_id", "hotel_name", "area"], how="left")


def spearman_table(
    frame: pd.DataFrame,
    metric_pairs: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    """Compute pairwise exploratory Spearman correlations with explicit sample sizes."""

    records: list[dict[str, Any]] = []
    for metric_x, metric_y in metric_pairs:
        if metric_x not in frame or metric_y not in frame:
            continue
        paired = frame[[metric_x, metric_y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(paired) < 3 or paired[metric_x].nunique() < 2 or paired[metric_y].nunique() < 2:
            rho = np.nan
            p_value = np.nan
        else:
            rho, p_value = spearmanr(paired[metric_x], paired[metric_y])
        if pd.isna(rho):
            interpretation = "Yeterli değişkenlik/gözlem yok."
        else:
            magnitude = abs(float(rho))
            strength = "çok güçlü" if magnitude >= 0.8 else "güçlü" if magnitude >= 0.6 else "orta" if magnitude >= 0.4 else "zayıf" if magnitude >= 0.2 else "çok zayıf"
            direction = "aynı yönlü" if rho >= 0 else "ters yönlü"
            interpretation = f"{strength.capitalize()}, {direction} sıralı birliktelik."
        records.append(
            {
                "metric_x": metric_x,
                "metric_y": metric_y,
                "n": len(paired),
                "spearman_rho": rho,
                "p_value": p_value,
                "interpretation": interpretation,
                "main_caution": "Exploratory; selected mapped-hotel sample, platform bias, no causality.",
            }
        )
    return pd.DataFrame(records)


def numeric_correlation_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return the Spearman coefficient matrix for available numeric columns."""

    available = [column for column in columns if column in frame]
    return frame[available].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
