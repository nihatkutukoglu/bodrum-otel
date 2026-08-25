"""Coverage-aware destinasyon zekâsı için tekrar kullanılabilir analiz fonksiyonları."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


EXPECTED_AREAS = [
    "Akyarlar",
    "Bitez",
    "Bodrum Merkez",
    "Göltürkbükü",
    "Gümbet",
    "Gümüşlük",
    "Gündoğan",
    "Güvercinlik",
    "Kadıkalesi",
    "Ortakent-Yahşi",
    "Torba",
    "Turgutreis",
    "Türkbükü",
    "Yalıkavak",
]

HIGH_CONFIDENCE_STATUS = "MATCHED_HIGH_CONFIDENCE"


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Eksik zorunlu kolonlar: {missing}")


def validate_area_set(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Beklenen 14 alan ile gerçek alan kümesini karşılaştırır."""

    require_columns(frame, ["area"])
    actual = set(frame["area"].dropna().astype(str))
    expected = set(EXPECTED_AREAS)
    return {
        "missing": sorted(expected - actual),
        "extra": sorted(actual - expected),
        "duplicates": sorted(
            frame.loc[frame["area"].duplicated(keep=False), "area"].astype(str).unique()
        ),
    }


def aggregate_hotel_metrics(hotels: pd.DataFrame) -> pd.DataFrame:
    """192 otellik hotel-level tablodan 14 destinasyon metriği üretir."""

    require_columns(
        hotels,
        [
            "hotel_id",
            "area",
            "google_rating",
            "weighted_google_rating",
            "google_review_count",
            "search_price_usd_snapshot",
            "rating_gap_from_area_median",
        ],
    )
    result = (
        hotels.groupby("area", observed=True)
        .agg(
            sample_hotel_count=("hotel_id", "size"),
            avg_google_rating=("google_rating", "mean"),
            median_google_rating=("google_rating", "median"),
            avg_weighted_google_rating=("weighted_google_rating", "mean"),
            avg_rating_gap_from_area_median=("rating_gap_from_area_median", "mean"),
            total_google_reviews=("google_review_count", "sum"),
            median_google_reviews=("google_review_count", "median"),
            price_observation_n=("search_price_usd_snapshot", "count"),
            median_price_snapshot=("search_price_usd_snapshot", "median"),
            mean_price_snapshot=("search_price_usd_snapshot", "mean"),
        )
        .reindex(EXPECTED_AREAS)
        .reset_index()
    )
    total_hotels = result["sample_hotel_count"].sum()
    result["sample_supply_share_pct"] = (
        100 * result["sample_hotel_count"] / total_hotels
    ).round(2)
    result["reviews_per_sample_hotel"] = (
        result["total_google_reviews"] / result["sample_hotel_count"]
    )
    result["price_coverage_pct"] = (
        100 * result["price_observation_n"] / result["sample_hotel_count"]
    ).round(1)
    return result


def aggregate_matched_official_metrics(hotels: pd.DataFrame) -> pd.DataFrame:
    """Yalnız high-confidence hotel matches üzerinden resmî kapasiteyi özetler."""

    require_columns(
        hotels,
        [
            "hotel_id",
            "area",
            "official_match_status",
            "official_star_rating_verified",
            "official_room_count",
            "official_bed_count",
            "official_type",
        ],
    )
    matched = hotels.loc[
        hotels["official_match_status"].eq(HIGH_CONFIDENCE_STATUS)
    ].copy()
    grouped = (
        matched.groupby("area", observed=True)
        .agg(
            official_matched_hotel_count=("hotel_id", "size"),
            official_attribute_n=("hotel_id", "size"),
            verified_star_n=("official_star_rating_verified", "count"),
            verified_five_star_count=(
                "official_star_rating_verified", lambda s: int(s.eq(5).sum())
            ),
            verified_four_star_count=(
                "official_star_rating_verified", lambda s: int(s.eq(4).sum())
            ),
            room_capacity_n=("official_room_count", "count"),
            bed_capacity_n=("official_bed_count", "count"),
            total_official_rooms=("official_room_count", lambda s: s.sum(min_count=1)),
            total_official_beds=("official_bed_count", lambda s: s.sum(min_count=1)),
            median_official_room_count=("official_room_count", "median"),
            median_official_bed_count=("official_bed_count", "median"),
            official_boutique_count=(
                "official_type", lambda s: int(s.eq("BUTİK OTEL").sum())
            ),
        )
        .reindex(EXPECTED_AREAS)
        .reset_index()
    )
    count_columns = [
        "official_matched_hotel_count",
        "official_attribute_n",
        "verified_star_n",
        "verified_five_star_count",
        "verified_four_star_count",
        "room_capacity_n",
        "bed_capacity_n",
        "official_boutique_count",
    ]
    grouped[count_columns] = grouped[count_columns].fillna(0).astype(int)
    return grouped


def build_destination_master(
    hotels: pd.DataFrame,
    destination_context: pd.DataFrame,
    *,
    official_match_rate_threshold: float = 40.0,
    verified_star_minimum: int = 3,
    low_sample_threshold: int = 7,
) -> pd.DataFrame:
    """Hotel aggregate, high-confidence official metrics ve V1 bağlamını birleştirir."""

    hotel_metrics = aggregate_hotel_metrics(hotels)
    official_metrics = aggregate_matched_official_metrics(hotels)
    context_columns = [
        "area",
        "has_marina_official_context",
        "has_weekly_market_official_context",
        "weekly_market_days_official_context",
    ]
    require_columns(destination_context, context_columns)
    context = destination_context[context_columns].drop_duplicates("area")
    master = hotel_metrics.merge(official_metrics, on="area", how="left", validate="one_to_one")
    master = master.merge(context, on="area", how="left", validate="one_to_one")
    master["official_match_rate_pct"] = (
        100 * master["official_matched_hotel_count"] / master["sample_hotel_count"]
    ).round(1)
    master["verified_star_coverage_pct"] = (
        100 * master["verified_star_n"] / master["sample_hotel_count"]
    ).round(1)
    master["room_coverage_pct"] = (
        100 * master["room_capacity_n"] / master["sample_hotel_count"]
    ).round(1)
    master["verified_five_star_share"] = np.where(
        master["verified_star_n"].gt(0),
        master["verified_five_star_count"] / master["verified_star_n"],
        np.nan,
    )
    master["beds_per_room_destination"] = np.where(
        master["total_official_rooms"].gt(0),
        master["total_official_beds"] / master["total_official_rooms"],
        np.nan,
    )
    overall_area_median = master["median_price_snapshot"].median()
    master["overall_area_median_price"] = overall_area_median
    master["price_index"] = np.where(
        overall_area_median > 0,
        100 * master["median_price_snapshot"] / overall_area_median,
        np.nan,
    )
    master["low_coverage_flag"] = (
        master["official_match_rate_pct"].lt(official_match_rate_threshold)
        | master["verified_star_n"].lt(verified_star_minimum)
    )
    master["coverage_flag"] = np.where(
        master["low_coverage_flag"], "LOW", "ADEQUATE_FOR_EXPLORATORY"
    )
    master["low_sample_flag"] = master["sample_hotel_count"].lt(low_sample_threshold)
    high_confidence = (
        master["sample_hotel_count"].ge(10)
        & master["price_coverage_pct"].ge(75)
        & master["official_match_rate_pct"].ge(official_match_rate_threshold)
        & master["verified_star_n"].ge(verified_star_minimum)
    )
    medium_confidence = (
        master["sample_hotel_count"].ge(low_sample_threshold)
        & master["price_coverage_pct"].ge(60)
        & master["official_matched_hotel_count"].ge(3)
    )
    master["score_confidence"] = np.select(
        [high_confidence, medium_confidence], ["high", "medium"], default="low"
    )
    return master


def minmax(series: pd.Series) -> pd.Series:
    """Eksikleri koruyan 0–100 min-max dönüşümü."""

    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    if valid.empty:
        return result
    minimum, maximum = valid.min(), valid.max()
    if np.isclose(minimum, maximum):
        result.loc[valid.index] = 50.0
    else:
        result.loc[valid.index] = 100 * (valid - minimum) / (maximum - minimum)
    return result


def weighted_available_mean(
    frame: pd.DataFrame,
    weights: dict[str, float],
    *,
    minimum_components: int,
) -> pd.Series:
    """Eksik bileşenleri doldurmadan yalnız mevcut ağırlıkları yeniden ölçekler."""

    values = frame[list(weights)].apply(pd.to_numeric, errors="coerce")
    weight_series = pd.Series(weights, dtype=float)
    available = values.notna()
    numerator = values.mul(weight_series, axis=1).sum(axis=1, min_count=1)
    denominator = available.mul(weight_series, axis=1).sum(axis=1)
    count = available.sum(axis=1)
    result = numerator / denominator.replace(0, np.nan)
    return result.where(count.ge(minimum_components))


def add_subindices(master: pd.DataFrame) -> pd.DataFrame:
    """Tek birleşik skor yerine açıklanabilir beş alt indeks üretir."""

    result = master.copy()
    result["quality_component_weighted_rating"] = minmax(
        result["avg_weighted_google_rating"]
    )
    result["popularity_component_review_intensity"] = minmax(
        result["reviews_per_sample_hotel"]
    )
    result["popularity_component_total_reviews"] = minmax(
        result["total_google_reviews"]
    )
    result["luxury_component_five_star_share"] = minmax(
        result["verified_five_star_share"]
    )
    result["luxury_component_five_star_count"] = minmax(
        result["verified_five_star_count"].where(result["verified_star_n"].gt(0))
    )
    result["luxury_component_price_index"] = minmax(result["price_index"])
    result["value_component_rating"] = result["quality_component_weighted_rating"]
    result["value_component_inverse_price"] = 100 - minmax(result["price_index"])
    result["supply_component_sample_hotels"] = minmax(result["sample_hotel_count"])
    result["supply_component_official_rooms"] = minmax(result["total_official_rooms"])
    result["supply_component_official_beds"] = minmax(result["total_official_beds"])

    result["quality_index"] = result["quality_component_weighted_rating"]
    result["popularity_index"] = weighted_available_mean(
        result,
        {
            "popularity_component_review_intensity": 0.5,
            "popularity_component_total_reviews": 0.5,
        },
        minimum_components=2,
    )
    result["luxury_index"] = weighted_available_mean(
        result,
        {
            "luxury_component_five_star_share": 0.4,
            "luxury_component_five_star_count": 0.3,
            "luxury_component_price_index": 0.3,
        },
        minimum_components=2,
    )
    result["value_index"] = weighted_available_mean(
        result,
        {"value_component_rating": 0.5, "value_component_inverse_price": 0.5},
        minimum_components=2,
    )
    result["supply_capacity_index"] = weighted_available_mean(
        result,
        {
            "supply_component_sample_hotels": 1 / 3,
            "supply_component_official_rooms": 1 / 3,
            "supply_component_official_beds": 1 / 3,
        },
        minimum_components=2,
    )
    result["luxury_rank_eligible"] = ~result["low_coverage_flag"]
    result["luxury_rank"] = result["luxury_index"].where(
        result["luxury_rank_eligible"]
    ).rank(method="min", ascending=False)
    return result


def value_sensitivity(master: pd.DataFrame) -> pd.DataFrame:
    """Value indeksi için dengeli/rating-heavy/price-heavy senaryolarını karşılaştırır."""

    scenarios = {
        "balanced": (0.5, 0.5),
        "rating_heavy": (0.7, 0.3),
        "price_heavy": (0.3, 0.7),
    }
    rows = []
    for scenario, (rating_weight, price_weight) in scenarios.items():
        score = (
            rating_weight * master["value_component_rating"]
            + price_weight * master["value_component_inverse_price"]
        )
        rank = score.rank(method="min", ascending=False)
        for area, area_score, area_rank in zip(master["area"], score, rank):
            rows.append(
                {
                    "area": area,
                    "scenario": scenario,
                    "rating_weight": rating_weight,
                    "inverse_price_weight": price_weight,
                    "value_index": area_score,
                    "rank": area_rank,
                }
            )
    result = pd.DataFrame(rows)
    spread = result.groupby("area")["rank"].agg(["min", "max"])
    spread["rank_spread"] = spread["max"] - spread["min"]
    spread["ranking_sensitive"] = spread["rank_spread"].ge(4)
    return result.merge(
        spread[["rank_spread", "ranking_sensitive"]],
        left_on="area",
        right_index=True,
        validate="many_to_one",
    )


def median_quadrant(
    frame: pd.DataFrame,
    x: str,
    y: str,
    labels: tuple[str, str, str, str],
    *,
    eligible: pd.Series | None = None,
) -> tuple[pd.Series, float, float]:
    """High-x/high-y, high-x/low-y, low-x/high-y, low-x/low-y etiketleri."""

    if eligible is None:
        eligible = pd.Series(True, index=frame.index)
    complete = eligible & frame[x].notna() & frame[y].notna()
    x_median = frame.loc[complete, x].median()
    y_median = frame.loc[complete, y].median()
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    result.loc[complete] = np.select(
        [
            frame.loc[complete, x].ge(x_median) & frame.loc[complete, y].ge(y_median),
            frame.loc[complete, x].ge(x_median) & frame.loc[complete, y].lt(y_median),
            frame.loc[complete, x].lt(x_median) & frame.loc[complete, y].ge(y_median),
        ],
        labels[:3],
        default=labels[3],
    )
    return result, float(x_median), float(y_median)


def add_quadrants(master: pd.DataFrame) -> pd.DataFrame:
    result = master.copy()
    result["value_quadrant"], _, _ = median_quadrant(
        result,
        "price_index",
        "avg_weighted_google_rating",
        (
            "High satisfaction + higher price",
            "Lower satisfaction + higher price",
            "High satisfaction + lower price",
            "Lower satisfaction + lower price",
        ),
    )
    result["popularity_satisfaction_quadrant"], _, _ = median_quadrant(
        result,
        "reviews_per_sample_hotel",
        "avg_weighted_google_rating",
        (
            "High popularity + high satisfaction",
            "High popularity + lower satisfaction",
            "Lower popularity + high satisfaction",
            "Lower popularity + lower satisfaction",
        ),
    )
    result["price_luxury_quadrant_eligible"] = ~result["low_coverage_flag"]
    result["price_luxury_quadrant"], _, _ = median_quadrant(
        result,
        "price_index",
        "verified_five_star_share",
        (
            "Higher price + higher five-star share",
            "Higher price + lower five-star share",
            "Lower price + higher five-star share",
            "Lower price + lower five-star share",
        ),
        eligible=result["price_luxury_quadrant_eligible"],
    )
    return result


def build_archetypes(master: pd.DataFrame) -> pd.DataFrame:
    """Gerçek indeks dağılımının üst çeyrek/medyan eşikleriyle kural tabanlı profil üretir."""

    result = master.copy()
    supply_q75 = result["supply_capacity_index"].quantile(0.75)
    luxury_q75 = result.loc[result["luxury_rank_eligible"], "luxury_index"].quantile(0.75)
    popularity_q75 = result["popularity_index"].quantile(0.75)
    value_q75 = result["value_index"].quantile(0.75)
    quality_median = result["quality_index"].median()
    popularity_median = result["popularity_index"].median()
    conditions = [
        result["supply_capacity_index"].ge(supply_q75),
        result["luxury_rank_eligible"] & result["luxury_index"].ge(luxury_q75),
        result["popularity_index"].ge(popularity_q75),
        result["value_index"].ge(value_q75),
        result["quality_index"].ge(quality_median)
        & result["popularity_index"].lt(popularity_median),
    ]
    labels = [
        "Capacity-Heavy Resort Area",
        "Premium / Luxury Concentrated",
        "High Popularity",
        "Value-Oriented",
        "High Satisfaction / Low Visibility",
    ]
    result["archetype"] = np.select(conditions, labels, default="Balanced")
    return result


def spearman_correlations(
    frame: pd.DataFrame, pairs: Iterable[tuple[str, str]]
) -> pd.DataFrame:
    rows = []
    for x, y in pairs:
        complete = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(complete) >= 3 and complete[x].nunique() > 1 and complete[y].nunique() > 1:
            rho, p_value = stats.spearmanr(complete[x], complete[y])
            note = "Exploratory; small destination n and coverage differences limit inference."
        else:
            rho, p_value = np.nan, np.nan
            note = "Skipped or unstable: fewer than 3 complete, non-constant observations."
        rows.append(
            {
                "metric_x": x,
                "metric_y": y,
                "n": len(complete),
                "spearman_rho": rho,
                "p_value": p_value,
                "interpretation_note": note,
            }
        )
    return pd.DataFrame(rows)


def input_consistency_check(
    current: pd.DataFrame,
    previous_eda: pd.DataFrame,
    destination_v1: pd.DataFrame,
    matched_capacity: pd.DataFrame,
) -> pd.DataFrame:
    """Aynı adlı metrikleri karşılaştırır; farklı evrenleri açıkça işaretler."""

    rows: list[dict[str, object]] = []

    def compare(
        source_a: str,
        source_b: str,
        frame_a: pd.DataFrame,
        frame_b: pd.DataFrame,
        metric_a: str,
        metric_b: str,
        *,
        tolerance: float = 1e-6,
        decision: str,
    ) -> None:
        left = frame_a.set_index("area")[metric_a]
        right = frame_b.set_index("area")[metric_b]
        for area in EXPECTED_AREAS:
            a = left.get(area, np.nan)
            b = right.get(area, np.nan)
            both_missing = pd.isna(a) and pd.isna(b)
            equal = both_missing or (
                pd.notna(a) and pd.notna(b) and np.isclose(float(a), float(b), atol=tolerance, rtol=tolerance)
            )
            rows.append(
                {
                    "area": area,
                    "metric": metric_a,
                    "source_a": source_a,
                    "value_a": a,
                    "source_b": source_b,
                    "value_b": b,
                    "absolute_difference": abs(float(a) - float(b)) if pd.notna(a) and pd.notna(b) else np.nan,
                    "comparison_status": "CONSISTENT" if equal else "DIFFERENT",
                    "authoritative_decision": decision,
                }
            )

    hotel_pairs = [
        ("sample_hotel_count", "hotel_count", "sample_hotel_count"),
        ("avg_google_rating", "avg_rating", "avg_google_rating"),
        ("median_google_rating", "median_rating", "median_google_rating"),
        ("total_google_reviews", "total_reviews", "total_google_reviews"),
        ("median_google_reviews", "median_reviews", "median_google_reviews"),
        ("price_observation_n", "price_n", "price_observation_n"),
        ("median_price_snapshot", "median_price", "median_price_usd_snapshot"),
        ("mean_price_snapshot", "mean_price", "mean_price_usd_snapshot"),
    ]
    for current_metric, previous_metric, v1_metric in hotel_pairs:
        compare(
            "current_hotels_enriched_aggregate",
            "eda_destination_profile",
            current,
            previous_eda,
            current_metric,
            previous_metric,
            tolerance=1e-4,
            decision="Use current hotels_enriched aggregate; previous EDA is a consistency reference.",
        )
        compare(
            "current_hotels_enriched_aggregate",
            "destination_intelligence_v1",
            current,
            destination_v1,
            current_metric,
            v1_metric,
            tolerance=1e-2,
            decision="Use current hotels_enriched aggregate; V1 hotel metrics are a consistency reference.",
        )

    official_pairs = [
        ("official_matched_hotel_count", "matched_hotel_count"),
        ("verified_star_n", "verified_star_count"),
        ("total_official_rooms", "total_official_rooms"),
        ("total_official_beds", "total_official_beds"),
    ]
    for current_metric, report_metric in official_pairs:
        compare(
            "current_high_confidence_match_aggregate",
            "hotel_attributes_destination_capacity",
            current,
            matched_capacity,
            current_metric,
            report_metric,
            tolerance=1e-6,
            decision="Use current high-confidence match aggregate; missing report rows mean zero matched coverage, not zero true capacity.",
        )

    v1_official_pairs = [
        ("official_matched_hotel_count", "official_facility_count_confidently_mapped"),
        ("total_official_rooms", "official_room_count_confidently_mapped"),
        ("total_official_beds", "official_bed_count_confidently_mapped"),
        ("verified_five_star_count", "official_5star_count_confidently_mapped"),
    ]
    for current_metric, v1_metric in v1_official_pairs:
        compare(
            "current_high_confidence_match_aggregate",
            "destination_intelligence_v1_official_universe",
            current,
            destination_v1,
            current_metric,
            v1_metric,
            tolerance=1e-6,
            decision="Different source scopes: retain high-confidence project-hotel matches for this notebook; use V1 only for marina/market context.",
        )
    result = pd.DataFrame(rows)
    official_v1_mask = result["source_b"].eq(
        "destination_intelligence_v1_official_universe"
    )
    result.loc[official_v1_mask & result["comparison_status"].eq("DIFFERENT"), "comparison_status"] = (
        "DIFFERENT_SOURCE_SCOPE_EXPECTED"
    )
    zero_coverage_missing = (
        result["source_b"].eq("hotel_attributes_destination_capacity")
        & result["value_a"].eq(0)
        & result["value_b"].isna()
    )
    result.loc[zero_coverage_missing, "comparison_status"] = "REPORT_OMITS_ZERO_COVERAGE_AREA"
    return result
