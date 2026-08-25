"""Resmî otel özellikleri EDA'sı için tekrar kullanılabilir özet fonksiyonları."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


HIGH_CONFIDENCE_STATUS = "MATCHED_HIGH_CONFIDENCE"
SIZE_LABELS = ("Small", "Medium", "Large")


@dataclass(frozen=True)
class CorrelationResult:
    analysis: str
    method: str
    n: int
    coefficient: float
    p_value: float


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Eksik zorunlu kolonları anlaşılır bir hata ile bildirir."""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Eksik zorunlu kolonlar: {missing}")


def official_analysis_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Yalnızca yüksek güvenli resmî eşleşmeleri analiz örneklemine alır."""

    require_columns(frame, ["official_match_status"])
    sample = frame.loc[
        frame["official_match_status"].eq(HIGH_CONFIDENCE_STATUS)
    ].copy()
    numeric_columns = [
        "official_star_rating_verified",
        "official_room_count",
        "official_bed_count",
        "google_rating",
        "google_review_count",
        "search_price_usd_snapshot",
        "weighted_google_rating",
        "review_confidence_weight",
        "rating_gap_from_area_median",
        "price_ratio_to_area_median",
        "price_percentile_within_area",
    ]
    for column in numeric_columns:
        if column in sample:
            sample[column] = pd.to_numeric(sample[column], errors="coerce")
    return sample


def coverage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Ana 192 otel evrenine göre resmî özellik ve fiyat kapsamını özetler."""

    require_columns(
        frame,
        [
            "hotel_id",
            "official_match_status",
            "official_star_rating_verified",
            "official_room_count",
            "official_bed_count",
            "official_type",
            "search_price_usd_snapshot",
        ],
    )
    total = len(frame)
    metrics = [
        ("total_hotels", total),
        (
            "high_confidence_official_match",
            int(frame["official_match_status"].eq(HIGH_CONFIDENCE_STATUS).sum()),
        ),
        ("verified_star_available", int(frame["official_star_rating_verified"].notna().sum())),
        ("room_count_available", int(frame["official_room_count"].notna().sum())),
        ("bed_count_available", int(frame["official_bed_count"].notna().sum())),
        ("official_type_available", int(frame["official_type"].notna().sum())),
        ("price_snapshot_available", int(frame["search_price_usd_snapshot"].notna().sum())),
    ]
    result = pd.DataFrame(metrics, columns=["metric", "hotel_count"])
    result["coverage_pct_of_192"] = np.where(
        total, 100 * result["hotel_count"] / total, np.nan
    ).round(1)
    return result


def star_summary(sample: pd.DataFrame) -> pd.DataFrame:
    """Yıldız dağılımı ile rating, yorum ve fiyat özetini üretir."""

    require_columns(
        sample,
        [
            "official_star_rating_verified",
            "google_rating",
            "google_review_count",
            "search_price_usd_snapshot",
        ],
    )
    valid = sample.dropna(subset=["official_star_rating_verified"]).copy()
    total = len(valid)
    summary = (
        valid.groupby("official_star_rating_verified", observed=True)
        .agg(
            hotel_count=("hotel_id", "size"),
            avg_google_rating=("google_rating", "mean"),
            median_google_rating=("google_rating", "median"),
            rating_std=("google_rating", "std"),
            rating_min=("google_rating", "min"),
            rating_max=("google_rating", "max"),
            median_review_count=("google_review_count", "median"),
            price_n=("search_price_usd_snapshot", "count"),
            median_price_snapshot=("search_price_usd_snapshot", "median"),
            mean_price_snapshot=("search_price_usd_snapshot", "mean"),
            price_q25=("search_price_usd_snapshot", lambda s: s.quantile(0.25)),
            price_q75=("search_price_usd_snapshot", lambda s: s.quantile(0.75)),
            median_weighted_rating=("weighted_google_rating", "median"),
            median_rating_gap_from_area=("rating_gap_from_area_median", "median"),
            median_price_ratio_to_area=("price_ratio_to_area_median", "median"),
        )
        .reset_index()
        .rename(columns={"official_star_rating_verified": "star"})
        .sort_values("star")
    )
    summary.insert(2, "share_pct", (100 * summary["hotel_count"] / total).round(1))
    return summary


def assign_size_groups(room_count: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    """Oda dağılımının üçte birlik eşiklerinden açıklanabilir boyut grupları üretir."""

    numeric = pd.to_numeric(room_count, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(pd.NA, index=room_count.index, dtype="string"), {}
    q33, q67 = valid.quantile([1 / 3, 2 / 3]).tolist()
    if q33 >= q67:
        median = float(valid.median())
        grouped = pd.Series(pd.NA, index=room_count.index, dtype="string")
        grouped.loc[numeric.notna() & (numeric <= median)] = "Small"
        grouped.loc[numeric.notna() & (numeric > median)] = "Large"
        return grouped, {"median_fallback": median}
    grouped = pd.cut(
        numeric,
        bins=[-np.inf, q33, q67, np.inf],
        labels=SIZE_LABELS,
        include_lowest=True,
    ).astype("string")
    return grouped, {"q33": float(q33), "q67": float(q67)}


def size_summary(sample: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Geçici oda bazlı tesis boyutu gruplarını ve özetlerini üretir."""

    working = sample.copy()
    working["hotel_size_group"], thresholds = assign_size_groups(
        working["official_room_count"]
    )
    summary = (
        working.dropna(subset=["hotel_size_group"])
        .groupby("hotel_size_group", observed=True)
        .agg(
            n=("hotel_id", "size"),
            min_room_count=("official_room_count", "min"),
            median_room_count=("official_room_count", "median"),
            max_room_count=("official_room_count", "max"),
            avg_google_rating=("google_rating", "mean"),
            median_google_rating=("google_rating", "median"),
            median_review_count=("google_review_count", "median"),
            median_price=("search_price_usd_snapshot", "median"),
        )
        .reindex(SIZE_LABELS)
        .dropna(how="all")
        .reset_index()
    )
    return summary, thresholds


def type_summary(sample: pd.DataFrame) -> pd.DataFrame:
    """Resmî tesis tiplerini örneklem ve temel metriklerle özetler."""

    return (
        sample.dropna(subset=["official_type"])
        .groupby("official_type", observed=True)
        .agg(
            n=("hotel_id", "size"),
            median_google_rating=("google_rating", "median"),
            median_review_count=("google_review_count", "median"),
            price_n=("search_price_usd_snapshot", "count"),
            median_price=("search_price_usd_snapshot", "median"),
            room_n=("official_room_count", "count"),
            median_room_count=("official_room_count", "median"),
            median_bed_count=("official_bed_count", "median"),
        )
        .reset_index()
        .sort_values(["n", "official_type"], ascending=[False, True])
    )


def destination_capacity(sample: pd.DataFrame) -> pd.DataFrame:
    """Yüksek güvenli eşleşmelerden destinasyon resmî kapasite profili üretir."""

    require_columns(sample, ["area", "hotel_id"])
    result = (
        sample.groupby("area", observed=True)
        .agg(
            matched_hotel_count=("hotel_id", "size"),
            room_count_available=("official_room_count", "count"),
            total_official_rooms=("official_room_count", "sum"),
            bed_count_available=("official_bed_count", "count"),
            total_official_beds=("official_bed_count", "sum"),
            median_room_count=("official_room_count", "median"),
            median_bed_count=("official_bed_count", "median"),
            verified_star_count=("official_star_rating_verified", "count"),
            verified_5star_count=(
                "official_star_rating_verified",
                lambda s: int(s.eq(5).sum()),
            ),
            verified_4star_count=(
                "official_star_rating_verified",
                lambda s: int(s.eq(4).sum()),
            ),
            avg_google_rating=("google_rating", "mean"),
            median_google_rating=("google_rating", "median"),
            avg_weighted_rating=("weighted_google_rating", "mean"),
            price_n=("search_price_usd_snapshot", "count"),
            median_price=("search_price_usd_snapshot", "median"),
        )
        .reset_index()
    )
    result["verified_5star_share_pct"] = np.where(
        result["verified_star_count"].gt(0),
        100 * result["verified_5star_count"] / result["verified_star_count"],
        np.nan,
    ).round(1)
    return result.sort_values(
        ["total_official_rooms", "area"], ascending=[False, True]
    ).reset_index(drop=True)


def correlation_results(
    frame: pd.DataFrame,
    x: str,
    y: str,
    analysis: str,
    methods: tuple[str, ...] = ("pearson", "spearman"),
) -> list[CorrelationResult]:
    """Eksikleri düşürüp sabit seri ve küçük örneklem güvenlikleriyle korelasyon hesaplar."""

    pair = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    results: list[CorrelationResult] = []
    if len(pair) < 3 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
        return [CorrelationResult(analysis, method, len(pair), np.nan, np.nan) for method in methods]
    for method in methods:
        if method == "pearson":
            coefficient, p_value = stats.pearsonr(pair[x], pair[y])
        elif method == "spearman":
            coefficient, p_value = stats.spearmanr(pair[x], pair[y])
        else:
            raise ValueError(f"Desteklenmeyen korelasyon yöntemi: {method}")
        results.append(
            CorrelationResult(
                analysis=analysis,
                method=method,
                n=len(pair),
                coefficient=float(coefficient),
                p_value=float(p_value),
            )
        )
    return results


def correlation_table(results: Iterable[CorrelationResult]) -> pd.DataFrame:
    return pd.DataFrame([result.__dict__ for result in results])


def kruskal_star_rating_test(sample: pd.DataFrame) -> pd.DataFrame:
    """Yeterli gruplarda yıldız-rating Kruskal-Wallis testi ve epsilon-kare üretir."""

    valid = sample.dropna(
        subset=["official_star_rating_verified", "google_rating"]
    )
    groups = [
        group["google_rating"].to_numpy()
        for _, group in valid.groupby("official_star_rating_verified")
        if len(group) >= 2
    ]
    group_sizes = [len(group) for group in groups]
    if len(groups) < 2:
        return pd.DataFrame(
            [
                {
                    "test": "Kruskal-Wallis: star vs google_rating",
                    "n": len(valid),
                    "group_count": len(groups),
                    "group_sizes": str(group_sizes),
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "effect_size_epsilon_squared": np.nan,
                    "status": "SKIPPED_INSUFFICIENT_GROUPS",
                }
            ]
        )
    statistic, p_value = stats.kruskal(*groups)
    n = sum(group_sizes)
    k = len(groups)
    epsilon_squared = max(0.0, (statistic - k + 1) / (n - k)) if n > k else np.nan
    return pd.DataFrame(
        [
            {
                "test": "Kruskal-Wallis: star vs google_rating",
                "n": n,
                "group_count": k,
                "group_sizes": str(group_sizes),
                "statistic": float(statistic),
                "p_value": float(p_value),
                "effect_size_epsilon_squared": float(epsilon_squared),
                "status": "COMPUTED",
            }
        ]
    )


def price_premiums(stars: pd.DataFrame) -> pd.DataFrame:
    """Komşu yıldız gruplarının medyan fiyat farkını hesaplar."""

    medians = stars.set_index("star")["median_price_snapshot"]
    counts = stars.set_index("star")["price_n"]
    rows = []
    for higher, lower in ((5, 4), (4, 3), (3, 2)):
        if higher not in medians or lower not in medians:
            continue
        high_value, low_value = medians.loc[higher], medians.loc[lower]
        premium = (
            100 * (high_value / low_value - 1)
            if pd.notna(high_value) and pd.notna(low_value) and low_value > 0
            else np.nan
        )
        rows.append(
            {
                "comparison": f"{higher}_star_vs_{lower}_star",
                "higher_star": higher,
                "lower_star": lower,
                "higher_price_n": int(counts.loc[higher]),
                "lower_price_n": int(counts.loc[lower]),
                "higher_median_price": high_value,
                "lower_median_price": low_value,
                "median_price_premium_pct": premium,
            }
        )
    return pd.DataFrame(rows)


def interesting_cases(sample: pd.DataFrame) -> pd.DataFrame:
    """Önceden tanımlı, açıklanabilir vaka kurallarıyla tek bir uzun tablo üretir."""

    columns = [
        "hotel_id",
        "hotel_name",
        "area",
        "official_star_rating_verified",
        "official_room_count",
        "google_rating",
        "weighted_google_rating",
        "google_review_count",
        "search_price_usd_snapshot",
        "price_ratio_to_area_median",
    ]
    cases: list[pd.DataFrame] = []

    def add(case_type: str, data: pd.DataFrame, ascending: bool, sort_col: str, n: int = 5) -> None:
        chosen = data.dropna(subset=[sort_col]).sort_values(sort_col, ascending=ascending).head(n)
        if not chosen.empty:
            chosen = chosen.reindex(columns=columns).copy()
            chosen.insert(0, "case_type", case_type)
            cases.append(chosen)

    add(
        "5_star_relatively_low_weighted_rating",
        sample.loc[sample["official_star_rating_verified"].eq(5)],
        True,
        "weighted_google_rating",
    )
    add(
        "up_to_3_star_high_weighted_rating",
        sample.loc[sample["official_star_rating_verified"].le(3)],
        False,
        "weighted_google_rating",
    )
    room_median = sample["official_room_count"].median()
    add(
        "large_capacity_relatively_low_rating",
        sample.loc[sample["official_room_count"].ge(room_median)],
        True,
        "google_rating",
    )
    room_q25 = sample["official_room_count"].quantile(0.25)
    add(
        "small_capacity_high_review_count",
        sample.loc[sample["official_room_count"].le(room_q25)],
        False,
        "google_review_count",
    )
    add(
        "high_area_price_ratio_lower_rating",
        sample.loc[sample["price_ratio_to_area_median"].ge(1.25)],
        True,
        "google_rating",
    )
    add(
        "low_area_price_ratio_higher_rating",
        sample.loc[sample["price_ratio_to_area_median"].le(0.80)],
        False,
        "google_rating",
    )
    if not cases:
        return pd.DataFrame(columns=["case_type", *columns])
    return pd.concat(cases, ignore_index=True)
