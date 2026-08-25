"""Muğla ve Bodrum turizm talebi için açıklanabilir özellik/özet fonksiyonları."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


MONTH_ORDER_TR = [
    "Ocak",
    "Şubat",
    "Mart",
    "Nisan",
    "Mayıs",
    "Haziran",
    "Temmuz",
    "Ağustos",
    "Eylül",
    "Ekim",
    "Kasım",
    "Aralık",
]

CORE_FLOW_COLUMNS = [
    "domestic_arrivals",
    "foreign_arrivals",
    "total_arrivals",
    "total_overnights",
]


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Eksik zorunlu kolonlar: {missing}")


def safe_ratio(numerator: pd.Series, denominator: pd.Series, scale: float = 1.0) -> pd.Series:
    """Sıfır paydayı NaN bırakır; inf üretmez."""

    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return (scale * numerator / denominator.where(denominator.gt(0))).replace(
        [np.inf, -np.inf], np.nan
    )


def dataset_quality_audit(
    frame: pd.DataFrame,
    dataset_name: str,
    period_column: str,
    *,
    occupancy_column: str | None = "occupancy_rate_pct",
    average_stay_column: str | None = None,
) -> pd.DataFrame:
    """Eksik, duplicate, aralık ve temel muhasebe kontrollerini uzun tabloda verir."""

    require_columns(frame, [period_column, *CORE_FLOW_COLUMNS])
    numeric_columns = [*CORE_FLOW_COLUMNS]
    if occupancy_column and occupancy_column in frame:
        numeric_columns.append(occupancy_column)
    if average_stay_column and average_stay_column in frame:
        numeric_columns.append(average_stay_column)
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    arrival_difference = (
        numeric["domestic_arrivals"]
        + numeric["foreign_arrivals"]
        - numeric["total_arrivals"]
    )
    checks = [
        ("row_count", len(frame), "INFO"),
        ("column_count", frame.shape[1], "INFO"),
        ("missing_cells", int(frame.isna().sum().sum()), "PASS" if not frame.isna().any().any() else "REVIEW"),
        ("duplicate_rows", int(frame.duplicated().sum()), "PASS" if not frame.duplicated().any() else "FAIL"),
        (
            f"duplicate_{period_column}",
            int(frame[period_column].duplicated().sum()),
            "PASS" if frame[period_column].is_unique else "FAIL",
        ),
        ("invalid_numeric_cells", int(numeric.isna().sum().sum()), "PASS" if not numeric.isna().any().any() else "FAIL"),
        ("negative_numeric_values", int(numeric.lt(0).sum().sum()), "PASS" if not numeric.lt(0).any().any() else "FAIL"),
        ("arrival_total_mismatch_rows", int(arrival_difference.ne(0).sum()), "PASS" if arrival_difference.eq(0).all() else "FAIL"),
        ("max_abs_arrival_total_difference", float(arrival_difference.abs().max()), "PASS" if arrival_difference.eq(0).all() else "FAIL"),
    ]
    if occupancy_column and occupancy_column in numeric:
        invalid_occupancy = ~numeric[occupancy_column].between(0, 100)
        checks.append(
            (
                "occupancy_out_of_0_100_rows",
                int(invalid_occupancy.sum()),
                "PASS" if not invalid_occupancy.any() else "FAIL",
            )
        )
    if average_stay_column and average_stay_column in numeric:
        invalid_stay = numeric[average_stay_column].le(0)
        checks.append(
            (
                "nonpositive_average_stay_rows",
                int(invalid_stay.sum()),
                "PASS" if not invalid_stay.any() else "FAIL",
            )
        )
    return pd.DataFrame(checks, columns=["check", "value", "status"]).assign(
        dataset_name=dataset_name
    )[["dataset_name", "check", "value", "status"]]


def add_annual_features(annual: pd.DataFrame) -> pd.DataFrame:
    """2009–2025 yıllık seriye türetilmiş kalite, büyüme ve 2019 kıyasları ekler."""

    require_columns(
        annual,
        [
            "year",
            *CORE_FLOW_COLUMNS,
            "avg_stay_nights",
            "occupancy_rate_pct",
            "foreign_arrival_share_pct",
        ],
    )
    result = annual.sort_values("year").reset_index(drop=True).copy()
    result["arrival_total_difference"] = (
        result["domestic_arrivals"]
        + result["foreign_arrivals"]
        - result["total_arrivals"]
    )
    result["derived_avg_stay_nights_recalculated"] = safe_ratio(
        result["total_overnights"], result["total_arrivals"]
    )
    result["avg_stay_difference"] = (
        result["avg_stay_nights"] - result["derived_avg_stay_nights_recalculated"]
    )
    result["derived_foreign_arrival_share_pct_recalculated"] = safe_ratio(
        result["foreign_arrivals"], result["total_arrivals"], 100
    )
    result["foreign_share_difference_pp"] = (
        result["foreign_arrival_share_pct"]
        - result["derived_foreign_arrival_share_pct_recalculated"]
    )
    for column in [
        "total_arrivals",
        "domestic_arrivals",
        "foreign_arrivals",
        "total_overnights",
    ]:
        result[f"{column}_yoy_pct"] = result[column].pct_change() * 100
    result["occupancy_yoy_change_pp"] = result["occupancy_rate_pct"].diff()
    benchmark_row = result.loc[result["year"].eq(2019)]
    if len(benchmark_row) != 1:
        raise ValueError("2019 benchmark satırı tekil değil veya eksik.")
    benchmark = benchmark_row.iloc[0]
    for column, output in [
        ("total_arrivals", "total_arrivals_vs_2019_pct"),
        ("foreign_arrivals", "foreign_arrivals_vs_2019_pct"),
        ("domestic_arrivals", "domestic_arrivals_vs_2019_pct"),
        ("total_overnights", "overnights_vs_2019_pct"),
    ]:
        result[output] = 100 * (result[column] / benchmark[column] - 1)
    result["occupancy_vs_2019_pp"] = (
        result["occupancy_rate_pct"] - benchmark["occupancy_rate_pct"]
    )
    result["analysis_period"] = np.select(
        [result["year"].le(2019), result["year"].between(2020, 2021)],
        ["pre_pandemic_2009_2019", "shock_transition_2020_2021"],
        default="post_shock_2022_2025",
    )
    return result


def add_monthly_features(
    monthly: pd.DataFrame, annual_2025: pd.Series
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Muğla 2025 aylık seri için kalite, pay, sezon ve normalize profil üretir."""

    require_columns(
        monthly,
        [
            "period",
            "year",
            "month_name_tr",
            *CORE_FLOW_COLUMNS,
            "occupancy_rate_pct",
            "derived_avg_stay_nights",
            "derived_foreign_arrival_share_pct",
        ],
    )
    result = monthly.copy()
    result["period_parsed"] = pd.PeriodIndex(result["period"], freq="M")
    result = result.sort_values("period_parsed").reset_index(drop=True)
    result["month_name_tr"] = pd.Categorical(
        result["month_name_tr"], categories=MONTH_ORDER_TR, ordered=True
    )
    result["arrival_total_difference"] = (
        result["domestic_arrivals"]
        + result["foreign_arrivals"]
        - result["total_arrivals"]
    )
    result["derived_avg_stay_nights_recalculated"] = safe_ratio(
        result["total_overnights"], result["total_arrivals"]
    )
    result["avg_stay_difference"] = (
        result["derived_avg_stay_nights"]
        - result["derived_avg_stay_nights_recalculated"]
    )
    result["derived_foreign_arrival_share_pct_recalculated"] = safe_ratio(
        result["foreign_arrivals"], result["total_arrivals"], 100
    )
    result["foreign_share_difference_pp"] = (
        result["derived_foreign_arrival_share_pct"]
        - result["derived_foreign_arrival_share_pct_recalculated"]
    )
    result["overnights_per_arrival"] = result[
        "derived_avg_stay_nights_recalculated"
    ]
    monthly_totals = result[CORE_FLOW_COLUMNS].sum()
    comparisons = {
        f"{column}_monthly_minus_annual": int(monthly_totals[column] - annual_2025[column])
        for column in CORE_FLOW_COLUMNS
    }
    totals_match = all(value == 0 for value in comparisons.values())
    arrival_denominator = (
        annual_2025["total_arrivals"] if totals_match else monthly_totals["total_arrivals"]
    )
    overnight_denominator = (
        annual_2025["total_overnights"] if totals_match else monthly_totals["total_overnights"]
    )
    denominator_source = (
        "annual_2025_official_total_matches_monthly_sum"
        if totals_match
        else "monthly_dataset_12_month_sum_due_to_source_difference"
    )
    result["monthly_arrival_share_pct"] = safe_ratio(
        result["total_arrivals"],
        pd.Series(arrival_denominator, index=result.index),
        100,
    )
    result["monthly_overnight_share_pct"] = safe_ratio(
        result["total_overnights"],
        pd.Series(overnight_denominator, index=result.index),
        100,
    )
    result["annual_share_denominator_source"] = denominator_source
    result["domestic_seasonality_index"] = safe_ratio(
        result["domestic_arrivals"],
        pd.Series(result["domestic_arrivals"].mean(), index=result.index),
        100,
    )
    result["foreign_seasonality_index"] = safe_ratio(
        result["foreign_arrivals"],
        pd.Series(result["foreign_arrivals"].mean(), index=result.index),
        100,
    )
    q33, q67 = result["total_arrivals"].quantile([1 / 3, 2 / 3])
    result["season_group"] = pd.cut(
        result["total_arrivals"],
        bins=[-np.inf, q33, q67, np.inf],
        labels=["LOW", "SHOULDER", "PEAK"],
        include_lowest=True,
    ).astype("string")
    metadata = {
        "monthly_totals_match_annual_2025": totals_match,
        "share_denominator_source": denominator_source,
        "arrival_share_denominator": int(arrival_denominator),
        "overnight_share_denominator": int(overnight_denominator),
        "season_q33_total_arrivals": float(q33),
        "season_q67_total_arrivals": float(q67),
        **comparisons,
    }
    return result, metadata


def seasonality_metrics(monthly_features: pd.DataFrame) -> pd.DataFrame:
    """2025 Muğla aylık yoğunlaşmasını açıklanabilir tek satırda özetler."""

    arrivals = monthly_features["total_arrivals"]
    overnights = monthly_features["total_overnights"]
    occupancy = monthly_features["occupancy_rate_pct"]
    arrival_shares = arrivals / arrivals.sum()
    overnight_shares = overnights / overnights.sum()
    return pd.DataFrame(
        [
            {
                "geography": "Muğla",
                "year": 2025,
                "monthly_observation_n": len(monthly_features),
                "peak_to_low_arrival_ratio": arrivals.max() / arrivals.min(),
                "peak_to_low_overnight_ratio": overnights.max() / overnights.min(),
                "top3_month_arrival_share_pct": 100 * arrivals.nlargest(3).sum() / arrivals.sum(),
                "top3_month_overnight_share_pct": 100 * overnights.nlargest(3).sum() / overnights.sum(),
                "top4_month_arrival_share_pct": 100 * arrivals.nlargest(4).sum() / arrivals.sum(),
                "coefficient_of_variation_arrivals": arrivals.std(ddof=1) / arrivals.mean(),
                "coefficient_of_variation_overnights": overnights.std(ddof=1) / overnights.mean(),
                "coefficient_of_variation_occupancy": occupancy.std(ddof=1) / occupancy.mean(),
                "hhi_monthly_arrival_concentration": float((arrival_shares**2).sum()),
                "hhi_monthly_overnight_concentration": float((overnight_shares**2).sum()),
            }
        ]
    )


def season_group_summary(monthly_features: pd.DataFrame) -> pd.DataFrame:
    result = (
        monthly_features.groupby("season_group", observed=True)
        .agg(
            month_count=("period", "size"),
            avg_total_arrivals=("total_arrivals", "mean"),
            avg_foreign_arrivals=("foreign_arrivals", "mean"),
            avg_domestic_arrivals=("domestic_arrivals", "mean"),
            avg_overnights=("total_overnights", "mean"),
            avg_occupancy=("occupancy_rate_pct", "mean"),
            avg_stay=("derived_avg_stay_nights_recalculated", "mean"),
            foreign_share_pct=("foreign_arrivals", "sum"),
            group_total_arrivals=("total_arrivals", "sum"),
        )
        .reset_index()
    )
    result["foreign_share_pct"] = safe_ratio(
        result["foreign_share_pct"], result["group_total_arrivals"], 100
    )
    return result.drop(columns="group_total_arrivals")


def period_summary(annual_features: pd.DataFrame) -> pd.DataFrame:
    """Farklı uzunluktaki pre/shock/post dönemlerini yıllık ortalamalarla özetler."""

    return (
        annual_features.groupby("analysis_period", observed=True)
        .agg(
            year_count=("year", "size"),
            start_year=("year", "min"),
            end_year=("year", "max"),
            average_annual_arrivals=("total_arrivals", "mean"),
            average_foreign_share_pct=("foreign_arrival_share_pct", "mean"),
            average_occupancy_pct=("occupancy_rate_pct", "mean"),
            average_stay_nights=("derived_avg_stay_nights_recalculated", "mean"),
        )
        .reset_index()
    )


def bodrum_profile(bodrum_annual: pd.DataFrame) -> pd.DataFrame:
    """Tek yıllık Bodrum satırını yeniden hesaplanan pay/kalış metrikleriyle verir."""

    if len(bodrum_annual) != 1:
        raise ValueError("Bodrum 2025 annual dataset tek satır olmalıdır.")
    result = bodrum_annual.copy()
    result["arrival_total_difference"] = (
        result["domestic_arrivals"]
        + result["foreign_arrivals"]
        - result["total_arrivals"]
    )
    result["domestic_share_pct"] = safe_ratio(
        result["domestic_arrivals"], result["total_arrivals"], 100
    )
    result["foreign_share_pct"] = safe_ratio(
        result["foreign_arrivals"], result["total_arrivals"], 100
    )
    result["avg_stay_nights_recalculated"] = safe_ratio(
        result["total_overnights"], result["total_arrivals"]
    )
    result["foreign_share_difference_pp"] = (
        result["derived_foreign_arrival_share_pct"] - result["foreign_share_pct"]
    )
    result["avg_stay_difference"] = (
        result["derived_avg_stay_nights"] - result["avg_stay_nights_recalculated"]
    )
    return result


def bodrum_vs_mugla_profile(
    bodrum: pd.DataFrame, mugla_2025: pd.Series
) -> pd.DataFrame:
    """Ortak konaklama metrikleri üzerinde Bodrum'un Muğla 2025 payını hesaplar."""

    row = bodrum.iloc[0]
    output = {
        "year": 2025,
        "definition_comparable": True,
        "comparison_scope_note": "Shared official accommodation-statistics scope; district versus province.",
    }
    for column in CORE_FLOW_COLUMNS:
        output[f"bodrum_{column}"] = row[column]
        output[f"mugla_{column}"] = mugla_2025[column]
        output[f"bodrum_share_of_mugla_{column}_pct"] = (
            100 * row[column] / mugla_2025[column]
            if mugla_2025[column] > 0
            else np.nan
        )
    output["bodrum_domestic_share_pct"] = row["domestic_share_pct"]
    output["mugla_domestic_share_pct"] = (
        100 * mugla_2025["domestic_arrivals"] / mugla_2025["total_arrivals"]
    )
    output["bodrum_foreign_share_pct"] = row["foreign_share_pct"]
    output["mugla_foreign_share_pct"] = (
        100 * mugla_2025["foreign_arrivals"] / mugla_2025["total_arrivals"]
    )
    output["bodrum_avg_stay_nights"] = row["avg_stay_nights_recalculated"]
    output["mugla_avg_stay_nights"] = (
        mugla_2025["total_overnights"] / mugla_2025["total_arrivals"]
    )
    return pd.DataFrame([output])


def monthly_correlations(
    monthly_features: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """n=12 için Pearson ve Spearman çiftlerini ayrı satırlarda üretir."""

    rows = []
    for index, x in enumerate(columns):
        for y in columns[index + 1 :]:
            pair = monthly_features[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            for method in ("pearson", "spearman"):
                if len(pair) >= 3 and pair[x].nunique() > 1 and pair[y].nunique() > 1:
                    if method == "pearson":
                        coefficient, p_value = stats.pearsonr(pair[x], pair[y])
                    else:
                        coefficient, p_value = stats.spearmanr(pair[x], pair[y])
                else:
                    coefficient, p_value = np.nan, np.nan
                rows.append(
                    {
                        "metric_x": x,
                        "metric_y": y,
                        "method": method,
                        "n": len(pair),
                        "coefficient": coefficient,
                        "p_value": p_value,
                        "interpretation_note": "Exploratory monthly association; n=12, correlation is not causation.",
                    }
                )
    return pd.DataFrame(rows)
