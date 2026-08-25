"""Milas-Bodrum Airport ve Muğla turizm aylık serileri için ortak analiz fonksiyonları."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


AIRPORT_SOURCE_COLUMNS = {
    "monthly_domestic_passengers": "airport_domestic_passengers",
    "monthly_international_passengers": "airport_international_passengers",
    "monthly_total_passengers": "airport_total_passengers",
}

TOURISM_SOURCE_COLUMNS = {
    "domestic_arrivals": "tourism_domestic_arrivals",
    "foreign_arrivals": "tourism_foreign_arrivals",
    "total_arrivals": "tourism_total_arrivals",
    "total_overnights": "tourism_total_overnights",
    "occupancy_rate_pct": "tourism_occupancy_rate_pct",
    "derived_avg_stay_nights_recalculated": "tourism_avg_stay_nights",
    "derived_foreign_arrival_share_pct_recalculated": "tourism_foreign_share_pct",
}


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Eksik zorunlu kolonlar: {missing}")


def safe_ratio(numerator: pd.Series, denominator: pd.Series, scale: float = 1.0) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return (scale * numerator / denominator.where(denominator.gt(0))).replace(
        [np.inf, -np.inf], np.nan
    )


def airport_quality_audit(airport: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """12 aylık passenger muhasebesi ve cumulative fark kontrolleri."""

    require_columns(
        airport,
        [
            "period",
            *AIRPORT_SOURCE_COLUMNS,
            "cumulative_domestic_passengers",
            "cumulative_international_passengers",
            "cumulative_total_passengers",
            "monthly_international_share_pct",
        ],
    )
    ordered = airport.sort_values("period").reset_index(drop=True).copy()
    ordered["passenger_total_difference"] = (
        ordered["monthly_domestic_passengers"]
        + ordered["monthly_international_passengers"]
        - ordered["monthly_total_passengers"]
    )
    cumulative_pairs = [
        ("domestic", "cumulative_domestic_passengers", "monthly_domestic_passengers"),
        ("international", "cumulative_international_passengers", "monthly_international_passengers"),
        ("total", "cumulative_total_passengers", "monthly_total_passengers"),
    ]
    for label, cumulative, monthly in cumulative_pairs:
        expected = ordered[cumulative].diff()
        expected.iloc[0] = ordered.loc[0, cumulative]
        ordered[f"derived_{label}_from_cumulative"] = expected
        ordered[f"cumulative_{label}_monthly_difference"] = expected - ordered[monthly]
    ordered["international_share_recalculated_pct"] = safe_ratio(
        ordered["monthly_international_passengers"],
        ordered["monthly_total_passengers"],
        100,
    )
    ordered["international_share_difference_pp"] = (
        ordered["monthly_international_share_pct"]
        - ordered["international_share_recalculated_pct"]
    )
    numeric = ordered[
        [
            *AIRPORT_SOURCE_COLUMNS,
            "cumulative_domestic_passengers",
            "cumulative_international_passengers",
            "cumulative_total_passengers",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    checks = pd.DataFrame(
        [
            ("row_count", len(ordered), "PASS" if len(ordered) == 12 else "FAIL"),
            ("unique_period", ordered["period"].nunique(), "PASS" if ordered["period"].is_unique else "FAIL"),
            ("missing_cells", int(ordered.isna().sum().sum()), "PASS" if not ordered.isna().any().any() else "REVIEW"),
            ("negative_passenger_values", int(numeric.lt(0).sum().sum()), "PASS" if not numeric.lt(0).any().any() else "FAIL"),
            ("monthly_passenger_total_mismatch_rows", int(ordered["passenger_total_difference"].ne(0).sum()), "PASS" if ordered["passenger_total_difference"].eq(0).all() else "FAIL"),
            ("cumulative_domestic_mismatch_rows", int(ordered["cumulative_domestic_monthly_difference"].ne(0).sum()), "PASS" if ordered["cumulative_domestic_monthly_difference"].eq(0).all() else "FAIL"),
            ("cumulative_international_mismatch_rows", int(ordered["cumulative_international_monthly_difference"].ne(0).sum()), "PASS" if ordered["cumulative_international_monthly_difference"].eq(0).all() else "FAIL"),
            ("cumulative_total_mismatch_rows", int(ordered["cumulative_total_monthly_difference"].ne(0).sum()), "PASS" if ordered["cumulative_total_monthly_difference"].eq(0).all() else "FAIL"),
            ("annual_sum_minus_year_end_cumulative", int(ordered["monthly_total_passengers"].sum() - ordered.iloc[-1]["cumulative_total_passengers"]), "PASS" if ordered["monthly_total_passengers"].sum() == ordered.iloc[-1]["cumulative_total_passengers"] else "FAIL"),
        ],
        columns=["check", "value", "status"],
    )
    return checks, ordered


def period_coverage(airport: pd.DataFrame, tourism: pd.DataFrame) -> pd.DataFrame:
    airport_periods = set(airport["period"].astype(str))
    tourism_periods = set(tourism["period"].astype(str))
    return pd.DataFrame(
        [
            ("airport_period_n", len(airport_periods)),
            ("tourism_period_n", len(tourism_periods)),
            ("matched_period_n", len(airport_periods & tourism_periods)),
            ("airport_only_periods", ";".join(sorted(airport_periods - tourism_periods))),
            ("tourism_only_periods", ";".join(sorted(tourism_periods - airport_periods))),
        ],
        columns=["metric", "value"],
    )


def mean_index(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mean_value = numeric.mean()
    if pd.isna(mean_value) or mean_value <= 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return 100 * numeric / mean_value


def build_joint_monthly(airport: pd.DataFrame, tourism: pd.DataFrame) -> pd.DataFrame:
    """Period coverage kontrolünden sonra 12 aylık ortak tablo ve proxy metrikleri üretir."""

    require_columns(airport, ["period", "year", "month_name_tr", *AIRPORT_SOURCE_COLUMNS])
    require_columns(tourism, ["period", "year", "month_name_tr", *TOURISM_SOURCE_COLUMNS, "season_group"])
    coverage = period_coverage(airport, tourism).set_index("metric")["value"]
    if int(coverage["matched_period_n"]) != 12 or coverage["airport_only_periods"] or coverage["tourism_only_periods"]:
        raise ValueError(f"Period coverage 12/12 değil: {coverage.to_dict()}")
    airport_selected = airport[
        ["period", "year", "month_name_tr", *AIRPORT_SOURCE_COLUMNS]
    ].rename(columns=AIRPORT_SOURCE_COLUMNS)
    tourism_selected = tourism[
        ["period", *TOURISM_SOURCE_COLUMNS, "season_group"]
    ].rename(columns=TOURISM_SOURCE_COLUMNS)
    result = airport_selected.merge(
        tourism_selected, on="period", how="inner", validate="one_to_one"
    ).sort_values("period").reset_index(drop=True)
    result["airport_monthly_share_pct"] = safe_ratio(
        result["airport_total_passengers"],
        pd.Series(result["airport_total_passengers"].sum(), index=result.index),
        100,
    )
    result["airport_international_share_pct"] = safe_ratio(
        result["airport_international_passengers"], result["airport_total_passengers"], 100
    )
    result["airport_domestic_share_pct"] = safe_ratio(
        result["airport_domestic_passengers"], result["airport_total_passengers"], 100
    )
    index_columns = {
        "airport_total_passengers": "airport_total_index",
        "airport_domestic_passengers": "airport_domestic_index",
        "airport_international_passengers": "airport_international_index",
        "tourism_total_arrivals": "tourism_total_arrivals_index",
        "tourism_domestic_arrivals": "tourism_domestic_index",
        "tourism_foreign_arrivals": "tourism_foreign_index",
        "tourism_total_overnights": "tourism_overnights_index",
        "tourism_occupancy_rate_pct": "tourism_occupancy_index",
    }
    for source, output in index_columns.items():
        result[output] = mean_index(result[source])
    result["share_gap_pp"] = (
        result["airport_international_share_pct"] - result["tourism_foreign_share_pct"]
    )
    result["tourism_arrivals_per_airport_passenger_proxy_ratio"] = safe_ratio(
        result["tourism_total_arrivals"], result["airport_total_passengers"]
    )
    result["foreign_arrivals_per_international_airport_passenger_proxy_ratio"] = safe_ratio(
        result["tourism_foreign_arrivals"], result["airport_international_passengers"]
    )
    result["domestic_arrivals_per_domestic_airport_passenger_proxy_ratio"] = safe_ratio(
        result["tourism_domestic_arrivals"], result["airport_domestic_passengers"]
    )
    result["total_index_gap"] = (
        result["airport_total_index"] - result["tourism_total_arrivals_index"]
    )
    result["international_foreign_index_gap"] = (
        result["airport_international_index"] - result["tourism_foreign_index"]
    )
    return result


def airport_seasonality_metrics(joint: pd.DataFrame) -> pd.DataFrame:
    total = joint["airport_total_passengers"]
    international = joint["airport_international_passengers"]
    domestic = joint["airport_domestic_passengers"]
    shares = total / total.sum()
    return pd.DataFrame(
        [
            {
                "year": 2025,
                "monthly_observation_n": len(joint),
                "airport_annual_total_passengers": total.sum(),
                "airport_peak_to_low_ratio": total.max() / total.min(),
                "airport_top3_month_share_pct": 100 * total.nlargest(3).sum() / total.sum(),
                "airport_top4_month_share_pct": 100 * total.nlargest(4).sum() / total.sum(),
                "airport_total_cv": total.std(ddof=1) / total.mean(),
                "airport_international_cv": international.std(ddof=1) / international.mean(),
                "airport_domestic_cv": domestic.std(ddof=1) / domestic.mean(),
                "airport_hhi_seasonality": float((shares**2).sum()),
            }
        ]
    )


def correlation_pair(frame: pd.DataFrame, x: str, y: str) -> dict[str, float | int | str]:
    pair = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
        return {
            "metric_x": x,
            "metric_y": y,
            "n": len(pair),
            "pearson_r": np.nan,
            "pearson_p_value": np.nan,
            "spearman_rho": np.nan,
            "spearman_p_value": np.nan,
        }
    pearson_r, pearson_p = stats.pearsonr(pair[x], pair[y])
    spearman_r, spearman_p = stats.spearmanr(pair[x], pair[y])
    return {
        "metric_x": x,
        "metric_y": y,
        "n": len(pair),
        "pearson_r": pearson_r,
        "pearson_p_value": pearson_p,
        "spearman_rho": spearman_r,
        "spearman_p_value": spearman_p,
    }


def peak_alignment(joint: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "airport_total": "airport_total_passengers",
        "airport_domestic": "airport_domestic_passengers",
        "airport_international": "airport_international_passengers",
        "tourism_total_arrivals": "tourism_total_arrivals",
        "tourism_domestic_arrivals": "tourism_domestic_arrivals",
        "tourism_foreign_arrivals": "tourism_foreign_arrivals",
        "tourism_overnights": "tourism_total_overnights",
        "tourism_occupancy": "tourism_occupancy_rate_pct",
    }
    rows = []
    for label, column in metrics.items():
        index = joint[column].idxmax()
        rows.append(
            {
                "metric": label,
                "source_column": column,
                "peak_period": joint.loc[index, "period"],
                "peak_month": joint.loc[index, "month_name_tr"],
                "peak_month_number": int(str(joint.loc[index, "period"])[-2:]),
                "peak_value": joint.loc[index, column],
            }
        )
    result = pd.DataFrame(rows)
    airport_total_month = int(
        result.loc[result["metric"].eq("airport_total"), "peak_month_number"].iloc[0]
    )
    airport_international_month = int(
        result.loc[result["metric"].eq("airport_international"), "peak_month_number"].iloc[0]
    )
    result["month_difference_vs_airport_total_peak"] = abs(
        result["peak_month_number"] - airport_total_month
    )
    result["month_difference_vs_airport_international_peak"] = abs(
        result["peak_month_number"] - airport_international_month
    )
    return result


def lag_correlations(joint: pd.DataFrame) -> pd.DataFrame:
    """Airport_t ile tourism_t ve tourism_t+1 hizalamasını yalnız lag 0/1 için hesaplar."""

    pairs = [
        ("total airport vs total tourism", "airport_total_passengers", "tourism_total_arrivals"),
        ("international airport vs foreign tourism", "airport_international_passengers", "tourism_foreign_arrivals"),
        ("domestic airport vs domestic tourism", "airport_domestic_passengers", "tourism_domestic_arrivals"),
    ]
    rows = []
    ordered = joint.sort_values("period").reset_index(drop=True)
    for label, airport_column, tourism_column in pairs:
        for lag in (0, 1):
            aligned = pd.DataFrame(
                {
                    "airport": ordered[airport_column],
                    "tourism": ordered[tourism_column].shift(-lag),
                }
            ).dropna()
            result = correlation_pair(aligned, "airport", "tourism")
            rows.append(
                {
                    "metric_pair": label,
                    "airport_metric": airport_column,
                    "tourism_metric": tourism_column,
                    "lag_months": lag,
                    "alignment": "airport_t vs tourism_t" if lag == 0 else "airport_t vs tourism_t+1",
                    "n": result["n"],
                    "pearson_r": result["pearson_r"],
                    "spearman_rho": result["spearman_rho"],
                    "p_value": result["spearman_p_value"],
                    "note": "Exploratory only; lag is not predictive or causal.",
                }
            )
    return pd.DataFrame(rows)


def season_summary(joint: pd.DataFrame) -> pd.DataFrame:
    return (
        joint.groupby("season_group", observed=True)
        .agg(
            month_count=("period", "size"),
            avg_airport_total=("airport_total_passengers", "mean"),
            avg_airport_domestic=("airport_domestic_passengers", "mean"),
            avg_airport_international=("airport_international_passengers", "mean"),
            avg_airport_international_share_pct=("airport_international_share_pct", "mean"),
            avg_tourism_arrivals=("tourism_total_arrivals", "mean"),
            avg_foreign_arrivals=("tourism_foreign_arrivals", "mean"),
            avg_overnights=("tourism_total_overnights", "mean"),
            avg_occupancy=("tourism_occupancy_rate_pct", "mean"),
        )
        .reset_index()
    )


def cross_domain_correlations(joint: pd.DataFrame) -> pd.DataFrame:
    airport_columns = [
        "airport_domestic_passengers",
        "airport_international_passengers",
        "airport_total_passengers",
    ]
    tourism_columns = [
        "tourism_domestic_arrivals",
        "tourism_foreign_arrivals",
        "tourism_total_arrivals",
        "tourism_total_overnights",
        "tourism_occupancy_rate_pct",
    ]
    rows = []
    for airport_column in airport_columns:
        for tourism_column in tourism_columns:
            result = correlation_pair(joint, airport_column, tourism_column)
            result["note"] = "Cross-domain co-movement only; n=12, not causal."
            rows.append(result)
    return pd.DataFrame(rows)
