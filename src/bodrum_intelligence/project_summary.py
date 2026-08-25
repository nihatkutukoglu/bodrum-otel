"""Project-level synthesis helpers used by the final intelligence notebook."""

from __future__ import annotations

from numbers import Real

import pandas as pd


def format_number_tr(value: Real, decimals: int = 0) -> str:
    """Format a number with Turkish thousands and decimal separators."""

    if pd.isna(value):
        return "—"
    formatted = f"{float(value):,.{decimals}f}"
    return formatted.translate(str.maketrans({",": ".", ".": ","}))


def format_pct_tr(value: Real, decimals: int = 1) -> str:
    """Format an already-percent-scaled value in Turkish notation."""

    return f"%{format_number_tr(value, decimals)}"


def interpret_spearman(rho: Real) -> str:
    """Return a cautious plain-language description of a Spearman coefficient."""

    if pd.isna(rho):
        return "Yeterli gözlem yok."
    magnitude = abs(float(rho))
    if magnitude >= 0.8:
        strength = "çok güçlü"
    elif magnitude >= 0.6:
        strength = "güçlü"
    elif magnitude >= 0.4:
        strength = "orta"
    elif magnitude >= 0.2:
        strength = "zayıf"
    else:
        strength = "çok zayıf"
    direction = "aynı yönlü" if float(rho) >= 0 else "ters yönlü"
    return f"{strength.capitalize()}, {direction} sıralı birliktelik; nedensellik göstermez."


def consistency_row(
    metric: str,
    notebook_report_value: Real,
    recomputed_value: Real,
    tolerance: float = 1e-9,
) -> dict[str, object]:
    """Build one transparent KPI consistency-check record."""

    report_value = float(notebook_report_value)
    calculated_value = float(recomputed_value)
    difference = calculated_value - report_value
    return {
        "metric": metric,
        "notebook_report_value": report_value,
        "recomputed_value": calculated_value,
        "difference": difference,
        "status": "PASS" if abs(difference) <= tolerance else "REVIEW",
    }
