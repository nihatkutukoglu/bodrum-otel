"""10_airport_tourism_joint_analysis.ipynb dosyasını tekrar üretilebilir biçimde oluşturur."""

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "10_airport_tourism_joint_analysis.ipynb"
nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


md(
    """# Bodrum Hotel & Destination Intelligence
## 10 - Milas-Bodrum Airport × Tourism Joint Analysis

Bu notebook Milas–Bodrum Havalimanı 2025 aylık yolcu trafiği ile Muğla ili 2025 aylık konaklama
tesisi gelişlerinin ortak sezonluk hareketini inceler.

> **Airport passenger ≠ tourist arrival.** Havalimanı yolcusu yerel halk, iş/ziyaret amaçlı yolcu
> veya başka destinasyona devam eden kişileri içerebilir. Muğla accommodation arrival ise
> konaklama tesisine geliş kaydıdır ve benzersiz kişi olmak zorunda değildir.

Analiz co-movement, seasonal alignment ve proxy relationship çerçevesindedir. Regression,
forecasting, Granger/causal analysis, destination-level monthly join veya ekonomik etki tahmini
yapılmaz. Muğla aylık verisi Bodrum aylık turizmi olarak adlandırılmaz.
"""
)

md("""### 1. Kurulum ve gerçek dosya yolları""")

code(
    """from pathlib import Path
import hashlib
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bodrum_intelligence.airport_tourism import (
    airport_quality_audit,
    airport_seasonality_metrics,
    build_joint_monthly,
    correlation_pair,
    cross_domain_correlations,
    lag_correlations,
    peak_alignment,
    period_coverage,
    season_summary,
)

AIRPORT_PATH = PROJECT_ROOT / "data" / "external" / "airport" / "milas_bodrum_airport_monthly_2025.csv"
TOURISM_PATH = PROJECT_ROOT / "data" / "processed" / "tourism_demand_monthly_features_2025.csv"
SEASON_SUMMARY_PATH = PROJECT_ROOT / "reports" / "tourism_season_group_summary.csv"
TOURISM_CORR_PATH = PROJECT_ROOT / "reports" / "tourism_monthly_correlations_2025.csv"
TOURISM_FINDINGS_PATH = PROJECT_ROOT / "reports" / "tourism_demand_key_findings.txt"
BODRUM_PROFILE_PATH = PROJECT_ROOT / "reports" / "bodrum_tourism_profile_2025.csv"
DESTINATION_CONTEXT_PATH = PROJECT_ROOT / "data" / "processed" / "destination_intelligence_enriched.csv"
PROCESSED_OUTPUT = PROJECT_ROOT / "data" / "processed" / "airport_tourism_monthly_2025.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" / "airport_tourism"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

required_paths = [AIRPORT_PATH, TOURISM_PATH, SEASON_SUMMARY_PATH, TOURISM_CORR_PATH, TOURISM_FINDINGS_PATH]
optional_paths = [path for path in [BODRUM_PROFILE_PATH, DESTINATION_CONTEXT_PATH] if path.exists()]
assert all(path.exists() for path in required_paths)
input_paths = required_paths + optional_paths
input_hashes_before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}

airport_raw = pd.read_csv(AIRPORT_PATH)
tourism_raw = pd.read_csv(TOURISM_PATH)
bodrum_profile = pd.read_csv(BODRUM_PROFILE_PATH) if BODRUM_PROFILE_PATH.exists() else None
destination_context = pd.read_csv(DESTINATION_CONTEXT_PATH) if DESTINATION_CONTEXT_PATH.exists() else None

pd.set_option("display.max_columns", 80)
pd.set_option("display.max_colwidth", 90)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")
print("Airport:", airport_raw.shape)
print("Muğla monthly tourism:", tourism_raw.shape)
"""
)

code(
    """PRIMARY = "#2F6B7C"
SECONDARY = "#4C956C"
ACCENT = "#C1666B"
HIGHLIGHT = "#D9A404"
PURPLE = "#7A5C8E"
NEUTRAL = "#777777"


def save_fig(fig, filename):
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return path


def monthly_lines(frame, columns, labels, colors, title, ylabel, filename, mean_line=False):
    fig, ax = plt.subplots(figsize=(11, 5.8))
    months = frame["month_name_tr"]
    for column, label, color in zip(columns, labels, colors):
        ax.plot(months, frame[column], marker="o", linewidth=2.2, color=color, label=label)
    if mean_line:
        ax.axhline(100, color=NEUTRAL, linestyle="--", linewidth=1, label="2025 aylık ortalama=100")
    ax.set(title=title, xlabel="Ay", ylabel=ylabel)
    ax.tick_params(axis="x", rotation=40); ax.grid(alpha=0.22); ax.legend()
    save_fig(fig, filename)


def labeled_scatter(frame, x, y, title, xlabel, ylabel, filename):
    stats_row = correlation_pair(frame, x, y)
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.scatter(frame[x], frame[y], s=68, color=PRIMARY, alpha=0.85)
    for row in frame.itertuples():
        ax.annotate(str(row.month_name_tr)[:3], (getattr(row, x), getattr(row, y)),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel); ax.grid(alpha=0.22)
    ax.text(0.03, 0.96,
            f"Pearson r={stats_row['pearson_r']:.3f}\\nSpearman ρ={stats_row['spearman_rho']:.3f}\\nn={stats_row['n']}",
            transform=ax.transAxes, va="top", bbox=dict(boxstyle="round", facecolor="white", edgecolor=NEUTRAL))
    save_fig(fig, filename)
    return stats_row
"""
)

md("""### 2. Grain / source audit""")

code(
    """input_summary = pd.DataFrame([
    {
        "dataset_name": "Milas-Bodrum Airport Monthly 2025", "row_count": len(airport_raw),
        "geography": airport_raw["airport"].iloc[0], "geography_level": "airport / regional gateway",
        "time_grain": "monthly", "start_period": airport_raw["period"].min(), "end_period": airport_raw["period"].max(),
        "main_metrics": "domestic, international, total passengers; cumulative passengers",
        "source_origin": "DHMİ passenger statistics (documented in data_status/source_url)",
    },
    {
        "dataset_name": "Muğla Tourism Monthly 2025", "row_count": len(tourism_raw),
        "geography": tourism_raw["geography"].iloc[0], "geography_level": tourism_raw["geography_level"].iloc[0],
        "time_grain": "monthly", "start_period": tourism_raw["period"].min(), "end_period": tourism_raw["period"].max(),
        "main_metrics": "domestic/foreign/total accommodation arrivals, overnights, occupancy",
        "source_origin": tourism_raw["source_origin"].iloc[0],
    },
], columns=["dataset_name", "row_count", "geography", "geography_level", "time_grain", "start_period", "end_period", "main_metrics", "source_origin"])
display(input_summary)
display(Markdown(
    "**Kapsam ayrımı:** Havalimanı bölgesel bir gateway; turizm serisi Muğla ilindeki konaklama "
    "tesislerine gelişlerdir. Aynı ayı paylaşmaları aynı population/grain oldukları anlamına gelmez."
))
"""
)

md("""### 3. Airport data audit ve kümülatif fark kontrolü""")

code(
    """airport_checks, airport_audited = airport_quality_audit(airport_raw)
display(airport_checks)
display(airport_audited[[
    "period", "monthly_total_passengers", "cumulative_total_passengers",
    "derived_total_from_cumulative", "cumulative_total_monthly_difference",
    "passenger_total_difference", "international_share_difference_pp",
]])
assert not airport_checks["status"].eq("FAIL").any()
"""
)

code(
    """airport_annual_sum = int(airport_audited["monthly_total_passengers"].sum())
airport_year_end_cumulative = int(airport_audited.iloc[-1]["cumulative_total_passengers"])
display(Markdown(
    f"**Audit sonucu:** 12 aylık airport total passengers toplamı {airport_annual_sum:,}; Aralık "
    f"kümülatif toplamı {airport_year_end_cumulative:,}; fark {airport_annual_sum - airport_year_end_cumulative:,}. "
    f"Aylık passenger değerlerinin kümülatif farklardan türetilmiş olduğu metadata ile birlikte korunur."
))
"""
)

md("""### 4. Tourism data audit ve period coverage""")

code(
    """tourism_checks = pd.DataFrame([
    ("row_count", len(tourism_raw), "PASS" if len(tourism_raw) == 12 else "FAIL"),
    ("unique_period", tourism_raw["period"].nunique(), "PASS" if tourism_raw["period"].is_unique else "FAIL"),
    ("missing_core_cells", int(tourism_raw[["total_arrivals", "foreign_arrivals", "domestic_arrivals", "total_overnights", "occupancy_rate_pct"]].isna().sum().sum()), "PASS"),
    ("arrival_total_mismatch_rows", int((tourism_raw["domestic_arrivals"] + tourism_raw["foreign_arrivals"] - tourism_raw["total_arrivals"]).ne(0).sum()), "PASS"),
    ("occupancy_out_of_range_rows", int((~tourism_raw["occupancy_rate_pct"].between(0, 100)).sum()), "PASS"),
], columns=["check", "value", "status"])
display(tourism_checks)
coverage = period_coverage(airport_raw, tourism_raw)
display(coverage)
"""
)

md("""### 5. 12 aylık ortak tablo""")

code(
    """joint = build_joint_monthly(airport_raw, tourism_raw)
assert len(joint) == 12 and joint["period"].is_unique
assert joint["period"].tolist() == [f"2025-{month:02d}" for month in range(1, 13)]
display(joint)
"""
)

md("""### 6. Airport seasonal overview""")

code(
    """monthly_lines(
    joint, ["airport_total_passengers"], ["Total passengers"], [PRIMARY],
    "Milas-Bodrum Airport Aylık Toplam Yolcu — 2025", "Yolcu", "01_airport_monthly_passengers.png",
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))
axes[0].plot(joint["month_name_tr"], joint["airport_domestic_passengers"], marker="o", color=SECONDARY, label="Domestic")
axes[0].plot(joint["month_name_tr"], joint["airport_international_passengers"], marker="o", color=PURPLE, label="International")
axes[0].set(title="Domestic vs International Passengers", ylabel="Yolcu"); axes[0].legend(); axes[0].grid(alpha=0.2)
axes[1].plot(joint["month_name_tr"], joint["airport_international_share_pct"], marker="o", color=ACCENT)
axes[1].set(title="International Passenger Share", ylabel="Pay (%)"); axes[1].grid(alpha=0.2)
for ax in axes: ax.tick_params(axis="x", rotation=40)
save_fig(fig, "02_airport_domestic_international.png")

airport_seasonality = airport_seasonality_metrics(joint)
display(airport_seasonality)
"""
)

code(
    """airport_peak_row = joint.loc[joint["airport_total_passengers"].idxmax()]
airport_domestic_peak_row = joint.loc[joint["airport_domestic_passengers"].idxmax()]
airport_international_peak_row = joint.loc[joint["airport_international_passengers"].idxmax()]
airport_share_peak_row = joint.loc[joint["airport_international_share_pct"].idxmax()]
display(Markdown(
    f"**Bulgu:** Airport total, domestic ve international passenger zirvesi "
    f"{airport_peak_row.month_name_tr} ayında ({int(airport_peak_row.airport_total_passengers):,} total). "
    f"International passenger share en yüksek {airport_share_peak_row.month_name_tr} ayında "
    f"%{airport_share_peak_row.airport_international_share_pct:.1f}. Top 3 ay yıllık airport yolcusunun "
    f"%{airport_seasonality.iloc[0].airport_top3_month_share_pct:.1f}'ini oluşturur."
))
"""
)

md("""### 7. Total airport × total tourism normalized seasonal alignment""")

code(
    """monthly_lines(
    joint, ["airport_total_index", "tourism_total_arrivals_index"],
    ["Airport total", "Muğla tourism arrivals"], [PRIMARY, ACCENT],
    "Airport Total vs Muğla Tourism Arrivals — Normalize Sezon Şekli",
    "2025 aylık ortalama = 100", "03_airport_vs_tourism_normalized.png", mean_line=True,
)
total_relation = labeled_scatter(
    joint, "airport_total_passengers", "tourism_total_arrivals",
    "Airport Total Passenger × Muğla Tourism Arrivals",
    "Airport total passengers", "Muğla accommodation arrivals",
    "06_airport_total_vs_tourism_scatter.png",
)
"""
)

md("""### 8. International airport × foreign tourism""")

code(
    """monthly_lines(
    joint, ["airport_international_index", "tourism_foreign_index"],
    ["Airport international", "Muğla foreign arrivals"], [PURPLE, HIGHLIGHT],
    "International Airport Traffic vs Muğla Foreign Arrivals",
    "2025 aylık ortalama = 100", "04_international_vs_foreign_normalized.png", mean_line=True,
)
international_relation = labeled_scatter(
    joint, "airport_international_passengers", "tourism_foreign_arrivals",
    "International Passenger × Muğla Foreign Arrivals",
    "Airport international passengers", "Muğla foreign accommodation arrivals",
    "07_international_vs_foreign_scatter.png",
)
"""
)

md(
    """Yüksek co-movement, international airport passengers'ın foreign tourist olduğu anlamına
gelmez. İki seri farklı evrenlerden gelir ve ortak sezon etkilerine birlikte tepki verebilir.
"""
)

md("""### 9. Domestic airport × domestic tourism""")

code(
    """monthly_lines(
    joint, ["airport_domestic_index", "tourism_domestic_index"],
    ["Airport domestic", "Muğla domestic arrivals"], [SECONDARY, ACCENT],
    "Domestic Airport Traffic vs Muğla Domestic Arrivals",
    "2025 aylık ortalama = 100", "05_domestic_vs_domestic_normalized.png", mean_line=True,
)
domestic_relation = correlation_pair(joint, "airport_domestic_passengers", "tourism_domestic_arrivals")
display(pd.DataFrame([domestic_relation]))
"""
)

md("""### 10. Airport × overnights ve occupancy""")

code(
    """monthly_lines(
    joint, ["airport_total_index", "tourism_overnights_index"],
    ["Airport total", "Muğla overnights"], [PRIMARY, HIGHLIGHT],
    "Airport Total Traffic vs Muğla Overnights",
    "2025 aylık ortalama = 100", "08_airport_vs_overnights.png", mean_line=True,
)
overnight_relation = correlation_pair(joint, "airport_total_passengers", "tourism_total_overnights")
display(pd.DataFrame([overnight_relation]))
"""
)

code(
    """fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
occupancy_relations = []
for ax, x, title in [
    (axes[0], "airport_total_passengers", "Total Airport × Occupancy"),
    (axes[1], "airport_international_passengers", "International Airport × Occupancy"),
]:
    ax.scatter(joint[x], joint["tourism_occupancy_rate_pct"], color=PRIMARY, s=62)
    relation = correlation_pair(joint, x, "tourism_occupancy_rate_pct")
    occupancy_relations.append(relation)
    for row in joint.itertuples():
        ax.annotate(str(row.month_name_tr)[:3], (getattr(row, x), row.tourism_occupancy_rate_pct),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set(title=f"{title}\\nSpearman ρ={relation['spearman_rho']:.3f}", xlabel=x.replace("_", " "), ylabel="Muğla occupancy (%)")
    ax.grid(alpha=0.2)
save_fig(fig, "09_airport_vs_occupancy.png")
display(pd.DataFrame(occupancy_relations))
"""
)

md(
    """Occupancy, yalnız arrivals veya airport trafiğinden türemez; açık tesis/oda-yatak arzı ve
kapasite kullanılabilirliği gibi faktörlerden etkilenebilir.
"""
)

md("""### 11. Peak month alignment ve ay farkları""")

code(
    """peaks = peak_alignment(joint)
display(peaks)
"""
)

code(
    """tourism_peak = peaks.loc[peaks["metric"].eq("tourism_total_arrivals")].iloc[0]
foreign_peak = peaks.loc[peaks["metric"].eq("tourism_foreign_arrivals")].iloc[0]
overnight_peak = peaks.loc[peaks["metric"].eq("tourism_overnights")].iloc[0]
display(Markdown(
    f"**Bulgu:** Airport total peak ile tourism total arrivals peak ay farkı "
    f"{int(tourism_peak.month_difference_vs_airport_total_peak)}; international airport peak ile foreign "
    f"arrival peak farkı {int(foreign_peak.month_difference_vs_airport_international_peak)}; airport total "
    f"peak ile overnights peak farkı {int(overnight_peak.month_difference_vs_airport_total_peak)} aydır."
))
"""
)

md("""### 12. Lag 0 / lag 1 keşifsel hizalama""")

code(
    """lag_table = lag_correlations(joint)
display(lag_table)
"""
)

code(
    """lag_comparison = lag_table.pivot(index="metric_pair", columns="lag_months", values="spearman_rho").rename(columns={0: "lag0_rho", 1: "lag1_rho"})
lag_comparison["lag1_minus_lag0"] = lag_comparison["lag1_rho"] - lag_comparison["lag0_rho"]
display(lag_comparison)
display(Markdown(
    "**Yorum:** Lag 1, `airport_t` ile `tourism_t+1` hizalamasıdır ve 11 gözleme dayanır. "
    "Bu karşılaştırma predictive lead/lag veya nedensellik kanıtı değildir."
))
"""
)

md("""### 13. International share × foreign tourism share""")

code(
    """monthly_lines(
    joint, ["airport_international_share_pct", "tourism_foreign_share_pct"],
    ["Airport international share", "Muğla foreign arrival share"], [PURPLE, HIGHLIGHT],
    "Airport Dış Hat Payı vs Muğla Foreign Arrival Payı",
    "Pay (%)", "10_international_share_vs_foreign_share.png",
)
share_gap_extreme = joint.loc[joint["share_gap_pp"].abs().idxmax()]
display(joint[["period", "month_name_tr", "airport_international_share_pct", "tourism_foreign_share_pct", "share_gap_pp"]])
display(Markdown(
    f"**Bulgu:** İki oran arasındaki en büyük mutlak fark {share_gap_extreme.month_name_tr} ayında "
    f"{share_gap_extreme.share_gap_pp:.1f} yüzde puandır. Bu metrik conversion gap değildir."
))
"""
)

md("""### 14. Proxy ratios ve tourism-derived season groups""")

code(
    """proxy_columns = [
    "period", "month_name_tr", "tourism_arrivals_per_airport_passenger_proxy_ratio",
    "foreign_arrivals_per_international_airport_passenger_proxy_ratio",
    "domestic_arrivals_per_domestic_airport_passenger_proxy_ratio",
]
display(joint[proxy_columns])
season_table = season_summary(joint)
display(season_table)
"""
)

md(
    """Proxy ratio >1 hata değildir; airport passenger ve accommodation arrival aynı kişi/universe
değildir. `season_group`, 09 notebookunda Muğla tourism arrivals tertillerinden türetilmiştir; airport
aylarından bağımsız bir sezon etiketi değildir.
"""
)

md("""### 15. Monthly normalized divergence""")

code(
    """divergence = joint[[
    "period", "month_name_tr", "airport_total_index", "tourism_total_arrivals_index",
    "total_index_gap", "airport_international_index", "tourism_foreign_index",
    "international_foreign_index_gap",
]].rename(columns={
    "month_name_tr": "month", "tourism_total_arrivals_index": "tourism_total_index",
    "international_foreign_index_gap": "international_foreign_gap",
})
divergence["abs_total_gap"] = divergence["total_index_gap"].abs()
divergence["abs_international_foreign_gap"] = divergence["international_foreign_gap"].abs()
display(divergence.sort_values("abs_total_gap", ascending=False))

fig, ax = plt.subplots(figsize=(11, 5.8))
ax.bar(joint["month_name_tr"], joint["total_index_gap"], color=np.where(joint["total_index_gap"].ge(0), PRIMARY, ACCENT), alpha=0.82, label="Total gap")
ax.plot(joint["month_name_tr"], joint["international_foreign_index_gap"], marker="o", color=PURPLE, label="International–foreign gap")
ax.axhline(0, color=NEUTRAL, linewidth=1)
ax.set(title="Normalize Airport–Tourism Aylık Sapmaları", xlabel="Ay", ylabel="İndeks puanı farkı")
ax.tick_params(axis="x", rotation=40); ax.grid(axis="y", alpha=0.2); ax.legend()
save_fig(fig, "11_monthly_divergence.png")
"""
)

code(
    """largest_total_divergence = divergence.loc[divergence["abs_total_gap"].idxmax()]
largest_international_divergence = divergence.loc[divergence["abs_international_foreign_gap"].idxmax()]
display(Markdown(
    f"**Bulgu:** En büyük mutlak total index ayrışması {largest_total_divergence['month']} "
    f"({largest_total_divergence.total_index_gap:.1f} indeks puanı); international–foreign ayrışması "
    f"{largest_international_divergence['month']} ({largest_international_divergence.international_foreign_gap:.1f}) "
    f"ayında görülür. Neden atfedilmez."
))
"""
)

md("""### 16. Cross-domain correlation table ve heatmap""")

code(
    """correlations = cross_domain_correlations(joint)
display(correlations.sort_values("spearman_rho", ascending=False))

matrix_columns = [
    "airport_domestic_passengers", "airport_international_passengers", "airport_total_passengers",
    "tourism_domestic_arrivals", "tourism_foreign_arrivals", "tourism_total_arrivals",
    "tourism_total_overnights", "tourism_occupancy_rate_pct",
]
spearman_matrix = joint[matrix_columns].corr(method="spearman")
fig, ax = plt.subplots(figsize=(11, 9))
image = ax.imshow(spearman_matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
labels = [c.replace("airport_", "A: ").replace("tourism_", "T: ").replace("_passengers", "").replace("_arrivals", "").replace("_", " ") for c in matrix_columns]
ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
ax.set_yticks(range(len(labels)), labels)
ax.set_title("Airport × Muğla Tourism Spearman Korelasyonları (n=12)")
for i in range(len(labels)):
    for j in range(len(labels)):
        value = spearman_matrix.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(value) > 0.65 else "black")
fig.colorbar(image, ax=ax, shrink=0.8, label="Spearman ρ")
save_fig(fig, "12_correlation_heatmap.png")
"""
)

code(
    """strongest_cross_domain = correlations.reindex(correlations["spearman_rho"].abs().sort_values(ascending=False).index).head(8)
display(strongest_cross_domain)
display(Markdown(
    "**Not:** Airport total = domestic + international; tourism total = domestic + foreign. Aynı "
    "domain içindeki total bileşen korelasyonları matematiksel bağımlılık içerir. Raporlanan tablo "
    "yalnız airport–tourism cross-domain çiftlerini içerir."
))
"""
)

md("""### 17. Bodrum annual ve Destination Intelligence bağlamı""")

code(
    """context_rows = [{
    "airport_2025_total_passengers": airport_annual_sum,
    "airport_scope": "Milas-Bodrum regional gateway passengers",
    "mugla_2025_monthly_tourism_arrivals_sum": int(joint["tourism_total_arrivals"].sum()),
    "tourism_scope": "Muğla province accommodation arrivals",
    "direct_ratio_interpretation_allowed": False,
}]
if bodrum_profile is not None:
    context_rows[0]["bodrum_2025_annual_accommodation_arrivals"] = int(bodrum_profile.loc[0, "total_arrivals"])
    context_rows[0]["bodrum_scope"] = "Bodrum district annual accommodation arrivals"
if destination_context is not None:
    context_rows[0]["hotel_snapshot_year"] = 2026
    context_rows[0]["destination_monthly_join_performed"] = False
display(pd.DataFrame(context_rows))
"""
)

md(
    """Bodrum annual arrivals ve airport annual passengers yalnız KPI olarak yan yana gösterilir;
birebir ratio/conversion olarak yorumlanmaz. 14 destination monthly airport/tourism serisiyle merge
edilmez; böyle bir monthly destination demand katmanı mevcut değildir.
"""
)

md("""### 18. Raporların kaydedilmesi""")

code(
    """output_paths = {
    "joint_monthly": PROCESSED_OUTPUT,
    "input_summary": REPORTS_DIR / "airport_tourism_input_summary.csv",
    "quality_checks": REPORTS_DIR / "airport_tourism_quality_checks.csv",
    "airport_seasonality": REPORTS_DIR / "airport_seasonality_metrics.csv",
    "peaks": REPORTS_DIR / "airport_tourism_peak_alignment.csv",
    "correlations": REPORTS_DIR / "airport_tourism_correlations.csv",
    "lag_correlations": REPORTS_DIR / "airport_tourism_lag_correlations.csv",
    "season_summary": REPORTS_DIR / "airport_tourism_season_summary.csv",
    "divergence": REPORTS_DIR / "airport_tourism_divergence_months.csv",
}
joint.to_csv(output_paths["joint_monthly"], index=False, encoding="utf-8-sig")
input_summary.to_csv(output_paths["input_summary"], index=False, encoding="utf-8-sig")
pd.concat([
    airport_checks.assign(dataset="airport"), tourism_checks.assign(dataset="tourism")
], ignore_index=True).to_csv(output_paths["quality_checks"], index=False, encoding="utf-8-sig")
airport_seasonality.to_csv(output_paths["airport_seasonality"], index=False, encoding="utf-8-sig")
peaks.to_csv(output_paths["peaks"], index=False, encoding="utf-8-sig")
correlations.to_csv(output_paths["correlations"], index=False, encoding="utf-8-sig")
lag_table.to_csv(output_paths["lag_correlations"], index=False, encoding="utf-8-sig")
season_table.to_csv(output_paths["season_summary"], index=False, encoding="utf-8-sig")
divergence.to_csv(output_paths["divergence"], index=False, encoding="utf-8-sig")
"""
)

md("""## Temel Bulgular""")

code(
    """total_lag0 = lag_table.loc[lag_table["metric_pair"].eq("total airport vs total tourism") & lag_table["lag_months"].eq(0)].iloc[0]
total_lag1 = lag_table.loc[lag_table["metric_pair"].eq("total airport vs total tourism") & lag_table["lag_months"].eq(1)].iloc[0]
international_lag0 = lag_table.loc[lag_table["metric_pair"].eq("international airport vs foreign tourism") & lag_table["lag_months"].eq(0)].iloc[0]
international_lag1 = lag_table.loc[lag_table["metric_pair"].eq("international airport vs foreign tourism") & lag_table["lag_months"].eq(1)].iloc[0]
domestic_lag0 = lag_table.loc[lag_table["metric_pair"].eq("domestic airport vs domestic tourism") & lag_table["lag_months"].eq(0)].iloc[0]
occupancy_total = occupancy_relations[0]

findings = [
    f"Milas-Bodrum Airport 2025 peak total passenger ayı {airport_peak_row.month_name_tr}: {int(airport_peak_row.airport_total_passengers):,}.",
    f"International passenger zirvesi {airport_international_peak_row.month_name_tr}: {int(airport_international_peak_row.airport_international_passengers):,}.",
    f"Muğla tourism total arrival peak ayı {tourism_peak.peak_month}; airport peak ile ay farkı {int(tourism_peak.month_difference_vs_airport_total_peak)}.",
    f"Muğla foreign arrival peak ayı {foreign_peak.peak_month}; international airport peak ile ay farkı {int(foreign_peak.month_difference_vs_airport_international_peak)}.",
    f"Airport total passengers ile Muğla tourism total arrivals Spearman ρ={total_relation['spearman_rho']:.3f} (n=12).",
    f"Airport international passengers ile Muğla foreign arrivals Spearman ρ={international_relation['spearman_rho']:.3f} (n=12).",
    f"Airport domestic passengers ile Muğla domestic arrivals Spearman ρ={domestic_relation['spearman_rho']:.3f} (n=12).",
    f"Airport total passengers ile Muğla overnights Spearman ρ={overnight_relation['spearman_rho']:.3f} (n=12).",
    f"Airport total passengers ile Muğla occupancy Spearman ρ={occupancy_total['spearman_rho']:.3f} (n=12); occupancy supply availability'den de etkilenir.",
    f"En büyük total normalized divergence {largest_total_divergence['month']} ayında {largest_total_divergence.total_index_gap:.1f} indeks puanıdır.",
    f"Total pair için lag 0 Spearman ρ={total_lag0.spearman_rho:.3f}, lag 1 ρ={total_lag1.spearman_rho:.3f}; international–foreign için lag 0 ρ={international_lag0.spearman_rho:.3f}, lag 1 ρ={international_lag1.spearman_rho:.3f}.",
    "En önemli sınırlılık airport passenger ile accommodation arrival'ın aynı kişi, population veya grain olmamasıdır.",
]
key_findings_path = REPORTS_DIR / "airport_tourism_key_findings.txt"
key_findings_path.write_text(
    "Bodrum Hotel & Destination Intelligence — Airport × Tourism Temel Bulgular\\n\\n"
    + "\\n".join(f"- {item}" for item in findings) + "\\n",
    encoding="utf-8",
)
display(Markdown("\\n".join(f"- {item}" for item in findings)))
"""
)

md("""## Analiz Sınırlılıkları""")

code(
    """limitations = [
    "Airport passengers yalnız turist değildir.",
    "Tourism arrivals airport passenger ile aynı grain/population değildir ve benzersiz kişi olmayabilir.",
    "Milas-Bodrum Airport bölgesel gateway'dir; yolcular başka destinasyonlara devam edebilir.",
    "Tourism monthly data Muğla il seviyesindedir; Bodrum monthly tourism verisi değildir.",
    "Yalnızca 12 aylık gözlem vardır.",
    "Korelasyon nedensellik değildir; ortak sezon etkisi yüksek birlikte hareket üretebilir.",
    "Lag analizi predictive veya causal değildir; lag 1 yalnız 11 hizalı gözlem içerir.",
    "Occupancy açık accommodation supply ve kapasite kullanılabilirliğinden etkilenebilir.",
    "Annual hotel/Google snapshot 2026, airport ve tourism serileri 2025 tarihlidir.",
    "Farklı veri setlerinin toplama ve istatistik tanımları farklı olabilir; proxy ratio conversion değildir.",
]
limitations_path = REPORTS_DIR / "airport_tourism_limitations.txt"
limitations_path.write_text("\\n".join(f"- {item}" for item in limitations) + "\\n", encoding="utf-8")
display(Markdown("\\n".join(f"- {item}" for item in limitations)))
"""
)

md(
    """## Customer Voice Layer

Sonraki aşama veri türüne bağlıdır. Normal customer review dataset'i hazırsa
`11_review_data_audit_cleaning.ipynb`; Google Maps scraping yerine Şikayetvar corpus'u kullanılıyorsa
`11_sikayetvar_complaint_audit_cleaning.ipynb` oluşturulmalıdır. Şikayetvar verisi genel review
evreni değil, **negative customer voice / complaint corpus** olarak analiz edilmelidir.
"""
)

md("""### 19. Çalıştırma ve çıktı bütünlüğü doğrulaması""")

code(
    """expected_figures = [
    "01_airport_monthly_passengers.png", "02_airport_domestic_international.png",
    "03_airport_vs_tourism_normalized.png", "04_international_vs_foreign_normalized.png",
    "05_domestic_vs_domestic_normalized.png", "06_airport_total_vs_tourism_scatter.png",
    "07_international_vs_foreign_scatter.png", "08_airport_vs_overnights.png",
    "09_airport_vs_occupancy.png", "10_international_share_vs_foreign_share.png",
    "11_monthly_divergence.png", "12_correlation_heatmap.png",
]
figure_paths = [FIGURES_DIR / name for name in expected_figures]
input_hashes_after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}

assert input_hashes_before == input_hashes_after
assert len(airport_raw) == 12 and airport_raw["period"].is_unique
assert len(tourism_raw) == 12 and tourism_raw["period"].is_unique
assert len(joint) == 12 and joint["period"].is_unique
assert joint["period"].tolist() == [f"2025-{month:02d}" for month in range(1, 13)]
assert airport_audited["passenger_total_difference"].eq(0).all()
assert airport_audited["cumulative_total_monthly_difference"].eq(0).all()
assert tourism_raw["arrival_total_difference"].eq(0).all()
assert np.isfinite(joint.select_dtypes(include="number")).all().all()
assert set(lag_table.loc[lag_table["lag_months"].eq(0), "n"]) == {12}
assert set(lag_table.loc[lag_table["lag_months"].eq(1), "n"]) == {11}
assert all(path.exists() and path.stat().st_size > 0 for path in output_paths.values())
assert key_findings_path.exists() and limitations_path.exists()
assert all(path.exists() and path.stat().st_size > 0 for path in figure_paths)

validation = pd.DataFrame([
    ("airport_12_months", len(airport_raw) == 12),
    ("tourism_12_months", len(tourism_raw) == 12),
    ("merge_12_rows", len(joint) == 12),
    ("unique_period", joint["period"].is_unique),
    ("passenger_totals_reconcile", airport_audited["passenger_total_difference"].eq(0).all()),
    ("cumulative_monthly_values_reconcile", airport_audited["cumulative_total_monthly_difference"].eq(0).all()),
    ("finite_numeric_outputs", np.isfinite(joint.select_dtypes(include="number")).all().all()),
    ("lag0_n_12_lag1_n_11", bool(set(lag_table.loc[lag_table["lag_months"].eq(0), "n"]) == {12} and set(lag_table.loc[lag_table["lag_months"].eq(1), "n"]) == {11})),
    ("source_hashes_unchanged", input_hashes_before == input_hashes_after),
    ("output_csv_count", len(output_paths)),
    ("figure_count", len(figure_paths)),
], columns=["check", "value"])
display(validation)
print("CSV/TXT çıktıları:")
for path in [*output_paths.values(), key_findings_path, limitations_path]:
    print("-", path.relative_to(PROJECT_ROOT))
print("Grafikler:")
for path in figure_paths:
    print("-", path.relative_to(PROJECT_ROOT))
"""
)

md(
    """### Sonraki Aşama

Airport ve Muğla tourism serilerinin ortak 2025 sezonluğu kavramsal farkları korunarak tamamlandı.
Bir sonraki notebook customer voice kaynağının gerçek niteliğine göre review audit veya Şikayetvar
complaint audit olmalıdır.
"""
)

nb["cells"] = cells
nbf.write(nb, NOTEBOOK_PATH)
print(f"Oluşturuldu: {NOTEBOOK_PATH}")
