"""08_destination_intelligence_analysis.ipynb dosyasını tekrar üretilebilir biçimde oluşturur."""

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "08_destination_intelligence_analysis.ipynb"

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
## 08 - Destination Intelligence Analysis

Bu notebook 14 Bodrum destinasyonunu tek bir ortalama puana indirgemeden; **arz, resmî eşleşme
kapasitesi, müşteri memnuniyeti, popülerlik, fiyat konumu, doğrulanmış lüks arz, değer sinyali ve
mevcut resmî destinasyon bağlamları** üzerinden inceler.

Temel kapsam ilkesi:

> Resmî yıldız/oda/yatak göstergeleri yalnızca 07 aşamasındaki yüksek güvenli proje-otel
> eşleşmelerini temsil eder. Eşleşmeyen alanlarda sıfır “gerçek kapasite” varsayılmaz; kapasite
> bilinmiyor (`NaN`) olarak korunur.

Restoran, bar, gece hayatı, plaj, Blue Flag, yürünebilirlik, havalimanı süresi, POI yoğunluğu,
erişilebilirlik, koordinat veya alan merkezi üretilmez. Harita, kümeleme, tahmin, anomali modeli,
NLP, sentiment ve öneri sistemi bu notebookun kapsamı dışındadır.
"""
)

md("""### 1. Kurulum, yollar ve kaynak bütünlüğü""")

code(
    """from pathlib import Path
import hashlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bodrum_intelligence.destination_intelligence import (
    EXPECTED_AREAS,
    add_quadrants,
    add_subindices,
    build_archetypes,
    build_destination_master,
    input_consistency_check,
    spearman_correlations,
    validate_area_set,
    value_sensitivity,
)

HOTELS_PATH = PROJECT_ROOT / "data" / "processed" / "hotels_enriched.csv"
DESTINATION_V1_PATH = PROJECT_ROOT / "data" / "external" / "destination" / "destination_intelligence_v1.csv"
EDA_PROFILE_PATH = PROJECT_ROOT / "reports" / "eda_destination_profile.csv"
MATCHED_CAPACITY_PATH = PROJECT_ROOT / "reports" / "hotel_attributes_destination_capacity.csv"
STAR_SUMMARY_PATH = PROJECT_ROOT / "reports" / "hotel_attributes_star_summary.csv"
COVERAGE_PATH = PROJECT_ROOT / "reports" / "hotel_attributes_match_coverage_by_area.csv"
ATTR_FINDINGS_PATH = PROJECT_ROOT / "reports" / "hotel_attributes_key_findings.txt"
PROCESSED_OUTPUT = PROJECT_ROOT / "data" / "processed" / "destination_intelligence_enriched.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" / "destination_intelligence"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

required_paths = [
    HOTELS_PATH, DESTINATION_V1_PATH, EDA_PROFILE_PATH, MATCHED_CAPACITY_PATH,
    STAR_SUMMARY_PATH, COVERAGE_PATH, ATTR_FINDINGS_PATH,
]
missing_paths = [path for path in required_paths if not path.exists()]
assert not missing_paths, f"Eksik girdiler: {missing_paths}"

input_hashes_before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in required_paths}
hotels = pd.read_csv(HOTELS_PATH, dtype={"phone": "string"})
destination_v1 = pd.read_csv(DESTINATION_V1_PATH)
eda_profile = pd.read_csv(EDA_PROFILE_PATH)
matched_capacity = pd.read_csv(MATCHED_CAPACITY_PATH)
star_summary_input = pd.read_csv(STAR_SUMMARY_PATH)
coverage_input = pd.read_csv(COVERAGE_PATH)

pd.set_option("display.max_columns", 80)
pd.set_option("display.max_colwidth", 85)
pd.set_option("display.float_format", lambda value: f"{value:,.3f}")
print(f"Hotel-level girdi: {hotels.shape}")
print(f"Destination V1: {destination_v1.shape}")
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


def horizontal_bar(frame, value, title, xlabel, filename, color=PRIMARY, note_col=None, percent=False):
    plot = frame.dropna(subset=[value]).sort_values(value)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(plot["area"], plot[value], color=color, alpha=0.9)
    for y, row in enumerate(plot.itertuples()):
        number = getattr(row, value)
        label = f"%{number:.1f}" if percent else f"{number:,.1f}"
        if note_col:
            label += f" (n={getattr(row, note_col)})"
        ax.text(number + max(plot[value].max() * 0.008, 0.02), y, label, va="center", fontsize=8)
    ax.set(title=title, xlabel=xlabel)
    ax.grid(axis="x", alpha=0.2)
    save_fig(fig, filename)


def quadrant_plot(frame, x, y, label_col, title, xlabel, ylabel, filename, eligible=None):
    plot = frame.copy()
    if eligible is not None:
        plot = plot.loc[eligible].copy()
    plot = plot.dropna(subset=[x, y])
    x_median, y_median = plot[x].median(), plot[y].median()
    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.scatter(plot[x], plot[y], s=78, color=PRIMARY, alpha=0.82)
    for row in plot.itertuples():
        ax.annotate(row.area, (getattr(row, x), getattr(row, y)), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    ax.axvline(x_median, color=NEUTRAL, linestyle="--", label=f"x medyan={x_median:.2f}")
    ax.axhline(y_median, color=NEUTRAL, linestyle=":", label=f"y medyan={y_median:.2f}")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.2); ax.legend(fontsize=8)
    save_fig(fig, filename)
    return x_median, y_median


def index_heatmap(frame, columns, filename):
    plot = frame.set_index("area")[columns]
    fig, ax = plt.subplots(figsize=(10, 7.5))
    image = ax.imshow(plot.to_numpy(), cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(columns)), [c.replace("_index", "").replace("_", " ").title() for c in columns], rotation=25, ha="right")
    ax.set_yticks(range(len(plot.index)), plot.index)
    ax.set_title("Coverage-Aware Destinasyon Alt İndeksleri (0–100)")
    for i in range(plot.shape[0]):
        for j in range(plot.shape[1]):
            value = plot.iloc[i, j]
            ax.text(j, i, "—" if pd.isna(value) else f"{value:.0f}", ha="center", va="center",
                    fontsize=8, color="white" if pd.notna(value) and value > 58 else "black")
    fig.colorbar(image, ax=ax, shrink=0.78, label="İndeks")
    save_fig(fig, filename)
"""
)

md(
    """### 2. Grain ve source audit

Her kaynak kendi grain ve kapsamıyla tanıtılır. `destination_intelligence_v1.csv` içindeki resmî
tesis kapasitesi, proje otelleriyle yüksek güvenli eşleşme evreninden farklıdır. Bu notebookta V1
yalnız marina/pazar bağlamı için kullanılır; resmî yıldız/oda/yatak metrikleri 07 eşleşmelerinden
yeniden hesaplanır.
"""
)

code(
    """source_audit = pd.DataFrame([
    {
        "dataset_name": "hotels_enriched.csv", "grain": "hotel-level", "row_count": len(hotels),
        "primary_key": "hotel_id", "date_scope": "2026-08-24 snapshot",
        "source_scope": "Google hotel sample + high-confidence official matches",
        "main_variables": "rating, reviews, price snapshot, weighted rating, matched star/room/bed",
    },
    {
        "dataset_name": "destination_intelligence_v1.csv", "grain": "destination-level", "row_count": len(destination_v1),
        "primary_key": "area", "date_scope": "retrieved 2026-08-25; source periods differ",
        "source_scope": "14-area V1 aggregate + official marina/market context + separately mapped official facilities",
        "main_variables": "hotel aggregates, marina/market context, separate official-facility universe",
    },
    {
        "dataset_name": "eda_destination_profile.csv", "grain": "destination-level", "row_count": len(eda_profile),
        "primary_key": "area", "date_scope": "2026-08-24 hotel snapshot",
        "source_scope": "05 notebook hotel sample aggregate",
        "main_variables": "hotel count, rating, reviews, price",
    },
    {
        "dataset_name": "hotel_attributes_destination_capacity.csv", "grain": "destination-level official-match aggregate", "row_count": len(matched_capacity),
        "primary_key": "area (only areas with matched hotels)", "date_scope": "07 notebook output",
        "source_scope": "52 high-confidence project-hotel matches",
        "main_variables": "matched count, verified stars, official rooms/beds",
    },
], columns=["dataset_name", "grain", "row_count", "primary_key", "date_scope", "source_scope", "main_variables"])
display(source_audit)

for name, frame in {
    "hotels_enriched": hotels,
    "destination_v1": destination_v1,
    "eda_profile": eda_profile,
    "coverage_report": coverage_input,
}.items():
    audit = validate_area_set(frame)
    print(name, audit)
"""
)

md("""### 3. 14 destinasyonun korunması ve master tablonun kurulması""")

code(
    """OFFICIAL_MATCH_RATE_THRESHOLD = 40.0
VERIFIED_STAR_MINIMUM = 3
LOW_SAMPLE_THRESHOLD = 7

master = build_destination_master(
    hotels,
    destination_v1,
    official_match_rate_threshold=OFFICIAL_MATCH_RATE_THRESHOLD,
    verified_star_minimum=VERIFIED_STAR_MINIMUM,
    low_sample_threshold=LOW_SAMPLE_THRESHOLD,
)
master = add_subindices(master)
master = add_quadrants(master)
master = build_archetypes(master)

assert master["area"].tolist() == EXPECTED_AREAS
assert master["area"].is_unique and len(master) == 14
display(master)
"""
)

code(
    """consistency = input_consistency_check(master, eda_profile, destination_v1, matched_capacity)
display(consistency["comparison_status"].value_counts().to_frame("row_count"))
display(consistency.loc[consistency["comparison_status"].ne("CONSISTENT")].head(40))
"""
)

code(
    """display(Markdown(
    f"**Kaynak kararı:** 192 otellik güncel tablodan yeniden hesaplanan hotel metrikleri ana kaynaktır. "
    f"Resmî eşleşme kapsamı için yalnız 07'nin {int(master['official_matched_hotel_count'].sum())} yüksek "
    f"güvenli eşleşmesi kullanılır. V1'in ayrı resmî tesis evrenindeki kapasite değerleri birleştirilmez; "
    f"marina ve haftalık pazar bağlamı korunur."
))
"""
)

md(
    """### 4. Coverage analizi

Gerçek dağılımda resmî eşleşme oranlarının medyanı yaklaşık %28'dir. Konservatif olarak
`official_match_rate < %40` **veya** `verified_star_n < 3` alanlar düşük resmî coverage kabul edilir.
`sample_hotel_count < 7` yalnız örneklem büyüklüğü uyarısıdır. Bu eşikler gerçek kapasiteyi tahmin
etmez; lüks sıralaması ve price×luxury quadrant uygunluğunu sınırlar.
"""
)

code(
    """coverage_columns = [
    "area", "sample_hotel_count", "official_attribute_n", "verified_star_n", "room_capacity_n",
    "price_observation_n", "official_match_rate_pct", "verified_star_coverage_pct",
    "room_coverage_pct", "price_coverage_pct", "coverage_flag", "low_sample_flag", "score_confidence",
]
coverage_table = master[coverage_columns]
display(coverage_table)

coverage_matrix = master.set_index("area")[[
    "official_match_rate_pct", "verified_star_coverage_pct", "room_coverage_pct", "price_coverage_pct"
]]
fig, ax = plt.subplots(figsize=(10, 7.5))
image = ax.imshow(coverage_matrix.to_numpy(), cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(4), ["Official match", "Verified star", "Room", "Price"])
ax.set_yticks(range(14), coverage_matrix.index)
ax.set_title("Destinasyon Bazında Veri Coverage (%)")
for i in range(coverage_matrix.shape[0]):
    for j in range(coverage_matrix.shape[1]):
        value = coverage_matrix.iloc[i, j]
        ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8)
fig.colorbar(image, ax=ax, shrink=0.75, label="Coverage (%)")
save_fig(fig, "coverage_by_area.png")
"""
)

code(
    """low_coverage_areas = master.loc[master["low_coverage_flag"], "area"].tolist()
display(Markdown(
    f"**Bulgu:** {len(low_coverage_areas)}/14 alan düşük resmî coverage bayrağı taşıyor: "
    + ", ".join(low_coverage_areas)
    + ". Gümüşlük ve Güvercinlik'te yüksek güvenli eşleşme yoktur; kapasite bilinmiyor olarak korunur."
))
"""
)

md("""### 5. Supply ve resmî eşleşme kapasitesi""")

code(
    """horizontal_bar(master, "sample_hotel_count", "Örneklemdeki Otel Arzı", "Otel sayısı",
               "sample_hotel_count_by_area.png", note_col=None)
horizontal_bar(master, "total_official_rooms", "Yüksek Güvenli Eşleşmelerde Resmî Oda Kapasitesi",
               "Toplam resmî oda", "official_room_capacity_by_area.png", color=PRIMARY, note_col="room_capacity_n")
horizontal_bar(master, "total_official_beds", "Yüksek Güvenli Eşleşmelerde Resmî Yatak Kapasitesi",
               "Toplam resmî yatak", "official_bed_capacity_by_area.png", color=SECONDARY, note_col="bed_capacity_n")
"""
)

code(
    """capacity_complete = master.dropna(subset=["total_official_rooms"])
supply_capacity_rho = capacity_complete["sample_hotel_count"].corr(
    capacity_complete["total_official_rooms"], method="spearman"
)
fig, ax = plt.subplots(figsize=(9.5, 6.3))
ax.scatter(capacity_complete["sample_hotel_count"], capacity_complete["total_official_rooms"], s=75, color=PRIMARY)
for row in capacity_complete.itertuples():
    ax.annotate(row.area, (row.sample_hotel_count, row.total_official_rooms), xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.set(title="Örneklem Otel Sayısı ile Eşleşen Resmî Oda Kapasitesi", xlabel="Örneklem otel sayısı", ylabel="Toplam eşleşen resmî oda")
ax.text(0.03, 0.96, f"Spearman ρ={supply_capacity_rho:.2f}\\nn={len(capacity_complete)}", transform=ax.transAxes,
        va="top", bbox=dict(boxstyle="round", facecolor="white", edgecolor=NEUTRAL))
ax.grid(alpha=0.2)
save_fig(fig, "sample_supply_vs_official_capacity.png")
"""
)

md("""### 6. Capacity yapısı""")

code(
    """capacity_columns = [
    "area", "total_official_rooms", "total_official_beds", "median_official_room_count",
    "median_official_bed_count", "beds_per_room_destination", "room_capacity_n",
]
display(master[capacity_columns])
horizontal_bar(master, "beds_per_room_destination", "Destinasyon Düzeyinde Oda Başına Yatak Kapasitesi",
               "Toplam yatak / toplam oda", "beds_per_room_by_area.png", color=PURPLE, note_col="room_capacity_n")
"""
)

code(
    """display(Markdown(
    "**Yorum:** `beds_per_room_destination`, yalnız eşleşen tesislerin toplam yatak/toplam oda "
    "oranıdır. Oda tipi, gerçek doluluk veya kişi kapasitesi dağılımını doğrudan göstermez."
))
"""
)

md("""### 7. Customer Satisfaction""")

code(
    """satisfaction = master[[
    "area", "avg_google_rating", "median_google_rating", "avg_weighted_google_rating",
    "avg_rating_gap_from_area_median", "sample_hotel_count",
]].sort_values("avg_weighted_google_rating", ascending=False)
display(satisfaction)
horizontal_bar(master, "avg_weighted_google_rating", "Yorum Güveni Ağırlıklı Ortalama Google Puanı",
               "Ağırlıklı ortalama puan", "weighted_rating_by_area.png", color=SECONDARY,
               note_col="sample_hotel_count")

raw_rank = master.set_index("area")["avg_google_rating"].rank(ascending=False, method="min")
weighted_rank = master.set_index("area")["avg_weighted_google_rating"].rank(ascending=False, method="min")
rating_rank_change = pd.DataFrame({"raw_rank": raw_rank, "weighted_rank": weighted_rank})
rating_rank_change["rank_change"] = rating_rank_change["weighted_rank"] - rating_rank_change["raw_rank"]
display(rating_rank_change.sort_values("rank_change", key=abs, ascending=False))
"""
)

md("""### 8. Popularity: toplam görünürlük ve otel başına yoğunluk""")

code(
    """horizontal_bar(master, "total_google_reviews", "Destinasyona Göre Toplam Google Yorumu",
               "Toplam yorum", "total_reviews_by_area.png", color=PRIMARY, note_col="sample_hotel_count")
horizontal_bar(master, "reviews_per_sample_hotel", "Otel Başına Google Yorum Yoğunluğu",
               "Toplam yorum / örneklem oteli", "review_intensity_by_area.png", color=ACCENT,
               note_col="sample_hotel_count")
"""
)

md(
    """### 9. Price Position

Fiyat yalnız 2026-08-24 arama snapshot'ıdır. `price_index=100`, 14 alan medyan fiyatlarının
medyanına eşittir; yıllık veya gerçek piyasa fiyat seviyesi değildir.
"""
)

code(
    """price_table = master[[
    "area", "price_observation_n", "price_coverage_pct", "median_price_snapshot",
    "mean_price_snapshot", "overall_area_median_price", "price_index",
]].sort_values("price_index", ascending=False)
display(price_table)
horizontal_bar(master, "price_index", "Destinasyon Medyan Fiyat Snapshot İndeksi",
               "14 alan medyanına göre indeks (100=medyan)", "median_price_index_by_area.png",
               color=HIGHLIGHT, note_col="price_observation_n")
"""
)

md("""### 10. Luxury supply bileşenleri""")

code(
    """luxury_components = master[[
    "area", "verified_star_n", "verified_five_star_count", "verified_five_star_share",
    "official_boutique_count", "median_official_room_count", "price_index",
    "official_match_rate_pct", "low_coverage_flag", "luxury_index", "luxury_rank_eligible", "luxury_rank",
]]
display(luxury_components.sort_values("verified_five_star_count", ascending=False))
horizontal_bar(master, "verified_five_star_share", "Verified Tesisler İçinde 5 Yıldız Payı",
               "5 yıldız payı (%)", "five_star_share_by_area.png", color=HIGHLIGHT,
               note_col="verified_star_n", percent=True)
"""
)

md(
    """Pay grafiği `verified_star_n` ile birlikte okunmalıdır. Düşük coverage alanları lüks sırasına
alınmaz; 0 doğrulanmış yıldız, “lüks arz yok” değil, “bu eşleşme katmanında bilinmiyor” anlamına gelir.
"""
)

md("""### 11. Value potential: fiyat × ağırlıklı memnuniyet""")

code(
    """value_x_median, value_y_median = quadrant_plot(
    master, "price_index", "avg_weighted_google_rating", "value_quadrant",
    "Fiyat Konumu × Ağırlıklı Müşteri Memnuniyeti",
    "Fiyat indeksi (100=14 alan medyanı)", "Ağırlıklı ortalama Google puanı",
    "price_satisfaction_value_quadrant.png",
)
display(master[["area", "price_index", "avg_weighted_google_rating", "value_index", "value_quadrant"]].sort_values("value_index", ascending=False))
"""
)

md("""### 12. Popularity × Satisfaction quadrant""")

code(
    """pop_x_median, pop_y_median = quadrant_plot(
    master, "reviews_per_sample_hotel", "avg_weighted_google_rating", "popularity_satisfaction_quadrant",
    "Otel Başına Popülerlik × Ağırlıklı Memnuniyet",
    "Otel başına Google yorumu", "Ağırlıklı ortalama Google puanı",
    "popularity_satisfaction_quadrant.png",
)
popularity_quadrants = master[[
    "area", "sample_hotel_count", "reviews_per_sample_hotel", "avg_weighted_google_rating",
    "popularity_satisfaction_quadrant", "score_confidence",
]]
display(popularity_quadrants.sort_values(["popularity_satisfaction_quadrant", "reviews_per_sample_hotel"]))
"""
)

md("""### 13. Price × Luxury quadrant (coverage uygun alanlar)""")

code(
    """eligible_luxury = master["price_luxury_quadrant_eligible"]
luxury_x_median, luxury_y_median = quadrant_plot(
    master, "price_index", "verified_five_star_share", "price_luxury_quadrant",
    "Fiyat Konumu × Doğrulanmış 5 Yıldız Payı",
    "Fiyat indeksi", "Verified tesislerde 5 yıldız payı",
    "price_luxury_quadrant.png", eligible=eligible_luxury,
)
price_luxury_quadrants = master[[
    "area", "price_index", "verified_star_n", "verified_five_star_share",
    "official_match_rate_pct", "price_luxury_quadrant_eligible", "price_luxury_quadrant",
]]
display(price_luxury_quadrants)
"""
)

md("""### 14. Ayrı destinasyon alt indeksleri ve score confidence""")

code(
    """subindex_columns = ["quality_index", "popularity_index", "luxury_index", "value_index", "supply_capacity_index"]
subindices = master[[
    "area", *subindex_columns, "score_confidence", "low_coverage_flag",
    "luxury_rank_eligible", "luxury_rank",
]]
display(subindices)
index_heatmap(master, subindex_columns, "destination_subindices_heatmap.png")
"""
)

code(
    """component_inventory = pd.DataFrame([
    ("quality_index", "avg_weighted_google_rating", "single component", "No official-data dependency"),
    ("popularity_index", "reviews_per_sample_hotel; total_google_reviews", "50/50", "Both hotel-sample components required"),
    ("luxury_index", "verified_five_star_share; verified_five_star_count; price_index", "40/30/30", "Minimum 2 components; rank only if coverage adequate"),
    ("value_index", "weighted rating; inverse price index", "50/50", "No missing component imputation"),
    ("supply_capacity_index", "sample_hotel_count; official rooms; official beds", "1/3 each", "Minimum 2 components; zero-match capacity remains NaN"),
], columns=["index", "components", "weights", "coverage_rule"])
display(component_inventory)
"""
)

md("""### 15. Value index weight sensitivity""")

code(
    """sensitivity = value_sensitivity(master)
display(sensitivity.sort_values(["area", "scenario"]))

rank_ranges = sensitivity.groupby("area").agg(
    min_rank=("rank", "min"), max_rank=("rank", "max"),
    rank_spread=("rank_spread", "first"), ranking_sensitive=("ranking_sensitive", "first"),
).reset_index().sort_values("rank_spread", ascending=False)
display(rank_ranges)

fig, ax = plt.subplots(figsize=(9, 6.5))
plot_ranges = rank_ranges.sort_values("rank_spread")
for y, row in enumerate(plot_ranges.itertuples()):
    ax.plot([row.min_rank, row.max_rank], [y, y], color=ACCENT if row.ranking_sensitive else PRIMARY, linewidth=4)
    ax.scatter([row.min_rank, row.max_rank], [y, y], color="white", edgecolor=NEUTRAL, zorder=3)
ax.set_yticks(range(len(plot_ranges)), plot_ranges["area"])
ax.invert_xaxis()
ax.set(title="Value İndeksi Ağırlık Senaryolarında Sıra Aralığı", xlabel="Sıra (1 daha yüksek)")
ax.grid(axis="x", alpha=0.2)
save_fig(fig, "value_index_rank_sensitivity.png")
"""
)

md("""### 16. Açıklanabilir destination archetypes""")

code(
    """archetypes = master[[
    "area", "archetype", "quality_index", "popularity_index", "luxury_index",
    "value_index", "supply_capacity_index", "score_confidence", "coverage_flag",
]]
display(archetypes.sort_values(["archetype", "area"]))
"""
)

md("""### 17. Marina / haftalık pazar bağlamı""")

code(
    """context_comparison = (
    master.groupby("has_marina_official_context", dropna=False)
    .agg(
        destination_n=("area", "size"),
        median_price_index=("price_index", "median"),
        median_weighted_rating=("avg_weighted_google_rating", "median"),
        median_review_intensity=("reviews_per_sample_hotel", "median"),
    ).reset_index()
)
market_comparison = (
    master.groupby("has_weekly_market_official_context", dropna=False)
    .agg(
        destination_n=("area", "size"),
        median_price_index=("price_index", "median"),
        median_weighted_rating=("avg_weighted_google_rating", "median"),
        median_review_intensity=("reviews_per_sample_hotel", "median"),
    ).reset_index()
)
display(context_comparison)
display(market_comparison)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
metrics = ["median_price_index", "median_weighted_rating", "median_review_intensity"]
titles = ["Medyan fiyat indeksi", "Medyan ağırlıklı rating", "Medyan yorum yoğunluğu"]
for ax, metric, title in zip(axes, metrics, titles):
    ax.bar(context_comparison["has_marina_official_context"].astype(str), context_comparison[metric], color=[NEUTRAL, PRIMARY])
    ax.set_title(title); ax.grid(axis="y", alpha=0.2)
    for x, row in enumerate(context_comparison.itertuples()):
        ax.text(x, getattr(row, metric), f"n={row.destination_n}", ha="center", va="bottom", fontsize=8)
fig.suptitle("Resmî Marina Bağlamı: Yalnız Betimsel Karşılaştırma", y=1.02)
save_fig(fig, "marina_context_descriptive_comparison.png")
"""
)

code(
    """display(Markdown(
    "**Sınırlılık:** Marina/pazar grupları küçüktür; test yapılmamıştır. Bu bağlamlar fiyat, rating "
    "veya görünürlüğün nedeni olarak yorumlanamaz."
))
"""
)

md("""### 18. Keşifsel Spearman korelasyonları""")

code(
    """correlation_pairs = [
    ("price_index", "avg_weighted_google_rating"),
    ("price_index", "verified_five_star_share"),
    ("total_official_rooms", "reviews_per_sample_hotel"),
    ("total_official_rooms", "avg_weighted_google_rating"),
    ("verified_five_star_share", "price_index"),
]
correlations = spearman_correlations(master, correlation_pairs)
display(correlations)
"""
)

md(
    """14 destinasyon küçük bir örneklemdir. P-değeri aşırı yorumlanmaz; normal dağılım varsayımı
zorlanmaz; “anlamlı değil” sonucu ilişki yokluğu kanıtı sayılmaz ve korelasyon nedensellik değildir.
Doğrulanmış yıldız/kapasite kullanan korelasyonlar ayrıca coverage farklarından etkilenir.
"""
)

md("""### 19. Çıktıların kaydedilmesi""")

code(
    """output_paths = {
    "processed_master": PROCESSED_OUTPUT,
    "profile": REPORTS_DIR / "destination_intelligence_profile.csv",
    "consistency": REPORTS_DIR / "destination_input_consistency_check.csv",
    "luxury_components": REPORTS_DIR / "destination_luxury_components.csv",
    "subindices": REPORTS_DIR / "destination_subindices.csv",
    "sensitivity": REPORTS_DIR / "destination_index_sensitivity.csv",
    "popularity_quadrants": REPORTS_DIR / "destination_popularity_satisfaction_quadrants.csv",
    "price_luxury_quadrants": REPORTS_DIR / "destination_price_luxury_quadrants.csv",
    "archetypes": REPORTS_DIR / "destination_archetypes.csv",
    "correlations": REPORTS_DIR / "destination_correlations.csv",
}

master.to_csv(output_paths["processed_master"], index=False, encoding="utf-8-sig")
master.to_csv(output_paths["profile"], index=False, encoding="utf-8-sig")
consistency.to_csv(output_paths["consistency"], index=False, encoding="utf-8-sig")
luxury_components.to_csv(output_paths["luxury_components"], index=False, encoding="utf-8-sig")
subindices.to_csv(output_paths["subindices"], index=False, encoding="utf-8-sig")
sensitivity.to_csv(output_paths["sensitivity"], index=False, encoding="utf-8-sig")
popularity_quadrants.to_csv(output_paths["popularity_quadrants"], index=False, encoding="utf-8-sig")
price_luxury_quadrants.to_csv(output_paths["price_luxury_quadrants"], index=False, encoding="utf-8-sig")
archetypes.to_csv(output_paths["archetypes"], index=False, encoding="utf-8-sig")
correlations.to_csv(output_paths["correlations"], index=False, encoding="utf-8-sig")
"""
)

md("""## Temel Bulgular""")

code(
    """def top_areas(column, n=3, eligible=None):
    frame = master if eligible is None else master.loc[eligible]
    return frame.dropna(subset=[column]).nlargest(n, column)[["area", column]]

top_quality = top_areas("avg_weighted_google_rating")
top_total_popularity = top_areas("total_google_reviews")
top_intensity = top_areas("reviews_per_sample_hotel")
top_price = top_areas("price_index")
top_capacity = top_areas("total_official_rooms")
top_luxury = top_areas("luxury_index", eligible=master["luxury_rank_eligible"])
top_value = top_areas("value_index")
sensitive_areas = rank_ranges.loc[rank_ranges["ranking_sensitive"], "area"].tolist()
zero_match_areas = master.loc[master["official_matched_hotel_count"].eq(0), "area"].tolist()

findings = [
    f"Beklenen 14 destinasyonun tamamı bulundu; eksik, fazla veya duplicate area yoktur.",
    f"Örneklem arzı en yüksek alanlar: " + ", ".join(f"{r.area} ({int(r.sample_hotel_count)})" for r in top_areas('sample_hotel_count').itertuples()) + ".",
    f"Ağırlıklı müşteri memnuniyeti en yüksek alanlar: " + ", ".join(f"{r.area} ({r.avg_weighted_google_rating:.3f})" for r in top_quality.itertuples()) + ".",
    f"Toplam Google görünürlüğü en yüksek alanlar: " + ", ".join(f"{r.area} ({int(r.total_google_reviews):,})" for r in top_total_popularity.itertuples()) + ".",
    f"Otel başına yorum yoğunluğu en yüksek alanlar: " + ", ".join(f"{r.area} ({r.reviews_per_sample_hotel:,.0f})" for r in top_intensity.itertuples()) + ".",
    f"Fiyat snapshot indeksi en yüksek alanlar: " + ", ".join(f"{r.area} ({r.price_index:.1f})" for r in top_price.itertuples()) + ".",
    f"Coverage uygun alanlarda lüks alt indeksi en yüksek alanlar: " + ", ".join(f"{r.area} ({r.luxury_index:.1f})" for r in top_luxury.itertuples()) + ".",
    f"Dengeli value sinyali en yüksek alanlar: " + ", ".join(f"{r.area} ({r.value_index:.1f})" for r in top_value.itertuples()) + ".",
    f"Yüksek güvenli eşleşmelerde resmî oda kapasitesi en yüksek alanlar: " + ", ".join(f"{r.area} ({r.total_official_rooms:,.0f})" for r in top_capacity.itertuples()) + ".",
    f"{len(low_coverage_areas)}/14 alan düşük resmî coverage taşır: " + ", ".join(low_coverage_areas) + ".",
    f"Yüksek güvenli resmî eşleşmesi olmayan alanlar: " + ", ".join(zero_match_areas) + "; kapasite sıfır değil NaN tutulmuştur.",
    f"Örneklem arzı ile eşleşen resmî oda kapasitesi Spearman ilişkisi ρ={supply_capacity_rho:.2f} (n={len(capacity_complete)}).",
    f"Value ağırlık senaryolarında ranking-sensitive alanlar: " + (", ".join(sensitive_areas) if sensitive_areas else "yok") + ".",
    f"Price×luxury quadrant yalnız {int(master['price_luxury_quadrant_eligible'].sum())}/14 coverage-uygun alanda hesaplanmıştır.",
]

key_findings_path = REPORTS_DIR / "destination_intelligence_key_findings.txt"
key_findings_path.write_text(
    "Bodrum Hotel & Destination Intelligence — Destination Intelligence Temel Bulgular\\n\\n"
    + "\\n".join(f"- {finding}" for finding in findings) + "\\n",
    encoding="utf-8",
)
display(Markdown("\\n".join(f"- {finding}" for finding in findings)))
"""
)

md("""## Analiz Sınırlılıkları""")

code(
    """limitations = [
    "192 otellik Google kaynaklı örneklem Bodrum'daki tüm konaklama tesisleri evreni değildir.",
    "High-confidence official hotel matching coverage sınırlıdır; doğrulanmış yıldız/oda/yatak tüm otellerde yoktur.",
    "Destinasyon metrikleri alan başına farklı sample ve official coverage oranlarından etkilenir.",
    "Google rating ve review count platform görünürlüğü ve kullanıcı davranışı yanlılığı taşır.",
    "Fiyat tek tarihli arama snapshot'ıdır; yıllık veya sezon geneli fiyat seviyesi değildir.",
    "14 destinasyon istatistiksel çıkarım ve korelasyon analizi için küçük örneklemdir.",
    "Marina ve haftalık pazar bağlamları betimseldir; nedensel değişken olarak yorumlanamaz.",
    "Restoran, plaj, gece hayatı, Blue Flag, POI, yürünebilirlik ve güvenilir erişim süresi katmanları henüz yoktur.",
    "Alt indeksler açıklanabilir karşılaştırma araçlarıdır; tek ve kesin bir Destination Score değildir.",
    "Min-max indeksleri bu 14 alanın mevcut örneklem aralığına bağlıdır ve yeni veri geldiğinde değişebilir.",
]
limitations_path = REPORTS_DIR / "destination_intelligence_limitations.txt"
limitations_path.write_text("\\n".join(f"- {item}" for item in limitations) + "\\n", encoding="utf-8")
display(Markdown("\\n".join(f"- {item}" for item in limitations)))
"""
)

md(
    """## Tourism Demand Layer

Sonraki notebook `notebooks/09_tourism_demand_analysis.ipynb` olacaktır. Bu aşamadaki 14 alanlık
profil daha sonra tourism demand, airport traffic, review NLP, hotel segmentation ve recommendation
katmanlarıyla grain ve tarih kapsamı korunarak bağlanabilir. Aylık/yıllık talep verisi destinasyon
tablosuna sırf birleştirmek için join edilmemelidir.
"""
)

md("""### 20. Çalıştırma ve çıktı bütünlüğü doğrulaması""")

code(
    """expected_figures = [
    "coverage_by_area.png", "sample_hotel_count_by_area.png", "official_room_capacity_by_area.png",
    "official_bed_capacity_by_area.png", "sample_supply_vs_official_capacity.png",
    "beds_per_room_by_area.png", "weighted_rating_by_area.png", "total_reviews_by_area.png",
    "review_intensity_by_area.png", "median_price_index_by_area.png", "five_star_share_by_area.png",
    "price_satisfaction_value_quadrant.png", "popularity_satisfaction_quadrant.png",
    "price_luxury_quadrant.png", "destination_subindices_heatmap.png",
    "value_index_rank_sensitivity.png", "marina_context_descriptive_comparison.png",
]
figure_paths = [FIGURES_DIR / name for name in expected_figures]
input_hashes_after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in required_paths}

assert input_hashes_before == input_hashes_after, "Kaynak girdilerden biri değiştirildi."
assert len(master) == 14 and master["area"].is_unique
assert master["area"].tolist() == EXPECTED_AREAS
assert hotels.shape[0] == 192 and hotels["hotel_id"].nunique() == 192
assert master.loc[master["official_matched_hotel_count"].eq(0), "total_official_rooms"].isna().all()
assert master.loc[master["low_coverage_flag"], "luxury_rank"].isna().all()
assert all(path.exists() and path.stat().st_size > 0 for path in output_paths.values())
assert all(path.exists() and path.stat().st_size > 0 for path in figure_paths)
assert key_findings_path.exists() and limitations_path.exists()

validation = pd.DataFrame([
    ("all_14_expected_areas", master["area"].tolist() == EXPECTED_AREAS),
    ("unique_area", master["area"].is_unique),
    ("hotel_rows_preserved", len(hotels) == 192),
    ("source_hashes_unchanged", input_hashes_before == input_hashes_after),
    ("zero_match_capacity_propagates_nan", bool(master.loc[master["official_matched_hotel_count"].eq(0), "total_official_rooms"].isna().all())),
    ("low_coverage_not_luxury_ranked", bool(master.loc[master["low_coverage_flag"], "luxury_rank"].isna().all())),
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

Destination Intelligence katmanı; 14 alan, coverage bayrakları, ayrı alt indeksler, ağırlık
hassasiyeti, quadrantlar ve açıklanabilir archetype profilleriyle tamamlandı. Sonraki aşama
`09_tourism_demand_analysis.ipynb` olmalıdır.
"""
)

nb["cells"] = cells
nbf.write(nb, NOTEBOOK_PATH)
print(f"Oluşturuldu: {NOTEBOOK_PATH}")
