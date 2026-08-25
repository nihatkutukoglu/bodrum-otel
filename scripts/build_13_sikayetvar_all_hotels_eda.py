"""13_sikayetvar_all_hotels_eda.ipynb dosyasını tekrar üretilebilir biçimde oluşturur."""

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "13_sikayetvar_all_hotels_eda.ipynb"
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
## 13 - Şikayetvar All-Hotels Exploratory Data Analysis
### Negative Customer Voice — Complaint Volume, Time, Visibility, Replies and Response Behavior

Bu notebook Notebook 12'de oluşturulan canonical-unique, complaint-level matched clean corpusu
structured-field EDA ile inceler. Sentiment, topic, aspect, embedding, clustering, supervised ML
ve hotel kalite sıralaması yapılmaz.

> **Ana metodolojik uyarı:** Şikayetvar self-selected bir complaint corpusudur. Complaint count
> hotel kalitesi veya gerçek complaint rate değildir. Daha çok complaint; platform görünürlüğü,
> müşteri hacmi, marka bilinirliği ve mapping coverage ile birlikte değişebilir. Şikayetvar ve
> Google Reviews farklı kullanıcı evrenleridir. Analiz yalnız güvenilir complaint eşleşmesi bulunan
> seçili hotel örneklemini kapsar; mapping coverage 192 project hotel arasında eşit değildir.
"""
)

md("""## 01. Amaç ve metodolojik çerçeve""")

md(
    """Araştırma soruları complaint corpusunun dağılımı, zaman kapsamı, metin uzunluğu, görüntülenme,
company response/reply davranışı ve cross-platform descriptive context üzerinedir. Hiçbir oran
gerçek müşteri complaint rate olarak adlandırılmaz. Official star, kapasite, fiyat ve Google
metrikleri yalnız coverage'ı açık keşifsel bağlam olarak kullanılır.

Hotel-level oran/medyan grafiklerinde `min_complaints_for_rate_chart=5` uygulanır. Bu eşik corpus
hotel medyanının 5 complaint olmasıyla uyumludur; düşük-n hotel'ler summary tabloda korunur ve
`small_n_flag=True` taşır.
"""
)

code(
    """from pathlib import Path
import hashlib
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from scipy import stats
from scipy.stats import spearmanr

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bodrum_intelligence.project_summary import format_number_tr, format_pct_tr, interpret_spearman
from bodrum_intelligence.sikayetvar_eda import (
    build_area_eda_summary,
    build_hotel_eda_summary,
    concentration_metrics,
    numeric_correlation_matrix,
    response_time_summary,
    spearman_table,
)

PROCESSED_DIR = PROJECT_ROOT / "data/processed"
RAW_DIR = PROJECT_ROOT / "data/raw/sikayetvar"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures/sikayetvar_eda"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

paths = {
    "complaints": PROCESSED_DIR / "sikayetvar_all_hotels_complaints_clean.csv",
    "replies": PROCESSED_DIR / "sikayetvar_all_hotels_replies_clean.csv",
    "review_required": PROCESSED_DIR / "sikayetvar_complaints_review_required.csv",
    "hotel_master": PROCESSED_DIR / "hotels_enriched.csv",
    "hotel_mapping": RAW_DIR / "sikayetvar_hotel_mapping.csv",
    "cleaning_summary": REPORTS_DIR / "sikayetvar_cleaning_summary.csv",
    "clean_hotel_coverage": REPORTS_DIR / "sikayetvar_clean_coverage_by_hotel.csv",
    "clean_area_coverage": REPORTS_DIR / "sikayetvar_clean_coverage_by_area.csv",
    "scraper_consistency": REPORTS_DIR / "sikayetvar_scraper_vs_audit_consistency.csv",
    "tourism_monthly": PROCESSED_DIR / "tourism_demand_monthly_features_2025.csv",
}
missing = [str(path.relative_to(PROJECT_ROOT)) for path in paths.values() if not path.exists()]
assert not missing, f"Eksik Notebook 12 / project girdisi: {missing}"

protected_clean_paths = [paths["complaints"], paths["replies"], paths["review_required"]]
clean_hashes_before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected_clean_paths}

complaints = pd.read_csv(
    paths["complaints"],
    parse_dates=["complaint_date", "company_response_date_parsed", "first_reply_date", "last_reply_date"],
)
replies = pd.read_csv(paths["replies"], parse_dates=["reply_date"])
review_required = pd.read_csv(paths["review_required"])
hotel_master = pd.read_csv(paths["hotel_master"])
hotel_mapping = pd.read_csv(paths["hotel_mapping"])
cleaning_summary = pd.read_csv(paths["cleaning_summary"]).set_index("metric")["value"]
clean_hotel_coverage = pd.read_csv(paths["clean_hotel_coverage"])
clean_area_coverage = pd.read_csv(paths["clean_area_coverage"])
scraper_consistency = pd.read_csv(paths["scraper_consistency"])
tourism_monthly = pd.read_csv(paths["tourism_monthly"])

MIN_COMPLAINTS_FOR_RATE_CHART = 5
complaints["response_time_days"] = (
    complaints["company_response_date_parsed"] - complaints["complaint_date"]
).dt.total_seconds() / 86400
complaints["response_time_negative_flag"] = complaints["response_time_days"].lt(0)
complaints.loc[complaints["response_time_negative_flag"], "response_time_days"] = np.nan

pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 100)
print(f"Clean corpus: {len(complaints)} complaint | {complaints['hotel_id'].nunique()} hotel | {complaints['area'].nunique()} area")
"""
)

code(
    """def save_fig(fig, filename):
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return path.relative_to(PROJECT_ROOT)


def explain_figure(how_to_read, observation, importance, caution):
    display(Markdown(
        f"### Grafik nasıl okunur?\\n{how_to_read}\\n\\n"
        f"### Ne görüyoruz?\\n{observation}\\n\\n"
        f"### Neden önemli?\\n{importance}\\n\\n"
        f"### Dikkat edilmesi gereken nokta\\n{caution}"
    ))
"""
)

md("""## 02. Clean corpus validation""")

code(
    """clean_orphan_replies = replies.loc[
    ~replies["canonical_complaint_url"].isin(set(complaints["canonical_complaint_url"]))
]
validation = pd.DataFrame([
    ("clean_complaint_row_count", int(cleaning_summary["processed_complaint_count"]), len(complaints)),
    ("unique_canonical_complaints", len(complaints), complaints["canonical_complaint_url"].nunique()),
    ("hotel_count_with_complaints", int((clean_hotel_coverage["matched_complaint_count"] > 0).sum()), complaints["hotel_id"].nunique()),
    ("area_count_represented", int((clean_area_coverage["matched_complaint_count"] > 0).sum()), complaints["area"].nunique()),
    ("duplicate_canonical_url", 0, int(complaints["canonical_complaint_url"].duplicated().sum())),
    ("missing_complaint_text", int(cleaning_summary["text_missing_count"]), int(complaints["complaint_text_clean"].isna().sum())),
    ("clean_reply_count", int(cleaning_summary["processed_reply_count"]), len(replies)),
    ("orphan_clean_reply", 0, len(clean_orphan_replies)),
], columns=["check", "notebook_12_value", "recomputed_value"])
validation["difference"] = validation["recomputed_value"] - validation["notebook_12_value"]
validation["status"] = np.where(validation["difference"].eq(0), "PASS", "WARNING")
display(validation)
assert validation["status"].eq("PASS").all(), "Notebook 12 ile clean corpus tutarsız; EDA durduruldu."
assert complaints["entity_match_status"].eq("COMPLAINT_MATCHED").all()
assert scraper_consistency["status"].eq("PASS").all()
"""
)

code(
    """inventory = pd.DataFrame([
    ("Clean complaints", "complaint", len(complaints), complaints["hotel_id"].nunique(), complaints["area"].nunique(), f"{complaints['complaint_date'].min():%Y-%m-%d} → {complaints['complaint_date'].max():%Y-%m-%d}", "Main EDA corpus"),
    ("Clean replies", "reply", len(replies), replies["hotel_id"].nunique(), np.nan, f"{replies['reply_date'].min():%Y-%m-%d} → {replies['reply_date'].max():%Y-%m-%d}", "Reply behavior"),
    ("Hotel master", "hotel", len(hotel_master), hotel_master["hotel_id"].nunique(), hotel_master["area"].nunique(), "2026-08-24 snapshot", "Cross-platform metadata"),
    ("Hotel mapping", "hotel mapping", len(hotel_mapping), hotel_mapping["hotel_id"].nunique(), hotel_mapping["area"].nunique(), "2026-08-25 discovery", "Coverage denominator"),
], columns=["dataset", "grain", "row_count", "unique_hotels", "unique_areas", "date_range", "main_role"])
display(inventory)
"""
)

md("""### Bölüm Sonucu

Notebook 12 ile sekiz kritik KPI birebir uyumludur. Clean corpus 236 canonical-unique, complaint-level
matched kayıttan oluşur; 12 review-required kayıt bu EDA'nın dışında kalır.
""")

md("""## 03. Data coverage""")

code(
    """mapping_status = (
    hotel_mapping["match_status"].value_counts(dropna=False)
    .rename_axis("mapping_status").reset_index(name="hotel_count")
)
mapping_status["share_pct"] = 100 * mapping_status["hotel_count"] / len(hotel_mapping)
display(mapping_status)

fig, ax = plt.subplots(figsize=(9.2, 5.2))
mapping_plot = mapping_status.sort_values("hotel_count")
colors = ["#2F6B7C" if status in {"FOUND_EXACT", "FOUND_HIGH_CONFIDENCE"} else "#B7B7B7" for status in mapping_plot["mapping_status"]]
ax.barh(mapping_plot["mapping_status"], mapping_plot["hotel_count"], color=colors)
for index, value in enumerate(mapping_plot["hotel_count"]):
    ax.text(value + 1, index, str(value), va="center")
ax.set(title="Project Hotels by Şikayetvar Mapping Status — Coverage, not performance", xlabel="Project hotel count", ylabel="Mapping status")
ax.set_xlim(0, mapping_plot["hotel_count"].max() * 1.13); ax.grid(axis="x", alpha=0.2)
save_fig(fig, "01_mapping_coverage_status.png")
"""
)

code(
    """explain_figure(
    "Her çubuk 192 project hotelin discovery/mapping statüsündeki sayısını gösterir; renkli çubuklar güvenilir hotel-level mapping statüleridir.",
    f"Clean complaint corpusu {complaints['hotel_id'].nunique()} hotel içerirken project master {len(hotel_master)} hotel içeriyor.",
    "Hotel ve area karşılaştırmalarının seçilmiş/heterojen bir coverage evreninden geldiğini gösterir.",
    "NOT_FOUND, hiç complaint olmadığı anlamına gelmez; PAGE_FOUND_NO_COMPLAINT ile de aynı değildir. Bu grafik performans grafiği değildir."
)
"""
)

md("""## 08. Temporal distribution

Tarih coverage'ı ve approximate-date kayıtları sonuçla birlikte raporlanır. 2023 ve 2026 tam yıl
değildir; özellikle 2026, scrape erişilebilirliği ve yılın henüz tamamlanmaması nedeniyle trend
karşılaştırmasına uygun değildir.
""")

code(
    """dated = complaints.dropna(subset=["complaint_date"]).copy()
dated["year"] = dated["complaint_date"].dt.year.astype(int)
dated["month"] = dated["complaint_date"].dt.month.astype(int)
yearly = dated.groupby("year").size().rename("complaint_count").reset_index()
monthly = pd.DataFrame({"month": np.arange(1, 13)}).merge(
    dated.groupby("month").size().rename("complaint_count").reset_index(), on="month", how="left"
).fillna({"complaint_count": 0})
monthly["complaint_count"] = monthly["complaint_count"].astype(int)
monthly["month_name"] = monthly["month"].map({1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})
temporal_summary = pd.concat([
    yearly.assign(granularity="year", period=yearly["year"].astype(str))[["granularity", "period", "complaint_count"]],
    monthly.assign(granularity="month_of_year", period=monthly["month_name"])[["granularity", "period", "complaint_count"]],
], ignore_index=True)
display(pd.DataFrame({
    "metric": ["dated_records", "date_coverage_pct", "approximate_date_count", "min_date", "max_date"],
    "value": [len(dated), 100*len(dated)/len(complaints), int(complaints["complaint_date_is_approximate"].sum()), dated["complaint_date"].min().date(), dated["complaint_date"].max().date()],
}))
display(yearly)
"""
)

code(
    """fig, ax = plt.subplots(figsize=(8.2, 5.0))
bars = ax.bar(yearly["year"].astype(str), yearly["complaint_count"], color=["#B8B8B8", "#6C91BF", "#5B8E7D", "#D98E73"])
ax.bar_label(bars)
ax.set(title="Dated Clean Complaints by Calendar Year", xlabel="Year", ylabel="Complaint count")
ax.grid(axis="y", alpha=0.2)
save_fig(fig, "08_complaints_by_year.png")
"""
)

code(
    """explain_figure(
    "Yalnız complaint_date parse edilen clean kayıtlar takvim yılına göre sayılır.",
    f"Date coverage {format_pct_tr(100*len(dated)/len(complaints))}; gözlenen aralık {dated['complaint_date'].min().date()}–{dated['complaint_date'].max().date()}.",
    "Corpusun hangi yıllarda yoğunlaştığını ve zaman analizinin fiili penceresini gösterir.",
    "2023 ve 2026 kısmi yıldır; scrape/mapping erişilebilirliği zaman içinde sabit değildir. Yıllar arası büyüme sonucu çıkarılamaz."
)
"""
)

code(
    """fig, ax = plt.subplots(figsize=(9.0, 5.0))
bars = ax.bar(monthly["month_name"], monthly["complaint_count"], color="#6C91BF")
ax.bar_label(bars, fontsize=8)
ax.set(title="Dated Clean Complaints by Month of Year", xlabel="Month", ylabel="Complaint count")
ax.grid(axis="y", alpha=0.2)
save_fig(fig, "09_complaints_by_month.png")
"""
)

code(
    """peak_month = monthly.loc[monthly["complaint_count"].idxmax()]
explain_figure(
    "Tüm yıllardaki dated complaints takvim ayına göre birleştirilir; sıfır gözlenen aylar da gösterilir.",
    f"En yüksek corpus hacmi {peak_month['month_name']} ayında {int(peak_month['complaint_count'])} complaint'tir.",
    "Seasonality olasılığı için ilk descriptif görünümü sağlar.",
    "Farklı yılların coverage'ı eşit değildir; yaklaşık tarihler ve scrape erişilebilirliği mevsim profilini bozabilir."
)
"""
)

code(
    """tourism_2025 = tourism_monthly[tourism_monthly["year"] == 2025].copy()
tourism_2025["month"] = pd.to_datetime(tourism_2025["period"]).dt.month
complaints_2025 = pd.DataFrame({"month": np.arange(1, 13)}).merge(
    dated[dated["year"] == 2025].groupby("month").size().rename("complaint_count").reset_index(), on="month", how="left"
).fillna({"complaint_count": 0})
tourism_compare = complaints_2025.merge(tourism_2025[["month", "total_arrivals"]], on="month", how="left")
tourism_compare["complaint_index"] = 100 * tourism_compare["complaint_count"] / tourism_compare["complaint_count"].max()
tourism_compare["mugla_arrival_index"] = 100 * tourism_compare["total_arrivals"] / tourism_compare["total_arrivals"].max()
temporal_tourism_corr = stats.spearmanr(tourism_compare["complaint_count"], tourism_compare["total_arrivals"], nan_policy="omit")
fig, ax = plt.subplots(figsize=(9.0, 5.2))
ax.plot(tourism_compare["month"], tourism_compare["complaint_index"], marker="o", label="Şikayetvar clean complaint index")
ax.plot(tourism_compare["month"], tourism_compare["mugla_arrival_index"], marker="o", label="Muğla total arrival index")
ax.set_xticks(range(1, 13)); ax.set(title="2025 Monthly Complaint Visibility vs Muğla Tourism Context", xlabel="Month", ylabel="Index (within-series peak = 100)")
ax.grid(alpha=0.2); ax.legend()
save_fig(fig, "10_2025_complaints_vs_mugla_tourism.png")
"""
)

code(
    """explain_figure(
    "2025 monthly complaint corpus ile Muğla total arrivals ayrı ayrı peak=100 endeksine çevrilip şekil olarak karşılaştırılır.",
    f"2025 clean dated complaint n={int(tourism_compare['complaint_count'].sum())}; aylık Spearman rho={temporal_tourism_corr.statistic:.3f}, n={int(tourism_compare['total_arrivals'].notna().sum())}.",
    "Complaint görünürlüğünün geniş turizm sezonuyla birlikte hareket edip etmediğine descriptif bağlam verir.",
    "Muğla, Bodrum'dan geniş coğrafyadır; arrival bir complaint denominator'ı değildir. Platform erişimi, complaint delay'i ve approximate dates vardır; nedensellik/trend sonucu çıkarılmaz."
)
"""
)

md("""## 09. Complaint text length""")

code(
    """word_values = complaints["complaint_word_count"].dropna()
word_stats = word_values.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).rename("value").reset_index().rename(columns={"index":"statistic"})
display(word_stats)
display(hotel_summary.loc[hotel_summary["matched_complaint_count"].ge(MIN_COMPLAINTS_FOR_RATE_CHART), [
    "hotel_name", "matched_complaint_count", "median_complaint_word_count", "small_n_flag"
]].sort_values("median_complaint_word_count", ascending=False))
display(area_summary.loc[area_summary["matched_complaint_count"].ge(MIN_COMPLAINTS_FOR_RATE_CHART), [
    "area", "matched_complaint_count", "median_word_count", "coverage_flag"
]].sort_values("median_word_count", ascending=False))
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), gridspec_kw={"width_ratios":[3, 1]})
axes[0].hist(word_values, bins=25, color="#7A5C8E", alpha=0.85)
axes[0].set(title="Complaint Word Count Distribution", xlabel="Word count", ylabel="Complaints")
axes[0].grid(axis="y", alpha=0.2)
axes[1].boxplot(word_values, vert=True, showfliers=True)
axes[1].set(title="Boxplot", ylabel="Word count", xticks=[])
save_fig(fig, "11_complaint_word_count_distribution.png")
"""
)

code(
    """explain_figure(
    "Histogram ve boxplot non-missing complaint text'lerin türetilmiş word count dağılımını gösterir.",
    f"Geçerli text n={len(word_values)}; median={word_values.median():.0f}, Q1={word_values.quantile(.25):.0f}, Q3={word_values.quantile(.75):.0f} kelime.",
    "NLP input uzunluğu, token bütçesi ve kısa/uzun metin dengesini planlamaya yardım eder.",
    "7 missing complaint text bu dağılıma girmez; uzunluk içerik kalitesi veya complaint ciddiyeti değildir."
)
"""
)

md("""## 10. View count distribution""")

code(
    """view_values = complaints["view_count_numeric"].dropna()
view_stats = view_values.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).rename("value").reset_index().rename(columns={"index":"statistic"})
display(view_stats)
display(complaints.nlargest(10, "view_count_numeric")[["complaint_id", "hotel_name", "area", "complaint_date", "view_count_numeric", "complaint_title"]])
fig, ax = plt.subplots(figsize=(8.7, 5.2))
ax.hist(np.log10(view_values + 1), bins=25, color="#D98E73", alpha=0.85)
ax.set(title="Complaint View Count Distribution (Log-transformed)", xlabel="log10(view count + 1)", ylabel="Complaints")
ax.grid(axis="y", alpha=0.2)
save_fig(fig, "12_view_count_distribution.png")
"""
)

code(
    """explain_figure(
    "View count sağa çarpık olduğu için x ekseninde log10(view+1) dönüşümüyle dağılım gösterilir; summary table ham sayıları korur.",
    f"View coverage={format_pct_tr(100*len(view_values)/len(complaints))}; median={view_values.median():.0f}, P95={view_values.quantile(.95):.0f}.",
    "Platform içi görünürlüğün tipik seviyesini ve uzun kuyruğunu görünür kılar.",
    "View count zaman içinde birikir; eski complaint'ler daha fazla exposure alabilir. View, unique reader veya etki ölçüsü değildir."
)
"""
)

code(
    """complaint_word_view_pair = complaints[["complaint_word_count", "view_count_numeric", "complaint_date"]].dropna(subset=["complaint_word_count", "view_count_numeric"])
complaint_word_view_corr = stats.spearmanr(complaint_word_view_pair["complaint_word_count"], complaint_word_view_pair["view_count_numeric"])
fig, ax = plt.subplots(figsize=(8.4, 5.3))
ax.scatter(complaint_word_view_pair["complaint_word_count"], complaint_word_view_pair["view_count_numeric"], alpha=.55, color="#40798C")
ax.set_yscale("log")
ax.set(title="Complaint Text Length vs View Count", xlabel="Complaint word count", ylabel="View count (log scale)")
ax.grid(alpha=.2)
save_fig(fig, "18_word_count_vs_views.png")
"""
)

code(
    """explain_figure(
    "Her nokta bir clean complaint; x text word count, y platform view count'tur. View ekseni sağa çarpıklık nedeniyle log ölçektedir.",
    f"Spearman rho={complaint_word_view_corr.statistic:.3f}, p={complaint_word_view_corr.pvalue:.4f}, n={len(complaint_word_view_pair)}.",
    "Daha uzun complaint metinlerinin platform görünürlüğüyle birlikte değişip değişmediğini descriptif olarak gösterir.",
    "View zamanla birikir; complaint yaşı, marka görünürlüğü ve platform sıralaması karıştırıcıdır. Nedensellik veya içerik ciddiyeti sonucu çıkarılamaz."
)
"""
)

md("""## 06. Complaint distribution by area""")

code(
    """area_summary = build_area_eda_summary(complaints, hotel_mapping, hotel_master)
assert int(area_summary["matched_complaint_count"].sum()) == len(complaints)
area_summary["small_n_flag"] = area_summary["matched_complaint_count"].lt(MIN_COMPLAINTS_FOR_RATE_CHART)
display(area_summary[[
    "area", "project_hotel_count", "mapped_hotel_count", "hotels_with_complaints",
    "mapping_coverage_pct", "matched_complaint_count", "complaint_share_pct", "small_n_flag",
]])
"""
)

code(
    """area_plot = area_summary.sort_values("matched_complaint_count")
fig, ax = plt.subplots(figsize=(9.5, 6.3))
ax.barh(area_plot["area"], area_plot["matched_complaint_count"], color="#3B8C88")
for index, value in enumerate(area_plot["matched_complaint_count"]):
    ax.text(value + 0.5, index, str(int(value)), va="center", fontsize=8)
ax.set(title="Clean Complaint Corpus by Bodrum Area", xlabel="Matched clean complaint count", ylabel="Area")
ax.grid(axis="x", alpha=0.2)
save_fig(fig, "04_complaints_by_area.png")
"""
)

code(
    """top_area = area_summary.iloc[0]
explain_figure(
    "Area çubukları, hotel-level clean complaint sayılarını lokasyon bazında toplar.",
    f"En büyük corpus hacmi {top_area['area']} alanında {int(top_area['matched_complaint_count'])} complaint'tir.",
    "Destination içindeki corpus dağılımını, hotel sayısı ve mapping coverage ile birlikte okumayı sağlar.",
    "Area toplamları gerçek turizm talebi veya gerçek complaint rate değildir; alanlardaki hotel ve mapping coverage farklıdır."
)
"""
)

code(
    """fig, ax = plt.subplots(figsize=(8.5, 5.5))
sizes = 40 + 10 * area_summary["project_hotel_count"]
ax.scatter(area_summary["mapping_coverage_pct"], area_summary["matched_complaint_count"], s=sizes, alpha=0.75, color="#D98E73")
for _, row in area_summary.iterrows():
    ax.annotate(row["area"], (row["mapping_coverage_pct"], row["matched_complaint_count"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
ax.set(title="Area Mapping Coverage vs Clean Complaint Volume", xlabel="Hotel mapping coverage (%)", ylabel="Matched clean complaint count")
ax.grid(alpha=0.2)
save_fig(fig, "05_area_coverage_vs_complaints.png")
"""
)

code(
    """area_cov_corr = stats.spearmanr(area_summary["mapping_coverage_pct"], area_summary["matched_complaint_count"], nan_policy="omit")
explain_figure(
    "Her nokta bir area; x ekseni project hotel master içindeki mapping coverage, y ekseni clean complaint sayısıdır. Nokta boyutu project hotel sayısını gösterir.",
    f"Area-level Spearman ilişki rho={area_cov_corr.statistic:.3f}; n={len(area_summary)}. Sıfır complaint'li alanlar da grafiktedir.",
    "Yüksek/düşük area hacminin coverage ile birlikte değerlendirilmesi gerektiğini gösterir.",
    "Yalnız 14 area vardır; ilişki nedensel değildir ve platform erişilebilirliği ile hotel karması sonucu etkiler."
)
"""
)

md("""## 07. Cross-platform visibility context

`cross_platform_complaint_visibility_per_1000_google_reviews` gerçek bir complaint rate değildir. Numeratör Şikayetvar'daki
clean mapped corpus; denominator Google'daki all-time review count'tur. Dönem, kullanıcı tabanı ve
platform davranışı aynı değildir. Bu nedenle metrik yalnız **cross-platform visibility index** olarak
yorumlanır.
""")

code(
    """cross_platform_summary = hotel_summary[[
    "hotel_name", "area", "matched_complaint_count", "google_review_count", "google_rating",
    "cross_platform_complaint_visibility_per_1000_google_reviews", "low_google_review_denominator_flag", "small_n_flag",
]].copy()
cross_platform_summary["interpretation_caution"] = "Visibility index; not a real complaint rate"
display(cross_platform_summary.sort_values("cross_platform_complaint_visibility_per_1000_google_reviews", ascending=False).head(10))
print("Lowest-quartile Google review denominator threshold:", google_review_denominator_threshold)
"""
)

code(
    """google_pair = hotel_summary.dropna(subset=["google_review_count", "matched_complaint_count"])
google_corr = stats.spearmanr(google_pair["google_review_count"], google_pair["matched_complaint_count"])
fig, ax = plt.subplots(figsize=(8.6, 5.7))
ax.scatter(google_pair["google_review_count"], google_pair["matched_complaint_count"], alpha=0.75, color="#40798C")
for _, row in google_pair.nlargest(7, "matched_complaint_count").iterrows():
    ax.annotate(row["hotel_name"], (row["google_review_count"], row["matched_complaint_count"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
ax.set_xscale("log")
ax.set(title="Google Review Count vs Clean Complaint Corpus Volume", xlabel="Google review count (log scale)", ylabel="Matched clean complaint count")
ax.grid(alpha=0.2)
save_fig(fig, "06_google_reviews_vs_complaints.png")
"""
)

code(
    """explain_figure(
    "Her nokta bir hotel; Google review count genel online görünürlüğe bağlam, clean complaint count ise bu projenin Şikayetvar corpus hacmini verir.",
    f"Spearman rho={google_corr.statistic:.3f}, p={google_corr.pvalue:.4f}, n={len(google_pair)}; daha görünür hotel'lerde corpus hacmi de çoğunlukla daha yüksektir.",
    "Ham complaint count karşılaştırmalarında exposure/visibility bağlamının neden gerekli olduğunu gösterir.",
    "Platformlar, zaman pencereleri ve kullanıcı davranışı farklıdır; Google review count müşteri sayısı değildir."
)
"""
)

code(
    """visibility_pair = hotel_summary.dropna(subset=["google_rating", "cross_platform_complaint_visibility_per_1000_google_reviews"])
visibility_corr = stats.spearmanr(visibility_pair["google_rating"], visibility_pair["cross_platform_complaint_visibility_per_1000_google_reviews"])
colors = np.where(visibility_pair["low_google_review_denominator_flag"], "#C1666B", "#5B8E7D")
fig, ax = plt.subplots(figsize=(8.6, 5.7))
ax.scatter(visibility_pair["google_rating"], visibility_pair["cross_platform_complaint_visibility_per_1000_google_reviews"], c=colors, alpha=0.8)
for _, row in visibility_pair.nlargest(6, "cross_platform_complaint_visibility_per_1000_google_reviews").iterrows():
    ax.annotate(row["hotel_name"], (row["google_rating"], row["cross_platform_complaint_visibility_per_1000_google_reviews"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
ax.set(title="Google Rating vs Cross-platform Complaint Visibility", xlabel="Google rating", ylabel="Clean complaints per 1,000 Google reviews")
ax.grid(alpha=0.2)
save_fig(fig, "07_visibility_vs_google_rating.png")
"""
)

code(
    """explain_figure(
    "Her nokta bir hotel; kırmızı noktalar Google review denominator'ı alt çeyrekte olan, daha oynak görünürlük indekslerini işaretler.",
    f"Spearman rho={visibility_corr.statistic:.3f}, p={visibility_corr.pvalue:.4f}, n={len(visibility_pair)}.",
    "Google rating ile Şikayetvar corpus görünürlüğü arasındaki yönsel ilişkiyi keşifsel olarak gösterir.",
    "Bu eksenler ortak bir memnuniyet ölçeği değildir. İndeks complaint rate değildir; düşük denominator ve küçük complaint sample'ları özellikle ihtiyat ister."
)
"""
)

code(
    """company_response_count = int(complaints["company_response_exists_clean"].fillna(False).sum())
complaints_with_replies = int(complaints["reply_count_total_derived"].gt(0).sum())
corpus_kpis = pd.DataFrame([
    ("Clean matched complaints", len(complaints), "Canonical-unique complaint-level matched"),
    ("Hotels with matched complaints", complaints["hotel_id"].nunique(), "Selected mapped-hotel sample"),
    ("Areas represented", complaints["area"].nunique(), "11/14 project area"),
    ("Company responses", company_response_count, format_pct_tr(100*company_response_count/len(complaints))),
    ("Total replies", len(replies), "COMPANY + USER + UNKNOWN"),
    ("Complaints with replies", complaints_with_replies, format_pct_tr(100*complaints_with_replies/len(complaints))),
    ("Median complaint words", complaints["complaint_word_count"].median(), "Missing text excluded"),
    ("Date coverage", int(complaints["complaint_date"].notna().sum()), format_pct_tr(100*complaints["complaint_date"].notna().mean())),
    ("View coverage", int(complaints["view_count_numeric"].notna().sum()), format_pct_tr(100*complaints["view_count_numeric"].notna().mean())),
], columns=["KPI", "Value", "Interpretation"])
display(corpus_kpis)
"""
)

md("""### Bölüm Sonucu

Ana EDA evreni project hotel masterının tamamı değil, complaint-level güvenilir eşleşmesi bulunan 32
hoteldir. Bu coverage sınırı tüm hotel, area ve cross-platform sonuçlarının parçasıdır.
""")

md("""## 04. Complaint volume by hotel""")

code(
    """hotel_summary, google_review_denominator_threshold = build_hotel_eda_summary(
    complaints, min_complaints_for_rate=MIN_COMPLAINTS_FOR_RATE_CHART, denominator_quantile=0.25
)
assert hotel_summary["matched_complaint_count"].sum() == len(complaints)
hotel_summary.insert(0, "rank_in_corpus", np.arange(1, len(hotel_summary) + 1))
display(hotel_summary.head(10)[[
    "rank_in_corpus", "hotel_name", "area", "matched_complaint_count", "share_of_corpus_pct",
    "google_review_count", "google_rating", "company_response_count", "company_response_rate_in_corpus",
]])
"""
)

code(
    """hotel_plot = hotel_summary.sort_values("matched_complaint_count")
fig, ax = plt.subplots(figsize=(10.5, 10.5))
ax.barh(hotel_plot["hotel_name"], hotel_plot["matched_complaint_count"], color="#2F6B7C")
for index, value in enumerate(hotel_plot["matched_complaint_count"]):
    ax.text(value + 0.25, index, str(value), va="center", fontsize=8)
ax.set(title=f"Clean Complaint Corpus by Hotel — n={len(complaints)}, hotels={len(hotel_summary)}", xlabel="Matched clean complaint count", ylabel="Hotel")
ax.set_xlim(0, hotel_plot["matched_complaint_count"].max() * 1.12); ax.grid(axis="x", alpha=0.2)
save_fig(fig, "02_complaint_count_by_hotel.png")
"""
)

code(
    """top_hotel = hotel_summary.iloc[0]
explain_figure(
    "Her çubuk clean corpusta bir hotele bağlanan canonical-unique complaint sayısını gösterir.",
    f"En yüksek corpus hacmi {top_hotel['hotel_name']} için {int(top_hotel['matched_complaint_count'])} kayıttır; toplam {len(hotel_summary)} hotel temsil edilir.",
    "Corpusun hotel düzeyinde ne kadar dengesiz dağıldığını ve hangi gruplarda EDA/NLP sample'ının daha büyük olduğunu gösterir.",
    "Bu sayı hotel kalitesi, gerçek müşteri complaint rate veya 'en kötü hotel' sıralaması değildir. Platform görünürlüğü ve müşteri hacmi bilinmiyor."
)
"""
)

md("""## 05. Complaint concentration""")

code(
    """concentration = concentration_metrics(hotel_summary)
concentration_table = pd.DataFrame({"metric": list(concentration), "value": list(concentration.values())})
display(concentration_table)

ranked = hotel_summary.sort_values("matched_complaint_count", ascending=False).copy()
ranked["hotel_rank"] = np.arange(1, len(ranked) + 1)
ranked["cumulative_complaint_share_pct"] = 100 * ranked["matched_complaint_count"].cumsum() / len(complaints)
fig, ax = plt.subplots(figsize=(9.2, 5.2))
ax.plot(ranked["hotel_rank"], ranked["cumulative_complaint_share_pct"], marker="o", color="#7A5C8E")
ax.axhline(80, linestyle="--", color="#C1666B", label="%80 corpus share")
ax.axvline(concentration["hotels_to_reach_80pct"], linestyle=":", color="#777777", label=f"{concentration['hotels_to_reach_80pct']} hotel")
ax.set(title="Cumulative Clean Complaint Share by Ranked Hotel", xlabel="Hotel rank by corpus complaint count", ylabel="Cumulative complaint share (%)", ylim=(0, 105))
ax.grid(alpha=0.2); ax.legend()
save_fig(fig, "03_complaint_concentration.png")
"""
)

code(
    """explain_figure(
    "Hotel'ler corpus complaint sayısına göre sıralanır; çizgi ilk N hotelin toplam corpus payını gösterir.",
    f"Top-5 hotel corpusun {format_pct_tr(concentration['top5_hotel_complaint_share_pct'])}'ini taşır; %80 paya {int(concentration['hotels_to_reach_80pct'])} hotelde ulaşılır.",
    "NLP ve hotel-level karşılaştırmalarda sample büyüklüğünün az sayıda hotelde yoğunlaştığını gösterir.",
    "Concentration kalite sorunu değildir; yalnız scraped/mapped corpus dağılımıdır."
)
"""
)

md("""## 11. Company response behavior""")

code(
    """response_overall = pd.DataFrame({
    "metric": ["company_response_count", "no_company_response_count", "company_response_rate_in_corpus_pct", "rate_chart_min_n"],
    "value": [int(complaints["company_response_exists_clean"].sum()), int((~complaints["company_response_exists_clean"]).sum()), 100*complaints["company_response_exists_clean"].mean(), MIN_COMPLAINTS_FOR_RATE_CHART],
})
display(response_overall)
response_group_comparison = complaints.assign(
    response_group=np.where(complaints["company_response_exists_clean"], "RESPONDED", "NOT_RESPONDED")
).groupby("response_group").agg(
    complaint_n=("complaint_id", "size"),
    median_word_count=("complaint_word_count", "median"),
    median_view_count=("view_count_numeric", "median"),
    median_company_response_word_count=("company_response_word_count", "median"),
).reset_index()
display(response_group_comparison)
display(area_summary[["area", "matched_complaint_count", "company_response_count", "company_response_rate_in_corpus", "small_n_flag"]])
response_chart = hotel_summary[hotel_summary["matched_complaint_count"] >= MIN_COMPLAINTS_FOR_RATE_CHART].sort_values("company_response_rate_in_corpus")
fig, ax = plt.subplots(figsize=(9.5, 6.7))
ax.barh(response_chart["hotel_name"], response_chart["company_response_rate_in_corpus"], color="#5B8E7D")
for index, row in enumerate(response_chart.itertuples()):
    ax.text(row.company_response_rate_in_corpus + 1, index, f"n={row.matched_complaint_count}", va="center", fontsize=7)
ax.set(title=f"Company Response Share in Clean Corpus — Hotels with n≥{MIN_COMPLAINTS_FOR_RATE_CHART}", xlabel="Complaints with a matched company response (%)", ylabel="Hotel", xlim=(0, 108))
ax.grid(axis="x", alpha=0.2)
save_fig(fig, "13_company_response_share_by_hotel.png")
"""
)

code(
    """explain_figure(
    "Yalnız en az 5 clean complaint'i bulunan hotel'lerde, complaint'lerin kaçında matched company response görüldüğünü gösterir; etiket n'dir.",
    f"Overall clean corpus response count={int(complaints['company_response_exists_clean'].sum())}/{len(complaints)} ({format_pct_tr(100*complaints['company_response_exists_clean'].mean())}).",
    "Platform içinde kurumsal yanıt davranışının coverage-aware karşılaştırmasını sağlar.",
    "Response varlığı çözüm, memnuniyet veya yanıt kalitesi değildir. Küçük örneklem oynaklığı sürer; eşik dışı hotel'ler summary tabloda tutulur."
)
"""
)

md("""## 12. Company response time""")

code(
    """company_response_time_summary = response_time_summary(complaints)
display(company_response_time_summary)
valid_response_time = complaints.loc[
    complaints["company_response_exists_clean"] & complaints["response_time_days"].notna(), "response_time_days"
]
display(complaints.loc[
    complaints["company_response_exists_clean"] & complaints["response_time_days"].notna(),
    ["hotel_name", "complaint_id", "complaint_date", "company_response_date_parsed", "response_time_days", "complaint_date_is_approximate"],
].nlargest(8, "response_time_days"))
display_q95 = valid_response_time.quantile(.95)
fig, ax = plt.subplots(figsize=(8.7, 5.1))
ax.hist(valid_response_time.clip(upper=display_q95), bins=20, color="#C1666B", alpha=0.85)
ax.axvline(valid_response_time.median(), linestyle="--", color="#333333", label=f"Median={valid_response_time.median():.2f} days")
ax.set(title="Company Response Time Distribution (Display Capped at P95)", xlabel="Days from complaint to company response", ylabel="Complaints")
ax.grid(axis="y", alpha=0.2); ax.legend()
save_fig(fig, "14_company_response_time_distribution.png")
"""
)

code(
    """explain_figure(
    "Complaint ve company response tarihleri bulunan, non-negative lag'li kayıtların response time dağılımıdır; okunabilirlik için yalnız grafikte P95 üstü P95'e kırpılır.",
    f"Valid response-time n={len(valid_response_time)}; median={valid_response_time.median():.2f} gün, P75={valid_response_time.quantile(.75):.2f}, max={valid_response_time.max():.2f}. Full değerler summary tabloda korunur.",
    "Tipik yanıt hızını ve sağa çarpık gecikme kuyruğunu gösterir.",
    "Approximate complaint dates vardır; tarih farkı operasyonel SLA değildir. Aykırılar grafikte kırpılmış, tabloda saklanmıştır."
)
"""
)

md("""## 13. Reply behavior""")

code(
    """reply_author_summary = replies.groupby("reply_author_type_clean", dropna=False).size().rename("reply_count").reset_index()
reply_count_dist = complaints["reply_count_total_derived"].value_counts().sort_index().rename_axis("reply_count_per_complaint").reset_index(name="complaint_count")
replied = complaints["reply_count_total_derived"] > 0
reply_behavior_summary = pd.DataFrame({
    "metric": ["clean_reply_count", "complaints_with_any_reply", "complaints_without_reply", "share_with_any_reply_pct", "median_replies_among_replied"],
    "value": [len(replies), int(replied.sum()), int((~replied).sum()), 100*replied.mean(), complaints.loc[replied, "reply_count_total_derived"].median()],
})
display(reply_behavior_summary)
display(reply_author_summary)
reply_group_comparison = complaints.assign(
    reply_group=np.where(replied, "REPLIED", "NOT_REPLIED")
).groupby("reply_group").agg(
    complaint_n=("complaint_id", "size"),
    median_word_count=("complaint_word_count", "median"),
    median_view_count=("view_count_numeric", "median"),
).reset_index()
display(reply_group_comparison)
display(hotel_summary.loc[hotel_summary["matched_complaint_count"].ge(MIN_COMPLAINTS_FOR_RATE_CHART), [
    "hotel_name", "matched_complaint_count", "complaints_with_replies", "total_reply_count", "median_reply_count", "max_reply_count", "reply_coverage_pct"
]].sort_values("total_reply_count", ascending=False))
"""
)

code(
    """fig, ax = plt.subplots(figsize=(7.5, 4.8))
bars = ax.bar(reply_author_summary["reply_author_type_clean"].astype(str), reply_author_summary["reply_count"], color="#6C91BF")
ax.bar_label(bars)
ax.set(title="Clean Replies by Author Type", xlabel="Reply author type", ylabel="Reply count")
ax.grid(axis="y", alpha=0.2)
save_fig(fig, "15_reply_author_distribution.png")
"""
)

code(
    """explain_figure(
    "Her çubuk clean reply kayıtlarını normalized reply author type'a göre sayar.",
    f"Toplam {len(replies)} clean reply, {int(replied.sum())} complaint'e bağlıdır.",
    "Company/user reply ayrımını ve gelecekteki conversation-level NLP evrenini gösterir.",
    "Reply sayısı çözüm veya interaction kalitesi değildir; clean corpus dışında kalan review-required replies dahil değildir."
)
"""
)

code(
    """fig, ax = plt.subplots(figsize=(7.8, 4.8))
bars = ax.bar(reply_count_dist["reply_count_per_complaint"].astype(str), reply_count_dist["complaint_count"], color="#7A5C8E")
ax.bar_label(bars)
ax.set(title="Replies per Clean Complaint", xlabel="Clean reply count per complaint", ylabel="Complaint count")
ax.grid(axis="y", alpha=0.2)
save_fig(fig, "16_replies_per_complaint.png")
"""
)

code(
    """explain_figure(
    "Clean complaint'ler kendilerine bağlanan clean reply sayısına göre gruplandırılır; sıfır reply de dahildir.",
    f"Any-reply share={format_pct_tr(100*replied.mean())}; replied complaints içinde median={complaints.loc[replied, 'reply_count_total_derived'].median():.0f} reply.",
    "Thread yoğunluğunu ve conversation-style NLP için mevcut interaction derinliğini gösterir.",
    "Scraper yalnız erişilebilen reply'ları görür; reply count konuşmanın eksiksizliği veya çözüm sonucu değildir."
)
"""
)

md("""## 14. Limited official star, capacity and price context""")

code(
    """official_context = hotel_summary[[
    "hotel_name", "matched_complaint_count", "cross_platform_complaint_visibility_per_1000_google_reviews", "official_star_rating_verified",
    "official_room_count", "official_bed_count", "search_price_usd_snapshot",
]].copy()
official_coverage = pd.DataFrame({
    "field": ["official_star_rating_verified", "official_room_count", "search_price_usd_snapshot"],
    "non_missing_hotels": [official_context[c].notna().sum() for c in ["official_star_rating_verified", "official_room_count", "search_price_usd_snapshot"]],
    "hotel_universe": len(official_context),
})
official_coverage["coverage_pct"] = 100*official_coverage["non_missing_hotels"]/official_coverage["hotel_universe"]
display(official_coverage)
display(hotel_summary.groupby("official_star_rating_verified", dropna=False).agg(
    hotel_count=("hotel_id", "size"),
    clean_complaints=("matched_complaint_count", "sum"),
    median_complaints=("matched_complaint_count", "median"),
    median_cross_platform_visibility=("cross_platform_complaint_visibility_per_1000_google_reviews", "median"),
).reset_index())
"""
)

md("""### Bölüm Sonucu

Official room capacity yalnız 10/32 hotel için bulunduğundan kapasite ilişkileri keşifsel kalır.
Fiyat tek bir source-date snapshot'ıdır ve sezon/oda tipi/paket farklarını temsil etmez. Star/capacity/
price karşılaştırmaları hiçbir biçimde kalite sıralaması olarak kullanılmaz.
""")

md("""## 15. Hotel-level exploratory correlations""")

code(
    """correlation_pairs = [
    ("google_review_count", "matched_complaint_count"),
    ("google_rating", "matched_complaint_count"),
    ("google_rating", "cross_platform_complaint_visibility_per_1000_google_reviews"),
    ("official_room_count", "matched_complaint_count"),
    ("official_room_count", "cross_platform_complaint_visibility_per_1000_google_reviews"),
    ("search_price_usd_snapshot", "matched_complaint_count"),
    ("search_price_usd_snapshot", "cross_platform_complaint_visibility_per_1000_google_reviews"),
    ("median_complaint_word_count", "median_view_count"),
    ("matched_complaint_count", "company_response_rate_in_corpus"),
]
hotel_level_correlations = spearman_table(hotel_summary, correlation_pairs)
display(hotel_level_correlations)

complaint_word_view_pair = complaints[["complaint_word_count", "view_count_numeric"]].dropna()
complaint_word_view_corr = stats.spearmanr(complaint_word_view_pair["complaint_word_count"], complaint_word_view_pair["view_count_numeric"])
display(pd.DataFrame([{
    "level":"complaint", "metric_x":"complaint_word_count", "metric_y":"view_count_numeric",
    "n":len(complaint_word_view_pair), "spearman_rho":complaint_word_view_corr.statistic,
    "p_value":complaint_word_view_corr.pvalue,
    "caution":"View accumulates with age; association is exploratory and non-causal."
}]))
"""
)

code(
    """heatmap_columns = [
    "matched_complaint_count", "google_review_count", "google_rating",
    "cross_platform_complaint_visibility_per_1000_google_reviews", "official_room_count",
    "search_price_usd_snapshot", "median_complaint_word_count", "median_view_count",
    "company_response_rate_in_corpus", "reply_coverage_pct",
]
corr_matrix = numeric_correlation_matrix(hotel_summary, heatmap_columns)
fig, ax = plt.subplots(figsize=(11.5, 9.0))
image = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns, rotation=55, ha="right")
ax.set_yticks(np.arange(len(corr_matrix.index)), labels=corr_matrix.index)
for i in range(len(corr_matrix.index)):
    for j in range(len(corr_matrix.columns)):
        value = corr_matrix.iloc[i, j]
        ax.text(j, i, "" if pd.isna(value) else f"{value:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(image, ax=ax, shrink=.75, label="Spearman rho")
ax.set_title("Hotel-level Exploratory Spearman Correlation Matrix")
save_fig(fig, "17_hotel_level_correlation_heatmap.png")
"""
)

code(
    """explain_figure(
    "Hücreler hotel-level değişken çiftlerinin Spearman rho değerini gösterir; kırmızı aynı yön, mavi ters yön birlikteliktir.",
    "En dikkat çekici ham ilişki room coverage bulunan küçük alt örneklemde capacity ile complaint count arasındadır; normalized visibility ilişkisi daha zayıftır.",
    "Exposure, kapasite, fiyat, response ve text yapısını tek bir keşifsel haritada karşılaştırır.",
    "Her hücrenin n'i farklı olabilir ve heatmap bunu göstermez; kesin n ve p-value CSV tablosundadır. Official room n çok düşüktür, price snapshot'tır, nedensellik yoktur."
)
"""
)

md("""## 16. Notable cases — investigation queue, not ranking""")

code(
    """notable_specs = [
    ("matched_complaint_count", "HIGH_CORPUS_VOLUME", "High clean-corpus visibility; exposure context required", "largest"),
    ("cross_platform_complaint_visibility_per_1000_google_reviews", "HIGH_CROSS_PLATFORM_VISIBILITY", "High cross-platform visibility index; not a complaint rate", "largest"),
    ("median_view_count", "HIGH_MEDIAN_VIEW_MIN_N", "High median view among hotels with complaint n>=5; age confounding", "largest"),
    ("company_response_rate_in_corpus", "LOW_RESPONSE_SHARE_MIN_N", "Low matched-response share among hotels with complaint n>=5; not resolution", "smallest"),
]
notable_records = []
for metric, reason_flag, reason, direction in notable_specs:
    pool = hotel_summary if metric in {"matched_complaint_count", "cross_platform_complaint_visibility_per_1000_google_reviews"} else hotel_summary.loc[hotel_summary["matched_complaint_count"].ge(MIN_COMPLAINTS_FOR_RATE_CHART)]
    selected = pool.nlargest(5, metric) if direction == "largest" else pool.nsmallest(5, metric)
    for row in selected.itertuples():
        notable_records.append({
            "hotel_name": row.hotel_name,
            "area": row.area,
            "metric": metric,
            "value": getattr(row, metric),
            "matched_complaint_count": row.matched_complaint_count,
            "reason_flag": reason_flag,
            "selection_reason": reason,
            "small_n_flag": row.small_n_flag,
            "low_google_review_denominator_flag": row.low_google_review_denominator_flag,
            "caution_note": "Selected mapped corpus; platform visibility and sample size differ; no quality ranking.",
        })
notable_cases = pd.DataFrame(notable_records).sort_values(["reason_flag", "value"], ascending=[True, False])
display(notable_cases)
"""
)

md("""## 17. What can and cannot be concluded""")

md("""### Güvenle söylenebilecekler

- Clean corpus 236 canonical-unique complaint, 32 reliably mapped hotel ve 11 represented area içerir.
- Corpus hacmi hotel/area düzeyinde eşit dağılmaz; concentration sayısal olarak raporlanabilir.
- Parse edilen tarih, view, structured response ve clean reply alanlarının mevcut coverage'ı ölçülebilir.
- Hotel-level Google context ve sınırlı official metadata ile yalnız keşifsel sıralı birliktelikler hesaplanabilir.

### Söylenemeyecekler

- Hangi hotel'in “en iyi/en kötü” olduğu veya gerçek complaint rate'i.
- Şikayetvar complaint sayısının toplam müşteri sayısına oranı.
- Company response varlığının complaint'i çözdüğü veya kullanıcıyı memnun ettiği.
- 2023–2026 corpus sayılarının gerçek pazar trendi olduğu.
- Korelasyonların nedensel, platformlar arası doğrudan karşılaştırılabilir veya tüm Bodrum hotel evrenine genellenebilir olduğu.
""")

md("""## 18. NLP sample readiness""")

code(
    """nlp_sample_readiness = hotel_summary[[
    "hotel_id", "hotel_name", "area", "matched_complaint_count", "complaints_with_replies",
    "company_response_count", "median_complaint_word_count", "small_n_flag", "nlp_sample_tier",
]].copy()
nlp_sample_readiness["tier_rule"] = np.select(
    [nlp_sample_readiness["matched_complaint_count"].ge(15), nlp_sample_readiness["matched_complaint_count"].ge(5)],
    ["HIGH_SAMPLE: n>=15", "MEDIUM_SAMPLE: 5<=n<15"],
    default="LOW_SAMPLE: n<5",
)
nlp_sample_readiness["recommended_use"] = nlp_sample_readiness["nlp_sample_tier"].map({
    "HIGH_SAMPLE":"Hotel-level topic/aspect exploration with uncertainty reporting",
    "MEDIUM_SAMPLE":"Pooled or area-level analysis; hotel findings only as indicative",
    "LOW_SAMPLE":"Pool into broader corpus; avoid hotel-level inference",
})
nlp_sample_readiness["main_caution"] = "Document count alone does not establish representativeness."
display(nlp_sample_readiness.groupby("nlp_sample_tier").agg(hotel_count=("hotel_id", "size"), complaint_count=("matched_complaint_count", "sum")).reset_index())
display(nlp_sample_readiness)
"""
)

md("""### Topic-modeling suitability

Genel corpus 229 non-missing complaint text ile pooled topic/aspect discovery için yeterli bir başlangıç
evrenidir. Hotel-level topic modelleme yalnız `HIGH_SAMPLE` dört hotelde bile küçük-n belirsizliğiyle
ele alınmalıdır. `MEDIUM_SAMPLE` grubu area/pool yaklaşımına, `LOW_SAMPLE` ise yalnız corpus-wide
modellemeye katılmalıdır. Notebook 14 önce language/boilerplate, minimum document frequency,
rare-topic stability ve human validation kontrolleri uygulamalıdır.
""")

md("""## 19. Key findings""")

code(
    """top5_names = ", ".join(hotel_summary.head(5)["hotel_name"])
top_visibility_names = ", ".join(hotel_summary.nlargest(5, "cross_platform_complaint_visibility_per_1000_google_reviews")["hotel_name"])
tier_counts = nlp_sample_readiness["nlp_sample_tier"].value_counts()
key_findings = [
    f"Clean EDA corpus: {len(complaints)} canonical-unique complaint, {complaints['hotel_id'].nunique()} hotel, {complaints['area'].nunique()} represented area.",
    f"Top-5 corpus-volume hotels: {top5_names}.",
    f"Top-3 / Top-5 / Top-10 complaint corpus shares: {concentration['top3_hotel_complaint_share_pct']:.1f}% / {concentration['top5_hotel_complaint_share_pct']:.1f}% / {concentration['top10_hotel_complaint_share_pct']:.1f}%.",
    f"{int(concentration['hotels_to_reach_80pct'])} hotels account for 80% of the clean complaint corpus; HHI={concentration['hhi_hotel_complaint_concentration']:.3f}.",
    f"Highest area corpus volume is {area_summary.iloc[0]['area']} with {int(area_summary.iloc[0]['matched_complaint_count'])} complaints; area results must be read with mapping coverage (rho={area_cov_corr.statistic:.3f}, n={len(area_summary)}).",
    f"Google review count vs clean complaint count: Spearman rho={google_corr.statistic:.3f}, n={len(google_pair)}; raw volume requires visibility context.",
    f"Google rating vs cross-platform visibility index: rho={visibility_corr.statistic:.3f}, n={len(visibility_pair)}; index is not a real complaint rate.",
    f"Highest cross-platform visibility candidates: {top_visibility_names}; low Google denominators and small complaint n are flagged.",
    f"Date coverage: {100*len(dated)/len(complaints):.1f}%; observed {dated['complaint_date'].min().date()} to {dated['complaint_date'].max().date()}, with {int(complaints['complaint_date_is_approximate'].sum())} approximate dates.",
    f"Complaint text coverage: {len(word_values)}/{len(complaints)}; median {word_values.median():.0f} words (Q1 {word_values.quantile(.25):.0f}, Q3 {word_values.quantile(.75):.0f}).",
    f"View coverage: {len(view_values)}/{len(complaints)}; median {view_values.median():.0f}; views accumulate with age.",
    f"Matched company response observed for {int(complaints['company_response_exists_clean'].sum())}/{len(complaints)} complaints ({100*complaints['company_response_exists_clean'].mean():.1f}%); response does not imply resolution.",
    f"Among hotels with n>={MIN_COMPLAINTS_FOR_RATE_CHART}, matched-response shares span {response_chart['company_response_rate_in_corpus'].min():.1f}% to {response_chart['company_response_rate_in_corpus'].max():.1f}%; this is behavior, not outcome quality.",
    f"Valid response lag n={len(valid_response_time)}; median {valid_response_time.median():.2f} days, P75 {valid_response_time.quantile(.75):.2f}, max {valid_response_time.max():.2f}; approximate dates limit precision.",
    f"Clean replies: {len(replies)} across {int(replied.sum())} complaints; any-reply share {100*replied.mean():.1f}%.",
    f"Official room coverage: {int(hotel_summary['official_room_count'].notna().sum())}/{len(hotel_summary)} hotels; capacity correlations are highly exploratory.",
    f"NLP readiness: HIGH={int(tier_counts.get('HIGH_SAMPLE', 0))}, MEDIUM={int(tier_counts.get('MEDIUM_SAMPLE', 0))}, LOW={int(tier_counts.get('LOW_SAMPLE', 0))} hotels; low samples should be pooled.",
    "Notebook 14 should prioritize pooled aspect mentions, sufficient-N hotel/area contrasts, distinctive terms and company response by aspect with human validation.",
]
for finding in key_findings:
    print("-", finding)
"""
)

md("""## 20. Limitations""")

code(
    """limitations = [
    "Şikayetvar is a self-selected negative-customer-voice platform; the corpus is not representative of all stays.",
    "Complaint count is corpus volume, not hotel quality and not a true complaint rate.",
    "Only reliably mapped complaint-level records are included: 32 of 192 project hotels and 11 of 14 areas have clean complaints.",
    "Mapping statuses and platform accessibility are uneven; zero complaints may mean no trusted mapping or no accessible data.",
    "Google Reviews and Şikayetvar use different populations, time windows, prompts and behaviors.",
    "Complaints per 1,000 Google reviews is a cross-platform visibility index, not a statistically valid rate.",
    "2023 and 2026 are partial/accessibility-limited; yearly differences cannot establish trends.",
    "86 complaint dates are approximate and 10 are missing/unparsed, limiting daily response-time precision.",
    "View count accumulates over time and is not a unique-reader or impact measure.",
    "Company response existence does not mean resolution, response quality or customer satisfaction.",
    "Official room/star metadata coverage is limited; price is a single search snapshot.",
    "All correlations are exploratory, sample sizes vary, multiple testing is unadjusted and causality is not supported.",
    "Small-n hotel-level medians and rates are unstable even when the n>=5 display threshold is met.",
    "Seven missing complaint texts are excluded from text-length and future NLP content analyses.",
]
for item in limitations:
    print("-", item)
"""
)

md("""## 21. Exports, validation and Notebook 14 handoff""")

code(
    """output_paths = {
    "hotel_summary": REPORTS_DIR / "sikayetvar_hotel_eda_summary.csv",
    "area_summary": REPORTS_DIR / "sikayetvar_area_eda_summary.csv",
    "response_time": REPORTS_DIR / "sikayetvar_company_response_time_summary.csv",
    "correlations": REPORTS_DIR / "sikayetvar_hotel_level_correlations.csv",
    "notable_cases": REPORTS_DIR / "sikayetvar_eda_notable_cases.csv",
    "key_findings": REPORTS_DIR / "sikayetvar_eda_key_findings.txt",
    "limitations": REPORTS_DIR / "sikayetvar_eda_limitations.txt",
    "temporal": REPORTS_DIR / "sikayetvar_temporal_summary.csv",
    "cross_platform": REPORTS_DIR / "sikayetvar_cross_platform_visibility.csv",
    "reply_behavior": REPORTS_DIR / "sikayetvar_reply_behavior_summary.csv",
    "nlp_readiness": REPORTS_DIR / "sikayetvar_nlp_sample_readiness.csv",
}

hotel_summary.to_csv(output_paths["hotel_summary"], index=False)
area_summary.to_csv(output_paths["area_summary"], index=False)
company_response_time_summary.to_csv(output_paths["response_time"], index=False)
hotel_level_correlations.to_csv(output_paths["correlations"], index=False)
notable_cases.to_csv(output_paths["notable_cases"], index=False)
temporal_summary.to_csv(output_paths["temporal"], index=False)
cross_platform_summary.to_csv(output_paths["cross_platform"], index=False)
reply_behavior_summary.to_csv(output_paths["reply_behavior"], index=False)
nlp_sample_readiness.to_csv(output_paths["nlp_readiness"], index=False)
output_paths["key_findings"].write_text("\\n".join(f"- {item}" for item in key_findings) + "\\n", encoding="utf-8")
output_paths["limitations"].write_text("\\n".join(f"- {item}" for item in limitations) + "\\n", encoding="utf-8")

clean_hashes_after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected_clean_paths}
figure_paths = sorted(FIGURES_DIR.glob("*.png"))
expected_figures = [f"{index:02d}_" for index in range(1, 19)]
validation_checks = [
    ("clean_complaint_hashes_unchanged", clean_hashes_before == clean_hashes_after, "Notebook 12 clean files remain immutable"),
    ("hotel_aggregation_reconciles", int(hotel_summary["matched_complaint_count"].sum()) == len(complaints), "Hotel counts sum to corpus"),
    ("area_aggregation_reconciles", int(area_summary["matched_complaint_count"].sum()) == len(complaints), "Area counts sum to corpus"),
    ("response_count_reconciles", int(hotel_summary["company_response_count"].sum()) == int(complaints["company_response_exists_clean"].sum()), "Response counts reconcile"),
    ("cross_platform_denominators_positive", hotel_summary.loc[hotel_summary["cross_platform_complaint_visibility_per_1000_google_reviews"].notna(), "google_review_count"].gt(0).all(), "No invalid denominator"),
    ("no_infinite_visibility_ratios", np.isfinite(hotel_summary["cross_platform_complaint_visibility_per_1000_google_reviews"].dropna()).all(), "Ratios are finite"),
    ("all_18_figures_present", len(figure_paths) == 18 and all(any(path.name.startswith(prefix) for path in figure_paths) for prefix in expected_figures), "Expected chart inventory"),
    ("mandatory_outputs_present", all(path.exists() for path in output_paths.values()), "All required/recommended reports exist"),
]
output_validation = pd.DataFrame(validation_checks, columns=["check", "passed", "detail"])
output_validation.to_csv(REPORTS_DIR / "sikayetvar_eda_output_validation.csv", index=False)
assert output_validation["passed"].all(), output_validation.loc[~output_validation["passed"]].to_dict("records")
display(output_validation)
display(pd.DataFrame({"output": list(output_paths), "path": [str(path.relative_to(PROJECT_ROOT)) for path in output_paths.values()]}))
print(f"Validated {len(figure_paths)} figures. Notebook 12 clean data hashes unchanged.")
"""
)

md("""### Notebook 14 readiness decision

**Hazır, koşullu.** Pooled NLP analysis 229 non-missing texts üzerinde ilerleyebilir. Hotel-level
sonuçlar yalnız sample tier ve uncertainty ile raporlanmalı; LOW_SAMPLE hotel'ler pool edilmelidir.
Notebook 14 sentiment/aspect/topic aşamasına geçmeden önce Turkish text normalization,
boilerplate/PII-safe processing, rare-topic stability ve human-label validation tasarımını sabitlemelidir.
""")

# The builder was assembled incrementally; normalize notebook sections into numeric order.
prefix_cells = []
section_groups = {}
current_section = None
for cell in cells:
    if cell.cell_type == "markdown" and cell.source.startswith("## ") and len(cell.source) >= 5 and cell.source[3:5].isdigit():
        current_section = int(cell.source[3:5])
        section_groups[current_section] = [cell]
    elif current_section is None:
        prefix_cells.append(cell)
    else:
        section_groups[current_section].append(cell)
cells = prefix_cells + [cell for number in sorted(section_groups) for cell in section_groups[number]]

nb["cells"] = cells
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
