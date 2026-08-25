"""Build the executable Şikayetvar NLP and aspect-analysis notebook."""

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/14_sikayetvar_all_hotels_nlp_aspect_analysis.ipynb"
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


md("""# Bodrum Hotel & Destination Intelligence
## 14 - Şikayetvar NLP & Aspect Analysis
### Negative Customer Voice — Complaint Themes, Distinctive Terms and Response Patterns

> **Ana metodolojik uyarı:** Şikayetvar corpusu zaten complaint amacıyla yazılmış, self-selected
> metinlerden oluşur. Bu nedenle “negative sentiment oranı” ana çıktı değildir. Bu notebook hangi
> aspect/theme/termlerin clean complaint örnekleminde geçtiğini, hangi başlıkların birlikte
> görüldüğünü ve company response davranışını inceler. Aspect mention rate gerçek müşteri problem
> oranı değildir; hotel kalite puanı veya “en kötü hotel” sıralaması üretilmez.
""")

md("""## 01. Amaç ve metodolojik çerçeve

Analiz structured, açıklanabilir ve denetlenebilir bir rule-based aspect dictionary kullanır.
Bir complaint birden çok aspect taşıyabilir. Otomatik eşleşmeler ground truth değildir; ayrı bir
manual-validation örneklemi hazırlanır. Hotel karşılaştırmalarında Notebook 13'teki HIGH/MEDIUM/
LOW sample tier'ları ve minimum `n=5` kuralı korunur.
""")

code("""from pathlib import Path
import hashlib
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bodrum_intelligence.sikayetvar_nlp import (
    ASPECT_KEYWORDS,
    CANONICAL_ASPECTS,
    CORPUS_DISCOVERED_KEYWORDS,
    GENERIC_DOMAIN_STOPWORDS,
    add_aspect_columns,
    aspect_cooccurrence_table,
    aspect_dictionary_table,
    aspect_frequency_table,
    aspects_long_table,
    distinctive_terms_by_group,
    group_aspect_matrix,
    hotel_name_stopwords,
    make_ngrams,
    normalize_for_nlp,
    term_frequency_table,
    tokenize,
)

PROCESSED_DIR = PROJECT_ROOT / "data/processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures/sikayetvar_nlp_aspect"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

paths = {
    "complaints": PROCESSED_DIR / "sikayetvar_all_hotels_complaints_clean.csv",
    "replies": PROCESSED_DIR / "sikayetvar_all_hotels_replies_clean.csv",
    "hotel_eda": REPORTS_DIR / "sikayetvar_hotel_eda_summary.csv",
    "area_eda": REPORTS_DIR / "sikayetvar_area_eda_summary.csv",
    "readiness": REPORTS_DIR / "sikayetvar_nlp_sample_readiness.csv",
    "eda_findings": REPORTS_DIR / "sikayetvar_eda_key_findings.txt",
    "eda_limitations": REPORTS_DIR / "sikayetvar_eda_limitations.txt",
    "notebook_12": PROJECT_ROOT / "notebooks/12_sikayetvar_all_hotels_audit_cleaning.ipynb",
    "notebook_13": PROJECT_ROOT / "notebooks/13_sikayetvar_all_hotels_eda.ipynb",
}
missing = [str(path.relative_to(PROJECT_ROOT)) for path in paths.values() if not path.exists()]
assert not missing, f"Eksik prerequisite: {missing}"

protected_inputs = [paths["complaints"], paths["replies"]]
input_hashes_before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected_inputs}

complaints = pd.read_csv(paths["complaints"], parse_dates=["complaint_date", "company_response_date_parsed"])
replies = pd.read_csv(paths["replies"], parse_dates=["reply_date"])
hotel_eda = pd.read_csv(paths["hotel_eda"])
area_eda = pd.read_csv(paths["area_eda"])
readiness = pd.read_csv(paths["readiness"])

MIN_HOTEL_N = 5
MIN_AREA_N = 10
MIN_NGRAM_DF = 3
MIN_ASPECT_RESPONSE_N = 3

def save_fig(fig, filename):
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

print(f"Clean input: {len(complaints)} complaints | {complaints.hotel_id.nunique()} hotels | {complaints.area.nunique()} areas")
""")

md("""## 02. Clean corpus validation""")

code("""validation = pd.DataFrame([
    ("total_clean_complaints", 236, len(complaints)),
    ("unique_complaint_ids", len(complaints), complaints["complaint_id"].nunique()),
    ("unique_canonical_urls", len(complaints), complaints["canonical_complaint_url"].nunique()),
    ("unique_hotels", 32, complaints["hotel_id"].nunique()),
    ("unique_areas", 11, complaints["area"].nunique()),
    ("non_null_clean_text", 229, complaints["complaint_text_clean"].notna().sum()),
    ("company_response_count", 39, int(complaints["company_response_exists_clean"].sum())),
    ("clean_reply_count", 97, len(replies)),
    ("dated_complaints", 226, complaints["complaint_date"].notna().sum()),
], columns=["check", "expected_from_notebook_12_13", "actual"])
validation["consistent"] = validation["expected_from_notebook_12_13"] == validation["actual"]
display(validation)
assert validation["consistent"].all(), "Notebook 12/13 ile corpus tutarsız; NLP durduruldu."

tier_summary = readiness.groupby("nlp_sample_tier").agg(
    hotel_count=("hotel_id", "size"), complaint_count=("matched_complaint_count", "sum")
).reset_index()
display(tier_summary)
assert set(readiness["nlp_sample_tier"]) == {"HIGH_SAMPLE", "MEDIUM_SAMPLE", "LOW_SAMPLE"}
""")

md("""### Bölüm Sonucu

NLP evreni Notebook 12'nin immutable clean corpusudur: 236 row korunur; içerik analizi yalnız
`complaint_text_clean` dolu 229 dokümanda yapılır. Eksik 7 text hiçbir sözcük/aspect uydurulmadan
`no_aspect_detected_flag=True` olarak kalır.
""")

md("""## 03. NLP preprocessing""")

code("""nlp = complaints.copy()
nlp["nlp_text_normalized"] = nlp["complaint_text_clean"].fillna("").map(normalize_for_nlp)
brand_terms = hotel_name_stopwords(nlp["hotel_name"].dropna().unique())
domain_stopwords = GENERIC_DOMAIN_STOPWORDS | brand_terms
nlp["nlp_tokens"] = nlp["nlp_text_normalized"].map(tokenize)
nlp["nlp_tokens_domain_filtered"] = nlp["nlp_text_normalized"].map(
    lambda text: tokenize(text, domain_stopwords)
)
nlp["nlp_text_domain_filtered"] = nlp["nlp_tokens_domain_filtered"].map(" ".join)
nlp["document_token_count"] = nlp["nlp_tokens"].map(len)
nlp["unique_token_count"] = nlp["nlp_tokens"].map(lambda values: len(set(values)))
nlp["lexical_diversity"] = np.where(
    nlp["document_token_count"].gt(0), nlp["unique_token_count"] / nlp["document_token_count"], np.nan
)
assert nlp["complaint_text"].equals(complaints["complaint_text"])
assert nlp["complaint_title"].equals(complaints["complaint_title"])
display(pd.DataFrame({
    "decision": ["main_text", "normalization", "negation", "domain_filter", "morphology"],
    "implementation": [
        "complaint_text_clean; title retained separately and not duplicated",
        "NFKC, Turkish-aware lowercase, URL/email/phone masking, whitespace/punctuation cleanup",
        "değil/yok/olmadı/gelmedi/çalışmıyor retained",
        f"{len(domain_stopwords)} generic + hotel-name tokens, only for term/TF-IDF analysis",
        "Surface forms + curated keyword roots; no unvalidated heavy lemmatizer",
    ],
}))
""")

md("""## 04. Corpus vocabulary overview""")

code("""valid_documents = nlp.loc[nlp["nlp_text_normalized"].ne("")]
all_tokens = [token for tokens in valid_documents["nlp_tokens"] for token in tokens]
domain_tokens = [token for tokens in valid_documents["nlp_tokens_domain_filtered"] for token in tokens]
corpus_summary = pd.DataFrame([
    ("total_clean_rows", len(nlp)),
    ("text_documents", len(valid_documents)),
    ("total_tokens_standard", len(all_tokens)),
    ("vocabulary_size_standard", len(set(all_tokens))),
    ("total_tokens_domain_filtered", len(domain_tokens)),
    ("vocabulary_size_domain_filtered", len(set(domain_tokens))),
    ("median_tokens_per_document", valid_documents["document_token_count"].median()),
    ("hotel_count", nlp["hotel_id"].nunique()),
    ("area_count", nlp["area"].nunique()),
], columns=["metric", "value"])
display(corpus_summary)
""")

md("""## 05. Unigram analysis""")

code("""unigram_standard = term_frequency_table(nlp["complaint_text_clean"], n=1, min_document_count=3)
unigram_domain = term_frequency_table(
    nlp["complaint_text_clean"], n=1, min_document_count=3, extra_stopwords=domain_stopwords
)
aspect_terms = {normalize_for_nlp(keyword, mask_pii=False) for keywords in ASPECT_KEYWORDS.values() for keyword in keywords if " " not in keyword}
unigram_standard["is_domain_stopword"] = unigram_standard["term"].isin(domain_stopwords)
unigram_standard["is_aspect_dictionary_term"] = unigram_standard["term"].isin(aspect_terms)
display(unigram_domain.head(20))

top_unigrams = unigram_domain.head(20).sort_values("document_count")
fig, ax = plt.subplots(figsize=(9.0, 6.8))
ax.barh(top_unigrams["term"], top_unigrams["document_count"], color="#3B8C88")
ax.set(title="Domain-filtered Terms in Clean Complaint Text", xlabel="Documents containing term", ylabel="Term")
ax.grid(axis="x", alpha=.2)
save_fig(fig, "01_top_unigrams.png")
""")

code("""explain_figure(
    "Brand/hotel ve genel platform filler kelimeleri çıkarıldıktan sonra terimlerin kaç complaint dokümanında geçtiğini gösterir.",
    f"En yaygın domain-filtered terimler: {', '.join(unigram_domain.head(5)['term'])}.",
    "Token count yerine document count kullanmak tek bir uzun metnin sıralamayı domine etmesini azaltır.",
    "Surface forms ayrı kalabilir; terim sıklığı aspect rate veya gerçek müşteri problem oranı değildir."
)
""")

md("""## 07. TF-IDF distinctive terms""")

code("""distinctive_hotel = distinctive_terms_by_group(
    nlp, "hotel_name", "nlp_text_domain_filtered", minimum_group_n=MIN_HOTEL_N, top_k=10, min_df=3
).rename(columns={"group_n": "hotel_n"})
distinctive_area = distinctive_terms_by_group(
    nlp, "area", "nlp_text_domain_filtered", minimum_group_n=MIN_AREA_N, top_k=10, min_df=3
).rename(columns={"group_n": "area_n"})
display(distinctive_hotel.groupby("hotel_name").head(5).head(30))
display(distinctive_area.groupby("area").head(5))

selected_names = hotel_eda.head(4)["hotel_name"].tolist()
selected_terms = distinctive_hotel[
    distinctive_hotel["hotel_name"].isin(selected_names) & distinctive_hotel["rank"].le(5)
].copy()
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, hotel_name in zip(axes.ravel(), selected_names):
    part = selected_terms[selected_terms["hotel_name"] == hotel_name].sort_values("tfidf_score")
    ax.barh(part["term"], part["tfidf_score"], color="#7A5C8E")
    ax.set_title(hotel_name, fontsize=10); ax.set_xlabel("Mean TF-IDF")
save_fig(fig, "03_distinctive_terms_selected_hotels.png")
""")

code("""explain_figure(
    "Her panelde yeterli sample'a sahip seçili hotelin document-level TF-IDF ortalaması yüksek terimleri gösterilir.",
    "Terimler brand/domain stopwords çıkarıldıktan sonra group metinlerini diğer dokümanlara göre ayırt eder.",
    "Hotel corpuslarını tanımlayan ifade adaylarını human review ve aspect dictionary triangulation'ına taşır.",
    "TF-IDF 'hotelin problemi kesin budur' demek değildir; corpus içi ayırt ediciliktir ve sample büyüklüğünden etkilenir."
)
""")

md("""## 08. Aspect taxonomy""")

code("""taxonomy = pd.DataFrame({
    "aspect": CANONICAL_ASPECTS,
    "keyword_count": [len(ASPECT_KEYWORDS[aspect]) for aspect in CANONICAL_ASPECTS],
    "scope_note": ["Interpretable complaint theme; multi-label and rule-based" for _ in CANONICAL_ASPECTS],
})
display(taxonomy)
print(f"Final taxonomy: {len(CANONICAL_ASPECTS)} canonical aspects")
""")

md("""Taxonomy, cross-platform downstream feature uyumu için 18 canonical başlıkta tutuldu. Corpus
incelemesinde `küf`, `tahtakurusu`, `kan lekesi`, `damlama`, `yanıltıcı bilgilendirme`, `muhatap`,
`çözüm sunulmadı` ve `ayrımcı` gibi corpus-driven adaylar ilgili başlıklara eklendi. Sağlık/zehirlenme
ifadeleri güvenlik başlığında ihtiyatla tutulur; ayrı bir HEALTH taxonomy'si human validation olmadan
eklenmedi.
""")

md("""## 09. Aspect dictionary validation""")

code("""aspect_dictionary = aspect_dictionary_table(nlp["complaint_text_clean"])
keyword_collisions = (
    aspect_dictionary.groupby("keyword").filter(lambda group: group["aspect"].nunique() > 1)
    .sort_values(["keyword", "aspect"])
)
display(aspect_dictionary.sort_values(["document_count", "aspect"], ascending=[False, True]).head(30))
display(keyword_collisions)

kwic_keywords = ["servis", "oda", "giriş", "kart", "ses", "iade", "temizlik", "muhatap", "zehirlenme"]
kwic_records = []
for keyword in kwic_keywords:
    candidates = nlp[nlp["nlp_text_normalized"].str.contains(
        rf"(?<![a-zçğıöşü]){keyword}(?![a-zçğıöşü])", regex=True, na=False
    )].head(4)
    for row in candidates.itertuples():
        text = row.nlp_text_normalized
        position = text.find(keyword)
        start, end = max(position - 80, 0), min(position + len(keyword) + 80, len(text))
        kwic_records.append({
            "keyword": keyword, "complaint_id": row.complaint_id, "hotel_name": row.hotel_name,
            "context_preview": text[start:end], "review_status": "TO_REVIEW",
        })
keyword_context_samples = pd.DataFrame(kwic_records)
display(keyword_context_samples.head(20))
""")

md("""Dictionary audit trail `SEED` ve `CORPUS_DISCOVERED` kaynaklarını ayırır. Collision tablosu
özellikle `zehirlenme` gibi birden fazla başlıkla ilişkili olabilen terimleri görünür kılar. “Servis
aracı” phrase'i önce TRANSPORT_TRANSFER olarak eşleştirilir ve aynı phrase STAFF_SERVICE `servis`
eşleşmesini tetiklemez. Manual validation tamamlanana kadar tüm status'ler `TO_REVIEW` kalır.
""")

md("""## 10. Aspect detection""")

code("""nlp = add_aspect_columns(nlp, "nlp_text_normalized")
aspect_columns = [f"aspect_{aspect.lower()}" for aspect in CANONICAL_ASPECTS]
assert all(nlp[column].dtype == bool for column in aspect_columns)
assert (nlp[aspect_columns].sum(axis=1) == nlp["aspect_count"]).all()
assert nlp["complaint_id"].is_unique

aspects_long = aspects_long_table(nlp)
assert len(aspects_long) == len(nlp) * len(CANONICAL_ASPECTS)
assert int(aspects_long["matched"].sum()) == int(nlp["aspect_count"].sum())
display(nlp[[
    "complaint_id", "hotel_name", "aspect_count", "matched_aspects", "matched_aspect_keywords",
    "no_aspect_detected_flag",
]].head(10))
""")

code("""validation_parts = []
for aspect in CANONICAL_ASPECTS:
    candidates = nlp[nlp[f"aspect_{aspect.lower()}"]].sort_values(
        ["matched_keyword_count", "hotel_name"], ascending=[False, True]
    ).head(1)
    validation_parts.append(candidates)
validation_parts.append(nlp[nlp["no_aspect_detected_flag"]].head(7))
validation_parts.append(nlp[nlp["aspect_count"].ge(8)].drop_duplicates("hotel_name").head(10))
manual_sample = pd.concat(validation_parts, ignore_index=True).drop_duplicates("complaint_id")
remaining = nlp[~nlp["complaint_id"].isin(manual_sample["complaint_id"])].drop_duplicates("hotel_name")
manual_sample = pd.concat([manual_sample, remaining.head(max(0, 40-len(manual_sample)))], ignore_index=True).head(40)
manual_validation_sample = pd.DataFrame({
    "complaint_id": manual_sample["complaint_id"],
    "hotel_name": manual_sample["hotel_name"],
    "complaint_text_preview": manual_sample["nlp_text_normalized"].str.slice(0, 280),
    "predicted_aspects": manual_sample["matched_aspects"],
    "matched_keywords": manual_sample["matched_aspect_keywords"],
    "manual_review_status": "TO_REVIEW",
    "notes": "",
})
display(manual_validation_sample.head(10))
print("Manual validation sample rows:", len(manual_validation_sample))
""")

md("""## 11. Aspect coverage""")

code("""text_document_n = int(nlp["nlp_text_normalized"].ne("").sum())
with_aspect_n = int(nlp["aspect_count"].gt(0).sum())
no_aspect_n = int(nlp["no_aspect_detected_flag"].sum())
aspect_coverage = pd.DataFrame([
    ("total_clean_rows", len(nlp)),
    ("text_documents", text_document_n),
    ("complaints_with_at_least_one_aspect", with_aspect_n),
    ("no_aspect_count", no_aspect_n),
    ("aspect_coverage_pct_all_clean_rows", 100*with_aspect_n/len(nlp)),
    ("aspect_coverage_pct_text_documents", 100*with_aspect_n/text_document_n),
    ("mean_aspects_per_complaint", nlp["aspect_count"].mean()),
    ("median_aspects_per_complaint", nlp["aspect_count"].median()),
], columns=["metric", "value"])
display(aspect_coverage)

coverage_plot = pd.Series({"At least one aspect": with_aspect_n, "No aspect": no_aspect_n})
fig, ax = plt.subplots(figsize=(7.2, 4.7))
bars = ax.bar(coverage_plot.index, coverage_plot.values, color=["#5B8E7D", "#B8B8B8"])
ax.bar_label(bars); ax.set(title="Rule-based Aspect Dictionary Coverage", ylabel="Clean complaints")
ax.grid(axis="y", alpha=.2)
save_fig(fig, "04_aspect_coverage.png")
""")

code("""explain_figure(
    "Clean complaint rowları en az bir dictionary aspect eşleşmesi bulunan ve bulunmayan olarak ayrılır.",
    f"Coverage all-clean rows üzerinde %{100*with_aspect_n/len(nlp):.1f}; no-aspect n={no_aspect_n}.",
    "Dictionary'nin corpusun ne kadarını yapılandırabildiğini ve manual review ihtiyacını gösterir.",
    "Yüksek coverage doğruluk garantisi değildir; geniş keywords false positive, dar keywords false negative yaratabilir."
)
""")

md("""## 12. Overall aspect frequencies""")

code("""aspect_frequency = aspect_frequency_table(nlp)
display(aspect_frequency)
plot_frequency = aspect_frequency.sort_values("mention_rate_pct")
fig, ax = plt.subplots(figsize=(10, 7.5))
ax.barh(plot_frequency["aspect"], plot_frequency["mention_rate_pct"], color="#C1666B")
ax.set(title="Aspect Mention Rates in the Clean Complaint Corpus", xlabel="Complaints mentioning aspect (%)", ylabel="Aspect")
ax.grid(axis="x", alpha=.2)
save_fig(fig, "05_aspect_mention_rates.png")
""")

code("""top_aspect = aspect_frequency.iloc[0]
explain_figure(
    "Her bar, aspect'in geçtiği unique clean complaint sayısının 236 clean rowa oranıdır; multi-label olduğu için barlar toplamı %100 değildir.",
    f"En sık eşleşen aspect {top_aspect['aspect']}: n={int(top_aspect['complaint_count'])}, %{top_aspect['mention_rate_pct']:.1f}.",
    "Negative customer voice corpusunun başlıca hizmet/problem başlıklarını karşılaştırılabilir hale getirir.",
    "Mention rate gerçek müşteri problem oranı değildir; dictionary coverage ve multi-label overlap içerir."
)
""")

md("""## 06. Bigram / trigram analysis""")

code("""bigram_table = term_frequency_table(
    nlp["complaint_text_clean"], n=2, min_document_count=MIN_NGRAM_DF, extra_stopwords=domain_stopwords
)
trigram_table = term_frequency_table(
    nlp["complaint_text_clean"], n=3, min_document_count=4, extra_stopwords=domain_stopwords
)
bigram_table.insert(1, "n", 2)
trigram_table.insert(1, "n", 3)
ngram_table = pd.concat([bigram_table, trigram_table], ignore_index=True)
display(bigram_table.head(20)); display(trigram_table.head(15))

top_bigrams = bigram_table.head(20).sort_values("document_count")
fig, ax = plt.subplots(figsize=(9.2, 6.8))
ax.barh(top_bigrams["term"], top_bigrams["document_count"], color="#6C91BF")
ax.set(title=f"Complaint Bigrams — Minimum Document Frequency {MIN_NGRAM_DF}", xlabel="Documents containing phrase", ylabel="Bigram")
ax.grid(axis="x", alpha=.2)
save_fig(fig, "02_top_bigrams.png")
""")

code("""explain_figure(
    "İki kelimelik ifadeler en az üç ayrı complaint'te görülme koşuluyla document count'a göre sıralanır.",
    f"İlk bigramlar: {', '.join(bigram_table.head(5)['term'])}.",
    "Phrase düzeyi, tek kelimenin kaçırdığı iade/iletişim/hizmet bağlamlarını görünür kılar.",
    "Sık phrase otomatik aspect değildir; boilerplate veya complaint yazım kalıbı olabilir."
)
""")

md("""## 13. Hotel × aspect analysis""")

code("""hotel_aspect = group_aspect_matrix(
    nlp, ["hotel_id", "hotel_name"], small_n_threshold=MIN_HOTEL_N
).rename(columns={"group_n": "hotel_n"})
hotel_aspect = hotel_aspect.merge(
    readiness[["hotel_id", "nlp_sample_tier"]], on="hotel_id", how="left", validate="many_to_one"
)
eligible_hotels = readiness.loc[readiness["matched_complaint_count"].ge(MIN_HOTEL_N), "hotel_name"]
major_aspects = aspect_frequency.head(12)["aspect"].tolist()
hotel_heatmap = hotel_aspect[
    hotel_aspect["hotel_name"].isin(eligible_hotels) & hotel_aspect["aspect"].isin(major_aspects)
].pivot(index="hotel_name", columns="aspect", values="aspect_mention_rate_pct")
hotel_order = hotel_eda[hotel_eda["matched_complaint_count"].ge(MIN_HOTEL_N)]["hotel_name"]
hotel_heatmap = hotel_heatmap.reindex([name for name in hotel_order if name in hotel_heatmap.index])
fig, ax = plt.subplots(figsize=(14.5, 8.5))
sns.heatmap(hotel_heatmap, cmap="YlOrRd", vmin=0, vmax=100, annot=True, fmt=".0f", ax=ax, cbar_kws={"label":"Mention rate (%)"})
ax.set(title=f"Hotel × Aspect Mention Rate — MEDIUM/HIGH Sample (n≥{MIN_HOTEL_N})", xlabel="Aspect", ylabel="Hotel")
save_fig(fig, "06_hotel_aspect_heatmap.png")
""")

code("""explain_figure(
    "Renk complaint sayısı değil, her hotelin own clean complaint corpusunda aspect mention rate yüzdesidir; yalnız n>=5 hotel'ler gösterilir.",
    f"Heatmap {hotel_heatmap.shape[0]} yeterli-sample hotel ve {hotel_heatmap.shape[1]} major aspect içerir.",
    "Farklı corpus büyüklüklerindeki hotel'leri normalized relative emphasis ile karşılaştırır.",
    "n=5 hâlâ oynaktır; selection/mapping bias sürer. Koyu hücre hotel kalitesi veya gerçek risk rate değildir."
)
""")

code("""def top_aspect_summary(matrix, group_column, n_column, minimum_n):
    rows = []
    for group_name, part in matrix[matrix[n_column].ge(minimum_n)].groupby(group_column):
        ranked = part.sort_values(["aspect_mention_rate_pct", "aspect_count"], ascending=False).head(3)
        row = {group_column: group_name, n_column: int(part[n_column].iloc[0])}
        for rank, item in enumerate(ranked.itertuples(), 1):
            row[f"top_aspect_{rank}"] = item.aspect
            row[f"top_aspect_{rank}_rate_pct"] = item.aspect_mention_rate_pct
        rows.append(row)
    return pd.DataFrame(rows)

hotel_top_aspects = top_aspect_summary(hotel_aspect, "hotel_name", "hotel_n", MIN_HOTEL_N)
display(hotel_top_aspects)
""")

md("""## 14. Area × aspect analysis""")

code("""area_aspect = group_aspect_matrix(nlp, ["area"], small_n_threshold=MIN_AREA_N).rename(columns={"group_n": "area_n"})
area_aspect = area_aspect.merge(
    area_eda[["area", "mapping_coverage_pct", "coverage_flag"]], on="area", how="left", validate="many_to_one"
)
eligible_areas = area_aspect.loc[area_aspect["area_n"].ge(MIN_AREA_N), "area"].unique()
area_heatmap = area_aspect[
    area_aspect["area"].isin(eligible_areas) & area_aspect["aspect"].isin(major_aspects)
].pivot(index="area", columns="aspect", values="aspect_mention_rate_pct")
area_order = area_eda[area_eda["matched_complaint_count"].ge(MIN_AREA_N)]["area"]
area_heatmap = area_heatmap.reindex([area for area in area_order if area in area_heatmap.index])
fig, ax = plt.subplots(figsize=(14.0, 6.0))
sns.heatmap(area_heatmap, cmap="PuBuGn", vmin=0, vmax=100, annot=True, fmt=".0f", ax=ax, cbar_kws={"label":"Mention rate (%)"})
ax.set(title=f"Area × Aspect Mention Rate — Areas with n≥{MIN_AREA_N}", xlabel="Aspect", ylabel="Area")
save_fig(fig, "07_area_aspect_heatmap.png")
""")

code("""explain_figure(
    "Renk her area'nın matched clean complaint corpusunda aspect mention rate'idir; count değildir. n<10 area'lar ana heatmap dışında tutulur.",
    f"Heatmap {area_heatmap.shape[0]} area içerir; area mapping coverage ayrı CSV'de korunur.",
    "Destination complaint themes için normalized descriptive karşılaştırma sağlar.",
    "Area mapping coverage eşit değildir ve project hotel karması değişir; renkler tüm area müşterilerini temsil etmez."
)
""")

code("""area_top_aspects = top_aspect_summary(area_aspect, "area", "area_n", MIN_AREA_N)
display(area_top_aspects)
""")

md("""## 15. Aspect lift""")

code("""overall_rates = aspect_frequency.set_index("aspect")["mention_rate_pct"]
hotel_lift = hotel_aspect.copy()
hotel_lift["overall_mention_rate_pct"] = hotel_lift["aspect"].map(overall_rates)
hotel_lift["hotel_mention_rate_pct"] = hotel_lift["aspect_mention_rate_pct"]
hotel_lift["lift_pp"] = hotel_lift["hotel_mention_rate_pct"] - hotel_lift["overall_mention_rate_pct"]
hotel_lift = hotel_lift[[
    "hotel_name", "hotel_n", "aspect", "hotel_mention_rate_pct", "overall_mention_rate_pct", "lift_pp", "small_n_flag", "nlp_sample_tier"
]]

area_lift = area_aspect.copy()
area_lift["overall_mention_rate_pct"] = area_lift["aspect"].map(overall_rates)
area_lift["area_mention_rate_pct"] = area_lift["aspect_mention_rate_pct"]
area_lift["lift_pp"] = area_lift["area_mention_rate_pct"] - area_lift["overall_mention_rate_pct"]

top_hotel_lift = hotel_lift[~hotel_lift["small_n_flag"]].sort_values(
    ["hotel_name", "lift_pp"], ascending=[True, False]
).groupby("hotel_name").head(3)
display(top_hotel_lift.head(30))

selected_lift = top_hotel_lift[top_hotel_lift["hotel_name"].isin(selected_names)]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, hotel_name in zip(axes.ravel(), selected_names):
    part = selected_lift[selected_lift["hotel_name"] == hotel_name].sort_values("lift_pp")
    ax.barh(part["aspect"], part["lift_pp"], color="#D98E73")
    ax.axvline(0, color="#555", linewidth=.8); ax.set_title(hotel_name, fontsize=10); ax.set_xlabel("Lift (percentage points)")
save_fig(fig, "08_hotel_aspect_lift.png")
""")

code("""explain_figure(
    "Lift, hotel mention rate eksi overall corpus mention rate'tir; pozitif değer yalnız corpus içi relative emphasis gösterir.",
    "Seçili dört yüksek-volume hotel için en yüksek üç pozitif aspect lift gösterilir.",
    "Normalized profile farklarını segmentation feature adayı haline getirir.",
    "Lift gerçek risk/kalite farkı değildir; küçük-n hotel'ler dışarıda olsa da selection bias ve dictionary hatası kalır."
)
""")

code("""eligible_area_lift = area_lift[area_lift["area_n"].ge(MIN_AREA_N)]
top_area_lift = eligible_area_lift.sort_values(["area", "lift_pp"], ascending=[True, False]).groupby("area").head(3)
fig, ax = plt.subplots(figsize=(10, 7))
plot_area_lift = top_area_lift.copy()
plot_area_lift["label"] = plot_area_lift["area"] + " — " + plot_area_lift["aspect"]
plot_area_lift = plot_area_lift.sort_values("lift_pp")
ax.barh(plot_area_lift["label"], plot_area_lift["lift_pp"], color="#6C91BF")
ax.axvline(0, color="#555", linewidth=.8)
ax.set(title="Top Positive Aspect Lift by Eligible Area", xlabel="Lift vs overall corpus (percentage points)", ylabel="Area — aspect")
save_fig(fig, "09_area_aspect_lift.png")
""")

code("""explain_figure(
    "Her yeterli-sample area için overall corpus oranına göre en yüksek üç pozitif aspect farkı gösterilir.",
    f"Yalnız area_n≥{MIN_AREA_N} olan gruplar dahildir.",
    "Area düzeyindeki relative emphasis'i sade bir investigation queue'ya dönüştürür.",
    "Mapping coverage ve hotel composition farklıdır; lift tüm destinasyon deneyimi veya problem rate değildir."
)
""")

md("""## 16. Aspect co-occurrence""")

code("""cooccurrence = aspect_cooccurrence_table(nlp, minimum_support=3)
display(cooccurrence.head(20))
co_matrix = pd.DataFrame(0, index=CANONICAL_ASPECTS, columns=CANONICAL_ASPECTS, dtype=float)
for aspect in CANONICAL_ASPECTS:
    co_matrix.loc[aspect, aspect] = int(nlp[f"aspect_{aspect.lower()}"].sum())
for row in cooccurrence.itertuples():
    co_matrix.loc[row.aspect_a, row.aspect_b] = row.cooccurrence_count
    co_matrix.loc[row.aspect_b, row.aspect_a] = row.cooccurrence_count
ordered_aspects = aspect_frequency["aspect"].tolist()
co_matrix = co_matrix.loc[ordered_aspects, ordered_aspects]
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(co_matrix, cmap="Blues", annot=True, fmt=".0f", ax=ax, cbar_kws={"label":"Co-occurring complaints"})
ax.set(title="Aspect Co-occurrence in Multi-label Complaints", xlabel="Aspect", ylabel="Aspect")
save_fig(fig, "10_aspect_cooccurrence_heatmap.png")
""")

code("""top_pair = cooccurrence.iloc[0]
explain_figure(
    "Off-diagonal hücreler iki aspect'in aynı complaint'te birlikte kaç kez geçtiğini; diagonal aspect'in total complaint count'unu gösterir.",
    f"En sık çift {top_pair['aspect_a']} × {top_pair['aspect_b']}: n={int(top_pair['cooccurrence_count'])}.",
    "Tekil problem başlıkları yerine birlikte yaşanan complaint örüntülerini gösterir.",
    "Broad keywords doğal olarak co-occurrence'ı yükseltebilir; count nedensellik veya problem şiddeti değildir."
)
""")

md("""## 17. Company response by aspect""")

code("""aspect_response = aspect_frequency[[
    "aspect", "complaint_count", "company_response_count", "company_response_rate_within_aspect",
    "median_view_count", "median_complaint_word_count",
]].copy()
eligible_response = aspect_response[aspect_response["complaint_count"].ge(10)].sort_values("company_response_rate_within_aspect")
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(eligible_response["aspect"], eligible_response["company_response_rate_within_aspect"], color="#5B8E7D")
for index, row in enumerate(eligible_response.itertuples()):
    ax.text(row.company_response_rate_within_aspect + .5, index, f"n={row.complaint_count}", va="center", fontsize=8)
ax.set(title="Company Response Share by Aspect — Aspect n≥10", xlabel="Complaints with matched company response (%)", ylabel="Aspect")
ax.grid(axis="x", alpha=.2)
save_fig(fig, "11_aspect_company_response_rate.png")
""")

code("""explain_figure(
    "Her bar, aspect geçen complaints içinde matched company response görülen payı; etiket aspect complaint n'ini gösterir.",
    f"Yeterli-support aspect'lerde response share %{eligible_response.company_response_rate_within_aspect.min():.1f}–%{eligible_response.company_response_rate_within_aspect.max():.1f} aralığındadır.",
    "Hangi complaint başlıklarında kurum yanıtının corpus içinde daha sık görüldüğünü descriptif gösterir.",
    "Company response çözüm değildir; hotel/brand response strategy ve aspect-hotel karması sonucu etkiler."
)
""")

code("""response_texts = nlp.loc[nlp["company_response_text_clean"].notna(), "company_response_text_clean"]
response_terms = term_frequency_table(response_texts, n=1, min_document_count=2, extra_stopwords=domain_stopwords)
display(response_terms.head(20))
fig, ax = plt.subplots(figsize=(8.8, 5.8))
plot_response_terms = response_terms.head(15).sort_values("document_count")
ax.barh(plot_response_terms["term"], plot_response_terms["document_count"], color="#7A5C8E")
ax.set(title="Common Terms in Matched Company Response Text", xlabel="Response documents containing term", ylabel="Term")
ax.grid(axis="x", alpha=.2)
save_fig(fig, "12_company_response_terms.png")
""")

code("""explain_figure(
    "Matched company response textlerinde standard/domain stopwords sonrası document frequency gösterilir.",
    f"Response-text corpus n={len(response_texts)}; ağır topic modeling uygulanmaz.",
    "Standardized communication phrase adaylarını görünür kılar.",
    "Benzer veya sık response language firmanın müşteriyi önemsemediği kanıtı değildir; template kullanımı olabilir."
)
""")

md("""## 18. Response time by aspect""")

code("""nlp["response_time_days"] = (
    nlp["company_response_date_parsed"] - nlp["complaint_date"]
).dt.total_seconds() / 86400
nlp.loc[nlp["response_time_days"].lt(0), "response_time_days"] = np.nan
response_rows = []
for aspect in CANONICAL_ASPECTS:
    subset = nlp[nlp[f"aspect_{aspect.lower()}"]]
    valid = subset["response_time_days"].dropna()
    response_rows.append({
        "aspect": aspect,
        "complaint_count": len(subset),
        "company_response_count": int(subset["company_response_exists_clean"].sum()),
        "response_date_available_n": len(valid),
        "company_response_rate_within_aspect": 100*subset["company_response_exists_clean"].mean() if len(subset) else np.nan,
        "median_response_time_days": valid.median(),
        "q25_response_time_days": valid.quantile(.25),
        "q75_response_time_days": valid.quantile(.75),
    })
aspect_response_summary = pd.DataFrame(response_rows).sort_values("complaint_count", ascending=False)
display(aspect_response_summary)
response_time_plot = aspect_response_summary[aspect_response_summary["response_date_available_n"].ge(MIN_ASPECT_RESPONSE_N)].sort_values("median_response_time_days")
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(response_time_plot["aspect"], response_time_plot["median_response_time_days"], color="#D98E73")
for index, row in enumerate(response_time_plot.itertuples()):
    ax.text(row.median_response_time_days + .05, index, f"response n={row.response_date_available_n}", va="center", fontsize=8)
ax.set(title=f"Median Company Response Time by Aspect — Response-date n≥{MIN_ASPECT_RESPONSE_N}", xlabel="Median days", ylabel="Aspect")
ax.grid(axis="x", alpha=.2)
save_fig(fig, "13_aspect_response_time.png")
""")

code("""explain_figure(
    "Yalnız en az üç calculable response lag'i bulunan aspect'lerde complaint-to-company-response median days gösterilir.",
    f"Grafikte {len(response_time_plot)} aspect vardır; exact n etiketlenir.",
    "Aspect bazında operasyonel response-speed farklılıklarını descriptif olarak gösterir.",
    "86 approximate complaint date, multi-label overlap ve düşük response n vardır; bu SLA veya çözüm süresi değildir."
)
""")

md("""## 19. Optional temporal aspect analysis""")

code("""dated_nlp = nlp[nlp["complaint_date"].notna()].copy()
dated_nlp["year"] = dated_nlp["complaint_date"].dt.year.astype(int)
temporal_aspects = aspect_frequency.head(6)["aspect"].tolist()
temporal_rows = []
for year, group in dated_nlp.groupby("year"):
    for aspect in temporal_aspects:
        count = int(group[f"aspect_{aspect.lower()}"].sum())
        temporal_rows.append({
            "year": year, "year_n": len(group), "aspect": aspect, "aspect_count": count,
            "aspect_mention_rate_pct": 100*count/len(group),
            "partial_year_flag": year in {dated_nlp["year"].min(), dated_nlp["year"].max()},
        })
temporal_aspect_summary = pd.DataFrame(temporal_rows)
fig, ax = plt.subplots(figsize=(10, 6))
for aspect, part in temporal_aspect_summary.groupby("aspect"):
    ax.plot(part["year"], part["aspect_mention_rate_pct"], marker="o", label=aspect)
ax.set_xticks(sorted(dated_nlp["year"].unique()))
ax.set(title="Major Aspect Mention Rates by Observed Calendar Year", xlabel="Year", ylabel="Mention rate within dated complaints (%)")
ax.grid(alpha=.2); ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
save_fig(fig, "14_major_aspects_by_year.png")
""")

code("""explain_figure(
    "Her çizgi major aspect'in ilgili yıldaki dated clean complaints içinde mention rate'ini gösterir.",
    f"Date coverage {len(dated_nlp)}/{len(nlp)}; yalnız top-6 overall aspect gösterilir.",
    "Theme mix'in gözlenen erişilebilir tarih penceresinde nasıl değiştiğine exploratory bakış verir.",
    "2023 ve 2026 partial/accessibility-limited; platform historical completeness bilinmiyor. Trend veya turizm sezonu sonucu çıkarılamaz."
)
""")

code("""plot_context = aspect_frequency.sort_values("complaint_count").copy()
fig, axes = plt.subplots(1, 2, figsize=(13, 7))
axes[0].barh(plot_context["aspect"], plot_context["median_view_count"], color="#40798C")
axes[0].set(title="Median View Count by Aspect", xlabel="Median views", ylabel="Aspect")
axes[1].barh(plot_context["aspect"], plot_context["median_complaint_word_count"], color="#7A5C8E")
axes[1].set(title="Median Complaint Length by Aspect", xlabel="Median words", ylabel="")
for ax in axes: ax.grid(axis="x", alpha=.2)
save_fig(fig, "15_aspect_view_and_text_length.png")
""")

code("""explain_figure(
    "Sol panel aspect geçen complaints'in median platform views; sağ panel median word count değeridir.",
    "Aspect'lerin görünürlük ve anlatım uzunluğu bağlamı aynı sırada karşılaştırılır.",
    "Hangi başlıkların daha uzun anlatıldığını veya daha çok görüntülendiğini descriptive gösterir.",
    "Views yaşla birikir; multi-label complaints panellerde birden fazla aspect'e katkı verir. Causality ve severity yorumu yapılamaz."
)
""")

md("""## 20. Optional topic modeling gate""")

code("""topic_gate = pd.DataFrame([
    ("text_document_count", len(valid_documents), "PASS", "229 non-empty complaint texts"),
    ("median_document_tokens", valid_documents["document_token_count"].median(), "PASS", "Narratives are long enough for term/aspect analysis"),
    ("hotel_distribution", readiness["nlp_sample_tier"].eq("LOW_SAMPLE").sum(), "CAUTION", "15/32 hotels are LOW_SAMPLE"),
    ("aspect_coverage_pct", 100*with_aspect_n/text_document_n, "PASS", "Rule-based taxonomy already structures most text documents"),
    ("human_topic_validation", 0, "FAIL", "No independently reviewed topic labels/representatives yet"),
    ("stability_evidence", 0, "FAIL", "No multi-seed coherence/stability benchmark approved"),
], columns=["gate", "value", "status", "note"])
topic_model_status = "TOPIC_MODEL_NOT_RELIABLE"
display(topic_gate)
print(topic_model_status)
""")

md("""NMF/BERTopic/LDA final analize dahil edilmedi. 229 uzun ve çoğunlukla multi-aspect doküman
terim/aspect analizi için yeterli olsa da topic label human validation ve multi-seed stability kanıtı
yoktur. Rule-based taxonomy %95+ coverage sağlarken yeni topic katmanının ek bilgi kazancı henüz
kanıtlanmamıştır. Bu gate metodolojik bir durdurma kararıdır, başarısızlık değildir.
""")

md("""## 21. NLP feature tables""")

code("""hotel_aspect_wide = hotel_aspect.pivot(
    index=["hotel_id", "hotel_name"], columns="aspect", values="aspect_mention_rate_pct"
).add_prefix("aspect_").add_suffix("_mention_rate_pct").reset_index()
hotel_features = hotel_eda.merge(hotel_aspect_wide, on=["hotel_id", "hotel_name"], how="inner", validate="one_to_one")
hotel_features["nlp_feature_reliability"] = hotel_features["nlp_sample_tier"].str.replace("_SAMPLE", "", regex=False)
hotel_feature_columns = [
    "hotel_id", "hotel_name", "area", "matched_complaint_count",
] + [column for column in hotel_features if column.startswith("aspect_") and column.endswith("_mention_rate_pct")] + [
    "company_response_rate_in_corpus", "median_response_time_days", "median_complaint_word_count",
    "median_view_count", "google_rating", "google_review_count", "official_star_rating_verified",
    "official_room_count", "nlp_sample_tier", "nlp_feature_reliability", "small_n_flag",
]
hotel_features = hotel_features[hotel_feature_columns].rename(columns={"matched_complaint_count":"complaint_n"})
assert hotel_features["hotel_id"].is_unique

area_aspect_wide = area_aspect.pivot(index="area", columns="aspect", values="aspect_mention_rate_pct").add_prefix("aspect_").add_suffix("_mention_rate_pct").reset_index()
area_features = area_eda.merge(area_aspect_wide, on="area", how="left", validate="one_to_one")
area_feature_columns = [
    "area", "project_hotel_count", "mapped_hotel_count", "hotels_with_complaints",
    "matched_complaint_count", "mapping_coverage_pct", "coverage_flag",
] + [column for column in area_features if column.startswith("aspect_") and column.endswith("_mention_rate_pct")] + [
    "company_response_count", "company_response_rate_in_corpus", "median_word_count", "median_view_count",
]
area_features = area_features[area_feature_columns].rename(columns={"matched_complaint_count":"complaint_n"})
assert area_features["area"].is_unique
display(hotel_features.head()); display(area_features)
""")

code("""rating_rows = []
eligible_rate_matrix = hotel_aspect[hotel_aspect["hotel_n"].ge(MIN_HOTEL_N)].merge(
    hotel_eda[["hotel_id", "google_rating"]], on="hotel_id", how="left", validate="many_to_one"
)
for aspect, group in eligible_rate_matrix.groupby("aspect"):
    pair = group[["aspect_mention_rate_pct", "google_rating"]].dropna()
    rho, p_value = stats.spearmanr(pair["aspect_mention_rate_pct"], pair["google_rating"]) if len(pair) >= 5 else (np.nan, np.nan)
    rating_rows.append({
        "aspect": aspect, "hotel_n": len(pair), "spearman_rho": rho, "p_value": p_value,
        "main_caution": "Exploratory, unadjusted multiple comparisons; selected hotels with complaint n>=5.",
    })
aspect_rating_correlations = pd.DataFrame(rating_rows).sort_values("spearman_rho")
display(aspect_rating_correlations)

official_context_gate = pd.DataFrame([
    ("official_star_rating_verified", int(hotel_features["official_star_rating_verified"].notna().sum()), len(hotel_features)),
    ("official_room_count", int(hotel_features["official_room_count"].notna().sum()), len(hotel_features)),
], columns=["field", "non_missing_hotels", "complaint_hotel_universe"])
official_context_gate["decision"] = "DESCRIPTIVE_ONLY_LOW_COVERAGE"
display(official_context_gate)
""")

code("""response_boilerplate = pd.DataFrame(columns=["response_a_complaint_id", "response_b_complaint_id", "cosine_similarity", "candidate_flag"])
response_frame = nlp.loc[nlp["company_response_text_clean"].notna(), ["complaint_id", "company_response_text_clean"]].copy()
if len(response_frame) >= 2:
    response_frame["normalized"] = response_frame["company_response_text_clean"].map(normalize_for_nlp)
    response_vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2), sublinear_tf=True)
    response_matrix = response_vectorizer.fit_transform(response_frame["normalized"])
    similarities = cosine_similarity(response_matrix)
    boilerplate_rows = []
    for i in range(len(response_frame)):
        for j in range(i+1, len(response_frame)):
            if similarities[i, j] >= .80:
                boilerplate_rows.append({
                    "response_a_complaint_id": response_frame.iloc[i]["complaint_id"],
                    "response_b_complaint_id": response_frame.iloc[j]["complaint_id"],
                    "cosine_similarity": similarities[i, j],
                    "candidate_flag": "STANDARDIZED_RESPONSE_PATTERN_CANDIDATE",
                })
    response_boilerplate = pd.DataFrame(boilerplate_rows, columns=response_boilerplate.columns)
display(response_boilerplate.head(20))
""")

md("""## 22. What we can / cannot say

### Bu NLP analizi ile ne söyleyebiliriz?

- Matched Şikayetvar complaint corpusunda hangi aspect/term/phrase'lerin daha sık geçtiğini.
- Yeterli-sample hotel ve area corpuslarında relative aspect emphasis'i.
- Hangi aspect'lerin aynı complaint içinde birlikte görüldüğünü.
- Hangi aspect'lerde matched company response'un daha sık görüldüğünü.
- Hangi terms'in belirli hotel/area corpuslarını TF-IDF ile ayırt ettiğini.

### Bu NLP analizi ile ne söyleyemeyiz?

- Tüm hotel müşterilerinin ne düşündüğünü veya gerçek problem oranını.
- “Hotel X'in en büyük problemi kesin temizliktir” gibi census iddiasını.
- Daha çok complaint/aspect görülen hotelin daha kötü olduğunu.
- Company response'un problemi çözdüğünü.
- Rule-based keyword eşleşmesinin %100 doğru olduğunu.
- Correlation'ın causation olduğunu veya complaint'i olmayan hotelde aspect rate'in sıfır olduğunu.
""")

md("""## 23. Key findings""")

code("""top_unigram_text = ", ".join(unigram_domain.head(10)["term"])
top_bigram_text = ", ".join(bigram_table.head(10)["term"])
top_aspect_text = "; ".join(
    f"{row.aspect} %{row.mention_rate_pct:.1f}" for row in aspect_frequency.head(10).itertuples()
)
top_pair_text = "; ".join(
    f"{row.aspect_a}×{row.aspect_b} n={row.cooccurrence_count}" for row in cooccurrence.head(5).itertuples()
)
top_response_text = "; ".join(
    f"{row.aspect} %{row.company_response_rate_within_aspect:.1f} (n={row.complaint_count})"
    for row in eligible_response.sort_values("company_response_rate_within_aspect", ascending=False).head(5).itertuples()
)
key_findings = [
    f"Clean NLP corpus keeps {len(nlp)} rows; {text_document_n} have non-empty complaint_text_clean across {nlp.hotel_id.nunique()} hotels and {nlp.area.nunique()} represented areas.",
    f"Standard preprocessing yields {len(all_tokens)} tokens and {len(set(all_tokens))} unique surface tokens; median document length is {valid_documents.document_token_count.median():.0f} tokens.",
    f"Top domain-filtered unigrams: {top_unigram_text}.",
    f"Top supported bigrams: {top_bigram_text}.",
    f"Final explainable taxonomy contains {len(CANONICAL_ASPECTS)} multi-label aspects; no overall sentiment model was run.",
    f"At least one aspect is detected in {with_aspect_n}/{len(nlp)} clean rows ({100*with_aspect_n/len(nlp):.1f}%); no-aspect n={no_aspect_n}.",
    f"Mean/median aspects per clean complaint are {nlp.aspect_count.mean():.2f}/{nlp.aspect_count.median():.0f}; totals exceed document n because detection is multi-label.",
    f"Top aspect mention rates: {top_aspect_text}.",
    f"Strongest supported co-occurrences: {top_pair_text}.",
    f"Major-aspect company response shares: {top_response_text}; response is not resolution.",
    f"Response-time analysis includes {len(response_time_plot)} aspects with response-date n>={MIN_ASPECT_RESPONSE_N}; approximate dates limit precision.",
    f"Hotel aspect heatmap includes {hotel_heatmap.shape[0]} HIGH/MEDIUM-sample hotels; LOW_SAMPLE hotels remain in CSV with caution flags.",
    f"Area heatmap includes {area_heatmap.shape[0]} areas with n>={MIN_AREA_N}; mapping coverage is unequal.",
    f"TF-IDF distinctive terms are produced for {distinctive_hotel.hotel_name.nunique()} hotels and {distinctive_area.area.nunique()} areas, as relative corpus markers rather than definitive problems.",
    f"Google-rating × aspect correlations are exploratory across {int(eligible_rate_matrix.hotel_id.nunique())} eligible hotels and are not multiple-comparison adjusted.",
    f"Official room coverage is {int(hotel_features.official_room_count.notna().sum())}/{len(hotel_features)} complaint hotels, so size/star aspect context is descriptive only.",
    f"Topic status is {topic_model_status}: independent human labels and multi-seed stability evidence are not yet available.",
    "Segmentation-ready features include aspect mention rates, response behavior, text/view medians, sample reliability, Google context and explicit coverage indicators.",
]
for finding in key_findings:
    print("-", finding)
""")

md("""## 24. Limitations""")

code("""limitations = [
    "Şikayetvar is a self-selected complaint platform; only reliably matched complaints are analyzed.",
    "Mapping coverage is unequal across hotels and areas; complaint absence must not be encoded as zero aspect risk.",
    "The complaint-hotel sample is 32 hotels and 15 are LOW_SAMPLE, so hotel-level rates are unstable.",
    "Rule-based dictionaries have false-positive and false-negative risk; manual validation remains TO_REVIEW.",
    "Turkish morphology and surface-form variation are only conservatively handled without a validated lemmatizer.",
    "Multi-label aspect overlap means aspect counts and rates do not sum to complaint count or 100%.",
    "Broad/ambiguous words such as service, room, entry, card and sound require context review.",
    "Company response existence, speed or template similarity does not establish resolution or satisfaction.",
    "Google and Şikayetvar represent different populations and time windows.",
    "Official star/capacity coverage is incomplete; capacity-related aspect comparisons are not robust.",
    "Topic modeling on a small selected corpus risks unstable and forced labels; it was gated out.",
    "Historical/platform completeness is uncertain and 86 complaint dates are approximate.",
    "TF-IDF is relative distinctiveness, not proof of a hotel's definitive problem.",
    "All rating/aspect correlations are exploratory, unadjusted for multiple comparisons and non-causal.",
]
for item in limitations:
    print("-", item)
""")

md("""## 25. Next step, exports and validation""")

code("""report_paths = {
    "corpus_summary": REPORTS_DIR / "sikayetvar_nlp_corpus_summary.csv",
    "unigrams": REPORTS_DIR / "sikayetvar_nlp_unigram_frequency.csv",
    "ngrams": REPORTS_DIR / "sikayetvar_nlp_ngrams.csv",
    "distinctive_hotel": REPORTS_DIR / "sikayetvar_nlp_distinctive_terms_by_hotel.csv",
    "distinctive_area": REPORTS_DIR / "sikayetvar_nlp_distinctive_terms_by_area.csv",
    "dictionary": REPORTS_DIR / "sikayetvar_aspect_dictionary.csv",
    "coverage": REPORTS_DIR / "sikayetvar_aspect_coverage_summary.csv",
    "frequency": REPORTS_DIR / "sikayetvar_aspect_frequency.csv",
    "hotel_matrix": REPORTS_DIR / "sikayetvar_hotel_aspect_matrix.csv",
    "area_matrix": REPORTS_DIR / "sikayetvar_area_aspect_matrix.csv",
    "hotel_lift": REPORTS_DIR / "sikayetvar_hotel_aspect_lift.csv",
    "area_lift": REPORTS_DIR / "sikayetvar_area_aspect_lift.csv",
    "cooccurrence": REPORTS_DIR / "sikayetvar_aspect_cooccurrence.csv",
    "response": REPORTS_DIR / "sikayetvar_aspect_response_summary.csv",
    "manual_sample": REPORTS_DIR / "sikayetvar_aspect_manual_validation_sample.csv",
    "kwic": REPORTS_DIR / "sikayetvar_aspect_keyword_context_samples.csv",
    "collisions": REPORTS_DIR / "sikayetvar_aspect_keyword_collisions.csv",
    "hotel_top": REPORTS_DIR / "sikayetvar_hotel_top_aspects.csv",
    "area_top": REPORTS_DIR / "sikayetvar_area_top_aspects.csv",
    "rating_corr": REPORTS_DIR / "sikayetvar_aspect_google_rating_correlations.csv",
    "topic_gate": REPORTS_DIR / "sikayetvar_topic_model_gate.csv",
    "temporal": REPORTS_DIR / "sikayetvar_temporal_aspect_summary.csv",
    "response_terms": REPORTS_DIR / "sikayetvar_company_response_terms.csv",
    "boilerplate": REPORTS_DIR / "sikayetvar_company_response_boilerplate.csv",
    "key_findings": REPORTS_DIR / "sikayetvar_nlp_key_findings.txt",
    "limitations": REPORTS_DIR / "sikayetvar_nlp_limitations.txt",
}
processed_paths = {
    "complaints_nlp": PROCESSED_DIR / "sikayetvar_all_hotels_complaints_nlp.csv",
    "aspects_long": PROCESSED_DIR / "sikayetvar_complaint_aspects_long.csv",
    "hotel_features": PROCESSED_DIR / "sikayetvar_hotel_nlp_features.csv",
    "area_features": PROCESSED_DIR / "sikayetvar_area_nlp_features.csv",
    "readme": PROCESSED_DIR / "README_sikayetvar_nlp_features.txt",
}

corpus_summary.to_csv(report_paths["corpus_summary"], index=False)
unigram_standard.to_csv(report_paths["unigrams"], index=False)
ngram_table.to_csv(report_paths["ngrams"], index=False)
distinctive_hotel.to_csv(report_paths["distinctive_hotel"], index=False)
distinctive_area.to_csv(report_paths["distinctive_area"], index=False)
aspect_dictionary.to_csv(report_paths["dictionary"], index=False)
aspect_coverage.to_csv(report_paths["coverage"], index=False)
aspect_frequency.to_csv(report_paths["frequency"], index=False)
hotel_aspect.to_csv(report_paths["hotel_matrix"], index=False)
area_aspect.to_csv(report_paths["area_matrix"], index=False)
hotel_lift.to_csv(report_paths["hotel_lift"], index=False)
area_lift.to_csv(report_paths["area_lift"], index=False)
cooccurrence.to_csv(report_paths["cooccurrence"], index=False)
aspect_response_summary.to_csv(report_paths["response"], index=False)
manual_validation_sample.to_csv(report_paths["manual_sample"], index=False)
keyword_context_samples.to_csv(report_paths["kwic"], index=False)
keyword_collisions.to_csv(report_paths["collisions"], index=False)
hotel_top_aspects.to_csv(report_paths["hotel_top"], index=False)
area_top_aspects.to_csv(report_paths["area_top"], index=False)
aspect_rating_correlations.to_csv(report_paths["rating_corr"], index=False)
topic_gate.to_csv(report_paths["topic_gate"], index=False)
temporal_aspect_summary.to_csv(report_paths["temporal"], index=False)
response_terms.to_csv(report_paths["response_terms"], index=False)
response_boilerplate.to_csv(report_paths["boilerplate"], index=False)
report_paths["key_findings"].write_text("\\n".join(f"- {item}" for item in key_findings) + "\\n", encoding="utf-8")
report_paths["limitations"].write_text("\\n".join(f"- {item}" for item in limitations) + "\\n", encoding="utf-8")

nlp_export = nlp.copy()
nlp_export["nlp_tokens"] = nlp_export["nlp_tokens"].map(lambda values: "|".join(values))
nlp_export["nlp_tokens_domain_filtered"] = nlp_export["nlp_tokens_domain_filtered"].map(lambda values: "|".join(values))
nlp_export.to_csv(processed_paths["complaints_nlp"], index=False)
aspects_long.to_csv(processed_paths["aspects_long"], index=False)
hotel_features.to_csv(processed_paths["hotel_features"], index=False)
area_features.to_csv(processed_paths["area_features"], index=False)

feature_readme = (
    "ŞİKAYETVAR NLP FEATURES\\n\\n"
    f"Corpus definition: Notebook 12 canonical-unique reliably matched clean complaints; {len(nlp)} rows, {text_document_n} non-empty texts.\\n"
    "Preprocessing: NFKC, Turkish lowercase, whitespace/punctuation cleanup, URL/email/phone masking in derived NLP text; raw text unchanged.\\n"
    f"Taxonomy: {len(CANONICAL_ASPECTS)} explainable multi-label aspects; phrase-first, word-boundary and limited Turkish surface-form rules.\\n"
    "Mention rate: unique complaints matching aspect / group clean complaint count * 100. It is not a real customer problem rate.\\n"
    "Hotel aggregation: one row per 32 complaint-bearing hotels; sample reliability uses Notebook 13 HIGH/MEDIUM/LOW tiers.\\n"
    "Area aggregation: all 14 project areas retained; areas without clean complaint data have missing aspect rates, not zero.\\n"
    "Small-n handling: hotel n<5 and area n<10 excluded from main heatmaps but retained with flags.\\n"
    "Company response: existence/rate/time are operational descriptors and do not imply resolution.\\n"
    "Missing coverage policy: hotels without trusted complaint data must not receive aspect rate=0; use data-availability and mapping-status indicators downstream.\\n"
    "Limitations: self-selected platform, unequal mapping coverage, Turkish morphology, rule-based errors, multi-label overlap, incomplete official metadata, non-causal correlations.\\n"
    "Intended use: coverage-aware segmentation/opportunity features with reliability indicators; not hotel quality scoring.\\n"
)
processed_paths["readme"].write_text(feature_readme, encoding="utf-8")

input_hashes_after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected_inputs}
figure_paths = sorted(FIGURES_DIR.glob("*.png"))
output_validation = pd.DataFrame([
    ("clean_inputs_unchanged", input_hashes_before == input_hashes_after),
    ("nlp_row_count_preserved", len(nlp_export) == len(complaints)),
    ("complaint_id_unique", nlp_export["complaint_id"].is_unique),
    ("aspect_columns_boolean", all(nlp[column].dtype == bool for column in aspect_columns)),
    ("aspect_count_reconciles", (nlp[aspect_columns].sum(axis=1) == nlp["aspect_count"]).all()),
    ("long_relation_reconciles", len(aspects_long) == len(nlp)*len(CANONICAL_ASPECTS) and int(aspects_long["matched"].sum()) == int(nlp["aspect_count"].sum())),
    ("no_aspect_reconciles", int(nlp["no_aspect_detected_flag"].sum()) == int(nlp["aspect_count"].eq(0).sum())),
    ("hotel_rates_valid", hotel_aspect["aspect_mention_rate_pct"].between(0, 100).all()),
    ("area_rates_valid", area_aspect["aspect_mention_rate_pct"].between(0, 100).all()),
    ("response_rates_valid", aspect_response_summary["company_response_rate_within_aspect"].dropna().between(0, 100).all()),
    ("hotel_feature_keys_unique", hotel_features["hotel_id"].is_unique),
    ("area_feature_keys_unique", area_features["area"].is_unique),
    ("small_n_rule_correct", (hotel_aspect["small_n_flag"] == hotel_aspect["hotel_n"].lt(MIN_HOTEL_N)).all()),
    ("manual_validation_sample_30_50", 30 <= len(manual_validation_sample) <= 50),
    ("topic_gate_respected", topic_model_status == "TOPIC_MODEL_NOT_RELIABLE"),
    ("all_15_figures_present", len(figure_paths) == 15),
    ("all_reports_present", all(path.exists() for path in report_paths.values())),
    ("all_processed_outputs_present", all(path.exists() for path in processed_paths.values())),
], columns=["check", "passed"])
output_validation.to_csv(REPORTS_DIR / "sikayetvar_nlp_output_validation.csv", index=False)
assert output_validation["passed"].all(), output_validation.loc[~output_validation["passed"]].to_dict("records")
display(output_validation)
print(f"Validated {len(figure_paths)} figures, {len(report_paths)} reports and {len(processed_paths)} processed outputs.")
""")

md("""### Sonraki aşama

Önerilen notebook: `notebooks/15_hotel_segmentation.ipynb`. Segmentation; hotel master,
destination context, official attributes ve yalnız coverage/reliability göstergeleriyle birlikte
NLP features kullanmalıdır. Complaint bulunmayan project hotel'lerde aspect rate **0 yapılmamalı**;
`sikayetvar_data_available` ve `sikayetvar_mapping_status` ayrı feature olarak taşınmalıdır.

**Hazırlık kararı:** Aspect/response/text features segmentation için hazırdır; LOW_SAMPLE hotel
rate'leri shrink/pool edilmeli veya ana clustering girdisi dışında sensitivity katmanında tutulmalıdır.
""")

# Normalize incrementally assembled sections to the required numeric order.
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
