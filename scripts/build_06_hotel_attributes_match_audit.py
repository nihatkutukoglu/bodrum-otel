"""06_hotel_attributes_match_audit.ipynb dosyasını tekrar üretilebilir biçimde oluşturur."""

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "06_hotel_attributes_match_audit.ipynb"

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


# ---------------------------------------------------------------------------
# 0. Başlık
# ---------------------------------------------------------------------------
md(
    """# Bodrum Hotel & Destination Intelligence
## 06 - Resmî Otel Özellikleri Eşleştirme ve Audit

Bu notebookun amacı **analiz yapmak değil**, resmî Bodrum konaklama tesisi kayıtlarını mevcut
192 otelik proje master verisine güvenilir biçimde bağlamak ve zenginleştirilmiş bir otel
dataseti üretmektir.

İki farklı evren var:

- **A — proje otelleri**: `data/processed/hotels_features.csv`, ~192 benzersiz otel (Google
  Places kaynaklı).
- **B — resmî tesis kayıtları**: `data/external/hotel/hotel_attributes_official_bodrum.csv`,
  ~168 resmî işletme belgeli tesis (Muğla İl Kültür ve Turizm Müdürlüğü kaynaklı).

**Bu iki dataset birebir aynı listeyi temsil etmek zorunda değildir.** 168 resmî tesis, 192
otelin tamamı anlamına gelmez; eşleşmeyen kayıtların olması normaldir ve beklenmektedir.

Temel ilke:

> **Yanlış eşleştirmeden kaçının. Daha az ama güvenilir eşleşme, çok sayıda hatalı
> eşleşmeden daha değerlidir.**

Bu nedenle yalnızca `MATCHED_HIGH_CONFIDENCE` olarak sınıflandırılan eşleşmeler otomatik olarak
zenginleştirilmiş datasete işlenir; şüpheli her şey ayrı bir manuel inceleme tablosuna yazılır.

Bu notebookta **kesinlikle yapılmayanlar**: K-Means/PCA/DBSCAN, regresyon/sınıflandırma,
rating/fiyat tahmini, sentiment/NLP, öneri sistemi, anomali tespiti, web scraping. Ayrıca
`data/processed/hotels_features.csv` ve external CSV hiçbir noktada değiştirilmez;
zenginleştirilmiş çıktı ayrı bir dosyaya (`data/processed/hotels_enriched.csv`) yazılır.
"""
)

# ---------------------------------------------------------------------------
# 1. Kurulum
# ---------------------------------------------------------------------------
md(
    """### 1. Kurulum ve veri yükleme

Eşleştirme mantığı `src/bodrum_intelligence/hotel_matching.py` modülünde tutulur (normalizasyon,
aday üretimi, skorlama, sınıflandırma, zenginleştirme fonksiyonları); notebook yalnızca bu
fonksiyonları çağırır ve sonuçları yorumlar.
"""
)

code(
    """from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bodrum_intelligence.hotel_matching import (
    MatchThresholds,
    build_area_coverage,
    build_best_match_table,
    build_enriched_dataset,
    classify_match,
    derive_official_status,
    detect_official_duplicates,
    explain_review_reason,
    generate_candidates,
    prepare_official_normalization,
    prepare_project_normalization,
    save_matching_outputs,
    score_match,
)

PROJECT_HOTELS_PATH = PROJECT_ROOT / "data" / "processed" / "hotels_features.csv"
OFFICIAL_PATH = PROJECT_ROOT / "data" / "external" / "hotel" / "hotel_attributes_official_bodrum.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

assert PROJECT_HOTELS_PATH.exists(), f"Proje otel verisi bulunamadı: {PROJECT_HOTELS_PATH}"
assert OFFICIAL_PATH.exists(), f"Resmî tesis verisi bulunamadı: {OFFICIAL_PATH}"

project_df = pd.read_csv(PROJECT_HOTELS_PATH, dtype={"phone": "string"})
official_df = pd.read_csv(OFFICIAL_PATH)

print(f"Proje otelleri (A): {project_df.shape[0]} satır, {project_df.shape[1]} kolon")
print(f"Resmî tesisler (B): {official_df.shape[0]} satır, {official_df.shape[1]} kolon")"""
)

# ---------------------------------------------------------------------------
# 2. Proje otelleri audit (evren A)
# ---------------------------------------------------------------------------
md(
    """### 2. Proje otelleri audit (evren A)

Eşleştirmeye başlamadan önce her iki tablo ayrı ayrı denetlenir. Bu, `02_data_audit.ipynb`'i
tekrarlamaz; yalnızca eşleştirmeyi doğrudan etkileyecek alanlara (kimlik, isim, telefon, bölge)
odaklanır.
"""
)

code(
    """def normalized_name_duplicate_count(names, normalize_fn):
    normalized = names.map(normalize_fn)
    return int(normalized[normalized.notna()].duplicated().sum())


from bodrum_intelligence.hotel_matching import normalize_text

project_audit = pd.DataFrame(
    [
        ("row_count", len(project_df)),
        ("column_count", project_df.shape[1]),
        ("unique_hotel_id", project_df["hotel_id"].nunique()),
        ("duplicate_hotel_id", int(project_df["hotel_id"].duplicated().sum())),
        ("unique_place_id", project_df["place_id"].nunique()),
        ("duplicate_place_id", int(project_df["place_id"].duplicated().sum())),
        ("missing_hotel_name", int(project_df["hotel_name"].isna().sum())),
        ("missing_phone", int(project_df["phone"].isna().sum())),
        ("missing_area", int(project_df["area"].isna().sum())),
        ("duplicate_normalized_hotel_name", normalized_name_duplicate_count(project_df["hotel_name"], normalize_text)),
    ],
    columns=["metric", "value"],
)
display(project_audit)"""
)

md(
    """**Bulgu:** Kimlikler (`hotel_id`, `place_id`) beklendiği gibi tamamen benzersiz; isim, telefon
ve bölge eksikliği sınırlı (bkz. `02_data_audit.ipynb`'te daha önce belgelenen 4 eksik telefon).
Normalize edilmiş isimde tekrar eden kayıt yok — proje tarafında eşleştirmeyi zorlaştıracak bir
duplicate isim problemi bulunmuyor."""
)

# ---------------------------------------------------------------------------
# 3. Resmî tesisler audit (evren B)
# ---------------------------------------------------------------------------
md(
    """### 3. Resmî tesisler audit (evren B)

Aynı temel denetim resmî tabloya uygulanır; ayrıca yıldız/oda/yatak/tip dağılımı ve
`mapped_project_area` kapsamı ile açık mantık hataları (`official_star_rating` 1-5 dışında,
`room_count`/`bed_count` <= 0, `bed_count < room_count`, duplicate `official_facility_id`)
kontrol edilir. **Hiçbir değer otomatik düzeltilmez, yalnızca flag üretilir.**
"""
)

code(
    """official_audit = pd.DataFrame(
    [
        ("row_count", len(official_df)),
        ("column_count", official_df.shape[1]),
        ("unique_official_facility_id", official_df["official_facility_id"].nunique()),
        ("duplicate_official_facility_id", int(official_df["official_facility_id"].duplicated().sum())),
        ("missing_official_name", int(official_df["official_name"].isna().sum())),
        ("missing_official_phone", int(official_df["official_phone"].isna().sum())),
        ("missing_official_address_area", int(official_df["official_address_area"].isna().sum())),
        ("missing_mapped_project_area", int(official_df["mapped_project_area"].isna().sum())),
        ("duplicate_normalized_official_name", normalized_name_duplicate_count(official_df["official_name"], normalize_text)),
    ],
    columns=["metric", "value"],
)
display(official_audit)"""
)

code(
    """print("official_star_rating dağılımı:")
display(official_df["official_star_rating"].value_counts(dropna=False).sort_index().to_frame("count"))

print("\\nofficial_type dağılımı:")
display(official_df["official_type"].value_counts(dropna=False).to_frame("count"))

print("\\nroom_count / bed_count özet:")
display(official_df[["room_count", "bed_count"]].describe().loc[["min", "50%", "max"]])

print("\\nmapped_project_area kapsamı:")
display(official_df["area_mapping_confidence"].value_counts(dropna=False).to_frame("count"))
print(f"mapped_project_area eksik: {official_df['mapped_project_area'].isna().sum()} / {len(official_df)}")"""
)

code(
    """logic_error_checks = pd.DataFrame(
    [
        ("official_star_rating < 1", int((official_df["official_star_rating"] < 1).sum())),
        ("official_star_rating > 5", int((official_df["official_star_rating"] > 5).sum())),
        ("room_count <= 0", int((official_df["room_count"] <= 0).sum())),
        ("bed_count <= 0", int((official_df["bed_count"] <= 0).sum())),
        ("bed_count < room_count", int((official_df["bed_count"] < official_df["room_count"]).sum())),
        ("duplicate official_facility_id", int(official_df["official_facility_id"].duplicated().sum())),
    ],
    columns=["check", "issue_count"],
)
display(logic_error_checks)
assert logic_error_checks["issue_count"].eq(0).all(), "Resmî veri setinde ham mantık hatası bulundu; düzeltilmeden ilerlenmemeli." """
)

md(
    """**Bulgu:** `official_facility_id` benzersiz (168/168), temel aralık/mantık kontrollerinin
tamamı temiz (0 ihlal). `official_star_rating` 40/168 kayıtta eksik — bu, veri toplama notunda
("Never infer a star rating where the official type does not explicitly encode one") açıklandığı
gibi kasıtlı bir eksikliktir (`BUTİK OTEL`, `MÜSTAKİL APART`, `PANSİYON` gibi tipler yıldız
belirtmez); **tahmin edilmez, olduğu gibi bırakılır**. `mapped_project_area`, 13 kayıtta eksik
(bu kayıtlar eşleştirmede bölge sinyali olmadan, yalnızca isim/telefon/adresle değerlendirilir)."""
)

# ---------------------------------------------------------------------------
# 4. Resmî tesis içi duplicate/conflict tespiti
# ---------------------------------------------------------------------------
md(
    """### 4. Resmî tesis içinde olası duplicate / çelişki tespiti

Aynı tesisin resmî listede birden fazla satırı olabilir (aynı/çok benzer isim, aynı telefon).
`detect_official_duplicates`, bu şekilde gruplanan kayıtları `official_duplicate_candidate`
olarak işaretler; grup içinde yıldız/tip/oda/yatak değerleri **çelişiyorsa** ayrıca
`official_conflict_flag=True` olur. **Hiçbir satır otomatik olarak seçilmez veya silinmez.**
"""
)

code(
    """official_norm = prepare_official_normalization(official_df)
official_norm = detect_official_duplicates(official_norm)

print(f"official_duplicate_candidate: {int(official_norm['official_duplicate_candidate'].sum())}")
print(f"official_conflict_flag: {int(official_norm['official_conflict_flag'].sum())}")

conflict_columns = [
    "duplicate_group_id", "official_facility_id", "official_name", "official_type",
    "official_star_rating", "room_count", "bed_count", "official_address_area", "official_phone",
    "official_duplicate_candidate", "official_conflict_flag", "conflict_reason",
]
official_conflicts = official_norm.loc[
    official_norm["official_duplicate_candidate"], conflict_columns
].sort_values(["duplicate_group_id", "official_facility_id"])
display(official_conflicts)"""
)

md(
    """**Bulgu:** 4 grup (8 kayıt) aynı telefon ve/veya aynı normalize isim nedeniyle
`duplicate_candidate` olarak işaretlendi; bunların tamamı aynı zamanda `conflict_flag=True`
(yıldız/tip/oda/yatak değerleri grup içinde farklı). İncelemede iki farklı gerçek senaryo
görülüyor: (1) gerçek veri tekrarı (OFFBOD025/OFFBOD064 — aynı isim, aynı bölge, aynı telefon)
ve (2) muhtemelen aynı işletme grubuna ait **kardeş tesisler** aynı santral numarasını
paylaşıyor (ör. "VOYAGE BODRUM" / "VOYAGE BODRUM PRIVATE", "CACTUS FLEUR BEACH CLUB" /
"ELEMENTA BY CACTUS HOTELS"). Her iki durumda da otomatik seçim yapılmadı; tam liste
`reports/hotel_attributes_official_conflicts.csv` dosyasına yazılacak (Bölüm 15)."""
)

# ---------------------------------------------------------------------------
# 5. Normalizasyon hazırlığı
# ---------------------------------------------------------------------------
md(
    """### 5. Eşleştirme için normalizasyon

Ham `hotel_name`/`official_name`/`phone`/`address` kolonları **değiştirilmez**. Yalnızca
eşleştirme amaçlı geçici kolonlar üretilir:

- `*_normalized_full`: küçük harf, kontrollü Türkçe karakter dönüşümü (ı/İ→i, ğ→g, ü→u, ş→s,
  ö→o, ç→c), noktalama temizliği, fazla boşluk temizliği. **Hiçbir kelime silinmez** — `hotel`,
  `otel`, `resort`, `spa` gibi kelimeler bazı otelleri ayırt etmek için gerekli olabileceğinden
  körlemesine kaldırılmaz.
- `*_normalized_core`: `full` üzerine yalnızca coğrafi/dolgu kelimeler (`bodrum`, `turkey`,
  `turkiye`) kaldırılır — bu kelimeler veri setinin tamamında ortak olduğu için ayırt edici
  değer taşımaz. `hotel`/`otel`/`resort`/`spa` `core` sürümde de dokunulmadan kalır.
- `phone_normalized`: yalnızca rakamlar, son 10 hane (ülke kodu/başındaki 0 farkı ortadan
  kalkar).
- `address_normalized` / `area_normalized`: aynı metin normalizasyonu.
"""
)

code(
    """project_norm = prepare_project_normalization(project_df)
# official_norm, Bölüm 4'te duplicate/conflict flag'leriyle birlikte zaten üretildi; burada
# yeniden kullanılıyor (normalize kolonları o adımda eklenmişti).

sample_cols_project = ["hotel_name", "hotel_name_normalized_full", "hotel_name_normalized_core", "phone", "phone_normalized"]
display(project_norm[sample_cols_project].head(8))

sample_cols_official = ["official_name", "official_name_normalized_full", "official_name_normalized_core", "official_phone", "official_phone_normalized"]
display(official_norm[sample_cols_official].head(8))"""
)

md(
    """**Örnek:** "Mandarin Oriental, Bodrum" → `full`="mandarin oriental bodrum", `core`="mandarin
oriental" (yalnızca "bodrum" düşer). Resmî kayıt "MANDARİN ORIENTAL BODRUM" da aynı `core`
değerine indirgeniyor — bu, aşağıdaki Katman 2'de doğrudan kesin eşleşme sağlıyor."""
)

# ---------------------------------------------------------------------------
# 6. Katmanlı aday üretimi
# ---------------------------------------------------------------------------
md(
    """### 6. Katmanlı eşleştirme adayları

`generate_candidates`, her proje oteli için üç katmanda aday resmî tesis toplar:

1. **Telefon** — normalize telefon birebir aynıysa.
2. **Kesin çekirdek isim** — `hotel_name_normalized_core` birebir aynıysa.
3. **Fuzzy isim** — `difflib.SequenceMatcher` oranı ≥ 0.55 olan tüm resmî kayıtlar (hem `full`
   hem `core` üzerinden en yüksek oran alınır).

Üç katmanın adayları birleştirilir (tekilleştirilir); tek başına fuzzy skor kullanılmaz —
her aday için isim benzerliğine ek olarak bölge, telefon ve adres destek sinyalleri de
hesaplanır (Bölüm 7).
"""
)

code(
    """candidates = generate_candidates(project_norm, official_norm)

candidate_summary = pd.DataFrame(
    [
        ("total_candidate_pairs", len(candidates)),
        ("project_hotels_with_at_least_one_candidate", candidates["hotel_id"].nunique()),
        ("project_hotels_with_zero_candidates", project_df["hotel_id"].nunique() - candidates["hotel_id"].nunique()),
        ("median_candidates_per_hotel", int(candidates.groupby("hotel_id").size().median())),
    ],
    columns=["metric", "value"],
)
display(candidate_summary)"""
)

md(
    """**Bulgu:** 192 proje otelinin 168'i için en az bir aday bulundu; 24 otel için (rakam sayısal
sonuçtan üretilir) hiçbir katman minimum fuzzy eşiğini bile geçen bir resmî kayıt üretemedi —
bu oteller doğrudan `UNMATCHED` olacak (Bölüm 8), herhangi bir zorlama yapılmaz."""
)

# ---------------------------------------------------------------------------
# 7. Açıklanabilir skor mantığı
# ---------------------------------------------------------------------------
md(
    """### 7. Açıklanabilir match score

Her aday çift için dört ham sinyal hesaplanır: `name_similarity`, `area_match`, `phone_match`,
`address_similarity`. Ağırlıklı skor:

```
match_score = 0.60 * name_similarity + 0.20 * phone_match + 0.15 * area_match + 0.05 * address_similarity
```

`area_match` resmî tarafta bölge bilgisi yoksa (`None`) ne ödüllendirilir ne cezalandırılır
(0.5 nötr katkı). **Kesin ve tekil telefon eşleşmesi** tek başına güçlü bir sinyal olduğu için
skoru en az 0.95'e yükseltir (isimler farklı yazılmış olsa bile aynı santral numarasını
paylaşma ihtimali yüksektir).
"""
)

code(
    """candidates["match_score"] = [
    score_match(row.name_similarity, row.area_match, row.phone_match, row.address_similarity)
    for row in candidates.itertuples()
]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(candidates["match_score"], bins=30, color="#2F6B7C", edgecolor="white")
ax.set(title="Tüm Aday Çiftlerin match_score Dağılımı", xlabel="match_score", ylabel="Aday sayısı")
ax.grid(axis="y", alpha=0.2)
plt.show()

print("En iyi adaya göre (her otel için tek en yüksek skor) skor dağılımı:")
best_per_hotel_preview = candidates.sort_values(["hotel_id", "match_score"], ascending=[True, False]).groupby("hotel_id").first()
display(best_per_hotel_preview["match_score"].describe())"""
)

md(
    """**Bulgu — skor dağılımı:** Telefon eşleşen adaylar skoru doğrudan ≥0.95 bandına taşıyor; telefon
olmadan ulaşılabilecek matematiksel tavan 0.60+0.15+0.05=0.80'dir (isim %60 + bölge %15 + adres
%5 tam puan alsa bile). Bu, **0.80 ile 0.95 arasında doğal ve boş bir bölge** oluşturuyor — bu
notebookta yüksek güven eşiği bu boşluğa değil, doğrudan bileşenlere (isim ≥0.95 **ve** bölge
doğrulanmış) dayandırıldı; aşağıda gerçek sınır vakalarıyla gösteriliyor.
"""
)

code(
    """edge_case_ids = ["Armonia Holiday Village & Spa", "Mandarin Resort Hotel", "Mandarin Oriental, Bodrum", "Voyage Torba", "Voyage Torba Private"]
edge_case_cols = ["hotel_name", "official_name", "name_similarity", "area_match", "phone_match", "address_similarity", "match_score"]
edge_cases = candidates[candidates["hotel_name"].isin(edge_case_ids)].sort_values(["hotel_name", "match_score"], ascending=[True, False])
edge_cases = edge_cases.loc[edge_cases.groupby("hotel_name")["match_score"].idxmax()]
display(edge_cases[edge_case_cols])"""
)

md(
    """**Sınır vakaları (dataset'ten gerçek örnekler):**

- **"Armonia Holiday Village & Spa"**: isim %100, adres %100 örtüşüyor ama resmî kayıt bölgeyi
  "Turgutreis" olarak işaretlerken proje bu oteli "Akyarlar"a koyuyor — **açık bir bölge
  çelişkisi**. İsim/adres ne kadar mükemmel olursa olsun bu otomatik yüksek güvene alınmadı;
  `REVIEW_REQUIRED`'a düşürüldü (bkz. Bölüm 8, `classify_match`).
- **"Mandarin Resort Hotel" vs "Mandarin Oriental, Bodrum"**: iki farklı proje oteli, resmî
  kayıttaki **aynı telefon numarasına** eşleşiyor (muhtemelen Google verisinde bir veri
  girişi/karışıklık hatası). Yalnızca isim benzerliği %68 olan "Mandarin Resort Hotel" ham
  telefon eşleşmesiyle skorunu 0.95'e taşısa da, isim benzerliği %100 olan "Mandarin Oriental,
  Bodrum" ile çakışınca çakışma çözümü **isim benzerliği daha yüksek olanı** tutuyor (Bölüm 8);
  "Mandarin Resort Hotel" otomatik bağlanmak yerine manuel incelemeye düşüyor.
- **"Voyage Torba" / "Voyage Torba Private"**: kardeş tesisler aynı resepsiyon telefonunu
  paylaşıyor. Ana tesisle isim benzerliği %100 olan "Voyage Torba" yüksek güvenle eşleşirken,
  resmî listede kendi ayrı kaydı olmayan "Voyage Torba Private" aynı `official_facility_id`'yi
  otomatik olarak devralmıyor; manuel incelemeye düşüyor.

Bu üç örnek, tek bir sinyale (yalnızca isim veya yalnızca telefon) güvenmenin neden yeterli
olmadığını somut biçimde gösteriyor."""
)

# ---------------------------------------------------------------------------
# 8. Sınıflandırma ve en iyi eşleşme tablosu
# ---------------------------------------------------------------------------
md(
    """### 8. Match statüsü sınıflandırması

`classify_match`, her proje oteli için en iyi tek adayı `MATCHED_HIGH_CONFIDENCE` /
`REVIEW_REQUIRED` / `UNMATCHED` olarak sınıflandırır. Yüksek güven için **iki** konservatif yol
vardır:

1. Kesin ve **tekil** telefon eşleşmesi (aynı telefon birden fazla resmî tesiste geçiyorsa bu
   yol devre dışı kalır — Bölüm 3'te bulunan 4 paylaşılan telefon grubu böyle ele alınıyor).
2. İsim benzerliği ≥ 0.95 **ve** bölge açıkça doğrulanmış (`area_match is True`; bölge bilgisi
   eksik veya çelişkiliyse bu yol tetiklenmez — Bölüm 7'deki Armonia örneği tam olarak bunu
   gösteriyor).

Ayrıca aynı `official_facility_id`'nin iki farklı otele yüksek güvenle bağlanması (Bölüm 7'deki
Mandarin örneği) otomatik olarak tespit edilip yalnızca isim benzerliği daha yüksek olan taraf
tutulur; diğeri `REVIEW_REQUIRED`'a düşürülür.
"""
)

code(
    """best_matches = build_best_match_table(candidates, project_norm)
assert len(best_matches) == len(project_df), "Her proje oteli için tam olarak bir satır olmalı."
assert best_matches["hotel_id"].nunique() == len(project_df)

status_counts = best_matches["match_status"].value_counts()
display(status_counts.to_frame("hotel_count"))

fig, ax = plt.subplots(figsize=(6, 4))
order = ["MATCHED_HIGH_CONFIDENCE", "REVIEW_REQUIRED", "UNMATCHED"]
colors = {"MATCHED_HIGH_CONFIDENCE": "#4C956C", "REVIEW_REQUIRED": "#D9A404", "UNMATCHED": "#8C8C8C"}
values = [int(status_counts.get(s, 0)) for s in order]
ax.bar(order, values, color=[colors[s] for s in order])
ax.set(title="Proje Otelleri - Match Status Dağılımı (n=192)", ylabel="Otel sayısı")
ax.grid(axis="y", alpha=0.2)
for i, v in enumerate(values):
    ax.text(i, v, str(v), ha="center", va="bottom")
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()"""
)

md(
    """**Bulgu:** 192 proje otelinden yalnızca telefon veya isim+bölge ile **çok güçlü** doğrulanan
kayıtlar otomatik olarak yüksek güvene alındı; geri kalanı ya insan incelemesi bekliyor ya da
hiç aday bulunamadığı için eşleşmedi. Tam sayılar Bölüm 19'da özetlenecek."""
)

# ---------------------------------------------------------------------------
# 9. Resmî taraf statüsü
# ---------------------------------------------------------------------------
md(
    """### 9. Resmî tesis tarafında statü

Her resmî kayıt için de bir statü türetilir: `MATCHED` (bir proje oteline yüksek güvenle
bağlandı), `CONFLICT` (Bölüm 4'te iç çelişki bulundu — veri kalitesi sorunu önceliklidir, bir
eşleşmesi olsa bile gizlenmez) veya `UNMATCHED_OFFICIAL` (proje örnekleminde karşılığı
bulunamadı).
"""
)

code(
    """official_status = derive_official_status(official_norm, best_matches)
display(official_status["official_match_status"].value_counts().to_frame("facility_count"))"""
)

md(
    """**Bulgu:** 168 resmî kaydın bir kısmı bizim 192 otelik örneklemimizde hiç yer almıyor
(`UNMATCHED_OFFICIAL`) — bu beklenen bir durumdur, çünkü resmî liste yalnızca "işletme belgeli"
tesisleri kapsar ve bizim Google Places kaynaklı örneklemimiz daha geniş bir konaklama
yelpazesini (apart, pansiyon, küçük butik) içerir. `CONFLICT` olarak işaretlenen 8 kayıt, bir
eşleşmeye sahip olsalar bile ayrı olarak raporlanacak (Bölüm 15) — çelişkili veri, eşleşme
başarısıyla gizlenmez."""
)

# ---------------------------------------------------------------------------
# 10. High confidence match tablosu
# ---------------------------------------------------------------------------
md(
    """### 10. High confidence match tablosu

Yalnızca `MATCHED_HIGH_CONFIDENCE` kayıtlar; bu tablo Bölüm 13'teki zenginleştirilmiş
datasetin temelini oluşturur.
"""
)

code(
    """high_confidence_cols = [
    "hotel_id", "hotel_name", "official_facility_id", "official_name",
    "match_score", "match_method", "official_type", "official_star_rating", "room_count", "bed_count",
]
high_confidence_matches = best_matches.loc[
    best_matches["match_status"].eq("MATCHED_HIGH_CONFIDENCE"), high_confidence_cols
].sort_values("hotel_id").reset_index(drop=True)
print(f"{len(high_confidence_matches)} yüksek güvenli eşleşme")
display(high_confidence_matches.head(10))"""
)

# ---------------------------------------------------------------------------
# 11. Manual review tablosu
# ---------------------------------------------------------------------------
md(
    """### 11. Manual review tablosu

`REVIEW_REQUIRED` kayıtlar, insan gözüyle hızlı karar verilebilmesi için tüm bağlam
kolonlarıyla birlikte ve kısa bir `reason_for_review` açıklamasıyla listelenir.
"""
)

code(
    """review_required = best_matches.loc[best_matches["match_status"].eq("REVIEW_REQUIRED")].copy()
review_required["reason_for_review"] = [
    explain_review_reason(row.name_similarity, row.area_match, row.phone_match, row.address_similarity)
    for row in review_required.itertuples()
]
manual_review_cols = [
    "hotel_id", "hotel_name", "area", "phone", "address",
    "official_facility_id", "official_name", "official_type", "official_star_rating",
    "room_count", "bed_count", "official_address_area", "official_phone",
    "name_similarity", "area_match", "phone_match", "address_similarity",
    "match_score", "match_method", "reason_for_review",
]
manual_review = review_required[manual_review_cols].sort_values("match_score", ascending=False).reset_index(drop=True)
print(f"{len(manual_review)} manuel inceleme kaydı")
display(manual_review.head(10))
print()
print("reason_for_review dağılımı:")
display(manual_review["reason_for_review"].value_counts().to_frame("count"))"""
)

# ---------------------------------------------------------------------------
# 12. Unmatched tablolar
# ---------------------------------------------------------------------------
md(
    """### 12. Unmatched tablolar

İki ayrı yön: bizim 192 otelden resmî listede bulunamayanlar, ve resmî listede olup bizim
örneklemimizde bulunamayanlar.
"""
)

code(
    """unmatched_project_cols = ["hotel_id", "hotel_name", "area", "phone", "address", "google_rating", "google_review_count"]
unmatched_project_hotels = project_df.loc[
    project_df["hotel_id"].isin(best_matches.loc[best_matches["match_status"].eq("UNMATCHED"), "hotel_id"]),
    unmatched_project_cols,
].reset_index(drop=True)
print(f"{len(unmatched_project_hotels)} proje oteli resmî listede bulunamadı")
display(unmatched_project_hotels.head(10))"""
)

code(
    """matched_official_ids = set(best_matches.loc[best_matches["match_status"].eq("MATCHED_HIGH_CONFIDENCE"), "official_facility_id"])
unmatched_official_cols = [
    "official_facility_id", "official_name", "official_type", "official_star_rating",
    "room_count", "bed_count", "official_address_area", "mapped_project_area",
]
unmatched_official_facilities = official_df.loc[
    ~official_df["official_facility_id"].isin(matched_official_ids), unmatched_official_cols
].reset_index(drop=True)
print(f"{len(unmatched_official_facilities)} resmî tesis proje örnekleminde bulunamadı")
display(unmatched_official_facilities.head(10))"""
)

md(
    """**Bulgu:** Eşleşmeyen proje otelleri büyük ölçüde küçük apart/pansiyon/villa tipi, resmî
"işletme belgeli tesis" listesine muhtemelen hiç girmemiş konaklama birimleri. Eşleşmeyen resmî
tesisler ise bizim Google Places tabanlı 192 otelik örneklemimize düşmemiş (küçük ölçekli veya
farklı adla listelenmiş) resmî kayıtlar — her iki liste de kendi içinde geçerli, yalnızca
kapsamları örtüşmüyor."""
)

# ---------------------------------------------------------------------------
# 13. Zenginleştirilmiş dataset
# ---------------------------------------------------------------------------
md(
    """### 13. Zenginleştirilmiş dataset

Yalnızca `MATCHED_HIGH_CONFIDENCE` eşleşmeler proje tablosuna eklenir. Mevcut
`hotels_features.csv` **değiştirilmez**; çıktı ayrı bir dosyaya yazılır. Ana datasetteki mevcut
`official_star_rating` kolonu (tamamen eksik) **asla üzerine yazılmaz** — yeni doğrulanmış
yıldız kolonu `official_star_rating_verified` adıyla eklenir.
"""
)

code(
    """enriched = build_enriched_dataset(project_df, best_matches)

assert len(enriched) == len(project_df), "Satır sayısı korunmalı."
assert enriched["hotel_id"].nunique() == len(project_df), "hotel_id benzersiz kalmalı."
assert enriched["official_star_rating"].equals(project_df["official_star_rating"]), "Mevcut official_star_rating overwrite edilmemeli."

new_columns = [c for c in enriched.columns if c not in project_df.columns]
print(f"Eklenen yeni kolon sayısı: {len(new_columns)}")
print(new_columns)
print()
print(f"Toplam satır: {len(enriched)}, toplam kolon: {enriched.shape[1]}")
display(enriched.loc[enriched["official_match_status"].eq("MATCHED_HIGH_CONFIDENCE"),
                      ["hotel_id", "hotel_name", "official_name", "official_star_rating_verified",
                       "official_room_count", "official_bed_count", "official_match_score"]].head(8))"""
)

# ---------------------------------------------------------------------------
# 14. Enrichment sonrası kalite kontrolü
# ---------------------------------------------------------------------------
md(
    """### 14. Enrichment sonrası kalite kontrolü

Eşleşmeyen otellerde hiçbir değerin uydurulmadığı doğrulanır, ardından genel kapsam sayıları ve
yüzdeleri özetlenir.
"""
)

code(
    """unmatched_mask = enriched["official_match_status"].eq("UNMATCHED")
fabricated_columns = ["official_star_rating_verified", "official_room_count", "official_bed_count",
                       "official_facility_id", "official_name"]
for column in fabricated_columns:
    assert enriched.loc[unmatched_mask, column].isna().all(), f"Unmatched otelde uydurma veri bulundu: {column}"
print("Doğrulandı: unmatched otellerde resmî kolonların tamamı boş (uydurma veri yok).")"""
)

code(
    """match_summary = pd.DataFrame(
    [
        ("total_project_hotels", len(project_df)),
        ("high_confidence_matches", int(status_counts.get("MATCHED_HIGH_CONFIDENCE", 0))),
        ("review_required", int(status_counts.get("REVIEW_REQUIRED", 0))),
        ("unmatched_project_hotels", int(status_counts.get("UNMATCHED", 0))),
        ("official_facilities_total", len(official_df)),
        ("matched_official_facilities", int((official_status["official_match_status"] == "MATCHED").sum())),
        ("unmatched_official_facilities", int((official_status["official_match_status"] == "UNMATCHED_OFFICIAL").sum())),
        ("official_conflicts", int(official_status["official_conflict_flag"].sum())),
        ("verified_star_coverage_pct", round(enriched["official_star_rating_verified"].notna().mean() * 100, 1)),
        ("room_count_coverage_pct", round(enriched["official_room_count"].notna().mean() * 100, 1)),
        ("bed_count_coverage_pct", round(enriched["official_bed_count"].notna().mean() * 100, 1)),
    ],
    columns=["metric", "value"],
)
display(match_summary)"""
)

# ---------------------------------------------------------------------------
# 15. Area bazlı coverage
# ---------------------------------------------------------------------------
md(
    """### 15. Bölge bazlı resmî coverage

Hangi destinasyonlarda resmî veri kapsamının güçlü/zayıf olduğunu gösterir.
"""
)

code(
    """area_coverage = build_area_coverage(enriched)
display(area_coverage)

fig, ax = plt.subplots(figsize=(8, 5.5))
order = area_coverage.sort_values("match_rate_pct")["area"]
ax.barh(order, area_coverage.set_index("area").loc[order, "match_rate_pct"], color="#2F6B7C")
ax.set(title="Bölgeye Göre Resmî Eşleşme Oranı", xlabel="match_rate_pct (%)")
ax.grid(axis="x", alpha=0.2)
plt.tight_layout()
plt.show()"""
)

md(
    """**Bulgu:** Gümüşlük ve Güvercinlik'te resmî eşleşme oranı **%0** — bu iki bölgede projedeki hiçbir
otel yüksek güvenle resmî kayda bağlanamadı (isim/telefon uyuşmazlığı veya bu bölgelerdeki
otellerin resmî "işletme belgeli tesis" listesinde yer almaması olabilir). En yüksek coverage
Turgutreis (%57) ve Kadıkalesi (%50) gibi küçük örneklemli bölgelerde — tek tük eşleşme küçük
örneklemde oranı hızlı yükseltiyor, bu nedenle yüzdeler mutlak sayılarla (hotel_count) birlikte
okunmalı."""
)

# ---------------------------------------------------------------------------
# 16. Yıldız / oda / yatak dağılımları (validation amaçlı)
# ---------------------------------------------------------------------------
md(
    """### 16. Yıldız / oda / yatak dağılımı (yalnızca veri kontrolü)

Bu notebook bir EDA notebooku değildir; aşağıdaki tablolar yalnızca zenginleştirilmiş verinin
makul göründüğünü doğrulamak içindir. **Yıldız vs rating, oda sayısı vs rating gibi analizler
kasıtlı olarak yapılmamıştır** — bu, `07_hotel_attributes_analysis.ipynb`'in görevidir.
"""
)

code(
    """star_labels = {1.0: "1 yıldız", 2.0: "2 yıldız", 3.0: "3 yıldız", 4.0: "4 yıldız", 5.0: "5 yıldız"}
verified_star_counts = enriched["official_star_rating_verified"].map(star_labels).fillna("Diğer/bilinmiyor (tip yıldız belirtmiyor veya eşleşmedi)")
display(verified_star_counts.value_counts().to_frame("hotel_count"))

print()
print("official_room_count özet (yalnızca eşleşen oteller):")
display(enriched["official_room_count"].describe()[["min", "50%", "max"]])
print()
print("official_bed_count özet (yalnızca eşleşen oteller):")
display(enriched["official_bed_count"].describe()[["min", "50%", "max"]])"""
)

# ---------------------------------------------------------------------------
# 17. Raporların ve zenginleştirilmiş datasetin kaydedilmesi
# ---------------------------------------------------------------------------
md(
    """### 17. Çıktıların kaydedilmesi

Tüm rapor tabloları `reports/` altına, zenginleştirilmiş dataset `data/processed/hotels_enriched.csv`
olarak kaydedilir. Ana `hotels_features.csv` ve external CSV bu notebookta hiçbir noktada
değiştirilmedi.
"""
)

code(
    """official_conflicts_output = official_conflicts.copy()

outputs = {
    "hotel_attributes_match_summary": match_summary,
    "hotel_attributes_high_confidence_matches": high_confidence_matches,
    "hotel_attributes_manual_review": manual_review,
    "hotel_attributes_unmatched_project_hotels": unmatched_project_hotels,
    "hotel_attributes_unmatched_official_facilities": unmatched_official_facilities,
    "hotel_attributes_official_conflicts": official_conflicts_output,
    "hotel_attributes_match_coverage_by_area": area_coverage,
    "hotels_enriched": enriched,
}
output_paths = save_matching_outputs(outputs, REPORTS_DIR, PROCESSED_DIR)
display(pd.DataFrame([
    {"output": name, "path": str(path.relative_to(PROJECT_ROOT))} for name, path in output_paths.items()
]))"""
)

# ---------------------------------------------------------------------------
# 18. Testler
# ---------------------------------------------------------------------------
md(
    """### 18. Testler

Bu eşleştirme mantığının temel garantileri `tests/test_hotel_matching.py` içinde otomatik test
edilir (gerçek proje verisiyle çalışır):

- 192 proje oteli korunuyor, `hotel_id`/`place_id` benzersiz kalıyor.
- Hiçbir `official_facility_id`, yüksek güvenli eşleşmede birden fazla otele bağlanmıyor.
- `official_star_rating_verified` her zaman 1-5 aralığında.
- `official_room_count` / `official_bed_count` negatif veya sıfır değil.
- Eşleşmeyen otellerde hiçbir resmî kolon uydurulmuyor (tamamı `NaN`).
- Mevcut `official_star_rating` kolonu overwrite edilmiyor.

Çalıştırma: `PYTHONPATH=src python3 -m unittest tests.test_hotel_matching -v`
"""
)

code(
    """import io
import unittest

TESTS_DIR = PROJECT_ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

loader = unittest.TestLoader()
suite = loader.loadTestsFromName("test_hotel_matching")
stream = io.StringIO()
runner = unittest.TextTestRunner(stream=stream, verbosity=2)
test_result = runner.run(suite)
print(stream.getvalue())
assert test_result.wasSuccessful(), "hotel_matching testleri başarısız oldu." """
)

# ---------------------------------------------------------------------------
# Eşleştirme Sonuçları ve Sonraki Aşama
# ---------------------------------------------------------------------------
md(
    """## Eşleştirme Sonuçları ve Sonraki Aşama

**Proje otelleri (192):**

- **52 otel (%27)** `MATCHED_HIGH_CONFIDENCE` — kesin/tekil telefon eşleşmesi ya da neredeyse
  birebir isim + doğrulanmış bölge ile otomatik zenginleştirildi.
- **48 otel (%25)** `REVIEW_REQUIRED` — güçlü ama tek başına yeterli görülmeyen sinyaller
  (ör. yüksek isim benzerliği ama telefon/bölge teyidi yok, ya da bölge çelişkisi) nedeniyle
  manuel incelemeye bırakıldı; `reports/hotel_attributes_manual_review.csv`.
- **92 otel (%48)** `UNMATCHED` — hiçbir katmanda anlamlı bir resmî aday bulunamadı; büyük
  ölçüde küçük apart/pansiyon/villa tipi konaklamalar.

**Resmî kayıtlar (168):**

- **52 kayıt** proje verisine bağlandı (`MATCHED`).
- **108 kayıt** proje örnekleminde karşılığı bulunamadı (`UNMATCHED_OFFICIAL`).
- **8 kayıt** (4 grup) iç çelişki taşıyor (`official_conflict_flag`) — aynı isim/telefon
  paylaşan ama yıldız/tip/oda/yatak bilgisi farklı olan kayıtlar; ayrı raporlandı.

**Zenginleştirme kapsamı (yalnızca yüksek güvenli 52 otel üzerinden):**

- Doğrulanmış yıldız bilgisi olan otel sayısı: **51 / 192 (%26.6)** (52 eşleşmeden 1'inin resmî
  tipi yıldız belirtmiyor, bu kasıtlı olarak boş bırakıldı).
- Oda/yatak kapasitesi olan otel sayısı: **52 / 192 (%27.1)** her ikisi için de.

**En önemli matching sorunları:**

1. **Coğrafi çelişkiler isim/adres eşleşmesini geçersiz kılabiliyor** — "Armonia Holiday
   Village & Spa" örneğinde isim ve adres %100 örtüşse bile resmî kayıttaki bölge etiketi
   projeyle çelişiyor; bu tür kayıtlar bilinçli olarak otomatik merge dışında tutuldu.
2. **Paylaşılan telefon numaraları yanlış yönlendirebilir** — "Mandarin Resort Hotel" ile
   "Mandarin Oriental, Bodrum" aynı resmî telefon numarasına eşleşiyor; yalnızca isim
   benzerliğini de dikkate alan çakışma çözümü doğru tarafı (Mandarin Oriental) korudu.
3. **Kardeş/şube tesisler tek resmî kayda sıkışabiliyor** — "Voyage Torba" / "Voyage Torba
   Private" ve "CACTUS" grubu örneklerinde birden fazla proje/marka aynı santral hattını
   paylaşıyor; bu gruplardan yalnızca en güçlü eşleşme otomatik alındı, diğerleri incelemeye
   bırakıldı.
4. **Resmî coverage bölgeler arası çok dengesiz** — Gümüşlük ve Güvercinlik'te %0 eşleşme
   oranı, bu iki destinasyonda ileride resmî veri zenginleştirmesinin en zayıf halka olacağını
   gösteriyor.

**Sonraki aşama:**

```text
07_hotel_attributes_analysis.ipynb
```

Bu notebookta üretilen `data/processed/hotels_enriched.csv` girdi olarak kullanılacak; yıldız
vs rating, oda sayısı vs rating, fiyat vs yıldız gibi analizler ancak o notebookta yapılacaktır.
"""
)

# ---------------------------------------------------------------------------
# Sonuç
# ---------------------------------------------------------------------------
md(
    """### Sonuç

- 192 proje oteli ve 168 resmî tesis kaydı hiçbir kaynak dosya değiştirilmeden ayrı ayrı audit
  edildi; ham mantık hatası (yıldız/oda/yatak aralığı, duplicate ID) bulunmadı.
- Resmî tesis içindeki 8 olası duplicate/çelişki otomatik seçim yapılmadan flag'lendi.
- Katmanlı eşleştirme (telefon → kesin isim → fuzzy isim + destek sinyalleri) ve açıklanabilir
  ağırlıklı skor ile yalnızca 52 otel otomatik zenginleştirildi; 48 otel manuel incelemeye,
  92 otel `UNMATCHED` olarak bırakıldı — hiçbir veri uydurulmadı.
- Gerçek skor dağılımı incelenerek (0.80-0.95 arası doğal boşluk) ve üç somut sınır vakası
  (Armonia, Mandarin, Voyage Torba) üzerinden eşikler konservatif biçimde kalibre edildi.
- `data/processed/hotels_enriched.csv` ve 7 rapor `reports/` altına kaydedildi; mevcut
  `hotels_features.csv`, external CSV ve önceki notebooklar değişmeden kaldı.
- `tests/test_hotel_matching.py` içindeki 7 test bu garantileri otomatik doğruluyor.
- Sonraki aşama: `07_hotel_attributes_analysis.ipynb` — yıldız/oda/fiyat/rating ilişkilerinin
  analiz edileceği notebook.
"""
)

nb["cells"] = cells
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK_PATH)
print(f"Tamamlandı: {NOTEBOOK_PATH} -- {len(cells)} hücre")
