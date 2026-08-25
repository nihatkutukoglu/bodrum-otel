"""04_feature_engineering.ipynb dosyasını tekrar üretilebilir biçimde oluşturur."""

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "04_feature_engineering.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# Bodrum Hotel & Destination Intelligence
## 04 - Temel Feature Engineering

Bu notebook, temizlenmiş tek ana `df` tablosuna EDA öncesinde kullanılabilecek temel ve açıklanabilir özellikleri ekler.

Kapsam bilinçli olarak sınırlıdır:

- Eksik fiyat, resmî yıldız veya diğer alanlar doldurulmaz.
- `value_score`, `luxury_score` ve `destination_score` gibi ağırlık seçimi gerektiren bileşik skorlar üretilmez.
- Bölgesel özellikler yalnızca mevcut ana tablodan türetilir; ayrı `area_summary` girdisi kullanılmaz.
- Fiyat özellikleri, sabit otel fiyatı değil, yalnızca 2026-08-24 tarihli snapshot bağlamını temsil eder.
"""
    ),
    nbf.v4.new_markdown_cell(
        """### 1. Kurulum ve temiz verinin yüklenmesi

Girdi `03_data_cleaning.ipynb` tarafından üretilen `data/processed/hotels_clean.csv` dosyasıdır. Notebook boyunca tek ana çalışma tablosu `df` olarak adlandırılır.
"""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import sys

import pandas as pd
from IPython.display import Markdown, display

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bodrum_intelligence.features import build_basic_features, save_feature_outputs

CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "hotels_clean.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

assert CLEAN_PATH.exists(), f'Temiz veri bulunamadı: {CLEAN_PATH}'

df = pd.read_csv(CLEAN_PATH, dtype={"phone": "string"})"""
    ),
    nbf.v4.new_markdown_cell(
        """### 2. Girdi bütünlüğü

Feature engineering başlamadan önce temiz tablonun satır, anahtar ve bölge sayısı bütünlüğü doğrulanır. Bu aşamada satır silinmez.
"""
    ),
    nbf.v4.new_code_cell(
        """input_summary = pd.DataFrame(
    {
        "metric": ["row_count", "column_count", "unique_hotel_id", "unique_place_id", "area_count"],
        "value": [
            len(df),
            df.shape[1],
            df["hotel_id"].nunique(),
            df["place_id"].nunique(),
            df["area"].nunique(),
        ],
    }
)
display(input_summary)

expected_area_count = df.groupby("area")["hotel_id"].transform("size")
assert df["area_hotel_count"].eq(expected_area_count).all(), 'area_hotel_count ana tabloyla eşleşmiyor.'"""
    ),
    nbf.v4.new_markdown_cell(
        """### 3. Temel özelliklerin üretilmesi

Özellikler dört grupta hazırlanır:

1. Bilgi varlık göstergeleri
2. Yorum hacmi ve şeffaf ağırlıklı puan
3. Bölge medyanına göre puan farkı
4. Bölge içi fiyat ve yorum konumu

Orijinal kolonlar değiştirilmez; yeni kolonlar aynı `df` tablosuna eklenir.
"""
    ),
    nbf.v4.new_code_cell(
        """result = build_basic_features(df)
df = result.df

display(pd.DataFrame({"metric": ["row_count", "total_columns", "new_feature_count"],
                      "value": [len(df), df.shape[1], len(result.feature_dictionary)]}))"""
    ),
    nbf.v4.new_markdown_cell(
        """### 4. Özellik sözlüğü ve hesaplama parametreleri

Her yeni kolonun tanımı, eksik değer davranışı ve yorumlama uyarısı açıkça kaydedilir. Ağırlıklı Google puanındaki iki parametre veriden hesaplanır ve ayrıca raporlanır.
"""
    ),
    nbf.v4.new_code_cell(
        """display(result.feature_dictionary)
display(result.parameters)"""
    ),
    nbf.v4.new_markdown_cell(
        """### 5. Yorum hacmi ve ağırlıklı puan

Yorum sayısı sağa çarpık olduğundan `log(1+x)` dönüşümü hazırlanır. Ağırlıklı puan formülü:

$$
WR = \\frac{v}{v+m}R + \\frac{m}{v+m}C
$$

Burada `R` otelin Google puanı, `v` yorum sayısı, `C` tüm otellerin ortalama Google puanı ve `m` medyan yorum sayısıdır.

Bu değer sıralama/EDA bağlamında kullanılabilir; doğrudan `google_rating` kullanılarak hesaplandığı için rating tahmin modelinde girdi yapılmamalıdır.
"""
    ),
    nbf.v4.new_code_cell(
        """rating_feature_preview = df[
    [
        "hotel_id",
        "hotel_name",
        "google_rating",
        "google_review_count",
        "review_count_log1p",
        "review_confidence_weight",
        "weighted_google_rating",
    ]
].head(10)
display(rating_feature_preview)"""
    ),
    nbf.v4.new_markdown_cell(
        """### 6. Bölgesel bağlam özellikleri

Puan ve fiyat karşılaştırmaları genel Bodrum ortalaması yerine otelin kendi bölgesindeki medyanı da dikkate alabilecek biçimde hazırlanır. Bunlar yalnızca göreli konum ölçüleridir; neden-sonuç veya fiyat/performans hükmü değildir.
"""
    ),
    nbf.v4.new_code_cell(
        """area_feature_preview = df[
    [
        "hotel_id",
        "area",
        "area_hotel_count",
        "google_rating",
        "area_median_google_rating",
        "rating_gap_from_area_median",
        "search_price_usd_snapshot",
        "area_median_price_snapshot",
        "price_gap_from_area_median",
        "price_ratio_to_area_median",
        "price_percentile_within_area",
        "review_count_percentile_within_area",
    ]
].head(10)
display(area_feature_preview)"""
    ),
    nbf.v4.new_markdown_cell(
        """### 7. Eksik değer davranışı

Eksik fiyatlar doldurulmaz. Bu satırlarda otele ait fiyat farkı, fiyat oranı ve fiyat yüzdelik sırası da eksik kalır. Bölge medyanı, yalnızca aynı bölgedeki mevcut fiyat gözlemlerinden hesaplanır ve imputasyon amacıyla kullanılmaz.
"""
    ),
    nbf.v4.new_code_cell(
        """missing_feature_report = pd.DataFrame(
    {
        "column": result.feature_dictionary["feature"],
        "missing_count": [int(df[column].isna().sum()) for column in result.feature_dictionary["feature"]],
        "missing_percentage": [float(df[column].isna().mean() * 100) for column in result.feature_dictionary["feature"]],
    }
).sort_values(["missing_count", "column"], ascending=[False, True], ignore_index=True)
display(missing_feature_report)"""
    ),
    nbf.v4.new_markdown_cell(
        """### 8. Doğrulama kontrolleri

Kaynak kolonların değişmediği, anahtarların ve satırların korunduğu, özellik aralıklarının geçerli olduğu ve eksik fiyatların doldurulmadığı otomatik olarak test edilir.
"""
    ),
    nbf.v4.new_code_cell(
        """display(result.validation_report)

failed_checks = result.validation_report.loc[result.validation_report["status"].eq("FAIL")]
assert failed_checks.empty, f'Başarısız feature kontrolleri:\\n{failed_checks.to_string(index=False)}'"""
    ),
    nbf.v4.new_markdown_cell(
        """### 9. Çıktıların kaydedilmesi

Zenginleştirilmiş tek tablo `data/processed/hotels_features.csv` olarak yazılır. Özellik sözlüğü, parametreler ve doğrulama raporu `reports/` altında saklanır.
"""
    ),
    nbf.v4.new_code_cell(
        """output_paths = save_feature_outputs(result, PROCESSED_DIR, REPORTS_DIR)
display(pd.DataFrame([
    {"output": name, "path": str(path.relative_to(PROJECT_ROOT))}
    for name, path in output_paths.items()
]))"""
    ),
    nbf.v4.new_markdown_cell(
        """### 10. Sonuç

- 192 otel satırı ve tüm kaynak kolonlar değişmeden korunmuştur.
- Aynı `df` içine 14 temel özellik eklenmiştir.
- Eksik fiyat ve resmî yıldız bilgileri doldurulmamıştır.
- Ayrı bölge özet tablosu kullanılmamıştır.
- Bileşik değer, lüks ve destinasyon skorları EDA sonrasına bırakılmıştır.
- Sonraki aşama `05_general_hotel_eda.ipynb` içinde hem kaynak değişkenlerin hem de bu temel özelliklerin dağılım ve ilişkilerini yorumlamaktır.
"""
    ),
]

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
