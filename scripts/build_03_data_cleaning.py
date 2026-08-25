"""03_data_cleaning.ipynb dosyasını tekrar üretilebilir biçimde oluşturur."""

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_data_cleaning.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# Bodrum Hotel & Destination Intelligence
## 03 - Data Cleaning

Bu notebook, `02_data_audit.ipynb` bulgularına dayanarak ana otel tablosunu kayıpsız ve açıklanabilir biçimde temizler.

Temel ilkeler:

- Ham dosya değiştirilmez.
- Eksik değerler tahmin edilmez veya doldurulmaz.
- Google müşteri puanı ile resmî yıldız sınıfı birbirinden ayrı tutulur.
- Fiyat alanı yalnızca tarihli bir arama snapshot'ı olarak korunur.
- Her dönüşüm ve doğrulama kontrolü ayrı rapora yazılır.
"""
    ),
    nbf.v4.new_markdown_cell(
        """### 1. Kurulum ve proje yolları

Notebook ister proje kökünden ister `notebooks/` klasöründen çalıştırılsın aynı dosyaları bulur. Temizleme mantığı tekrar kullanım için `src/bodrum_intelligence/cleaning.py` modülünde tutulur.
"""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import sys

import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bodrum_intelligence.cleaning import clean_hotels, load_raw_hotels, save_cleaning_outputs

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "bodrum_hotels_master_2026-08-24.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

assert RAW_PATH.exists(), f'Ham veri bulunamadı: {RAW_PATH}'"""
    ),
    nbf.v4.new_markdown_cell(
        """### 2. Ham verinin güvenli yüklenmesi

Telefon ve kimlik alanları baştaki `+` işareti veya olası sıfırlar kaybolmasın diye metin olarak okunur. Bu aşamada veri üzerinde değişiklik yapılmaz.
"""
    ),
    nbf.v4.new_code_cell(
        """raw_df = load_raw_hotels(RAW_PATH)

raw_summary = pd.DataFrame(
    {
        "metric": ["row_count", "column_count", "missing_cells", "duplicate_rows"],
        "value": [
            len(raw_df),
            raw_df.shape[1],
            int(raw_df.isna().sum().sum()),
            int(raw_df.duplicated().sum()),
        ],
    }
)
display(raw_summary)"""
    ),
    nbf.v4.new_markdown_cell(
        """### 3. Temizleme dönüşümleri

Uygulanan işlemler yalnızca baş/son boşluklarını temizleme, boş metinleri standart eksik değere çevirme, güvenli nullable veri tipleri oluşturma ve tarihi ISO biçimine getirmedir. Satır silme, aykırı değer baskılama ve imputasyon yapılmaz.
"""
    ),
    nbf.v4.new_code_cell(
        """result = clean_hotels(raw_df)
df = result.hotels

display(result.transformation_log)
display(df.dtypes.rename("dtype").astype(str).to_frame())"""
    ),
    nbf.v4.new_markdown_cell(
        """### 4. Eksik değerlerin korunması

Özellikle `official_star_rating`, `business_status` ve fiyat snapshot'ındaki eksiklikler bilgi yokluğunu temsil eder. Bu değerler başka kolonlardan türetilmez.
"""
    ),
    nbf.v4.new_code_cell(
        """important_missing = [
    "official_star_rating",
    "business_status",
    "search_price_usd_snapshot",
    "phone",
]

missing_comparison = pd.DataFrame(
    {
        "column": important_missing,
        "raw_missing": [int(raw_df[column].isna().sum()) for column in important_missing],
        "clean_missing": [int(df[column].isna().sum()) for column in important_missing],
    }
)
missing_comparison["difference"] = missing_comparison["clean_missing"] - missing_comparison["raw_missing"]
display(missing_comparison)"""
    ),
    nbf.v4.new_markdown_cell(
        """### 5. Doğrulama kontrolleri

Temiz veri; satır sayısı, anahtar bütünlüğü ve alanların geçerli aralıkları açısından kontrol edilir. Herhangi bir `FAIL`, sonraki analize geçmeden önce incelenmelidir.
"""
    ),
    nbf.v4.new_code_cell(
        """display(result.validation_report)

failed_checks = result.validation_report.loc[result.validation_report["status"].eq("FAIL")]
assert failed_checks.empty, f'Başarısız temizleme kontrolleri:\\n{failed_checks.to_string(index=False)}'"""
    ),
    nbf.v4.new_markdown_cell(
        """### 6. İşlenmiş verinin ve raporların kaydedilmesi

Temiz tablo `data/processed/` altına; dönüşüm günlüğü ve doğrulama raporu `reports/` altına yazılır. `data/raw/` içeriğine yazılmaz.
"""
    ),
    nbf.v4.new_code_cell(
        """output_paths = save_cleaning_outputs(result, PROCESSED_DIR, REPORTS_DIR)
pd.DataFrame(
    [{"output": name, "path": str(path.relative_to(PROJECT_ROOT))} for name, path in output_paths.items()]
)"""
    ),
    nbf.v4.new_markdown_cell(
        """### 7. Sonuç

- **192 kayıt korunmuştur;** satır eklenmemiş veya silinmemiştir.
- Tek çalışma tablosu `df` kullanılmış ve bölge otel sayısı `area_hotel_count` olarak bu tablodan türetilmiştir.
- Otel ve Google Place kimlikleri benzersiz kalmıştır.
- Telefon alanı metin, sayısal alanlar nullable sayısal tip olarak standardize edilmiştir.
- Resmî yıldız bilgisi uydurulmamış; mevcut eksiklik aynen korunmuştur.
- Fiyat snapshot'ı eksikleri doldurulmamış ve aykırı görünen değerler silinmemiştir.
- Temiz veri ve dönüşüm kanıtları ham kaynaktan ayrı kaydedilmiştir.

Bir sonraki aşamada `04_feature_engineering.ipynb`, yalnızca araştırma sorularına hizmet eden ve veri kapsamıyla desteklenen özellikleri üretmelidir.
"""
    ),
]

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
