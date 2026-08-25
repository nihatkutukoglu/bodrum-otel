# Bodrum Hotel & Destination Intelligence

Bodrum'daki konaklama tesislerini otel ve destinasyon düzeyinde, gerçek veri ve açıklanabilir analiz ilkeleriyle inceleyen veri bilimi projesidir.

## Mevcut durum

- `01_data_collection.ipynb`: kaynak envanteri ve şema tanıtımı
- `02_data_audit.ipynb`: eksiklik, benzersizlik, geçerlilik ve kapsam denetimi
- `03_data_cleaning.ipynb`: kayıpsız tip/biçim standardizasyonu ve doğrulama
- `04_feature_engineering.ipynb`: EDA öncesi temel ve açıklanabilir özellikler

Ana veri seti 192 benzersiz tesisten oluşur. `google_rating` müşteri puanıdır; `official_star_rating` değildir. `search_price_usd_snapshot` yalnızca toplama anındaki fiyat göstergesidir.

Notebooklarda tek ana çalışma tablosu `df` kullanılır. Eski bölge özet dosyası analiz girdisi değildir; `area_hotel_count` doğrudan ana tablodan `groupby(...).transform("size")` ile üretilir.

## Klasör yapısı

```text
data/raw/          Değiştirilmeyen kaynak kopyaları
data/processed/    Tekrar üretilebilen temiz/işlenmiş tablolar
notebooks/         Sıralı analiz notebookları
src/               Tekrar kullanılan veri işleme kodu
reports/           Denetim, dönüşüm ve doğrulama çıktıları
models/            İleride üretilecek model artefaktları
tests/             Veri temizleme kontrolleri
```

## Çalıştırma

```bash
python -m pip install -e .
python scripts/execute_notebook.py notebooks/03_data_cleaning.ipynb
PYTHONPATH=src python -m unittest discover -s tests -v
```

Notebooklar Türkçe açıklama, kod ise İngilizce değişken adları kullanır. Ham veri overwrite edilmez ve eksik alanlara tahmini değer yazılmaz.
