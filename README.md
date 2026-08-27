# Bodrum Hotel & Destination Intelligence

Bodrum'daki konaklama tesislerini otel ve destinasyon düzeyinde, gerçek veri ve açıklanabilir analiz ilkeleriyle inceleyen veri bilimi projesidir.

## Mevcut durum

- `01_data_collection.ipynb`: kaynak envanteri ve şema tanıtımı
- `02_data_audit.ipynb`: eksiklik, benzersizlik, geçerlilik ve kapsam denetimi
- `03_data_cleaning.ipynb`: kayıpsız tip/biçim standardizasyonu ve doğrulama
- `04_feature_engineering.ipynb`: EDA öncesi temel ve açıklanabilir özellikler
- `05_general_hotel_eda.ipynb`: genel otel ve destinasyon keşifsel veri analizi
- `06_hotel_attributes_match_audit.ipynb`: resmî tesis özellikleri eşleştirme ve kapsam denetimi
- `07_hotel_attributes_analysis.ipynb`: resmî yıldız, kapasite, tesis tipi ve destinasyon analizi
- `08_destination_intelligence_analysis.ipynb`: coverage-aware destinasyon profilleri, alt indeksler ve quadrant analizi
- `09_tourism_demand_analysis.ipynb`: Muğla uzun dönem trendi, 2025 sezonluğu ve Bodrum–Muğla yıllık profil karşılaştırması
- `10_airport_tourism_joint_analysis.ipynb`: Milas-Bodrum Airport ve Muğla aylık turizm serilerinin ortak sezonluk hareketi
- `11_project_intelligence_summary.ipynb`: ilk 10 notebookun doğrulanmış KPI, seçili grafik, sınırlılık ve bütünsel bulgularını birleştiren yönetici özeti
- `12_sikayetvar_all_hotels_audit_cleaning.ipynb`: all-hotels negative customer voice corpusunun entity, duplicate, missing, tarih, reply ve response audit'i ile izlenebilir clean dataset hazırlığı
- `13_sikayetvar_all_hotels_eda.ipynb`: clean Şikayetvar corpusunun coverage-aware hotel/area dağılımı, zaman, görünürlük, company response, reply davranışı ve NLP örneklem hazırlığı
- `14_sikayetvar_all_hotels_nlp_aspect_analysis.ipynb`: Türkçe preprocessing, distinctive terms, açıklanabilir multi-label aspect analizi, co-occurrence, company response ve segmentation-ready NLP features
- `21_sikayetvar_severity_analysis.ipynb`: complaint severity skorlama ve destekli örneklem üzerinden risk profili
- `22_sikayetvar_final_customer_voice_summary.ipynb`: local-evidence mapping closure (44 review/ambiguous vaka; 9'u kanıt yetersizliğinden bilinçli olarak açık bırakıldı) + hedefli tamamlayıcı complaint toplama (16 doğrulanmış ama hiç taranmamış sayfa) sonrası final clean-v3 corpus (353 complaint / 44 otel), aspect, reply-visibility ve cross-platform-hazırlık özeti
- `23_sikayetvar_google_travel_cross_source_customer_voice_alignment.ipynb`: Google Travel (genel review corpusu) ile Şikayetvar (complaint-odaklı corpus) arasında hotel_id/aspect-özet seviyesinde, açık bir 21-kategorili aspect crosswalk üzerinden hizalama; satır-seviyesi birleştirme yok, tek sentiment dağılımı gibi kıyaslanmaz

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

Final proje özeti ayrıca `reports/project_key_findings_master.csv`,
`reports/project_selected_figures.csv`, `reports/project_summary_consistency_checks.csv` ve
`reports/project_intelligence_summary.txt` çıktılarını üretir.

Şikayetvar complaint verisi normal müşteri yorumu veya kalite skoru değildir. Notebook 12 yalnız
complaint-level kesin eşleşmeleri `data/processed/sikayetvar_all_hotels_complaints_clean.csv`
dosyasına alır; review-required kayıtları ayrı tutar ve raw metinleri değiştirmez.
Notebook 13 complaint count'u kalite veya gerçek complaint rate olarak kullanmaz; cross-platform
oranları yalnız görünürlük bağlamı olarak adlandırır ve küçük örneklemleri açıkça işaretler.
Notebook 14 aspect mention rate'lerini gerçek müşteri problem oranı olarak yorumlamaz; otomatik
eşleşmeler için 40 satırlık ayrı manual-validation örneklemi ve sample-reliability göstergeleri üretir.
