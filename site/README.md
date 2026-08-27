# Bodrum Hotel Intelligence — yönetici sunum sitesi

Bu klasör, `hotel-reviews` ve `bodrum-otel` repolarındaki analiz çıktılarını tek bir
interaktif yönetici sunumunda birleştiren "Bodrum Hotel Intelligence" sitesinin
kaynağını ve son üretilmiş halini içerir.

**Canlı sürüm:** https://claude.ai/code/artifact/ad0662c4-7556-4fd9-ba0d-32925cad2fc9

**Bu klasördeki dosyalar birebir bu linkte yayınlanan son sürümle aynıdır** (hero
bölümü, Google Travel / Trip.com / Şikayetvar / Hotel 360° / Otel Karşılaştırma
modülleri dahil).

## İçerik

- `bodrum_hotel_intelligence_v3.html` — üretilen, tek dosyalık, bağımsız site.
  Herhangi bir tarayıcıda doğrudan açılabilir (yalnız Google Fonts için internet
  bağlantısı gerekir, başka harici bağımlılığı yoktur).
- `build_site.py` — HTML/CSS/JS'i üreten ana script. `data/` altındaki JSON
  dosyalarını okuyup tek dosyalık siteyi derler.
- `build_final_data.py` — `hotel_360_intelligence.csv`,
  `sikayetvar_final_customer_voice_master_v3.csv` ve Trip.com yorum verisini
  birleştirip `data/FINAL_DATA.json`'ı üretir.
- `gather_site_data.py`, `gather_site_data_bodrumotel.py`, `gather_quotes.py` —
  Google Travel / Trip.com / Şikayetvar tarafındaki agregat istatistikleri ve
  gerçek örnek yorumları ilgili repo'lardaki `data/processed/` ve `reports/`
  dosyalarından çıkaran yardımcı scriptler.
- `data/*.json` — yukarıdaki scriptlerin ürettiği ara ve son veri dosyaları.

## Yeniden üretilebilirlik notu

`build_site.py`, tasarımı büyük ölçüde korumak için önceki bir Artifact
sürümünün HTML/CSS'ini (`ORIG`) kaynak olarak okuyor; bu dosya bir Claude Code
oturumunun önbelleğinde tutulduğu için bu repoya dahil edilmedi. Bu yüzden
scriptler bu haliyle sıfırdan klonlanan bir ortamda tekrar çalıştırılamaz —
ama **`bodrum_hotel_intelligence_v3.html` ve `data/` altındaki tüm JSON'lar
tam ve kendi başına kullanılabilir** durumdadır; scriptler yöntemin şeffaf bir
kaydı olarak burada tutuluyor.

## Veri kaynakları

- Google Travel: `hotel-reviews/data/processed/google_travel_all_hotels_reviews_clean.csv` (2.336 yorum, 104 otel)
- Trip.com: `hotel-reviews/data/processed/tripcom_reviews_clean.csv` (3.013 yorum, 94 otel)
- Şikayetvar: `bodrum-otel/reports/sikayetvar_final_customer_voice_master_v3.csv` (353 kayıt, 44 otel)
- Hotel 360°: `hotel-reviews/data/processed/hotel_360_intelligence.csv` (192 otel, dört kaynağı birleştiren tekil tablo)
