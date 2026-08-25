ŞİKAYETVAR CLEAN DATASETS

PURPOSE
Negative customer voice complaint corpusunu EDA/NLP öncesi güvenilir ve izlenebilir biçimde sunar.

GRAIN AND KEYS
- complaints_clean: one row = one complaint; primary dedupe key canonical_complaint_url.
- replies_clean: one row = one reply; reply_id stable hash key.
- replies_clean.canonical_complaint_url -> complaints_clean.canonical_complaint_url.

ENTITY FILTER
Ana corpus yalnız COMPLAINT_MATCHED complaint-level kayıtları içerir. COMPLAINT_REVIEW_REQUIRED
kayıtları sikayetvar_complaints_review_required.csv içinde ayrı tutulur.

RAW VS CLEAN
Raw title/text/response/progress/reply kolonları overwrite edilmez. *_clean alanları yalnız minimal
Unicode, HTML entity/tag, zero-width/control-character ve whitespace standardizasyonu içerir.

MISSING SEMANTICS
Missing hiçbir zaman otomatik zero değildir. Özellikle support_count missing -> support_count_numeric NaN.

DATE PARSING
complaint_date deterministik Türkçe tarih parser'ıyla üretilir. Yılı görünmeyen kayıtlar collection
timestamp'ine göre türetilir ve complaint_date_is_approximate=True taşır. Parse edilemeyen tarih NaT kalır.

LIMITATIONS
Şikayetvar self-selected negative customer voice platformudur. Complaint count kalite/rate değildir;
mapping coverage heterojendir; company response resolution anlamına gelmez; Google reviews ile aynı
sampling process değildir.

CURRENT RELEASE COUNTS
Clean complaints: 236
Review-required rows: 12
Clean replies: 97
