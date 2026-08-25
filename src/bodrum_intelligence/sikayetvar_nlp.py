"""Explainable Turkish preprocessing and multi-label aspect helpers for Şikayetvar."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


CANONICAL_ASPECTS = [
    "STAFF_SERVICE", "CLEANLINESS_HYGIENE", "FOOD_BEVERAGE", "ROOM", "BEACH_SEA", "POOL",
    "FACILITIES_MAINTENANCE", "RESERVATION", "PAYMENT_REFUND", "PRICE_VALUE",
    "CHECKIN_CHECKOUT", "AIR_CONDITIONING", "NOISE", "FAMILY_CHILDREN",
    "TRANSPORT_TRANSFER", "MANAGEMENT_COMMUNICATION", "SPA_WELLNESS", "SAFETY_SECURITY",
]

# Negation-bearing words are intentionally absent.
TURKISH_STOPWORDS = {
    "acaba", "ama", "ancak", "artık", "aslında", "ayrıca", "bana", "bazen", "bazı",
    "belki", "ben", "bence", "beni", "benim", "beri", "bile", "bir", "biraz", "birçok",
    "biri", "birkaç", "biz", "bize", "bizi", "bizim", "bu", "buna", "bunda", "bundan",
    "bunu", "bunun", "böyle", "çok", "çünkü", "da", "daha", "de", "diye", "dolayı",
    "en", "fakat", "gibi", "göre", "hala", "hangi", "hani", "hatta", "hem", "hep", "her",
    "hiç", "için", "ile", "ise", "işte", "kadar", "kendi", "ki", "kim", "mı", "mi", "mu",
    "mü", "nasıl", "ne", "neden", "nerede", "o", "olan", "olarak", "oldukça", "olduğu",
    "onlar", "orada", "oysa", "sadece", "siz", "size", "sizi", "sonra", "şey", "şu", "tüm",
    "ve", "veya", "ya", "yani", "yine", "zaten",
}

GENERIC_DOMAIN_STOPWORDS = {
    "otel", "oteli", "otelin", "otelde", "otele", "hotel", "resort", "spa", "bodrum",
    "tatil", "konaklama", "konakladık", "konakladım", "şikayet", "şikayetim", "yaşadık",
    "yaşadım", "tarih", "tarihinde", "gün", "gece", "yıldızlı", "şekilde", "kesinlikle",
    "talep", "ediyorum", "rağmen", "ciddi", "son", "olduğunu", "tl", "arasında", "tarihleri",
    "nedeniyle", "zorunda", "herhangi", "özellikle", "ilgili", "yaşadığımız", "boyunca", "ilk",
    "üzerinden", "toplam", "süre", "süresince", "mevcut", "durum", "konuda", "tarafından",
    "tatilbudur", "com", "numaralı", "bin", "olarak",
}

ASPECT_KEYWORDS: dict[str, list[str]] = {
    "STAFF_SERVICE": [
        "personel", "çalışan", "görevli", "garson", "resepsiyonist", "hizmet", "servis",
        "ilgisiz", "kaba", "saygısız", "tecrübesiz", "profesyonellik", "yardımcı olmadı",
        "ilgi yok", "muhatap yok", "animasyon ekibi",
    ],
    "CLEANLINESS_HYGIENE": [
        "temizlik", "temiz", "kirli", "pis", "hijyen", "çarşaf", "havlu", "leke", "koku",
        "küf", "böcek", "haşere", "tahtakurusu", "örümcek ağı", "saç kılı", "kan lekesi",
        "oda temizliği", "kötü koku",
    ],
    "FOOD_BEVERAGE": [
        "yemek", "kahvaltı", "restoran", "restaurant", "büfe", "açık büfe", "içecek", "bar",
        "lezzet", "yiyecek", "mutfak", "snack", "alkol", "tabak", "çatal", "zehirlenme",
        "a la carte", "her şey dahil",
    ],
    "ROOM": [
        "oda", "yatak", "banyo", "duş", "tuvalet", "lavabo", "minibar", "balkon", "mobilya",
        "nevresim", "anahtar", "oda değişikliği", "oda manzarası", "oda teslimi",
    ],
    "BEACH_SEA": [
        "plaj", "sahil", "deniz", "iskele", "şezlong", "kumsal", "koy", "taşlı deniz",
        "kum plaj", "denize giriş",
    ],
    "POOL": ["havuz", "kaydırak", "aquapark", "su parkı", "çocuk havuzu", "havuz kenarı"],
    "FACILITIES_MAINTENANCE": [
        "bakım", "bakımsız", "bozuk", "kırık", "eski", "asansör", "kapı", "elektrik", "tamir",
        "arızalı", "damlama", "su kesintisi", "sıcak su", "spor salonu", "ortak alan",
    ],
    "RESERVATION": [
        "rezervasyon", "booking", "otelz", "tatilbudur", "acente", "tur şirketi", "iptal",
        "rezervasyon iptali", "yer ayırma", "tarih değişikliği", "yanıltıcı bilgilendirme",
    ],
    "PAYMENT_REFUND": [
        "iade", "geri ödeme", "ücret iadesi", "para iadesi", "ödeme", "tahsilat", "provizyon",
        "kredi kartı", "kesinti", "fatura", "para", "tutar",
    ],
    "PRICE_VALUE": [
        "fiyat", "pahalı", "ucuz", "değer", "ücret", "fiyat performans", "ödediğimiz",
        "karşılığını", "hak etmeyen", "yüksek fiyat", "ekstra ücret",
    ],
    "CHECKIN_CHECKOUT": [
        "check in", "check-in", "checkin", "check out", "check-out", "checkout", "giriş saati",
        "çıkış saati", "oda teslim", "erken giriş", "geç giriş",
    ],
    "AIR_CONDITIONING": [
        "klima", "havalandırma", "soğutmuyor", "ısıtma", "klimasız", "klima çalışmıyor",
    ],
    "NOISE": [
        "gürültü", "yüksek ses", "müzik sesi", "ses", "uyuyamadım", "uyuyamadık", "rahatsız edici müzik",
    ],
    "FAMILY_CHILDREN": [
        "çocuk", "bebek", "aile", "çocuk kulübü", "kids club", "mini club", "çocuğum", "oğlum", "kızım",
    ],
    "TRANSPORT_TRANSFER": [
        "transfer", "servis aracı", "shuttle", "taksi", "ulaşım", "havaalanı", "otopark", "araç servisi",
    ],
    "MANAGEMENT_COMMUNICATION": [
        "iletişim", "müdür", "yönetim", "yönetici", "yetkili", "telefon", "ulaşamadım", "geri dönüş",
        "çözüm sunulmadı", "muhatap", "müşteri hizmetleri", "cevap vermedi", "bilgilendirme",
    ],
    "SPA_WELLNESS": ["spa", "hamam", "sauna", "masaj", "wellness", "buhar odası"],
    "SAFETY_SECURITY": [
        "güvenlik", "güvenli", "hırsızlık", "kasa", "kayboldu", "çalındı", "tehlike", "kaza",
        "sağlık", "hastane", "doktor", "yaralandı", "zehirlenme", "can güvenliği", "ayrımcı",
    ],
}

CORPUS_DISCOVERED_KEYWORDS = {
    "muhatap", "küf", "tahtakurusu", "örümcek ağı", "kan lekesi", "damlama", "spor salonu",
    "yanıltıcı bilgilendirme", "hak etmeyen", "çözüm sunulmadı", "müşteri hizmetleri", "ayrımcı",
}

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?90\s*)?(?:0?\d[\s().*-]*){10,}(?!\w)")
_TOKEN_RE = re.compile(r"[a-zçğıöşü]+", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


def normalize_for_nlp(text: Any, mask_pii: bool = True) -> str:
    """Normalize Unicode and whitespace while preserving Turkish characters and negation."""

    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = _ZERO_WIDTH_RE.sub("", value)
    if mask_pii:
        value = _URL_RE.sub(" URLMASK ", value)
        value = _EMAIL_RE.sub(" EMAILMASK ", value)
        value = _PHONE_RE.sub(" PHONEMASK ", value)
    value = value.replace("İ", "i").replace("I", "ı").casefold()
    value = re.sub(r"[^a-zçğıöşü\s-]", " ", value)
    value = value.replace("-", " ")
    return _SPACE_RE.sub(" ", value).strip()


def hotel_name_stopwords(hotel_names: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for name in hotel_names:
        tokens.update(_TOKEN_RE.findall(normalize_for_nlp(name, mask_pii=False)))
    return {token for token in tokens if len(token) >= 3}


def tokenize(normalized_text: str, extra_stopwords: set[str] | None = None) -> list[str]:
    stopwords = TURKISH_STOPWORDS | (extra_stopwords or set())
    return [
        token for token in _TOKEN_RE.findall(normalized_text)
        if len(token) >= 2 and token not in stopwords and not token.endswith("mask")
    ]


def make_ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)] if len(tokens) >= n else []


def _contains_keyword(text: str, keyword: str) -> bool:
    normalized_keyword = normalize_for_nlp(keyword, mask_pii=False)
    escaped = re.escape(normalized_keyword)
    if " " in normalized_keyword:
        return bool(re.search(rf"(?<![a-zçğıöşü]){escaped}(?![a-zçğıöşü])", text))
    if normalized_keyword == "oda":
        suffix = r"(?:da|dan|ya|yı|nın|sı|mız|lar|ları|m|n)?"
    elif normalized_keyword == "ses":
        suffix = r"(?:i|in|ler|leri|siz|inden)?"
    elif len(normalized_keyword) >= 4 and normalized_keyword not in {"checkin", "checkout"}:
        # Turkish surface forms are retained; a conservative root-prefix rule captures common
        # inflections (yemekler, kirliydi, rezervasyonu) without external morphology dependencies.
        suffix = r"[a-zçğıöşü]{0,10}"
    else:
        suffix = ""
    return bool(re.search(rf"(?<![a-zçğıöşü]){escaped}{suffix}(?![a-zçğıöşü])", text))


def detect_aspects(normalized_text: str) -> tuple[list[str], dict[str, list[str]]]:
    """Phrase-first, word-boundary multi-label detector with a service context rule."""

    text = normalize_for_nlp(normalized_text)
    matched: dict[str, list[str]] = {}
    transport_service = any(_contains_keyword(text, phrase) for phrase in ("servis aracı", "araç servisi"))
    for aspect, keywords in ASPECT_KEYWORDS.items():
        ordered = sorted(keywords, key=lambda item: (-len(item.split()), -len(item)))
        hits: list[str] = []
        for keyword in ordered:
            if aspect == "STAFF_SERVICE" and keyword == "servis" and transport_service:
                continue
            if _contains_keyword(text, keyword):
                hits.append(keyword)
        if hits:
            matched[aspect] = sorted(set(hits), key=hits.index)
    return sorted(matched), matched


def term_frequency_table(
    texts: pd.Series,
    n: int = 1,
    min_document_count: int = 1,
    extra_stopwords: set[str] | None = None,
) -> pd.DataFrame:
    token_counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    document_n = int(texts.fillna("").ne("").sum())
    for text in texts.fillna(""):
        terms = make_ngrams(tokenize(normalize_for_nlp(text), extra_stopwords), n)
        token_counts.update(terms)
        doc_counts.update(set(terms))
    rows = [
        {
            "term": term,
            "token_count": token_counts[term],
            "document_count": count,
            "document_share_pct": 100 * count / document_n if document_n else np.nan,
        }
        for term, count in doc_counts.items() if count >= min_document_count
    ]
    columns = ["term", "token_count", "document_count", "document_share_pct"]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["document_count", "token_count", "term"], ascending=[False, False, True]
    ).reset_index(drop=True)


def aspect_dictionary_table(texts: pd.Series) -> pd.DataFrame:
    normalized = texts.fillna("").map(normalize_for_nlp)
    rows = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for keyword in keywords:
            count = int(normalized.map(lambda text: _contains_keyword(text, keyword)).sum())
            source = "CORPUS_DISCOVERED" if keyword in CORPUS_DISCOVERED_KEYWORDS else "SEED"
            rows.append({
                "aspect": aspect,
                "keyword": keyword,
                "keyword_type": "PHRASE" if " " in keyword else "UNIGRAM",
                "source": source,
                "document_count": count,
                "included": True,
                "notes": "Phrase-first and word-boundary matching; manual validation pending.",
            })
    return pd.DataFrame(rows)


def add_aspect_columns(frame: pd.DataFrame, text_column: str = "nlp_text_normalized") -> pd.DataFrame:
    result = frame.copy()
    detections = result[text_column].fillna("").map(detect_aspects)
    result["matched_aspects"] = detections.map(lambda item: "|".join(item[0]))
    result["matched_aspect_keywords"] = detections.map(
        lambda item: json.dumps(item[1], ensure_ascii=False, sort_keys=True)
    )
    result["aspect_count"] = detections.map(lambda item: len(item[0])).astype(int)
    result["matched_keyword_count"] = detections.map(
        lambda item: sum(len(values) for values in item[1].values())
    ).astype(int)
    result["no_aspect_detected_flag"] = result["aspect_count"].eq(0)
    for aspect in CANONICAL_ASPECTS:
        result[f"aspect_{aspect.lower()}"] = detections.map(lambda item, a=aspect: a in item[0]).astype(bool)
    return result


def aspects_long_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_columns = [
        "complaint_id", "hotel_id", "hotel_name", "area", "company_response_exists_clean",
        "complaint_date", "google_rating", "google_review_count",
    ]
    for row in frame.itertuples(index=False):
        base = {column: getattr(row, column) for column in base_columns if hasattr(row, column)}
        matched_map = json.loads(getattr(row, "matched_aspect_keywords"))
        for aspect in CANONICAL_ASPECTS:
            rows.append({
                **base,
                "aspect": aspect,
                "matched": aspect in matched_map,
                "matched_keywords": "|".join(matched_map.get(aspect, [])),
            })
    return pd.DataFrame(rows)


def aspect_frequency_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(frame)
    for aspect in CANONICAL_ASPECTS:
        column = f"aspect_{aspect.lower()}"
        subset = frame.loc[frame[column]]
        response_n = int(subset["company_response_exists_clean"].fillna(False).sum())
        rows.append({
            "aspect": aspect,
            "complaint_count": len(subset),
            "mention_rate_pct": 100 * len(subset) / total if total else np.nan,
            "unique_hotel_count": subset["hotel_id"].nunique(),
            "unique_area_count": subset["area"].nunique(),
            "company_response_count": response_n,
            "company_response_rate_within_aspect": 100 * response_n / len(subset) if len(subset) else np.nan,
            "median_view_count": subset["view_count_numeric"].median(),
            "median_complaint_word_count": subset["complaint_word_count"].median(),
        })
    return pd.DataFrame(rows).sort_values("complaint_count", ascending=False).reset_index(drop=True)


def group_aspect_matrix(
    frame: pd.DataFrame,
    group_columns: list[str],
    small_n_threshold: int = 5,
) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, keys))
        for aspect in CANONICAL_ASPECTS:
            count = int(group[f"aspect_{aspect.lower()}"].sum())
            rows.append({
                **base,
                "group_n": len(group),
                "aspect": aspect,
                "aspect_count": count,
                "aspect_mention_rate_pct": 100 * count / len(group),
                "small_n_flag": len(group) < small_n_threshold,
            })
    return pd.DataFrame(rows)


def aspect_cooccurrence_table(frame: pd.DataFrame, minimum_support: int = 3) -> pd.DataFrame:
    counts = {aspect: int(frame[f"aspect_{aspect.lower()}"].sum()) for aspect in CANONICAL_ASPECTS}
    rows = []
    for aspect_a, aspect_b in combinations(CANONICAL_ASPECTS, 2):
        both = int((frame[f"aspect_{aspect_a.lower()}"] & frame[f"aspect_{aspect_b.lower()}"]).sum())
        if both < minimum_support:
            continue
        union = counts[aspect_a] + counts[aspect_b] - both
        rows.append({
            "aspect_a": aspect_a,
            "aspect_b": aspect_b,
            "cooccurrence_count": both,
            "cooccurrence_rate_pct": 100 * both / len(frame),
            "jaccard_similarity": both / union if union else np.nan,
        })
    return pd.DataFrame(rows).sort_values(
        ["cooccurrence_count", "jaccard_similarity"], ascending=False
    ).reset_index(drop=True)


def distinctive_terms_by_group(
    frame: pd.DataFrame,
    group_column: str,
    text_column: str,
    minimum_group_n: int = 5,
    top_k: int = 10,
    min_df: int = 3,
) -> pd.DataFrame:
    """Return mean document-level TF-IDF terms for eligible groups."""

    work = frame.loc[frame[text_column].fillna("").ne("")].copy()
    if work.empty:
        return pd.DataFrame()
    vectorizer = TfidfVectorizer(
        tokenizer=str.split, preprocessor=None, token_pattern=None, lowercase=False,
        ngram_range=(1, 2), min_df=min_df, max_df=0.90, sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(work[text_column])
    terms = np.asarray(vectorizer.get_feature_names_out())
    records = []
    for group_name, indices in work.groupby(group_column).groups.items():
        positions = work.index.get_indexer(indices)
        if len(positions) < minimum_group_n:
            continue
        mean_scores = np.asarray(matrix[positions].mean(axis=0)).ravel()
        ranked = np.argsort(mean_scores)[::-1]
        rank = 0
        for term_index in ranked:
            if mean_scores[term_index] <= 0:
                continue
            term = terms[term_index]
            doc_count = int((matrix[positions, term_index] > 0).sum())
            rank += 1
            records.append({
                group_column: group_name,
                "group_n": len(positions),
                "term": term,
                "tfidf_score": mean_scores[term_index],
                "document_count": doc_count,
                "rank": rank,
            })
            if rank >= top_k:
                break
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Severity / escalation tiering
#
# Şikayetvar has no star rating, so every row is already a complaint — a plain
# positive/negative sentiment score would not discriminate. What is useful is how
# much the complainant *escalates*: legal threats, health/safety harm, fraud,
# harassment and absolute-strength language read very differently from a routine
# "room was small" complaint even though both are negative. This tiers that
# escalation intensity with the same explainable, keyword-based approach as the
# aspect detector above (see detect_aspects) rather than a black-box classifier.
# ---------------------------------------------------------------------------

SEVERITY_KEYWORDS: dict[str, list[str]] = {
    "HIGH": [
        "yasal", "avukat", "mahkeme", "dava", "tüketici hakem heyeti", "tüketici mahkemesi",
        "basına", "sosyal medyada", "ifşa", "teşhir",
        "rezillik", "rezalet", "skandal", "iğrenç", "berbat", "felaket", "dehşet",
        "zehirlenme", "hastalandık", "hastalandım", "ambulans", "can güvenliği", "tehlike",
        "dolandırıcılık", "hırsızlık", "çalındı", "gasp",
        "hakaret", "küfür", "tehdit", "taciz", "saldırı",
        "tazminat",
    ],
    "MEDIUM": [
        "hayal kırıklığı", "beklentimizin altında", "beklentimin altında", "memnun kalmadık",
        "memnun kalmadım", "asla", "bir daha gelmeyiz", "bir daha gelmem", "tavsiye etmiyorum",
        "tavsiye etmem", "üzücü", "yazık", "pişman", "mağdur", "kabul edilemez", "yetersiz",
    ],
}


def detect_severity(normalized_text: str) -> tuple[str, list[str], list[str]]:
    """Rule-based escalation tiering: HIGH (legal/health/fraud/threat language), MEDIUM
    (explicit strong dissatisfaction), BASELINE (aspect complaint, no escalation language).
    Returns (tier, high_keyword_hits, medium_keyword_hits). HIGH takes priority over MEDIUM
    when both are present."""

    text = normalize_for_nlp(normalized_text)
    high_hits = [kw for kw in SEVERITY_KEYWORDS["HIGH"] if _contains_keyword(text, kw)]
    medium_hits = [kw for kw in SEVERITY_KEYWORDS["MEDIUM"] if _contains_keyword(text, kw)]
    tier = "HIGH" if high_hits else ("MEDIUM" if medium_hits else "BASELINE")
    return tier, high_hits, medium_hits


def severity_dictionary_table(texts: pd.Series) -> pd.DataFrame:
    """Per-keyword document counts, for manual audit — same pattern as aspect_dictionary_table."""

    normalized = texts.fillna("").map(normalize_for_nlp)
    rows = []
    for tier, keywords in SEVERITY_KEYWORDS.items():
        for keyword in keywords:
            count = int(normalized.map(lambda text, k=keyword: _contains_keyword(text, k)).sum())
            rows.append({
                "severity_tier": tier, "keyword": keyword,
                "keyword_type": "PHRASE" if " " in keyword else "UNIGRAM",
                "document_count": count,
            })
    return pd.DataFrame(rows).sort_values(["severity_tier", "document_count"], ascending=[True, False]).reset_index(drop=True)


def add_severity_columns(frame: pd.DataFrame, text_column: str = "nlp_text_normalized") -> pd.DataFrame:
    result = frame.copy()
    detections = result[text_column].fillna("").map(detect_severity)
    result["severity_tier"] = detections.map(lambda item: item[0])
    result["severity_high_keywords"] = detections.map(lambda item: "|".join(item[1]))
    result["severity_medium_keywords"] = detections.map(lambda item: "|".join(item[2]))
    result["severity_keyword_count"] = detections.map(lambda item: len(item[1]) + len(item[2])).astype(int)
    return result


def severity_distribution_table(frame: pd.DataFrame) -> pd.DataFrame:
    total = len(frame)
    counts = frame["severity_tier"].value_counts()
    rows = []
    for tier in ("HIGH", "MEDIUM", "BASELINE"):
        n = int(counts.get(tier, 0))
        response_n = int(frame.loc[frame["severity_tier"] == tier, "company_response_exists_clean"].fillna(False).sum())
        rows.append({
            "severity_tier": tier, "complaint_count": n,
            "share_pct": 100 * n / total if total else np.nan,
            "company_response_count": response_n,
            "company_response_rate_within_tier": 100 * response_n / n if n else np.nan,
        })
    return pd.DataFrame(rows)


def severity_by_aspect_table(frame: pd.DataFrame) -> pd.DataFrame:
    """For each canonical aspect, what share of its complaints fall in each severity tier."""

    rows = []
    for aspect in CANONICAL_ASPECTS:
        column = f"aspect_{aspect.lower()}"
        subset = frame.loc[frame[column]]
        n = len(subset)
        if n == 0:
            continue
        tier_counts = subset["severity_tier"].value_counts()
        rows.append({
            "aspect": aspect, "complaint_count": n,
            "high_share_pct": 100 * int(tier_counts.get("HIGH", 0)) / n,
            "medium_share_pct": 100 * int(tier_counts.get("MEDIUM", 0)) / n,
            "baseline_share_pct": 100 * int(tier_counts.get("BASELINE", 0)) / n,
        })
    return pd.DataFrame(rows).sort_values("high_share_pct", ascending=False).reset_index(drop=True)
    return pd.DataFrame(records)
