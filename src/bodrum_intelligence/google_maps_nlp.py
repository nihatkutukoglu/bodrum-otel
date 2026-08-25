"""Aspect taxonomy, keyword detection, term stats and driver classification for review text.

Shared canonical aspect taxonomy (kept identical to the Şikayetvar branch so hotel-level
aggregates can be joined cross-platform on aspect name):
STAFF_SERVICE, CLEANLINESS_HYGIENE, FOOD_BEVERAGE, ROOM, BEACH_SEA, POOL,
FACILITIES_MAINTENANCE, RESERVATION, PAYMENT_REFUND, PRICE_VALUE, CHECKIN_CHECKOUT,
AIR_CONDITIONING, NOISE, FAMILY_CHILDREN, TRANSPORT_TRANSFER, MANAGEMENT_COMMUNICATION,
SPA_WELLNESS, SAFETY_SECURITY.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

TURKISH_STOPWORDS = {
    "ve", "bir", "bu", "da", "de", "çok", "için", "ile", "ama", "gibi", "her", "en", "daha",
    "olan", "olarak", "ise", "ya", "ki", "mi", "mı", "mu", "mü", "ben", "biz", "siz", "onlar",
    "o", "şu", "değil", "diye", "kadar", "sonra", "önce", "var", "yok", "oldu", "olduk",
    "olduğu", "hem", "ne", "nasıl", "tüm", "tum", "hep", "sadece", "yine", "yani", "ancak",
    "fakat", "veya", "ya da", "gayet", "bize", "bizi", "bunu", "bunun", "biraz", "tekrar",
    "birçok", "hiç", "artık", "zaten", "böyle", "şey", "şeyler", "kendi", "sizi", "size",
    # domain stopwords so hotel-generic filler doesn't dominate frequency tables (17.5)
    "otel", "otelin", "otelde", "oteli", "otele", "gün", "özellikle", "teşekkür", "ayrıca",
    "bence", "kesinlikle", "tekrar", "geliriz", "gideriz", "tavsiye", "ederim", "olduk", "aldık",
}


def brand_stopwords(hotel_names: list[str]) -> set[str]:
    """Tokenize hotel names into their own stopword set so brand tokens don't dominate term stats."""

    tokens: set[str] = set()
    for name in hotel_names:
        tokens.update(tokenize(normalize_for_nlp(name)))
    return tokens

CANONICAL_ASPECTS = [
    "STAFF_SERVICE", "CLEANLINESS_HYGIENE", "FOOD_BEVERAGE", "ROOM", "BEACH_SEA", "POOL",
    "FACILITIES_MAINTENANCE", "RESERVATION", "PAYMENT_REFUND", "PRICE_VALUE", "CHECKIN_CHECKOUT",
    "AIR_CONDITIONING", "NOISE", "FAMILY_CHILDREN", "TRANSPORT_TRANSFER",
    "MANAGEMENT_COMMUNICATION", "SPA_WELLNESS", "SAFETY_SECURITY",
]

# Google Travel corpus skews toward positive general-voice language, so the dictionary widens
# beyond the Şikayetvar complaint-oriented terms with praise vocabulary per 17.3.
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "STAFF_SERVICE": [
        "personel", "çalışan", "garson", "resepsiyon", "animasyon", "ilgili", "güler yüzlü",
        "yardımcı", "nazik", "kibar", "ilgisiz", "kaba", "hizmet", "servis", "ağırlama",
        "karşılama", "profesyonel",
    ],
    "CLEANLINESS_HYGIENE": [
        "temiz", "tertemiz", "hijyen", "hijyenik", "kirli", "pis", "lekeli", "koku", "çarşaf",
        "havlu", "temizlik",
    ],
    "FOOD_BEVERAGE": [
        "yemek", "kahvaltı", "restoran", "büfe", "açık büfe", "lezzetli", "çeşit", "çeşitlilik",
        "soğuk", "taze", "içecek", "bar", "menü", "mutfak", "a la carte",
    ],
    "ROOM": [
        "oda", "yatak", "banyo", "duş", "manzara", "geniş", "küçük oda", "eski", "konforlu",
        "balkon", "klima", "yatak odası",
    ],
    "BEACH_SEA": [
        "plaj", "deniz", "sahil", "iskele", "şezlong", "koy", "deniz suyu", "deniz manzarası",
    ],
    "POOL": ["havuz", "havuz kenarı", "çocuk havuzu", "kaydırak"],
    "FACILITIES_MAINTENANCE": [
        "bakım", "bakımsız", "yıpranmış", "eskimiş", "arızalı", "bozuk", "asansör", "tesis",
        "altyapı", "yenilenmiş",
    ],
    "RESERVATION": ["rezervasyon", "booking", "tarih değişikliği", "iptal"],
    "PAYMENT_REFUND": ["ödeme", "iade", "fatura", "ekstra ücret", "fiyat farkı", "kredi kartı"],
    "PRICE_VALUE": [
        "fiyat", "fiyatına göre", "değer", "pahalı", "ucuz", "uygun fiyat", "hesaplı",
        "fiyat performans", "bütçe",
    ],
    "CHECKIN_CHECKOUT": ["check-in", "check in", "checkin", "check-out", "check out", "checkout", "giriş", "çıkış"],
    "AIR_CONDITIONING": ["klima", "soğutma", "havalandırma"],
    "NOISE": ["gürültü", "ses", "sessiz", "sakin", "ses geçirgenliği", "gürültülü"],
    "FAMILY_CHILDREN": ["çocuk", "aile", "bebek", "mini club", "çocuk kulübü", "çocuklu"],
    "TRANSPORT_TRANSFER": ["transfer", "shuttle", "otopark", "ulaşım", "servis aracı"],
    "MANAGEMENT_COMMUNICATION": ["yönetim", "müdür", "iletişim", "şikayet", "ilgilenmedi", "geri dönüş"],
    "SPA_WELLNESS": ["spa", "masaj", "sauna", "hamam", "wellness"],
    "SAFETY_SECURITY": ["güvenlik", "güvenli", "tehlikeli", "kaza", "can güvenliği"],
}

NEGATION_MARKERS = ["değil", "yok", "olmadı", "gelmedi", "çalışmıyor", "kalmamış", "yetersiz"]

_ZERO_WIDTH_RE = re.compile(r"[​-‍﻿]")
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-zçğıöşü]+", re.IGNORECASE)


def normalize_for_nlp(text: Any) -> str:
    """Casefold + normalize Unicode while preserving Turkish characters and negation words."""

    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    normalized = normalized.replace("İ", "i").replace("I", "ı").casefold()
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def tokenize(normalized_text: str, extra_stopwords: set[str] | None = None) -> list[str]:
    """Turkish-aware word tokenizer with stopwords removed."""

    stopwords = TURKISH_STOPWORDS | (extra_stopwords or set())
    tokens = _TOKEN_RE.findall(normalized_text)
    return [tok for tok in tokens if tok not in stopwords and len(tok) > 2]


def ngrams(tokens: list[str], n: int) -> list[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def detect_aspects(normalized_text: str) -> tuple[list[str], dict[str, list[str]]]:
    """Multi-label keyword match against the canonical taxonomy. Returns (aspects, matched_keywords)."""

    matched: dict[str, list[str]] = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        hits = [kw for kw in keywords if kw in normalized_text]
        if hits:
            matched[aspect] = hits
    return sorted(matched.keys()), matched


def term_frequency_table(
    texts: pd.Series, n: int = 1, top_k: int = 25, extra_stopwords: set[str] | None = None
) -> pd.DataFrame:
    """Document-frequency ranked unigram/bigram/trigram table for one rating-group subset."""

    doc_freq: Counter = Counter()
    total_freq: Counter = Counter()
    for text in texts:
        tokens = tokenize(normalize_for_nlp(text), extra_stopwords)
        terms = set(ngrams(tokens, n))
        doc_freq.update(terms)
        total_freq.update(ngrams(tokens, n))
    rows = [
        {"term": term, "document_count": doc_freq[term], "total_count": total_freq[term]}
        for term in doc_freq
    ]
    table = pd.DataFrame(rows, columns=["term", "document_count", "total_count"])
    return table.sort_values(["document_count", "total_count"], ascending=False).head(top_k).reset_index(drop=True)


def distinctive_terms(
    group_texts: pd.Series,
    other_texts: pd.Series,
    n: int = 1,
    top_k: int = 20,
    min_doc_count: int = 3,
    extra_stopwords: set[str] | None = None,
) -> pd.DataFrame:
    """Rank terms by normalized document-share difference (group vs. the rest); simple and robust
    for small samples, per master-prompt 17.8/17.9 ("small sample → prefer a robust method")."""

    def doc_share(texts: pd.Series) -> Counter:
        n_docs = max(len(texts), 1)
        counter: Counter = Counter()
        for text in texts:
            terms = set(ngrams(tokenize(normalize_for_nlp(text), extra_stopwords), n))
            counter.update(terms)
        return Counter({term: count / n_docs for term, count in counter.items()}), n_docs

    group_share, group_n = doc_share(group_texts)
    other_share, other_n = doc_share(other_texts)
    group_raw_counts = Counter()
    for text in group_texts:
        group_raw_counts.update(set(ngrams(tokenize(normalize_for_nlp(text), extra_stopwords), n)))

    rows = []
    for term, g_share in group_share.items():
        if group_raw_counts[term] < min_doc_count:
            continue
        o_share = other_share.get(term, 0.0)
        rows.append(
            {
                "term": term,
                "group_document_count": group_raw_counts[term],
                "group_n": group_n,
                "group_share_pct": round(g_share * 100, 2),
                "other_share_pct": round(o_share * 100, 2),
                "share_diff_pp": round((g_share - o_share) * 100, 2),
            }
        )
    table = pd.DataFrame(
        rows, columns=["term", "group_document_count", "group_n", "group_share_pct", "other_share_pct", "share_diff_pp"]
    )
    return table.sort_values(
        ["share_diff_pp", "group_document_count", "term"],
        ascending=[False, False, True],
    ).head(top_k).reset_index(drop=True)


def classify_driver(
    low_rate_pct: float, high_rate_pct: float, low_n: int, high_n: int,
    gap_threshold_pp: float = 10.0, min_support: int = 5,
) -> tuple[str, bool]:
    """Data-driven driver classification per 17.15-17.21. Returns (driver_class, support_ok)."""

    support_ok = low_n >= min_support and high_n >= min_support
    gap = high_rate_pct - low_rate_pct
    if not support_ok:
        return "INSUFFICIENT_SAMPLE", support_ok
    if gap >= gap_threshold_pp:
        return "POSITIVE_DRIVER_CANDIDATE", support_ok
    if -gap >= gap_threshold_pp:
        return "NEGATIVE_DRIVER_CANDIDATE", support_ok
    if low_rate_pct >= 30 and high_rate_pct >= 30:
        return "EXPERIENCE_DEFINING", support_ok
    if low_rate_pct < 10 and high_rate_pct < 10:
        return "LOW_SIGNAL", support_ok
    return "NO_STRONG_SIGNAL", support_ok


def sample_tier(n: int, high: int = 30, medium: int = 10) -> str:
    """HIGH_SAMPLE / MEDIUM_SAMPLE / LOW_SAMPLE per master-prompt 16.18 starting thresholds."""

    if n >= high:
        return "HIGH_SAMPLE"
    if n >= medium:
        return "MEDIUM_SAMPLE"
    return "LOW_SAMPLE"
