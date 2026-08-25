"""Proje otel master verisi ile resmî tesis kayıtlarını eşleştirmek için yardımcı fonksiyonlar.

Eşleştirme katmanlı ilerler (telefon -> kesin isim -> fuzzy isim + destek sinyalleri) ve
sadece güçlü/açıklanabilir sinyallere sahip eşleşmeler otomatik olarak yüksek güvenli kabul
edilir. Hiçbir fonksiyon ham `hotel_name`/`official_name`/`phone`/`address` kolonlarını
değiştirmez; yalnızca eşleştirme amaçlı geçici normalize kolonlar üretir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd


_TURKISH_TRANSLATION = str.maketrans(
    {
        "ı": "i", "İ": "i", "I": "i",
        "ğ": "g", "Ğ": "g",
        "ü": "u", "Ü": "u",
        "ş": "s", "Ş": "s",
        "ö": "o", "Ö": "o",
        "ç": "c", "Ç": "c",
    }
)

# Yalnızca coğrafi/dolgu kelimeler; "hotel", "otel", "resort", "spa" gibi ayırt edici olabilecek
# kelimeler kasıtlı olarak burada değildir (bkz. modül docstring'i ve notebook Bölüm 6).
CORE_DROP_WORDS = {"bodrum", "turkey", "turkiye"}

_AREA_SUFFIX_WORDS = {"mahallesi", "mahalle", "mah", "mh"}


def normalize_text(value) -> str | None:
    """Küçük harfe çevirir, Türkçe karakterleri kontrollü ASCII karşılığına indirger,
    noktalamayı kaldırır ve fazla boşlukları tekilleştirir."""

    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return None
    text = str(value).translate(_TURKISH_TRANSLATION).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_hotel_name(name) -> tuple[str | None, str | None]:
    """`(normalized_name_full, normalized_name_core)` çiftini döndürür.

    `full`, tüm kelimeleri korur. `core`, yalnızca coğrafi/dolgu kelimeleri (`CORE_DROP_WORDS`)
    kaldırır; "hotel/otel/resort/spa" gibi ayırt edici olabilecek kelimeler kasıtlı olarak
    dokunulmadan bırakılır.
    """

    full = normalize_text(name)
    if full is None:
        return None, None
    core_tokens = [token for token in full.split() if token not in CORE_DROP_WORDS]
    core = " ".join(core_tokens) if core_tokens else full
    return full, core


def normalize_phone(phone) -> str | None:
    """Rakam dışındaki her şeyi atar, son 10 haneyi döndürür (ülke kodu/başındaki 0 fark etmez).

    7 haneden az rakam kalırsa (anlamlı bir numara olamayacak kadar kısa) `None` döner.
    """

    if phone is None or pd.isna(phone):
        return None
    digits = re.sub(r"\D", "", str(phone))
    if len(digits) < 7:
        return None
    return digits[-10:]


def normalize_address(address) -> str | None:
    """Adres metnini `normalize_text` ile normalize eder (ek bir işlem yapmaz)."""

    return normalize_text(address)


def calculate_name_similarity(name_a: str | None, name_b: str | None) -> float:
    """İki normalize edilmiş isim arasında `difflib.SequenceMatcher` oranı (0-1) döndürür."""

    if not name_a or not name_b:
        return 0.0
    return SequenceMatcher(None, name_a, name_b).ratio()


def calculate_address_similarity(project_address_norm: str | None, official_area_norm: str | None) -> float:
    """Resmî mahalle adının anlamlı kelimelerinin proje adresi içinde geçme oranını döndürür.

    `official_address_area` bir mahalle adıdır, `address` ise Google Places'ten gelen tam
    adres metnidir; bu iki alan farklı ayrıntı düzeyinde olduğu için tam dize benzerliği yerine
    içerme (containment) tabanlı bir skor kullanılır.
    """

    if not project_address_norm or not official_area_norm:
        return 0.0
    tokens = [t for t in official_area_norm.split() if t not in _AREA_SUFFIX_WORDS]
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in project_address_norm)
    return hits / len(tokens)


NAME_WEIGHT = 0.60
PHONE_WEIGHT = 0.20
AREA_WEIGHT = 0.15
ADDRESS_WEIGHT = 0.05
EXACT_PHONE_SCORE_FLOOR = 0.95


def score_match(
    name_similarity: float,
    area_match: bool | None,
    phone_match: bool,
    address_similarity: float,
) -> float:
    """Açıklanabilir ağırlıklı skor: isim %60, telefon %20, bölge %15, adres %5.

    `area_match=None` (resmî tarafta bölge eşlemesi yoksa) ne ödüllendirilir ne cezalandırılır;
    0.5 (nötr) katkı verir. Kesin telefon eşleşmesi tek başına güçlü bir sinyal olduğu için
    skoru en az `EXACT_PHONE_SCORE_FLOOR` seviyesine yükseltir (isimler çok farklı yazılmış
    olsa bile aynı işletmenin santral numarasını paylaşması olasıdır).
    """

    area_component = 0.5 if area_match is None else float(bool(area_match))
    weighted = (
        NAME_WEIGHT * name_similarity
        + PHONE_WEIGHT * float(phone_match)
        + AREA_WEIGHT * area_component
        + ADDRESS_WEIGHT * address_similarity
    )
    if phone_match:
        weighted = max(weighted, EXACT_PHONE_SCORE_FLOOR)
    return round(min(weighted, 1.0), 4)


@dataclass(frozen=True)
class MatchThresholds:
    """`classify_match` için gözlemlenen skor dağılımına göre ayarlanmış eşikler.

    Varsayılan değerler `06_hotel_attributes_match_audit.ipynb` içinde gerçek skor
    dağılımı incelenerek seçilmiştir; körlemesine seçilmemiştir. `high_confidence_score`
    kasıtlı olarak kullanılmaz: ağırlıklı skor telefon sinyali olmadan en fazla ~0.80'e
    ulaşabildiğinden (isim %60 + bölge %15 + adres %5), yüksek güven kararı `match_score`
    yerine doğrudan isim/bölge bileşenleri üzerinden verilir — aksi halde bu yol hiçbir zaman
    tetiklenmeyen "ölü" bir eşik olurdu.
    """

    high_confidence_min_name_similarity: float = 0.95
    review_required_score: float = 0.55


def classify_match(
    match_score: float,
    name_similarity: float,
    area_match: bool | None,
    phone_match: bool,
    thresholds: MatchThresholds = MatchThresholds(),
) -> str:
    """`MATCHED_HIGH_CONFIDENCE` / `REVIEW_REQUIRED` / `UNMATCHED` durumunu belirler.

    Yüksek güven için iki konservatif yol vardır: (1) kesin ve tekil telefon eşleşmesi, ya da
    (2) neredeyse birebir isim benzerliği **ve** bölgenin açıkça doğrulanmış olması
    (`area_match is True` — bölge bilgisi eksikse veya çelişiyorsa bu yol tetiklenmez). İsim
    ve adres birebir aynı olsa bile bölge açıkça **çelişiyorsa** (ör. "Armonia Holiday
    Village & Spa" örneği, bkz. notebook Bölüm 10) kasıtlı olarak REVIEW_REQUIRED'a düşer.
    """

    if phone_match:
        return "MATCHED_HIGH_CONFIDENCE"
    if name_similarity >= thresholds.high_confidence_min_name_similarity and area_match is True:
        return "MATCHED_HIGH_CONFIDENCE"
    if match_score >= thresholds.review_required_score:
        return "REVIEW_REQUIRED"
    return "UNMATCHED"


def prepare_project_normalization(project_df: pd.DataFrame) -> pd.DataFrame:
    """Proje otel tablosuna eşleştirme amaçlı geçici normalize kolonlar ekler (kopya döner)."""

    result = project_df.copy()
    normalized = result["hotel_name"].apply(normalize_hotel_name)
    result["hotel_name_normalized_full"] = normalized.apply(lambda pair: pair[0])
    result["hotel_name_normalized_core"] = normalized.apply(lambda pair: pair[1])
    result["phone_normalized"] = result["phone"].apply(normalize_phone)
    result["address_normalized"] = result["address"].apply(normalize_address)
    result["area_normalized"] = result["area"].apply(normalize_text)
    return result


def prepare_official_normalization(official_df: pd.DataFrame) -> pd.DataFrame:
    """Resmî tesis tablosuna eşleştirme amaçlı geçici normalize kolonlar ekler (kopya döner)."""

    result = official_df.copy()
    normalized = result["official_name"].apply(normalize_hotel_name)
    result["official_name_normalized_full"] = normalized.apply(lambda pair: pair[0])
    result["official_name_normalized_core"] = normalized.apply(lambda pair: pair[1])
    result["official_phone_normalized"] = result["official_phone"].apply(normalize_phone)
    result["official_address_normalized"] = result["official_address_area"].apply(normalize_address)
    result["mapped_project_area_normalized"] = result["mapped_project_area"].apply(normalize_text)
    return result


def detect_official_duplicates(official_norm: pd.DataFrame) -> pd.DataFrame:
    """Resmî tablo içinde aynı isim/telefonla görünen olası tekrar eden kayıtları işaretler.

    Otomatik olarak hiçbir satırı silmez veya seçmez; yalnızca `official_duplicate_candidate`,
    `official_conflict_flag`, `duplicate_group_id` ve `conflict_reason` kolonlarını üretir.
    """

    result = official_norm.copy()
    result["official_duplicate_candidate"] = False
    result["official_conflict_flag"] = False
    result["duplicate_group_id"] = pd.array([pd.NA] * len(result), dtype="Int64")
    result["conflict_reason"] = pd.NA

    conflict_columns = ["official_star_rating", "official_type", "room_count", "bed_count"]
    group_id = 0

    for key_column in ["official_name_normalized_core", "official_phone_normalized"]:
        for key_value, group_index in result.groupby(key_column).groups.items():
            if pd.isna(key_value) or len(group_index) < 2:
                continue
            group_id += 1
            result.loc[group_index, "official_duplicate_candidate"] = True
            existing_group = result.loc[group_index, "duplicate_group_id"]
            result.loc[group_index, "duplicate_group_id"] = existing_group.fillna(group_id)

            differing = [
                column
                for column in conflict_columns
                if result.loc[group_index, column].nunique(dropna=True) > 1
            ]
            if differing:
                result.loc[group_index, "official_conflict_flag"] = True
                reason = f"Farklı değerler: {', '.join(differing)}"
                result.loc[group_index, "conflict_reason"] = reason

    return result


def generate_candidates(
    project_norm: pd.DataFrame,
    official_norm: pd.DataFrame,
    fuzzy_name_floor: float = 0.55,
) -> pd.DataFrame:
    """Her proje oteli için aday resmî tesis satırlarını üç katmanda toplar.

    Katman 1 (telefon) ve Katman 2 (kesin çekirdek isim) doğrudan eşleşme arar; Katman 3
    (fuzzy) `fuzzy_name_floor` üzerindeki tüm resmî kayıtları aday havuzuna ekler. Adaylar
    daha sonra `score_match` ile puanlanmak üzere ham sinyal kolonlarıyla birlikte döner.
    """

    official_phone_counts = official_norm["official_phone_normalized"].value_counts()

    rows: list[dict] = []
    for _, project_row in project_norm.iterrows():
        p_full = project_row["hotel_name_normalized_full"]
        p_core = project_row["hotel_name_normalized_core"]
        p_phone = project_row["phone_normalized"]
        p_area = project_row["area_normalized"]
        p_address = project_row["address_normalized"]

        candidate_idx: set[int] = set()

        if p_phone is not None:
            candidate_idx.update(
                official_norm.index[official_norm["official_phone_normalized"] == p_phone]
            )
        if p_core is not None:
            candidate_idx.update(
                official_norm.index[official_norm["official_name_normalized_core"] == p_core]
            )
        if p_full is not None:
            for official_idx, official_row in official_norm.iterrows():
                name_sim = max(
                    calculate_name_similarity(p_full, official_row["official_name_normalized_full"]),
                    calculate_name_similarity(p_core, official_row["official_name_normalized_core"]),
                )
                if name_sim >= fuzzy_name_floor:
                    candidate_idx.add(official_idx)

        for official_idx in candidate_idx:
            official_row = official_norm.loc[official_idx]
            name_similarity = max(
                calculate_name_similarity(p_full, official_row["official_name_normalized_full"]),
                calculate_name_similarity(p_core, official_row["official_name_normalized_core"]),
            )
            official_phone = official_row["official_phone_normalized"]
            phone_match = bool(
                p_phone is not None
                and official_phone is not None
                and p_phone == official_phone
                and official_phone_counts.get(official_phone, 0) == 1
            )
            mapped_area = official_row["mapped_project_area_normalized"]
            if mapped_area is None:
                area_match = None
            else:
                area_match = mapped_area == p_area
            address_similarity = calculate_address_similarity(
                p_address, official_row["official_address_normalized"]
            )

            rows.append(
                {
                    "hotel_id": project_row["hotel_id"],
                    "hotel_name": project_row["hotel_name"],
                    "area": project_row["area"],
                    "phone": project_row["phone"],
                    "address": project_row["address"],
                    "official_facility_id": official_row["official_facility_id"],
                    "official_name": official_row["official_name"],
                    "official_type": official_row["official_type"],
                    "official_star_rating": official_row["official_star_rating"],
                    "room_count": official_row["room_count"],
                    "bed_count": official_row["bed_count"],
                    "official_address_area": official_row["official_address_area"],
                    "official_phone": official_row["official_phone"],
                    "mapped_project_area": official_row["mapped_project_area"],
                    "source_url": official_row.get("source_url"),
                    "name_similarity": round(name_similarity, 4),
                    "area_match": area_match,
                    "phone_match": phone_match,
                    "address_similarity": round(address_similarity, 4),
                }
            )

    columns = [
        "hotel_id", "hotel_name", "area", "phone", "address",
        "official_facility_id", "official_name", "official_type", "official_star_rating",
        "room_count", "bed_count", "official_address_area", "official_phone", "mapped_project_area",
        "source_url", "name_similarity", "area_match", "phone_match", "address_similarity",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_best_match_table(
    candidates: pd.DataFrame,
    project_norm: pd.DataFrame,
    thresholds: MatchThresholds = MatchThresholds(),
) -> pd.DataFrame:
    """Her `hotel_id` için en iyi tek adayı seçer, skorlar ve durumunu sınıflandırır.

    Sıralama önceliği: telefon eşleşmesi > match_score > isim benzerliği. Hiçbir aday
    üretilmemiş proje otelleri için (candidates tablosunda hiç satırı olmayanlar) tek satırlık
    bir `UNMATCHED` kaydı eklenir; resmî kolonlar `NaN` kalır, hiçbir değer uydurulmaz.
    """

    scored = candidates.copy()
    if not scored.empty:
        scored["match_score"] = [
            score_match(row.name_similarity, row.area_match, row.phone_match, row.address_similarity)
            for row in scored.itertuples()
        ]
        scored["match_method"] = np.select(
            [scored["phone_match"], scored["name_similarity"] >= 0.999],
            ["phone_exact", "name_exact"],
            default="fuzzy",
        )
        scored = scored.sort_values(
            by=["phone_match", "match_score", "name_similarity"], ascending=False
        )
        best = scored.groupby("hotel_id", as_index=False).first()
        best["match_status"] = [
            classify_match(row.match_score, row.name_similarity, row.area_match, row.phone_match, thresholds)
            for row in best.itertuples()
        ]
    else:
        best = scored.assign(match_score=pd.Series(dtype="float64"), match_method=pd.Series(dtype="object"),
                              match_status=pd.Series(dtype="object"))

    matched_ids = set(best["hotel_id"])
    missing = project_norm.loc[~project_norm["hotel_id"].isin(matched_ids)]
    if not missing.empty:
        no_candidate_rows = pd.DataFrame(
            {
                "hotel_id": missing["hotel_id"].values,
                "hotel_name": missing["hotel_name"].values,
                "area": missing["area"].values,
                "phone": missing["phone"].values,
                "address": missing["address"].values,
                "official_facility_id": pd.NA,
                "official_name": pd.NA,
                "official_type": pd.NA,
                "official_star_rating": np.nan,
                "room_count": np.nan,
                "bed_count": np.nan,
                "official_address_area": pd.NA,
                "official_phone": pd.NA,
                "mapped_project_area": pd.NA,
                "source_url": pd.NA,
                "name_similarity": 0.0,
                "area_match": None,
                "phone_match": False,
                "address_similarity": 0.0,
                "match_score": 0.0,
                "match_method": "no_candidate",
                "match_status": "UNMATCHED",
            }
        )
        best = pd.concat([best, no_candidate_rows], ignore_index=True)

    return resolve_duplicate_official_assignments(best)


def resolve_duplicate_official_assignments(best: pd.DataFrame) -> pd.DataFrame:
    """Aynı `official_facility_id`'nin birden fazla otele yüksek güvenle bağlanmasını önler.

    Böyle bir çakışma varsa (ör. iki proje oteli aynı telefonu paylaşıyorsa) yalnızca en
    yüksek **isim benzerlikli** satır `MATCHED_HIGH_CONFIDENCE` olarak kalır; diğerleri
    otomatik bağlanmak yerine `REVIEW_REQUIRED`'a düşürülür (kayıt silinmez, yalnızca durumu
    değişir). Tie-break kasıtlı olarak `match_score` değil `name_similarity` üzerinden yapılır:
    kesin telefon eşleşmesi skoru yapay biçimde yükselttiği için (bkz. `score_match`), düşük
    isim benzerlikli tesadüfi bir telefon eşleşmesi, çok daha yüksek isim benzerlikli gerçek
    eşleşmenin önüne geçebilir (bkz. notebook Bölüm 10 — "Mandarin Resort Hotel" örneği).
    """

    result = best.copy()
    high_confidence = result.loc[result["match_status"].eq("MATCHED_HIGH_CONFIDENCE")]
    duplicated_ids = high_confidence.loc[
        high_confidence["official_facility_id"].duplicated(keep=False), "official_facility_id"
    ].unique()

    for official_id in duplicated_ids:
        group = result.loc[
            (result["official_facility_id"] == official_id) & (result["match_status"].eq("MATCHED_HIGH_CONFIDENCE"))
        ]
        keep_index = group["name_similarity"].idxmax()
        downgrade_index = group.index.difference([keep_index])
        result.loc[downgrade_index, "match_status"] = "REVIEW_REQUIRED"

    return result


def explain_review_reason(
    name_similarity: float,
    area_match: bool | None,
    phone_match: bool,
    address_similarity: float,
) -> str:
    """`REVIEW_REQUIRED` kaydı için kısa, okunabilir bir gerekçe metni üretir."""

    if phone_match and area_match is False:
        return "Telefon eşleşiyor ama bölge çelişiyor"
    if name_similarity >= 0.95 and area_match is False:
        return "İsim (ve muhtemelen adres) neredeyse birebir ama bölge çelişiyor"
    if name_similarity >= 0.95 and area_match is None:
        return "İsim neredeyse birebir ama resmi kayıtta bölge bilgisi yok"
    if name_similarity >= 0.80 and area_match is True:
        return "Yüksek isim benzerliği ve bölge doğrulanıyor ama telefon teyidi yok"
    if name_similarity >= 0.80 and address_similarity >= 0.5:
        return "Yüksek isim ve adres benzerliği ama bölge doğrulanamıyor"
    if name_similarity >= 0.80:
        return "Yüksek isim benzerliği ama bölge/adres doğrulanamıyor"
    if phone_match:
        return "Telefon eşleşiyor ama isim benzerliği düşük (olası hatalı/paylaşılan numara)"
    return "Orta düzey isim benzerliği, güçlü destek sinyali yok"


def build_enriched_dataset(
    project_df: pd.DataFrame,
    best_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Ana proje tablosuna, yalnızca `MATCHED_HIGH_CONFIDENCE` eşleşmelerden gelen resmî
    kolonları ekler. Eşleşmeyen otellerde yeni kolonlar `NaN` kalır; hiçbir değer uydurulmaz.
    Mevcut `official_star_rating` kolonu asla üzerine yazılmaz.
    """

    high_confidence = best_matches.loc[best_matches["match_status"].eq("MATCHED_HIGH_CONFIDENCE")]
    enrichment = high_confidence[[
        "hotel_id", "official_facility_id", "official_name", "official_type",
        "official_star_rating", "room_count", "bed_count", "official_phone",
        "mapped_project_area", "match_score", "match_method", "source_url",
    ]].rename(
        columns={
            "official_star_rating": "official_star_rating_verified",
            "room_count": "official_room_count",
            "bed_count": "official_bed_count",
            "official_phone": "official_phone_verified",
            "mapped_project_area": "official_address_area",
            "match_score": "official_match_score",
            "match_method": "official_match_method",
            "source_url": "official_source_url",
        }
    )
    enrichment["official_match_status"] = "MATCHED_HIGH_CONFIDENCE"

    enriched = project_df.merge(enrichment, on="hotel_id", how="left")
    enriched["official_match_status"] = enriched["official_match_status"].fillna("UNMATCHED")
    return enriched


def derive_official_status(official_norm: pd.DataFrame, best_matches: pd.DataFrame) -> pd.DataFrame:
    """Resmî tesis tablosuna `official_match_status` kolonu ekler: `CONFLICT` / `MATCHED` /
    `UNMATCHED_OFFICIAL`. Bir kayıt hem çakışma taşıyıp hem yüksek güvenle eşleşmiş olsa bile
    (nadir), veri kalitesi sorununu gizlememek için `CONFLICT` önceliklidir."""

    result = official_norm.copy()
    matched_ids = set(
        best_matches.loc[best_matches["match_status"].eq("MATCHED_HIGH_CONFIDENCE"), "official_facility_id"]
    )
    conditions = [
        result["official_conflict_flag"],
        result["official_facility_id"].isin(matched_ids),
    ]
    choices = ["CONFLICT", "MATCHED"]
    result["official_match_status"] = np.select(conditions, choices, default="UNMATCHED_OFFICIAL")
    return result


def build_area_coverage(enriched: pd.DataFrame) -> pd.DataFrame:
    """Her proje bölgesi için resmî veri kapsamını (eşleşme oranı, yıldız/oda/yatak
    doluluğu) özetler; hangi destinasyonlarda resmî coverage zayıf olduğunu gösterir."""

    frame = enriched.copy()
    frame["is_matched"] = frame["official_match_status"].eq("MATCHED_HIGH_CONFIDENCE")

    coverage = frame.groupby("area").agg(
        hotel_count=("hotel_id", "size"),
        matched_count=("is_matched", "sum"),
        verified_star_count=("official_star_rating_verified", "count"),
        room_count_available=("official_room_count", "count"),
        bed_count_available=("official_bed_count", "count"),
    )
    coverage["match_rate_pct"] = (coverage["matched_count"] / coverage["hotel_count"] * 100).round(1)
    coverage = coverage[
        ["hotel_count", "matched_count", "match_rate_pct", "verified_star_count",
         "room_count_available", "bed_count_available"]
    ]
    return coverage.sort_values("match_rate_pct", ascending=False).reset_index()


def save_matching_outputs(outputs: dict[str, pd.DataFrame], reports_dir: str | Path, processed_dir: str | Path) -> dict[str, Path]:
    """Rapor DataFrame'lerini `reports/` altına, zenginleştirilmiş tabloyu `data/processed/`
    altına yazar. `outputs` içinde `"hotels_enriched"` anahtarı varsa processed dizine,
    diğerleri reports dizinine yazılır."""

    reports_dir = Path(reports_dir)
    processed_dir = Path(processed_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for name, frame in outputs.items():
        target_dir = processed_dir if name == "hotels_enriched" else reports_dir
        path = target_dir / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = path
    return paths
