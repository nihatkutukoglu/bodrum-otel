"""Otel ana tablosu için kayıpsız ve açıklanabilir temizleme adımları."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


IDENTIFIER_COLUMNS = ["hotel_id", "place_id"]
TEXT_COLUMNS = [
    "hotel_id",
    "place_id",
    "hotel_name",
    "area",
    "district",
    "province",
    "country",
    "property_category",
    "price_note",
    "address",
    "phone",
    "business_status",
    "source_url",
]
FLOAT_COLUMNS = ["official_star_rating", "google_rating", "search_price_usd_snapshot"]
INTEGER_COLUMNS = ["google_rating_scale", "google_review_count"]


@dataclass(frozen=True)
class CleaningResult:
    """Temiz veri ile dönüşüm ve doğrulama raporlarını birlikte taşır."""

    hotels: pd.DataFrame
    transformation_log: pd.DataFrame
    validation_report: pd.DataFrame


def load_raw_hotels(path: str | Path) -> pd.DataFrame:
    """Ham CSV'yi kimlik ve telefon alanlarını metin olarak koruyarak yükler."""

    return pd.read_csv(
        path,
        dtype={column: "string" for column in TEXT_COLUMNS},
        encoding="utf-8-sig",
    )


def _record(log: list[dict], step: str, column: str, affected_rows: int, note: str) -> None:
    log.append(
        {
            "step": step,
            "column": column,
            "affected_rows": int(affected_rows),
            "note": note,
        }
    )


def _validation_row(check: str, passed: bool, issue_count: int, detail: str) -> dict:
    return {
        "check": check,
        "status": "PASS" if passed else "FAIL",
        "issue_count": int(issue_count),
        "detail": detail,
    }


def clean_hotels(raw: pd.DataFrame) -> CleaningResult:
    """Eksik değer üretmeden veya doldurmadan güvenli tip/boşluk temizliği yapar."""

    hotels = raw.copy(deep=True)
    log: list[dict] = []

    for column in TEXT_COLUMNS:
        if column not in hotels:
            continue
        original = hotels[column].astype("string")
        stripped = original.str.strip()
        empty_mask = stripped.eq("").fillna(False)
        changed_mask = original.notna() & stripped.ne(original).fillna(False)
        hotels[column] = stripped.mask(empty_mask, pd.NA)
        _record(log, "trim_whitespace", column, changed_mask.sum(), "Baş/son boşlukları kaldırıldı.")
        _record(log, "empty_string_to_null", column, empty_mask.sum(), "Boş metinler eksik değer olarak işaretlendi.")

    for column in FLOAT_COLUMNS:
        if column not in hotels:
            continue
        before_missing = hotels[column].isna()
        converted = pd.to_numeric(hotels[column], errors="coerce").astype("Float64")
        newly_missing = (~before_missing & converted.isna()).sum()
        hotels[column] = converted
        _record(log, "convert_to_nullable_float", column, newly_missing, "Sayısala çevrilemeyen dolu değer sayısıdır.")

    for column in INTEGER_COLUMNS:
        if column not in hotels:
            continue
        before_missing = hotels[column].isna()
        numeric = pd.to_numeric(hotels[column], errors="coerce")
        non_integer = (numeric.notna() & numeric.mod(1).ne(0)).sum()
        converted = numeric.where(numeric.mod(1).eq(0) | numeric.isna()).astype("Int64")
        newly_missing = (~before_missing & converted.isna()).sum()
        hotels[column] = converted
        _record(log, "convert_to_nullable_integer", column, newly_missing, "Sayısala çevrilemeyen veya tam sayı olmayan değer sayısıdır.")
        _record(log, "non_integer_values", column, non_integer, "Tam sayı kolonunda bulunan ondalıklı değer sayısıdır.")

    if "collected_at" in hotels:
        original_date = hotels["collected_at"].astype("string")
        parsed_date = pd.to_datetime(original_date, errors="coerce")
        invalid_dates = (original_date.notna() & parsed_date.isna()).sum()
        hotels["collected_at"] = parsed_date.dt.strftime("%Y-%m-%d").astype("string")
        _record(log, "parse_iso_date", "collected_at", invalid_dates, "Geçersiz dolu tarih sayısıdır.")

    hotels["area_hotel_count"] = (
        hotels.groupby("area", dropna=False)["hotel_id"].transform("size").astype("Int64")
    )
    _record(
        log,
        "derive_area_hotel_count",
        "area_hotel_count",
        len(hotels),
        "Ayrı bölge özet dosyası kullanmadan ana DataFrame içinden hesaplandı.",
    )

    duplicate_mask = hotels.duplicated(subset=IDENTIFIER_COLUMNS, keep=False)
    missing_id_mask = hotels[IDENTIFIER_COLUMNS].isna().any(axis=1)
    rating_invalid = hotels["google_rating"].notna() & ~hotels["google_rating"].between(0, 5)
    scale_invalid = hotels["google_rating_scale"].notna() & hotels["google_rating_scale"].ne(5)
    reviews_invalid = hotels["google_review_count"].notna() & hotels["google_review_count"].lt(0)
    price_invalid = hotels["search_price_usd_snapshot"].notna() & hotels["search_price_usd_snapshot"].le(0)
    stars_invalid = hotels["official_star_rating"].notna() & ~hotels["official_star_rating"].between(1, 5)
    expected_area_count = hotels.groupby("area", dropna=False)["hotel_id"].transform("size")
    area_count_invalid = hotels["area_hotel_count"].ne(expected_area_count)

    checks = [
        _validation_row("row_count_preserved", len(hotels) == len(raw), abs(len(hotels) - len(raw)), "Temizleme satır eklememeli veya silmemelidir."),
        _validation_row("required_ids_present", not missing_id_mask.any(), missing_id_mask.sum(), "hotel_id ve place_id zorunludur."),
        _validation_row("required_ids_unique", not duplicate_mask.any(), duplicate_mask.sum(), "hotel_id/place_id çifti benzersiz olmalıdır."),
        _validation_row("google_rating_in_range", not rating_invalid.any(), rating_invalid.sum(), "Google müşteri puanı 0-5 aralığında olmalıdır."),
        _validation_row("google_rating_scale_is_five", not scale_invalid.any(), scale_invalid.sum(), "Mevcut puan ölçeği 5 olmalıdır."),
        _validation_row("review_count_non_negative", not reviews_invalid.any(), reviews_invalid.sum(), "Yorum sayısı negatif olamaz."),
        _validation_row("snapshot_price_positive", not price_invalid.any(), price_invalid.sum(), "Dolu fiyat snapshot değeri pozitif olmalıdır."),
        _validation_row("official_star_in_range", not stars_invalid.any(), stars_invalid.sum(), "Dolu resmî yıldız 1-5 aralığında olmalıdır."),
        _validation_row("area_hotel_count_matches", not area_count_invalid.any(), area_count_invalid.sum(), "Bölge otel sayısı ana tablodan türetilmelidir."),
    ]

    return CleaningResult(
        hotels=hotels,
        transformation_log=pd.DataFrame(log),
        validation_report=pd.DataFrame(checks),
    )


def save_cleaning_outputs(result: CleaningResult, processed_dir: str | Path, reports_dir: str | Path) -> dict[str, Path]:
    """Temiz tabloyu ve açıklama raporlarını ham veriden ayrı kaydeder."""

    processed_dir = Path(processed_dir)
    reports_dir = Path(reports_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "clean_hotels": processed_dir / "hotels_clean.csv",
        "transformation_log": reports_dir / "cleaning_transformation_log.csv",
        "validation_report": reports_dir / "cleaning_validation_report.csv",
    }
    result.hotels.to_csv(output_paths["clean_hotels"], index=False, encoding="utf-8-sig")
    result.transformation_log.to_csv(output_paths["transformation_log"], index=False, encoding="utf-8-sig")
    result.validation_report.to_csv(output_paths["validation_report"], index=False, encoding="utf-8-sig")
    return output_paths
