"""EDA öncesi temel ve açıklanabilir otel özellikleri."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "hotel_id",
    "place_id",
    "area",
    "official_star_rating",
    "google_rating",
    "google_review_count",
    "search_price_usd_snapshot",
    "phone",
    "business_status",
    "area_hotel_count",
}


@dataclass(frozen=True)
class FeatureEngineeringResult:
    """Zenginleştirilmiş tablo ile özellik dokümantasyonunu birlikte taşır."""

    df: pd.DataFrame
    feature_dictionary: pd.DataFrame
    parameters: pd.DataFrame
    validation_report: pd.DataFrame


def _validation_row(check: str, passed: bool, issue_count: int, detail: str) -> dict:
    return {
        "check": check,
        "status": "PASS" if passed else "FAIL",
        "issue_count": int(issue_count),
        "detail": detail,
    }


def _feature_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        ("has_price_snapshot", "availability", "Fiyat snapshot değerinin bulunup bulunmadığı.", "Eksik fiyat doldurulmaz; False olarak işaretlenir.", "Fiyatın kalitesi veya güncelliği hakkında hüküm vermez."),
        ("has_official_star_rating", "availability", "Doğrulanmış resmî yıldız bilgisinin bulunup bulunmadığı.", "Eksik yıldız doldurulmaz; False olarak işaretlenir.", "Google müşteri puanından yıldız türetilmez."),
        ("has_phone", "availability", "Telefon bilgisinin bulunup bulunmadığı.", "Eksik telefon doldurulmaz; False olarak işaretlenir.", "İletişim kalitesini ölçmez."),
        ("has_business_status", "availability", "Google business status bilgisinin bulunup bulunmadığı.", "Eksik durum doldurulmaz; False olarak işaretlenir.", "Eksiklik işletmenin kapalı olduğu anlamına gelmez."),
        ("review_count_log1p", "distribution", "Google yorum sayısının log(1+x) dönüşümü.", "Yorum sayısı eksikse sonuç eksik kalır.", "Orijinal yorum sayısının yerine geçmez."),
        ("review_confidence_weight", "rating", "v/(v+m); v yorum sayısı, m veri seti medyan yorum sayısıdır.", "Puan veya yorum sayısı eksikse sonuç eksik kalır.", "İstatistiksel güven aralığı değildir; şeffaf bir ağırlıktır."),
        ("weighted_google_rating", "rating", "Yorum hacmine göre genel ortalamaya daraltılmış Google puanı.", "Puan veya yorum sayısı eksikse sonuç eksik kalır.", "Rating tahmininde girdi olarak kullanılırsa hedef sızıntısı yaratır."),
        ("area_median_google_rating", "area_context", "Otelin bölgesindeki medyan Google müşteri puanı.", "Bölgedeki mevcut puanlardan hesaplanır.", "Küçük bölgelerde örneklem hassasiyeti vardır."),
        ("rating_gap_from_area_median", "area_context", "Otel puanı eksi bölge medyan puanı.", "Gerekli değer eksikse sonuç eksik kalır.", "Nedensel etki göstermez."),
        ("area_median_price_snapshot", "price_context", "Aynı bölgedeki dolu fiyat snapshotlarının medyanı.", "Eksik fiyatlar medyan hesabında yok sayılır; doldurulmaz.", "Yalnızca aynı snapshot bağlamında yorumlanmalıdır."),
        ("price_gap_from_area_median", "price_context", "Otel fiyat snapshotı eksi bölge medyan fiyat snapshotı.", "Otel fiyatı eksikse sonuç eksik kalır.", "Kalıcı otel fiyatı değildir."),
        ("price_ratio_to_area_median", "price_context", "Otel fiyat snapshotının bölge medyanına oranı.", "Otel fiyatı eksik veya medyan sıfırsa sonuç eksik kalır.", "Oda ve rezervasyon koşulları bilinmeden performans skoru değildir."),
        ("price_percentile_within_area", "price_context", "Fiyatın kendi bölgesindeki yüzdelik sırası.", "Eksik fiyatın sırası eksik kalır.", "Az fiyat gözlemli bölgelerde dikkatle yorumlanmalıdır."),
        ("review_count_percentile_within_area", "popularity_context", "Yorum sayısının kendi bölgesindeki yüzdelik sırası.", "Eksik yorum sayısının sırası eksik kalır.", "Yorum sayısı tek başına talep veya kalite ölçüsü değildir."),
    ]
    rows = []
    for feature, group, definition, missing_policy, caution in definitions:
        rows.append(
            {
                "feature": feature,
                "dtype": str(df[feature].dtype),
                "feature_group": group,
                "definition": definition,
                "missing_policy": missing_policy,
                "caution": caution,
            }
        )
    return pd.DataFrame(rows)


def build_basic_features(clean_df: pd.DataFrame) -> FeatureEngineeringResult:
    """Temiz otel tablosuna EDA öncesi temel özellikleri ekler."""

    missing_columns = sorted(REQUIRED_COLUMNS.difference(clean_df.columns))
    if missing_columns:
        raise ValueError(f"Zorunlu kolonlar eksik: {missing_columns}")

    source = clean_df.copy(deep=True)
    df = clean_df.copy(deep=True)

    rating = pd.to_numeric(df["google_rating"], errors="coerce")
    reviews = pd.to_numeric(df["google_review_count"], errors="coerce")
    price = pd.to_numeric(df["search_price_usd_snapshot"], errors="coerce")

    global_mean_rating = float(rating.mean())
    review_count_median = float(reviews.median())

    df["has_price_snapshot"] = price.notna().astype("boolean")
    df["has_official_star_rating"] = df["official_star_rating"].notna().astype("boolean")
    df["has_phone"] = df["phone"].notna().astype("boolean")
    df["has_business_status"] = df["business_status"].notna().astype("boolean")

    df["review_count_log1p"] = np.log1p(reviews).astype("Float64")
    df["review_confidence_weight"] = (reviews / (reviews + review_count_median)).astype("Float64")
    df["weighted_google_rating"] = (
        df["review_confidence_weight"] * rating
        + (1 - df["review_confidence_weight"]) * global_mean_rating
    ).astype("Float64")

    df["area_median_google_rating"] = df.groupby("area")["google_rating"].transform("median").astype("Float64")
    df["rating_gap_from_area_median"] = (rating - df["area_median_google_rating"]).astype("Float64")

    df["area_median_price_snapshot"] = df.groupby("area")["search_price_usd_snapshot"].transform("median").astype("Float64")
    df["price_gap_from_area_median"] = (price - df["area_median_price_snapshot"]).astype("Float64")
    valid_area_median = df["area_median_price_snapshot"].ne(0)
    df["price_ratio_to_area_median"] = (
        price.div(df["area_median_price_snapshot"]).where(valid_area_median).astype("Float64")
    )
    df["price_percentile_within_area"] = (
        df.groupby("area")["search_price_usd_snapshot"].rank(method="average", pct=True).astype("Float64")
    )
    df["review_count_percentile_within_area"] = (
        df.groupby("area")["google_review_count"].rank(method="average", pct=True).astype("Float64")
    )

    parameters = pd.DataFrame(
        [
            {
                "parameter": "global_mean_google_rating",
                "value": global_mean_rating,
                "definition": "Ağırlıklı puanda kullanılan tüm otellerin ortalama Google puanı.",
            },
            {
                "parameter": "review_count_median_m",
                "value": review_count_median,
                "definition": "Ağırlıklı puanda kullanılan veri seti medyan yorum sayısı.",
            },
        ]
    )

    raw_columns_unchanged = [
        column for column in source.columns if not source[column].equals(df[column])
    ]
    expected_area_count = df.groupby("area", dropna=False)["hotel_id"].transform("size")
    area_count_invalid = df["area_hotel_count"].ne(expected_area_count)
    confidence_invalid = df["review_confidence_weight"].notna() & ~df["review_confidence_weight"].between(0, 1)
    weighted_rating_invalid = df["weighted_google_rating"].notna() & ~df["weighted_google_rating"].between(0, 5)
    price_missing_changed = df["has_price_snapshot"].ne(price.notna())
    price_feature_imputed = price.isna() & df[
        ["price_gap_from_area_median", "price_ratio_to_area_median", "price_percentile_within_area"]
    ].notna().any(axis=1)

    validation_report = pd.DataFrame(
        [
            _validation_row("row_count_preserved", len(df) == len(source), abs(len(df) - len(source)), "Satır eklenmemeli veya silinmemelidir."),
            _validation_row("source_columns_unchanged", not raw_columns_unchanged, len(raw_columns_unchanged), f"Değişen kaynak kolonlar: {raw_columns_unchanged}"),
            _validation_row("keys_preserved", df[["hotel_id", "place_id"]].equals(source[["hotel_id", "place_id"]]), 0 if df[["hotel_id", "place_id"]].equals(source[["hotel_id", "place_id"]]) else 1, "Otel kimlikleri ve sırası korunmalıdır."),
            _validation_row("area_hotel_count_matches", not area_count_invalid.any(), area_count_invalid.sum(), "Bölge sayısı ana tablodan hesaplanmalıdır."),
            _validation_row("confidence_weight_in_range", not confidence_invalid.any(), confidence_invalid.sum(), "Güven ağırlığı 0-1 aralığında olmalıdır."),
            _validation_row("weighted_rating_in_range", not weighted_rating_invalid.any(), weighted_rating_invalid.sum(), "Ağırlıklı puan 0-5 aralığında olmalıdır."),
            _validation_row("price_availability_matches", not price_missing_changed.any(), price_missing_changed.sum(), "Fiyat varlık göstergesi gerçek eksiklikle eşleşmelidir."),
            _validation_row("missing_prices_not_imputed", not price_feature_imputed.any(), price_feature_imputed.sum(), "Eksik otel fiyatından otel düzeyi fiyat özelliği üretilmemelidir."),
        ]
    )

    return FeatureEngineeringResult(
        df=df,
        feature_dictionary=_feature_dictionary(df),
        parameters=parameters,
        validation_report=validation_report,
    )


def save_feature_outputs(
    result: FeatureEngineeringResult,
    processed_dir: str | Path,
    reports_dir: str | Path,
) -> dict[str, Path]:
    """Özellik tablosu ile sözlük, parametre ve doğrulama raporlarını kaydeder."""

    processed_dir = Path(processed_dir)
    reports_dir = Path(reports_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "feature_table": processed_dir / "hotels_features.csv",
        "feature_dictionary": reports_dir / "feature_dictionary.csv",
        "parameters": reports_dir / "feature_engineering_parameters.csv",
        "validation_report": reports_dir / "feature_validation_report.csv",
    }
    result.df.to_csv(paths["feature_table"], index=False, encoding="utf-8-sig")
    result.feature_dictionary.to_csv(paths["feature_dictionary"], index=False, encoding="utf-8-sig")
    result.parameters.to_csv(paths["parameters"], index=False, encoding="utf-8-sig")
    result.validation_report.to_csv(paths["validation_report"], index=False, encoding="utf-8-sig")
    return paths
