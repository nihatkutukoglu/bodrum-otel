"""Safe, resumable Google Maps public-UI collection for all project hotels.

This module intentionally does not use private endpoints, authentication bypass,
CAPTCHA handling, proxy rotation, or stealth browser techniques.  A challenge page
causes a durable ``BLOCKED_SAFE_STOP`` result and collection stops immediately.
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
import shutil
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

MAX_REVIEWS_PER_HOTEL = 75
RATING_TARGETS = {"LOW": 25, "MIXED": 15, "HIGH": 35}
MAPPING_STATUSES = {
    "FOUND_EXACT_PLACE_ID",
    "FOUND_HIGH_CONFIDENCE",
    "REVIEW_REQUIRED",
    "NOT_FOUND",
    "BLOCKED_SAFE_STOP",
    "ERROR",
}

TARGET_FIELDS = (
    "hotel_id", "place_id", "hotel_name", "area", "address", "google_rating",
    "google_review_count", "source_url", "target_review_cap", "collection_priority",
)
RAW_FIELDS = (
    "review_id", "hotel_id", "place_id", "hotel_name", "area", "review_rating",
    "review_date_raw", "review_text", "review_language", "reviewer_name_raw",
    "source_url", "review_url", "source_platform", "collected_at", "collection_order",
    "rating_group", "collection_batch", "is_rating_only", "extraction_confidence",
)
STATUS_FIELDS = (
    "hotel_id", "hotel_name", "area", "mapping_status", "page_accessible",
    "total_google_review_count_master", "reviews_collected", "reviews_with_text",
    "rating_1_n", "rating_2_n", "rating_3_n", "rating_4_n", "rating_5_n",
    "low_n", "mixed_n", "high_n", "target_cap", "target_reached", "blocked",
    "error_type", "last_checkpoint", "collection_completed_at",
)
MAPPING_FIELDS = (
    "hotel_id", "place_id", "hotel_name", "area", "source_url", "detected_hotel_name",
    "detected_address", "mapping_status", "name_similarity", "page_accessible",
    "review_panel_found", "checked_at", "mapping_note",
)
CHALLENGE_MARKERS = (
    "unusual traffic", "our systems have detected", "not a robot", "robot olmadığınızı",
    "captcha", "access denied", "erişim reddedildi", "trafik algıladı",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def name_similarity(expected: str, detected: str) -> float:
    from difflib import SequenceMatcher

    left, right = normalize_name(expected), normalize_name(detected)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def identity_status(expected: str, detected: str, place_id: str, source_url: str) -> tuple[str, float]:
    """Fail closed: a place-id URL plus a strong visible-name match is exact."""
    score = name_similarity(expected, detected)
    place_bound = bool(place_id and f"query_place_id={place_id}" in source_url)
    if place_bound and score >= 0.84:
        return "FOUND_EXACT_PLACE_ID", score
    if score >= 0.92:
        return "FOUND_HIGH_CONFIDENCE", score
    if not detected:
        return "NOT_FOUND", score
    return "REVIEW_REQUIRED", score


def rating_group(value: Any) -> str | None:
    try:
        rating = int(float(value))
    except (TypeError, ValueError):
        return None
    if rating not in range(1, 6):
        return None
    return "LOW" if rating <= 2 else "MIXED" if rating == 3 else "HIGH"


def stable_review_id(hotel_id: str, rating: Any, date_raw: str, text: str, native_id: str = "") -> str:
    if native_id:
        digest = hashlib.sha256(native_id.encode("utf-8")).hexdigest()[:24]
    else:
        normalized = re.sub(r"\s+", " ", (text or "").casefold()).strip()
        payload = f"{hotel_id}\0{rating}\0{date_raw}\0{normalized}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"gma_{digest}"


def build_targets(
    master: pd.DataFrame,
    current_nlp_hotel_ids: Iterable[str] = (),
    sikayetvar_hotel_ids: Iterable[str] = (),
    cap: int = MAX_REVIEWS_PER_HOTEL,
) -> pd.DataFrame:
    required = {"hotel_id", "place_id", "hotel_name", "area", "address", "google_rating", "google_review_count", "source_url"}
    missing = required.difference(master.columns)
    if missing:
        raise ValueError(f"Master hotel columns missing: {sorted(missing)}")
    if cap < 1:
        raise ValueError("target review cap must be positive")
    work = master[list(required)].copy()
    work["google_review_count"] = pd.to_numeric(work["google_review_count"], errors="coerce").fillna(0).astype(int)
    work["_missing_current"] = ~work["hotel_id"].astype(str).isin(set(map(str, current_nlp_hotel_ids)))
    work["_sikayetvar"] = work["hotel_id"].astype(str).isin(set(map(str, sikayetvar_hotel_ids)))
    # Priority is execution order only.  High visibility leads, then missing current
    # coverage and future cross-platform overlap; stable area/name keys break ties.
    work = work.sort_values(
        ["google_review_count", "_missing_current", "_sikayetvar", "area", "hotel_name"],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    work["target_review_cap"] = int(cap)
    work["collection_priority"] = range(1, len(work) + 1)
    return work[list(TARGET_FIELDS)]


def select_smoke_targets(targets: pd.DataFrame, n: int = 10, regression_ids: Iterable[str] = ()) -> pd.DataFrame:
    """Choose deterministic volume/area/regression coverage without collecting."""
    if n < 1 or targets.empty:
        return targets.iloc[0:0].copy()
    pool = targets.copy().sort_values("google_review_count", ascending=False, kind="stable")
    chosen: list[str] = []
    for hotel_id in regression_ids:
        if hotel_id in set(pool["hotel_id"].astype(str)) and hotel_id not in chosen:
            chosen.append(str(hotel_id))
        if len(chosen) >= min(2, n):
            break
    quantile_indices = sorted({0, len(pool) // 4, len(pool) // 2, (3 * len(pool)) // 4, len(pool) - 1})
    for idx in quantile_indices:
        hotel_id = str(pool.iloc[idx]["hotel_id"])
        if hotel_id not in chosen:
            chosen.append(hotel_id)
    for _, row in pool.groupby("area", sort=True).head(1).sort_values("google_review_count", ascending=False).iterrows():
        hotel_id = str(row["hotel_id"])
        if hotel_id not in chosen:
            chosen.append(hotel_id)
        if len(chosen) >= n:
            break
    for hotel_id in pool["hotel_id"].astype(str):
        if hotel_id not in chosen:
            chosen.append(hotel_id)
        if len(chosen) >= n:
            break
    order = {hotel_id: idx for idx, hotel_id in enumerate(chosen[:n])}
    selected = targets[targets["hotel_id"].astype(str).isin(order)].copy()
    selected["_order"] = selected["hotel_id"].astype(str).map(order)
    return selected.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def write_csv(path: Path, rows: pd.DataFrame | Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    frame = frame.reindex(columns=list(fields))
    frame.to_csv(path, index=False, encoding="utf-8-sig")


@dataclass
class CollectionResult:
    mapping: dict[str, Any]
    status: dict[str, Any]
    reviews: list[dict[str, Any]]
    safe_stop: bool = False


class MapsPublicCollector:
    """Selenium collector restricted to the public rendered Google Maps UI."""

    def __init__(self, headless: bool = True, delay_min: float = 2.0, delay_max: float = 5.0, retries: int = 2):
        if delay_min < 0 or delay_max < delay_min:
            raise ValueError("invalid delay range")
        if retries not in range(0, 4):
            raise ValueError("retries must be between 0 and 3")
        self.headless, self.delay_min, self.delay_max, self.retries = headless, delay_min, delay_max, retries
        self.driver = None

    def __enter__(self):
        from selenium import webdriver

        options = webdriver.ChromeOptions()
        options.add_argument("--lang=tr-TR")
        options.add_argument("--disable-notifications")
        options.add_argument("--window-size=1400,1000")
        options.add_experimental_option("prefs", {"intl.accept_languages": "tr-TR,tr,en-US,en"})
        if self.headless:
            options.add_argument("--headless=new")
        self.driver = webdriver.Chrome(options=options)
        self.driver.set_page_load_timeout(30)
        return self

    def __exit__(self, *_: object) -> None:
        if self.driver is not None:
            self.driver.quit()

    def _delay(self) -> None:
        time.sleep(random.uniform(self.delay_min, self.delay_max))

    def _body(self) -> str:
        from selenium.webdriver.common.by import By

        try:
            return self.driver.find_element(By.TAG_NAME, "body").text
        except Exception:
            return ""

    def _challenge(self) -> bool:
        body = self._body().casefold()
        return any(marker in body for marker in CHALLENGE_MARKERS)

    def _open(self, url: str) -> None:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                self.driver.get(url)
                self._delay()
                return
            except Exception as error:
                last = error
                if attempt < self.retries:
                    self._delay()
        raise RuntimeError(f"PAGE_OPEN_FAILED: {last}")

    def _visible_text(self, selector: str) -> str:
        from selenium.webdriver.common.by import By

        for node in self.driver.find_elements(By.CSS_SELECTOR, selector):
            try:
                text = node.text.strip()
                if node.is_displayed() and text:
                    return text
            except Exception:
                continue
        return ""

    def _click_reviews_tab(self, expected_name: str) -> bool:
        from selenium.webdriver.common.by import By

        candidates = self.driver.find_elements(By.CSS_SELECTOR, "[role='tab']")
        for tab in candidates:
            label = f"{tab.get_attribute('aria-label') or ''} {tab.text or ''}".casefold()
            if any(word in label for word in ("yorum", "review")):
                try:
                    self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", tab)
                    tab.click()
                    self._delay()
                    return True
                except Exception:
                    continue
        return False

    def _expand_review_texts(self) -> None:
        from selenium.webdriver.common.by import By

        for card in self.driver.find_elements(By.CSS_SELECTOR, ".jftiEf[data-review-id]"):
            for button in card.find_elements(By.CSS_SELECTOR, "button.w8nwRe"):
                try:
                    if button.is_displayed() and (button.get_attribute("aria-label") or "").casefold() in {"daha fazla göster", "more"}:
                        self.driver.execute_script("arguments[0].click();", button)
                except Exception:
                    continue

    def _extract_cards(self, hotel: pd.Series, batch: str) -> list[dict[str, Any]]:
        from selenium.webdriver.common.by import By

        self._expand_review_texts()
        rows: list[dict[str, Any]] = []
        for order, card in enumerate(self.driver.find_elements(By.CSS_SELECTOR, ".jftiEf[data-review-id]"), start=1):
            def text(css: str) -> str:
                try:
                    return card.find_element(By.CSS_SELECTOR, css).text.strip()
                except Exception:
                    return ""

            native_id = card.get_attribute("data-review-id") or ""
            rating_raw = text(".fzvQIb")
            rating_match = re.search(r"([1-5])\s*/\s*5", rating_raw)
            rating = int(rating_match.group(1)) if rating_match else None
            date_source = text(".xRkPPb")
            date_raw = re.sub(r"^Google\s*", "", date_source, flags=re.I).strip(" ()")
            review_text = text(".wiI7pd")
            reviewer = (card.get_attribute("aria-label") or "").strip()
            group = rating_group(rating)
            rows.append({
                "review_id": stable_review_id(str(hotel.hotel_id), rating, date_raw, review_text, native_id),
                "hotel_id": hotel.hotel_id, "place_id": hotel.place_id, "hotel_name": hotel.hotel_name,
                "area": hotel.area, "review_rating": rating, "review_date_raw": date_raw,
                "review_text": review_text, "review_language": "", "reviewer_name_raw": reviewer,
                "source_url": hotel.source_url, "review_url": "", "source_platform": "Google Maps",
                "collected_at": utc_now(), "collection_order": order, "rating_group": group,
                "collection_batch": batch, "is_rating_only": not bool(review_text),
                "extraction_confidence": "HIGH" if rating is not None and native_id else "MEDIUM",
            })
        return rows

    def _scroll_review_panel(self) -> bool:
        return bool(self.driver.execute_script("""
            const card = document.querySelector('.jftiEf[data-review-id]');
            if (!card) return false;
            let node = card.parentElement;
            while (node && node !== document.body) {
              if (node.scrollHeight > node.clientHeight + 20) {
                const before = node.scrollTop; node.scrollTop = node.scrollHeight;
                return node.scrollTop !== before;
              }
              node = node.parentElement;
            }
            return false;
        """))

    def collect_hotel(self, hotel: pd.Series, cap: int, batch: str) -> CollectionResult:
        started = utc_now()
        empty_mapping = {
            "hotel_id": hotel.hotel_id, "place_id": hotel.place_id, "hotel_name": hotel.hotel_name,
            "area": hotel.area, "source_url": hotel.source_url, "detected_hotel_name": "",
            "detected_address": "", "mapping_status": "ERROR", "name_similarity": 0,
            "page_accessible": False, "review_panel_found": False, "checked_at": started,
            "mapping_note": "",
        }
        try:
            self._open(str(hotel.source_url))
            if self._challenge():
                empty_mapping.update(mapping_status="BLOCKED_SAFE_STOP", mapping_note="Challenge detected; no bypass attempted.")
                return CollectionResult(empty_mapping, self._status(hotel, [], cap, empty_mapping, "BLOCKED_SAFE_STOP"), [], True)
            detected = self._visible_text("main h1, h1")
            address = ""
            for node in self.driver.find_elements("css selector", "button[aria-label^='Adres:'], button[data-item-id='address']"):
                address = (node.get_attribute("aria-label") or node.text or "").removeprefix("Adres:").strip()
                if address:
                    break
            mapping_status, score = identity_status(str(hotel.hotel_name), detected, str(hotel.place_id), str(hotel.source_url))
            empty_mapping.update(
                detected_hotel_name=detected, detected_address=address, mapping_status=mapping_status,
                name_similarity=round(score, 4), page_accessible=True,
                mapping_note="place_id-bound master URL and visible hotel name checked",
            )
            if mapping_status not in {"FOUND_EXACT_PLACE_ID", "FOUND_HIGH_CONFIDENCE"}:
                return CollectionResult(empty_mapping, self._status(hotel, [], cap, empty_mapping, mapping_status), [])
            panel = self._click_reviews_tab(str(hotel.hotel_name))
            empty_mapping["review_panel_found"] = panel
            if not panel:
                # Identity is still place-id exact.  Keep mapping truth separate
                # from collection availability and do not pretend the hotel needs
                # manual entity review merely because anonymous Maps is limited.
                empty_mapping["mapping_note"] += "; public anonymous reviews tab not available"
                return CollectionResult(
                    empty_mapping,
                    self._status(hotel, [], cap, empty_mapping, "PUBLIC_REVIEW_PANEL_NOT_ACCESSIBLE"),
                    [],
                )

            by_id: dict[str, dict[str, Any]] = {}
            stable_rounds = 0
            while len(by_id) < cap and stable_rounds < 4:
                if self._challenge():
                    empty_mapping["mapping_status"] = "BLOCKED_SAFE_STOP"
                    return CollectionResult(empty_mapping, self._status(hotel, list(by_id.values()), cap, empty_mapping, "BLOCKED_SAFE_STOP"), list(by_id.values()), True)
                before = len(by_id)
                for row in self._extract_cards(hotel, batch):
                    by_id.setdefault(str(row["review_id"]), row)
                if len(by_id) >= cap:
                    break
                moved = self._scroll_review_panel()
                self._delay()
                stable_rounds = stable_rounds + 1 if len(by_id) == before or not moved else 0
            rows = list(by_id.values())[:cap]
            for order, row in enumerate(rows, start=1):
                row["collection_order"] = order
            return CollectionResult(empty_mapping, self._status(hotel, rows, cap, empty_mapping, ""), rows)
        except Exception as error:
            empty_mapping["mapping_note"] = str(error)[:500]
            return CollectionResult(empty_mapping, self._status(hotel, [], cap, empty_mapping, type(error).__name__), [])

    @staticmethod
    def _status(hotel: pd.Series, rows: list[dict[str, Any]], cap: int, mapping: dict[str, Any], error: str) -> dict[str, Any]:
        ratings = pd.Series([row.get("review_rating") for row in rows], dtype="object")
        groups = pd.Series([row.get("rating_group") for row in rows], dtype="object")
        blocked = mapping.get("mapping_status") == "BLOCKED_SAFE_STOP"
        return {
            "hotel_id": hotel.hotel_id, "hotel_name": hotel.hotel_name, "area": hotel.area,
            "mapping_status": mapping.get("mapping_status", "ERROR"),
            "page_accessible": mapping.get("page_accessible", False),
            "total_google_review_count_master": hotel.google_review_count,
            "reviews_collected": len(rows), "reviews_with_text": sum(bool(row.get("review_text")) for row in rows),
            **{f"rating_{n}_n": int((ratings == n).sum()) for n in range(1, 6)},
            "low_n": int((groups == "LOW").sum()), "mixed_n": int((groups == "MIXED").sum()),
            "high_n": int((groups == "HIGH").sum()), "target_cap": cap,
            "target_reached": len(rows) >= cap, "blocked": blocked, "error_type": error,
            "last_checkpoint": utc_now(), "collection_completed_at": utc_now(),
        }


def merge_checkpoint(path: Path, new_rows: list[dict[str, Any]], force: bool = False) -> pd.DataFrame:
    """Append/merge by review_id.  Force creates a timestamped backup first."""
    existing = pd.DataFrame(columns=RAW_FIELDS)
    if path.exists() and path.stat().st_size:
        existing = pd.read_csv(path, dtype=str).reindex(columns=RAW_FIELDS)
        if force:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy2(path, path.with_suffix(f".csv.bak_{stamp}"))
            existing = existing.iloc[0:0]
    incoming = pd.DataFrame(new_rows).reindex(columns=RAW_FIELDS)
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.drop_duplicates("review_id", keep="first")
    write_csv(path, combined, RAW_FIELDS)
    return combined


def summarize_collection(targets: pd.DataFrame, statuses: pd.DataFrame, reviews: pd.DataFrame) -> dict[str, Any]:
    attempted_mask = statuses["mapping_status"].fillna("").astype(str).str.strip().ne("") if not statuses.empty else pd.Series(dtype=bool)
    attempted = statuses[attempted_mask] if not statuses.empty else statuses
    counts = statuses.get("mapping_status", pd.Series(dtype=str)).value_counts()
    per_hotel = reviews.groupby("hotel_id").size() if not reviews.empty else pd.Series(dtype=int)
    text_reviews = reviews[reviews.get("review_text", pd.Series(index=reviews.index, dtype=str)).fillna("").str.strip().ne("")] if not reviews.empty else reviews
    groups = reviews.get("rating_group", pd.Series(dtype=str)).value_counts()
    return {
        "project_hotel_count": len(targets), "hotels_attempted": len(attempted),
        "hotels_exact_high_confidence_mapped": int(statuses.get("mapping_status", pd.Series(dtype=str)).isin(["FOUND_EXACT_PLACE_ID", "FOUND_HIGH_CONFIDENCE"]).sum()),
        "hotels_with_reviews": int((per_hotel > 0).sum()),
        "hotels_with_text_reviews": int(text_reviews["hotel_id"].nunique()) if not text_reviews.empty else 0,
        "total_reviews_collected": len(reviews), "total_text_reviews": len(text_reviews),
        "median_reviews_per_hotel": float(per_hotel.median()) if len(per_hotel) else 0.0,
        "hotels_reaching_75": int((per_hotel >= 75).sum()),
        "low_n": int(groups.get("LOW", 0)), "mixed_n": int(groups.get("MIXED", 0)), "high_n": int(groups.get("HIGH", 0)),
        "hotels_blocked": int(counts.get("BLOCKED_SAFE_STOP", 0)),
        "hotels_not_found": int(counts.get("NOT_FOUND", 0)),
        "hotels_review_required": int(counts.get("REVIEW_REQUIRED", 0)),
    }
