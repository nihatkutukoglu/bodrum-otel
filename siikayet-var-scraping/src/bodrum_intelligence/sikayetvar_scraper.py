"""
Sikayetvar.com complaint scraper for the Bodrum Hotel & Destination
Intelligence project (Top 3 Most-Reviewed Bodrum Hotels dataset).

Generalized from three Beko-specific reference scripts. No brand/product
slug is hard-coded anywhere in this module: target hotels, their source
pages and Selectum entity-matching rules all come from
config/sikayetvar_targets.json.

Verified 2026-08-25 via plain `requests` (no Selenium needed): the site
server-renders complaint listing, pagination (?page=N) and complaint
detail content (including brand replies) as static HTML, so requests +
BeautifulSoup is used throughout per the project's stability preference.
No CAPTCHA/anti-bot challenge was encountered during verification; the
block-detection guard below exists for if that changes during a full run.
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

SITE_ROOT = "https://www.sikayetvar.com"
REQUEST_DELAY_SECONDS = 1.5
REQUEST_TIMEOUT = 15

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

LINKS_FIELDS = [
    "canonical_hotel_name", "area", "source_page", "complaint_url",
    "discovered_page", "collected_at",
]
COMPLAINTS_FIELDS = [
    "complaint_id", "complaint_url", "hotel_name", "area",
    "complaint_title", "complaint_text", "complaint_date_raw",
    "view_count", "support_count", "category", "product_name",
    "company_response_exists", "company_response_date", "company_response_text",
    "progress_exists", "progress_date", "progress_text",
    "user_reply_count", "first_user_reply_date", "first_user_reply_text",
    "source_page", "entity_match_status", "entity_match_reason", "collected_at",
]
REPLIES_FIELDS = [
    "complaint_id", "complaint_url", "reply_order", "reply_author_type",
    "reply_date", "reply_text",
]
STATUS_FIELDS = [
    "hotel_name", "source_page", "links_found", "matched_links",
    "review_required", "details_success", "details_failed", "last_page",
    "status", "last_updated",
]

MATCHED = "MATCHED"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
EXCLUDED = "EXCLUDED_OTHER_SELECTUM_PROPERTY"


class AntiBotBlock(Exception):
    """Raised when the site appears to show a CAPTCHA / access-block page."""


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class Target:
    canonical_hotel_name: str
    area: str
    source_pages: list
    requires_entity_validation: bool
    aliases: list = field(default_factory=list)
    match_patterns: list = field(default_factory=list)
    exclude_patterns: list = field(default_factory=list)
    ambiguous_terms: list = field(default_factory=list)


def load_targets(config_path: str) -> list:
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    return [Target(**t) for t in raw["targets"]]


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    })
    return session


_BLOCK_MARKERS = (
    "captcha", "access denied", "are you a human", "erisim engellendi",
    "cf-challenge", "attention required! | cloudflare", "just a moment...",
)


def fetch_soup(session: requests.Session, url: str) -> BeautifulSoup:
    """GET a page and return parsed HTML, or raise AntiBotBlock if the
    site appears to be showing a challenge/block page."""
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    if response.status_code in (403, 429, 503):
        raise AntiBotBlock(f"HTTP {response.status_code} on {url}")
    lowered = response.text[:4000].lower()
    if any(marker in lowered for marker in _BLOCK_MARKERS):
        raise AntiBotBlock(f"Block/challenge page detected on {url}")
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def polite_sleep(delay: float = REQUEST_DELAY_SECONDS) -> None:
    time.sleep(delay)


# --------------------------------------------------------------------------
# URL canonicalization
# --------------------------------------------------------------------------

def canonicalize_url(url: str) -> str:
    """Strip query params/fragment/trailing slash so the same complaint
    discovered from different source pages or with tracking params
    collapses to one primary key."""
    if url.startswith("/"):
        url = SITE_ROOT + url
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    return urlunsplit((("https"), "www.sikayetvar.com", path, "", ""))


def listing_page_url(source_page: str, page_num: int) -> str:
    if page_num <= 1:
        return source_page
    return f"{source_page}?page={page_num}"


# --------------------------------------------------------------------------
# Listing page parsing / link discovery
# --------------------------------------------------------------------------

def source_page_prefix(source_page: str) -> str:
    """The site renders 'related complaint' cards from *other* companies
    on thin/near-empty listing pages (observed on green-bay-resort, which
    has no complaints of its own). Complaint cards are only trustworthy if
    their href starts with the source page's own first path segment, e.g.
    '/green-bay-resort/' or '/selectum-hotels/' (shared prefix for all
    Selectum properties, disambiguated later via entity_match)."""
    first_segment = urlsplit(source_page).path.strip("/").split("/")[0]
    return f"/{first_segment}/"


def extract_complaint_cards(soup: BeautifulSoup, required_prefix: str = None) -> list:
    """Return [(href, title), ...] for each real complaint card on a
    listing page. Cards with no detail link (e.g. withdrawn/'solved'
    summary cards with no href) are skipped, as are cards whose href
    doesn't belong to this source page (see source_page_prefix)."""
    results = []
    for article in soup.find_all("article", attrs={"data-ga-element": "Complaint_Card"}):
        h3 = article.find("h3")
        link = h3.find("a", href=True) if h3 else None
        if not link:
            continue
        href = link["href"]
        # Real complaint detail links are /{slug}/{complaint-slug}; skip
        # anything else (e.g. member profile links) just in case.
        if href.count("/") < 2:
            continue
        if required_prefix and not href.startswith(required_prefix):
            continue
        results.append((canonicalize_url(href), link.get("title", "").strip()))
    return results


def collect_links_for_source_page(
    session: requests.Session,
    target: Target,
    source_page: str,
    max_pages=None,
    delay: float = REQUEST_DELAY_SECONDS,
    log=print,
) -> tuple:
    """Paginate a single source page until a page yields no new unique
    complaint URLs (or max_pages is hit). Returns (rows, last_page)."""
    rows = []
    seen_urls = set()
    page_num = 1
    last_page = 0
    blocked = False
    now = datetime.now(timezone.utc).isoformat()
    required_prefix = source_page_prefix(source_page)

    while True:
        if max_pages is not None and page_num > max_pages:
            log(f"    [stop] max_pages={max_pages} reached")
            break

        url = listing_page_url(source_page, page_num)
        log(f"    page {page_num}: {url}")
        try:
            soup = fetch_soup(session, url)
        except AntiBotBlock as exc:
            log(f"    [BLOCKED] {exc} -- stopping this source page, keeping rows collected so far")
            blocked = True
            break
        cards = extract_complaint_cards(soup, required_prefix=required_prefix)

        new_count = 0
        for href, _title in cards:
            if href not in seen_urls:
                seen_urls.add(href)
                new_count += 1
                rows.append({
                    "canonical_hotel_name": target.canonical_hotel_name,
                    "area": target.area,
                    "source_page": source_page,
                    "complaint_url": href,
                    "discovered_page": page_num,
                    "collected_at": now,
                })

        log(f"      -> {len(cards)} card(s), {new_count} new")
        last_page = page_num

        if new_count == 0:
            log("      -> no new complaints on this page, stopping pagination")
            break

        page_num += 1
        polite_sleep(delay)

    return rows, last_page, blocked


# --------------------------------------------------------------------------
# Entity matching (Selectum disambiguation)
# --------------------------------------------------------------------------

_TR_MAP = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u"})


def _normalize(text: str) -> str:
    text = (text or "").lower().translate(_TR_MAP)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text


def entity_match(target: Target, complaint_url: str, title: str, body_text: str,
                  category: str, product_name: str) -> tuple:
    """Returns (status, reason)."""
    if not target.requires_entity_validation:
        return MATCHED, "Dedicated single-property source page"

    haystack = _normalize(" ".join([
        title or "", body_text or "", category or "", product_name or "", complaint_url or "",
    ]))

    for pattern in target.exclude_patterns:
        if _normalize(pattern) in haystack:
            return EXCLUDED, f"Matched exclusion term: '{pattern}'"

    for pattern in target.match_patterns:
        if _normalize(pattern) in haystack:
            return MATCHED, f"Matched explicit reference: '{pattern}'"

    for term in target.ambiguous_terms:
        if _normalize(term) in haystack:
            return REVIEW_REQUIRED, f"Only generic/ambiguous reference found: '{term}'"

    return REVIEW_REQUIRED, "No hotel-name evidence found in title/text/URL"


# --------------------------------------------------------------------------
# Complaint detail parsing
# --------------------------------------------------------------------------

_DATE_RE = re.compile(
    r"\d{1,2}\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)"
    r"(\s+\d{4})?\s+\d{1,2}:\d{2}"
)


def _extract_date(text: str) -> str:
    if not text:
        return ""
    m = _DATE_RE.search(text)
    return m.group(0) if m else text.strip()


_TR_MONTHS = {
    "ocak": 1, "şubat": 2, "mart": 3, "nisan": 4, "mayıs": 5, "haziran": 6,
    "temmuz": 7, "ağustos": 8, "eylül": 9, "ekim": 10, "kasım": 11, "aralık": 12,
}
_DATE_PARSE_RE = re.compile(
    r"(?P<day>\d{1,2})\s+(?P<month>\w+)(?:\s+(?P<year>\d{4}))?\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})"
)


def parse_display_date(raw: str, assume_year: int = None):
    """Best-effort parse of a Sikayetvar display date ('22 Ağustos 13:45' or
    '21 Aralık 2023 11:40') into a sortable (year, month, day, hour, minute)
    tuple, for computing chronological min/max in reports. Returns None if
    unparseable. Complaints with no year shown are assumed to be from the
    current year the page was scraped (Sikayetvar omits the year for
    same-year dates)."""
    if not raw:
        return None
    m = _DATE_PARSE_RE.search(raw)
    if not m:
        return None
    month = _TR_MONTHS.get(m.group("month").lower())
    if month is None:
        return None
    year = int(m.group("year")) if m.group("year") else (assume_year or datetime.now().year)
    return (year, month, int(m.group("day")), int(m.group("hour")), int(m.group("minute")))


def _extract_view_count(header_div) -> str:
    if header_div is None:
        return ""
    use = header_div.find("use", href=re.compile("ic-view"))
    if not use:
        return ""
    span = use.find_parent("span")
    sibling_span = span.find("span") if span else None
    return sibling_span.get_text(strip=True) if sibling_span else ""


def _extract_support_count(article) -> str:
    button = article.find(attrs={"data-ga-element": "Engagement_Card_Upvote"})
    if not button:
        return ""
    m = re.search(r"\d+", button.get_text(" ", strip=True))
    return m.group(0) if m else ""


def _is_brand_message(message_div, company_hrefs: set) -> bool:
    if message_div.find(attrs={"data-ga-element": "Complaint_Answer_Brand"}):
        return True
    # Fallback signal: the message author link points at the company's own
    # profile slug (derived from the target's configured source pages, not
    # hard-coded to any single brand).
    for a in message_div.find_all("a", href=True):
        if a["href"] in company_hrefs:
            return True
    return False


def parse_complaint_detail(soup: BeautifulSoup, complaint_url: str, company_hrefs: set = frozenset()) -> tuple:
    """Returns (fields_dict, replies_list). Best-effort: any field that
    can't be found is left empty rather than guessed."""
    fields = {k: "" for k in COMPLAINTS_FIELDS}
    fields["complaint_url"] = complaint_url
    fields["complaint_id"] = urlsplit(complaint_url).path.strip("/").split("/")[-1]
    fields["company_response_exists"] = False
    fields["progress_exists"] = False
    fields["user_reply_count"] = 0
    replies = []

    article = soup.find("article")
    if article is None:
        return fields, replies

    h1 = soup.find("h1")
    if h1:
        # Some titles embed a "Brand_Name_In_Title" link before the actual
        # title text (e.g. "Selectum Hotels" + "Aldığımız Hizmetten...");
        # the real title text lives in a plain <span> sibling. Fall back to
        # the whole h1 text (space-joined) if that structure isn't there.
        title_span = h1.find("span")
        if title_span:
            fields["complaint_title"] = title_span.get_text(strip=True)
        else:
            fields["complaint_title"] = h1.get_text(" ", strip=True)

    content_end = article.find(attrs={"data-ga-element": "Complaint_Content_End"})
    body_div = None
    header_div = None
    if content_end is not None:
        body_div = content_end.find_previous_sibling(
            lambda t: t.name == "div" and "selection-share" in (t.get("class") or [])
        )
        if body_div is not None:
            header_div = body_div.find_previous_sibling("div")

    if body_div is not None:
        paragraphs = body_div.find_all("p")
        if paragraphs:
            fields["complaint_text"] = "\n".join(
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            )
        else:
            fields["complaint_text"] = body_div.get_text("\n", strip=True)

    if header_div is not None:
        fields["complaint_date_raw"] = _extract_date(header_div.get_text(" ", strip=True))
        fields["view_count"] = _extract_view_count(header_div)

    fields["support_count"] = _extract_support_count(article)

    # Breadcrumb -> category / product (rarely populated for hotels, which
    # have no product hierarchy, but kept for schema parity with the
    # reference Beko script).
    crumbs = [
        a.get_text(strip=True)
        for a in soup.find_all("a", attrs={"data-ga-element": "Breadcrumb_Link"})
    ]
    if len(crumbs) >= 4:
        fields["category"] = crumbs[-2]
        fields["product_name"] = crumbs[-1]
    elif len(crumbs) == 3:
        fields["category"] = crumbs[-1]

    # Replies (brand + user), in document order.
    messages_container = soup.find(id="messages")
    if messages_container is not None:
        message_divs = messages_container.find_all(
            "div", id=lambda x: x and x.startswith("message-")
        )
        user_reply_seen = False
        for order, message_div in enumerate(message_divs, start=1):
            is_brand = _is_brand_message(message_div, company_hrefs)
            msg_p = message_div.find("p")
            msg_text = msg_p.get_text(" ", strip=True) if msg_p else ""
            header_text = message_div.find("div", class_="min-w-0")
            msg_date = _extract_date(header_text.get_text(" ", strip=True)) if header_text else ""

            author_type = "COMPANY" if is_brand else "USER"
            replies.append({
                "complaint_id": fields["complaint_id"],
                "complaint_url": complaint_url,
                "reply_order": order,
                "reply_author_type": author_type,
                "reply_date": msg_date,
                "reply_text": msg_text,
            })

            if is_brand and not fields["company_response_exists"]:
                fields["company_response_exists"] = True
                fields["company_response_date"] = msg_date
                fields["company_response_text"] = msg_text
            elif not is_brand:
                fields["user_reply_count"] += 1
                if not user_reply_seen:
                    user_reply_seen = True
                    fields["first_user_reply_date"] = msg_date
                    fields["first_user_reply_text"] = msg_text

    return fields, replies


# --------------------------------------------------------------------------
# CSV / resume helpers
# --------------------------------------------------------------------------

def read_existing_values(csv_path: str, column: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        return {row[column] for row in csv.DictReader(f) if row.get(column)}


def append_rows(csv_path: str, fieldnames: list, rows: list) -> None:
    if not rows:
        return
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        f.flush()


def load_status(csv_path: str) -> dict:
    status = {}
    if not os.path.exists(csv_path):
        return status
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            status[(row["hotel_name"], row["source_page"])] = row
    return status


def update_status(csv_path: str, updates: dict) -> None:
    """updates: {(hotel_name, source_page): {field: value, ...}}"""
    status = load_status(csv_path)
    now = datetime.now(timezone.utc).isoformat()
    for key, fields_update in updates.items():
        row = status.get(key) or {f: "" for f in STATUS_FIELDS}
        row["hotel_name"], row["source_page"] = key
        row.update(fields_update)
        row["last_updated"] = now
        status[key] = row

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        for row in status.values():
            writer.writerow({k: row.get(k, "") for k in STATUS_FIELDS})
