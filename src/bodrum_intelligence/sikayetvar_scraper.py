"""Sikayetvar.com complaint listing/detail scraping primitives.

Generalized from a validated 3-hotel pilot (La Blanche Island Bodrum, Green
Bay Resort & Spa Bodrum, Selectum Colours Bodrum). No brand/hotel slug is
hard-coded here: callers pass in the source page and (for shared/chain
pages) the match/exclude terms to validate each complaint against.

Verified against the live site (2026-08-25): listing pages, pagination
(?page=N) and complaint detail content (including brand replies) are
server-rendered as static HTML, so requests + BeautifulSoup is used
throughout -- no Selenium/browser automation needed. A CAPTCHA/challenge
guard (`AntiBotBlock`) exists in case that changes; on trigger, callers are
expected to save partial progress and stop rather than working around it.
"""
from __future__ import annotations

import csv
import os
import re
import time
import unicodedata
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

MATCHED = "COMPLAINT_MATCHED"
REVIEW_REQUIRED = "COMPLAINT_REVIEW_REQUIRED"
EXCLUDED = "COMPLAINT_EXCLUDED_OTHER_PROPERTY"


class AntiBotBlock(Exception):
    """Raised when the site appears to show a CAPTCHA / access-block page."""


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


def fetch_soup(session: requests.Session, url: str, params: dict = None) -> BeautifulSoup:
    """GET a page and return parsed HTML, or raise AntiBotBlock if the site
    appears to be showing a challenge/block page."""
    response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    if response.status_code in (403, 429, 503):
        raise AntiBotBlock(f"HTTP {response.status_code} on {response.url}")
    lowered = response.text[:4000].lower()
    if any(marker in lowered for marker in _BLOCK_MARKERS):
        raise AntiBotBlock(f"Block/challenge page detected on {response.url}")
    return BeautifulSoup(response.text, "html.parser")


def fetch_response(session: requests.Session, url: str, params: dict = None) -> requests.Response:
    """Like fetch_soup but returns the raw Response (used by discovery,
    which needs status_code/final url, not just parsed HTML)."""
    response = session.get(url, params=params, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    if response.status_code in (403, 429):
        raise AntiBotBlock(f"HTTP {response.status_code} on {response.url}")
    lowered = response.text[:4000].lower()
    if any(marker in lowered for marker in _BLOCK_MARKERS):
        raise AntiBotBlock(f"Block/challenge page detected on {response.url}")
    return response


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
    return urlunsplit(("https", "www.sikayetvar.com", path, "", ""))


def listing_page_url(source_page: str, page_num: int) -> str:
    if page_num <= 1:
        return source_page
    return f"{source_page}?page={page_num}"


def source_page_prefix(source_page: str) -> str:
    """The site renders 'related complaint' cards from *other* companies on
    thin/near-empty listing pages. Complaint cards are only trustworthy if
    their href starts with the source page's own first path segment, e.g.
    '/green-bay-resort/' or '/selectum-hotels/' (a shared prefix for every
    property under that umbrella account, disambiguated later per-complaint
    via entity_match)."""
    first_segment = urlsplit(source_page).path.strip("/").split("/")[0]
    return f"/{first_segment}/"


# --------------------------------------------------------------------------
# Listing page parsing / link discovery
# --------------------------------------------------------------------------

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
        if href.count("/") < 2:
            continue
        if required_prefix and not href.startswith(required_prefix):
            continue
        results.append((canonicalize_url(href), link.get("title", "").strip()))
    return results


def paginate_listing(session: requests.Session, source_page: str, max_pages=None,
                      delay: float = REQUEST_DELAY_SECONDS, log=print) -> tuple:
    """Paginate a single source page until a page yields no new unique
    complaint URLs (or max_pages is hit). Returns (links, last_page, blocked)
    where links is [(complaint_url, title, discovered_page), ...]."""
    links = []
    seen_urls = set()
    page_num = 1
    last_page = 0
    blocked = False
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
            log(f"    [BLOCKED] {exc} -- stopping this source page, keeping links collected so far")
            blocked = True
            break

        cards = extract_complaint_cards(soup, required_prefix=required_prefix)
        new_count = 0
        for href, title in cards:
            if href not in seen_urls:
                seen_urls.add(href)
                new_count += 1
                links.append((href, title, page_num))

        log(f"      -> {len(cards)} card(s), {new_count} new")
        last_page = page_num
        if new_count == 0:
            log("      -> no new complaints on this page, stopping pagination")
            break

        page_num += 1
        polite_sleep(delay)

    return links, last_page, blocked


# --------------------------------------------------------------------------
# Complaint-level entity matching (chain / shared-page disambiguation)
# --------------------------------------------------------------------------

_TR_MAP = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u"})


def normalize_for_match(text: str) -> str:
    text = (text or "").lower().translate(_TR_MAP)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text


def entity_match(
    complaint_url: str, title: str, body_text: str, category: str, product_name: str,
    match_patterns: list, exclude_patterns: list, ambiguous_terms: list,
    requires_validation: bool = True,
) -> tuple:
    """Returns (status, reason). If requires_validation is False (dedicated,
    single-property source page), every complaint is auto-MATCHED."""
    if not requires_validation:
        return MATCHED, "Dedicated single-property source page"

    haystack = normalize_for_match(" ".join([
        title or "", body_text or "", category or "", product_name or "", complaint_url or "",
    ]))

    for pattern in exclude_patterns:
        if normalize_for_match(pattern) in haystack:
            return EXCLUDED, f"Matched exclusion term: '{pattern}'"

    for pattern in match_patterns:
        if normalize_for_match(pattern) in haystack:
            return MATCHED, f"Matched explicit reference: '{pattern}'"

    for term in ambiguous_terms:
        if normalize_for_match(term) in haystack:
            return REVIEW_REQUIRED, f"Only generic/ambiguous reference found: '{term}'"

    return REVIEW_REQUIRED, "No hotel-name evidence found in title/text/URL"


# --------------------------------------------------------------------------
# Complaint detail parsing
# --------------------------------------------------------------------------

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
    tuple. Returns None if unparseable. Dates with no year shown are
    assumed to be from the current year (the site omits same-year years)."""
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


def _extract_date(text: str) -> str:
    if not text:
        return ""
    m = _DATE_PARSE_RE.search(text)
    return m.group(0) if m else text.strip()


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
    """Returns '' (not '0') when the count isn't present in static HTML --
    the "Destekle" button doesn't render a number server-side in the
    samples checked, so 0 vs "not shown" must stay distinguishable."""
    button = article.find(attrs={"data-ga-element": "Engagement_Card_Upvote"})
    if not button:
        return ""
    m = re.search(r"\d+", button.get_text(" ", strip=True))
    return m.group(0) if m else ""


def _is_brand_message(message_div, company_hrefs: set) -> bool:
    if message_div.find(attrs={"data-ga-element": "Complaint_Answer_Brand"}):
        return True
    for a in message_div.find_all("a", href=True):
        if a["href"] in company_hrefs:
            return True
    return False


COMPLAINT_DETAIL_FIELDS = [
    "complaint_id", "complaint_url", "complaint_title", "complaint_text",
    "complaint_date_raw", "view_count", "support_count", "category", "product_name",
    "company_response_exists", "company_response_date", "company_response_text",
    "progress_exists", "progress_date", "progress_text",
    "user_reply_count", "first_user_reply_date", "first_user_reply_text",
]


def parse_complaint_detail(soup: BeautifulSoup, complaint_url: str, company_hrefs: set = frozenset()) -> tuple:
    """Returns (fields_dict, replies_list). Best-effort: any field that
    can't be found is left empty rather than guessed."""
    fields = {k: "" for k in COMPLAINT_DETAIL_FIELDS}
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
        # title text; the real title lives in a plain <span> sibling.
        title_span = h1.find("span")
        fields["complaint_title"] = (
            title_span.get_text(strip=True) if title_span else h1.get_text(" ", strip=True)
        )

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
        fields["complaint_text"] = (
            "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            if paragraphs else body_div.get_text("\n", strip=True)
        )

    if header_div is not None:
        fields["complaint_date_raw"] = _extract_date(header_div.get_text(" ", strip=True))
        fields["view_count"] = _extract_view_count(header_div)

    fields["support_count"] = _extract_support_count(article)

    crumbs = [
        a.get_text(strip=True)
        for a in soup.find_all("a", attrs={"data-ga-element": "Breadcrumb_Link"})
    ]
    if len(crumbs) >= 4:
        fields["category"] = crumbs[-2]
        fields["product_name"] = crumbs[-1]
    elif len(crumbs) == 3:
        fields["category"] = crumbs[-1]

    messages_container = soup.find(id="messages")
    if messages_container is not None:
        message_divs = messages_container.find_all("div", id=lambda x: x and x.startswith("message-"))
        user_reply_seen = False
        for order, message_div in enumerate(message_divs, start=1):
            is_brand = _is_brand_message(message_div, company_hrefs)
            msg_p = message_div.find("p")
            msg_text = msg_p.get_text(" ", strip=True) if msg_p else ""
            header_text = message_div.find("div", class_="min-w-0")
            msg_date = _extract_date(header_text.get_text(" ", strip=True)) if header_text else ""

            replies.append({
                "complaint_id": fields["complaint_id"],
                "canonical_complaint_url": complaint_url,
                "reply_order": order,
                "reply_author_type": "COMPANY" if is_brand else "USER",
                "reply_date_raw": msg_date,
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

def read_csv_rows(csv_path: str) -> list:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_existing_values(csv_path: str, column: str) -> set:
    return {row[column] for row in read_csv_rows(csv_path) if row.get(column)}


def append_rows(csv_path: str, fieldnames: list, rows: list) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        f.flush()


def write_csv(csv_path: str, fieldnames: list, rows: list) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
