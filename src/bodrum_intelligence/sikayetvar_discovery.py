"""Sikayetvar.com hotel entity discovery: turn a project hotel_name into
candidate Sikayetvar company-page slugs and check which ones are real.

Discovery method (see scripts/sikayetvar/README.md for the full writeup):
Sikayetvar's `/sikayetler?k=...` endpoint was probed and found to be a noisy
generic "trending brands" fallback rather than a real full-text search index
(confirmed empirically: a query for "selectum" returned unrelated footer
brands like Uber/Skoda/MINI mixed with a couple of relevant hits), so it is
used only as a *secondary*, filtered signal -- not the primary method.

The primary, reliable method is direct slug guessing: Sikayetvar company
slugs are near-verbatim slugified versions of the display name
("Rixos Premium Bodrum" -> /rixos-premium-bodrum). A guess is confirmed
real by checking the final response: a non-existent slug 302/308-redirects
to /sikayetler?k=<query> with an unrelated final URL, while a real page
stays on its own URL with a matching <title>. Chain/umbrella accounts whose
slug bears no resemblance to the property name (e.g. Selectum Colours
Bodrum living under /selectum-hotels/selectum-colours) cannot be found this
way; those must be seeded via config/sikayetvar_manual_aliases.json.
"""
from __future__ import annotations

import re
import unicodedata

from bodrum_intelligence.sikayetvar_scraper import (
    SITE_ROOT, canonicalize_url, fetch_response, fetch_soup,
)

_TR_MAP = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u"})

_DROP_SUFFIX_WORDS = {
    "hotel", "otel", "resort", "spa", "suites", "suite", "club",
    "boutique", "beach", "residence",
}


def slugify(text: str) -> str:
    text = (text or "").lower().translate(_TR_MAP)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def generate_slug_candidates(hotel_name: str, area: str = None, max_candidates: int = 5) -> list:
    """Ordered, de-duplicated list of candidate Sikayetvar slugs for a
    project hotel name (most-likely-correct first)."""
    candidates = []

    def add(slug: str):
        if slug and slug not in candidates:
            candidates.append(slug)

    full = slugify(hotel_name)
    add(full)

    tokens = full.split("-")
    trimmed = [t for t in tokens if t not in _DROP_SUFFIX_WORDS]
    if trimmed and trimmed != tokens:
        add("-".join(trimmed))

    if "bodrum" not in tokens:
        add(f"{full}-bodrum")

    no_ampersand = slugify((hotel_name or "").replace("&", ""))
    add(no_ampersand)

    if area:
        area_slug = slugify(area)
        if area_slug and area_slug not in tokens:
            add(f"{full}-{area_slug}")

    return candidates[:max_candidates]


def generate_search_queries(hotel_name: str, area: str = None, max_queries: int = 4) -> list:
    """Human-readable query variants, kept for the secondary /sikayetler?k=
    signal and for documenting what was tried in candidate rows."""
    queries = [hotel_name]
    if area and area.lower() not in hotel_name.lower():
        queries.append(f"{hotel_name} {area}")
    if "bodrum" not in hotel_name.lower():
        queries.append(f"{hotel_name} Bodrum")
    queries.append(f"{hotel_name} Şikayetvar")
    seen, out = set(), []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out[:max_queries]


_NOT_FOUND_PATH_MARKERS = ("/sikayetler", "/arama", "/search")


def check_slug(session, slug: str) -> dict:
    """GET https://www.sikayetvar.com/{slug} and report whether it's a real,
    accessible company page. Returns a dict with:
      exists, final_url, page_title, visible_complaint_count
    Never raises for a plain not-found; AntiBotBlock still propagates."""
    url = f"{SITE_ROOT}/{slug}"
    response = fetch_response(session, url)
    final_path = response.url.split("?")[0].rstrip("/")
    requested_path = url.rstrip("/")
    redirected_away = final_path != requested_path or any(
        marker in response.url for marker in _NOT_FOUND_PATH_MARKERS
    )
    if response.status_code >= 400 or redirected_away:
        return {"exists": False, "final_url": response.url, "page_title": "", "visible_complaint_count": ""}

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    if "sikayet" not in title.lower() and "şikayet" not in title.lower():
        # A 200 page that isn't a company/complaint page at all (e.g. a
        # static info page like /hakkimizda) is not a usable match.
        return {"exists": False, "final_url": response.url, "page_title": title,
                "company_name": "", "visible_complaint_count": ""}

    # Titles look like "{Company Name} Şikayet ve Yorumları - Şikayetvar" or
    # "{Company Name} Şikayetleri - Şikayetvar" for tag/filter pages. Strip
    # that boilerplate so name-similarity scoring compares names, not whole
    # sentences (comparing the raw title crushes the similarity score).
    company_name = re.split(r"\s+[Şş]ikayet", title, maxsplit=1)[0].strip()

    # The site's old `.complaint-count` badge no longer exists in the
    # current (Tailwind) template, so count the page's own complaint cards
    # directly instead -- filtered to hrefs actually belonging to this slug
    # (see source_page_prefix), since a near-empty legacy company page can
    # otherwise display *other* companies' "related complaint" cards and
    # look deceptively non-empty (observed on green-bay-resort). This also
    # catches the mirror case: a real, well-matched slug that turns out to
    # have zero complaints of its own because the chain actually posts
    # under a different shared account (observed: rixos-premium-bodrum is
    # a live page but every visible card belongs to /etstur/, /setur-turizm/
    # or the separate /rixos-hotels/ umbrella account) -- that distinction
    # feeds match_status PAGE_FOUND_NO_COMPLAINT in 01_discover_hotels.py.
    from bodrum_intelligence.sikayetvar_scraper import extract_complaint_cards, source_page_prefix
    own_cards = extract_complaint_cards(soup, required_prefix=source_page_prefix(f"{SITE_ROOT}/{slug}"))
    count_text = str(len(own_cards))

    # If this candidate has zero complaints of its own, the site's own
    # "related complaint" cards on that same (near-)empty page are a
    # useful lead: a chain often has a dead/legacy standalone slug
    # alongside the real active one under a slightly different name (e.g.
    # la-blanche-resort-bodrum -> real complaints live at
    # la-blanche-resort-**spa**-bodrum; green-bay-resort ->
    # green-bay-resort-**spa**). Surface those other slugs so the caller
    # can try them as extra candidates instead of giving up.
    related_slugs = []
    if not own_cards:
        from urllib.parse import urlsplit
        for other_href, _title in extract_complaint_cards(soup):
            # extract_complaint_cards already canonicalizes hrefs to full
            # URLs, so take the path's first segment, not the raw string.
            other_slug = urlsplit(other_href).path.strip("/").split("/")[0]
            if other_slug and other_slug != slug and other_slug not in related_slugs:
                related_slugs.append(other_slug)

    return {
        "exists": True,
        "final_url": canonicalize_url(response.url),
        "page_title": title,
        "company_name": company_name,
        "visible_complaint_count": count_text,
        "related_slugs": related_slugs,
    }


_GENERIC_NAV_SLUGS = {
    "hakkimizda", "iletisim", "cerez-politikasi", "kurumsal-uyelik", "rehber",
    "seffaflik-raporu-2025", "topluluk-kurallari", "trend-100", "tum-markalar",
    "tv", "uye-aydinlatma-metni", "uyelik-sozlesmesi", "giris", "uye-ol", "canli-yayin",
}


def search_fallback_candidates(session, query: str, hotel_core_tokens: set, max_results: int = 5) -> list:
    """Secondary discovery signal: fetch /sikayetler?k=<query> and keep only
    the company links whose slug shares a token with the hotel's own core
    name (filters out the generic trending-brand footer noise the endpoint
    otherwise mixes in -- see module docstring)."""
    soup = fetch_soup(session, f"{SITE_ROOT}/sikayetler", params={"k": query})
    results = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("/") or href.count("/") != 1:
            continue
        slug = href.strip("/")
        if not slug or slug in _GENERIC_NAV_SLUGS or slug in seen:
            continue
        slug_tokens = set(slug.split("-"))
        if not (slug_tokens & hotel_core_tokens):
            continue
        seen.add(slug)
        results.append({"slug": slug, "url": f"{SITE_ROOT}/{slug}", "link_text": a.get_text(strip=True)})
        if len(results) >= max_results:
            break
    return results
