"""Score Sikayetvar discovery candidates against project hotels and derive
per-hotel complaint-level validation terms for chain/shared pages.

Reuses `bodrum_intelligence.hotel_matching`'s Turkish-safe name
normalization and similarity so this pipeline's notion of "the same name"
stays consistent with the official-facility matching already validated in
`06_hotel_attributes_match_audit.ipynb`.

Scoring note (see module docstring in sikayetvar_discovery.py): unlike the
official-facility matching problem, Sikayetvar company pages essentially
never expose phone/address, so the initial weights below lean on
name+area/Bodrum evidence. This is a starting point, not a final answer --
tune `MatchThresholds`/`CANDIDATE_WEIGHTS` after inspecting the real score
distribution from a discovery run (see 02_review_discovery_matches.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from bodrum_intelligence.hotel_matching import (
    calculate_name_similarity, normalize_hotel_name,
)

FOUND_EXACT = "FOUND_EXACT"
FOUND_HIGH_CONFIDENCE = "FOUND_HIGH_CONFIDENCE"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
AMBIGUOUS = "AMBIGUOUS"
NOT_FOUND = "NOT_FOUND"
PAGE_FOUND_NO_COMPLAINT = "PAGE_FOUND_NO_COMPLAINT"
EXCLUDED_WRONG_ENTITY = "EXCLUDED_WRONG_ENTITY"
ERROR = "ERROR"

AUTO_ACCEPTED_STATUSES = {FOUND_EXACT, FOUND_HIGH_CONFIDENCE}

CANDIDATE_WEIGHTS = {"name": 0.70, "bodrum": 0.15, "area": 0.15}

# Generic filler words dropped when deriving a hotel's "brand" token(s) for
# sibling-conflict detection (kept in sync with sikayetvar_discovery's
# _DROP_SUFFIX_WORDS, but this set also drops "bodrum"/"island" since those
# are geography, not brand, and would falsely link e.g. every "Bodrum"
# hotel together).
_BRAND_STOPWORDS = {
    "hotel", "otel", "resort", "spa", "suites", "suite", "club", "boutique",
    "beach", "residence", "bodrum", "island", "adasi", "the", "by",
}


@dataclass(frozen=True)
class MatchThresholds:
    """Initial thresholds -- see module docstring. Tune after a real
    discovery run (section 34 of the spec: never hardcode blindly)."""

    exact_min_name_similarity: float = 0.90
    high_confidence_min_score: float = 0.75
    review_required_min_score: float = 0.45
    ambiguous_gap: float = 0.05


def brand_tokens(hotel_name: str) -> list:
    """The distinctive (non-generic, non-geography) tokens of a hotel name,
    in original order, used to find sibling/collision candidates within the
    project dataset and to derive a single "chain root alone" ambiguous
    term (see build_complaint_validation_terms)."""
    _, core = normalize_hotel_name(hotel_name)
    if not core:
        return []
    seen = []
    for t in core.split():
        if t not in _BRAND_STOPWORDS and t not in seen:
            seen.append(t)
    return seen


def score_candidate(hotel_name: str, candidate_title: str,
                     bodrum_evidence: bool, area_evidence: bool) -> tuple:
    """Returns (name_similarity, score). `candidate_title` is the
    Sikayetvar page's own <title>/company-name text."""
    _, hotel_core = normalize_hotel_name(hotel_name)
    _, title_core = normalize_hotel_name(candidate_title)
    name_similarity = calculate_name_similarity(hotel_core, title_core)
    score = (
        CANDIDATE_WEIGHTS["name"] * name_similarity
        + CANDIDATE_WEIGHTS["bodrum"] * float(bodrum_evidence)
        + CANDIDATE_WEIGHTS["area"] * float(area_evidence)
    )
    return round(name_similarity, 4), round(min(score, 1.0), 4)


def detect_negative_conflict(hotel_name: str, candidate_title: str,
                              other_project_hotel_names: list) -> tuple:
    """Checks whether `candidate_title` looks like it actually belongs to a
    *different* property. Two independent signals:

    1. The hotel's own primary brand token (e.g. "mira") is nowhere in the
       candidate title at all -- catches false positives where generic
       shared words ("beach resort bodrum") inflate whole-string
       similarity despite the actual distinguishing brand word differing
       entirely (observed: "Mira Beach Resort Bodrum" scoring 0.89 against
       "Amilla Beach Resort Bodrum" purely off the shared filler words).
    2. The candidate title matches a *different* project hotel that shares
       this hotel's brand token(s) better than it matches this one (the
       Selectum Colours Bodrum / Selectum Collection Bodrum case).

    Returns (conflict: bool, reason: str)."""
    this_brand = brand_tokens(hotel_name)
    _, title_core = normalize_hotel_name(candidate_title)
    if not this_brand or not title_core:
        return False, ""

    if this_brand[0] not in title_core.split():
        return True, f"Hotel's own brand word '{this_brand[0]}' does not appear in candidate title at all"

    this_sim = calculate_name_similarity(normalize_hotel_name(hotel_name)[1], title_core)
    for other_name in other_project_hotel_names:
        if other_name == hotel_name:
            continue
        other_brand = brand_tokens(other_name)
        if not (set(other_brand) & set(this_brand)):
            continue
        other_sim = calculate_name_similarity(normalize_hotel_name(other_name)[1], title_core)
        if other_sim > this_sim + 0.10:
            return True, f"Page title matches sibling project hotel better: '{other_name}'"
    return False, ""


def classify_candidate(
    name_similarity: float, score: float, reliable_method: bool, negative_conflict: bool,
    runner_up_score: float = 0.0, thresholds: MatchThresholds = MatchThresholds(),
) -> str:
    """`reliable_method` = the winning candidate came from a direct slug
    guess or a seeded manual alias (as opposed to the noisier
    /sikayetler?k= search fallback) -- not tied to which rank it was tried
    at, since a hotel can have several equally-valid known_urls/guesses."""
    if negative_conflict:
        return REVIEW_REQUIRED
    if score - runner_up_score < thresholds.ambiguous_gap and runner_up_score >= thresholds.review_required_min_score:
        return AMBIGUOUS
    if reliable_method and name_similarity >= thresholds.exact_min_name_similarity:
        return FOUND_EXACT
    # The /sikayetler?k= search fallback is a noisier signal (see
    # sikayetvar_discovery module docstring); never let it auto-accept on
    # score alone, however high -- cap it at REVIEW_REQUIRED so a human
    # confirms it (this is what caught the Mira/Amilla false positive).
    if reliable_method and score >= thresholds.high_confidence_min_score:
        return FOUND_HIGH_CONFIDENCE
    if score >= thresholds.review_required_min_score:
        return REVIEW_REQUIRED
    return NOT_FOUND


@dataclass
class ComplaintValidationTerms:
    match_patterns: list = field(default_factory=list)
    exclude_patterns: list = field(default_factory=list)
    ambiguous_terms: list = field(default_factory=list)
    requires_validation: bool = False


def build_complaint_validation_terms(
    hotel_id: str, hotel_name: str, sikayetvar_url: str,
    mapping_rows: list, manual_aliases: dict = None,
) -> ComplaintValidationTerms:
    """Builds the per-hotel match/exclude/ambiguous terms used by
    `sikayetvar_scraper.entity_match` for each collected complaint.

    A source page requires per-complaint validation whenever EITHER:
      (a) it is shared with at least one other *accepted* project-hotel
          mapping (a chain umbrella account with a sibling in this
          dataset -- the Selectum Colours/Collection case), or
      (b) the URL's own slug doesn't closely resemble the hotel's name at
          all (a multi-city chain umbrella account with NO sibling in this
          192-hotel dataset, e.g. Rixos Premium Bodrum's complaints living
          under the generic /rixos-hotels/ account alongside Rixos
          properties in other cities entirely -- there's no sibling to
          build exclude_patterns from, but match_patterns still requires
          the complaint to name this specific property, so validation
          still does real work here).
    A page whose slug basically *is* the hotel name (the common case) is
    trusted automatically.
    """
    manual_aliases = manual_aliases or {}
    siblings = [
        r for r in mapping_rows
        if r["sikayetvar_url"] == sikayetvar_url and r["hotel_id"] != hotel_id
    ]
    slug = sikayetvar_url.rstrip("/").split("/")[-1]
    slug_matches_hotel = calculate_name_similarity(
        normalize_hotel_name(hotel_name)[1], normalize_hotel_name(slug.replace("-", " "))[1]
    ) >= 0.75
    requires_validation = len(siblings) > 0 or not slug_matches_hotel

    alias_cfg = manual_aliases.get(hotel_id, {})
    match_patterns = list(dict.fromkeys(
        [hotel_name, normalize_hotel_name(hotel_name)[1] or hotel_name]
        + list(alias_cfg.get("aliases", []))
    ))
    exclude_patterns = list(dict.fromkeys(
        [s["hotel_name"] for s in siblings] + list(alias_cfg.get("exclude_aliases", []))
    ))
    # A mention of just the chain root (e.g. "Selectum" alone, without the
    # property-distinguishing modifier "Colours") is evidence but not proof;
    # only meaningful when there's a modifier to omit in the first place.
    this_brand = brand_tokens(hotel_name)
    ambiguous_terms = [this_brand[0]] if len(this_brand) >= 2 else []

    return ComplaintValidationTerms(
        match_patterns=match_patterns,
        exclude_patterns=exclude_patterns,
        ambiguous_terms=ambiguous_terms,
        requires_validation=requires_validation,
    )
