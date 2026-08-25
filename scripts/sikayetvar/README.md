# Sikayetvar all-hotel pipeline

Discovers, maps, and scrapes Sikayetvar.com complaints for the project's
192 Bodrum hotels. This is a **data collection** pipeline only -- no
cleaning, sentiment, TF-IDF, topic modeling, or classification happens
here (that's `notebooks/11_sikayetvar_all_hotels_audit_cleaning.ipynb`
onward).

## Pipeline stages

| # | Script | Input | Output |
|---|--------|-------|--------|
| 01 | `01_discover_hotels.py` | `data/processed/hotels_enriched.csv` | `sikayetvar_mapping_candidates.csv`, `sikayetvar_hotel_mapping.csv` |
| 02 | `02_review_discovery_matches.py` | mapping + candidates | `reports/sikayetvar_all_hotels_manual_review.csv`, mapping summary reports |
| 03 | `03_collect_complaint_links.py` | mapping | `sikayetvar_all_complaint_links.csv` |
| 04 | `04_collect_complaint_details.py` | links | `sikayetvar_all_hotels_complaints_raw.csv`, `sikayetvar_all_hotels_replies_raw.csv` |
| 05 | `05_validate_all_hotels.py` | everything above | validation + coverage reports |

`run_all_hotels_pipeline.py` runs 01 -> 02 -> 03 -> 04 -> 05 in sequence.
Raw outputs live under `data/raw/sikayetvar/`; every human-facing report
lives under `reports/`.

## Discovery method (why there's no "search")

Sikayetvar's `/sikayetler?k=...` endpoint looks like a search box but, when
probed directly, turned out to be a noisy "trending brands" fallback page
(a query for "selectum" returned Uber, Skoda and MINI mixed in with the
one or two actually-relevant hits) -- not a real full-text index. So the
**primary** discovery method is direct slug guessing: Sikayetvar company
slugs are near-verbatim slugified hotel names
(`Rixos Premium Bodrum` -> `/rixos-premium-bodrum`), confirmed real by
checking that the response doesn't redirect to the `/sikayetler` fallback.
The `/sikayetler?k=` page is used only as a last-resort **secondary**
signal, filtered to links sharing a name token with the hotel -- and even
then, a match found this way can never auto-accept (see below).

Chain/umbrella accounts whose slug bears no resemblance to the property
name (e.g. Selectum Colours Bodrum lives under
`/selectum-hotels/selectum-colours`) cannot be found by guessing at all;
those are seeded in `config/sikayetvar_manual_aliases.json`.

## Mapping statuses

`FOUND_EXACT` / `FOUND_HIGH_CONFIDENCE` -- auto-accepted for complaint
scraping. Both require the match to have come from a **reliable** method
(a direct slug guess or a seeded manual alias) -- a match found only via
the `/sikayetler?k=` fallback is capped at `REVIEW_REQUIRED` no matter how
high its score, because whole-string name similarity is easy to fool: an
early full run found "Mira Beach Resort Bodrum" scoring 0.89 against
"**Amilla** Beach Resort Bodrum" purely off the shared filler words ("beach
resort bodrum"). `detect_negative_conflict` now also independently checks
that the hotel's own primary brand word literally appears in the candidate
title; see `tests/test_sikayetvar_discovery.py`'s
`test_conflict_when_own_brand_word_missing_from_candidate` for that
regression case.

`REVIEW_REQUIRED` / `AMBIGUOUS` -- not scraped by default (`--allow-review-required`
overrides this, but that's a conscious per-run choice, never a default).
`NOT_FOUND` -- no candidate resolved to a real page at all.
`PAGE_FOUND_NO_COMPLAINT` -- a real page was found but shows 0 complaints.

## Chain / shared-page entity validation

A Sikayetvar company page can host complaints about more than one physical
property (Selectum's shared `/selectum-hotels/...` account is the proven
case; La Blanche Island Bodrum vs. La Blanche **Resort** Bodrum is another
sibling pair in this same 192-hotel dataset). `03` crawls each *unique*
Sikayetvar URL once; when more than one project hotel maps to it, `04`
validates every complaint separately against **each** sharing hotel's own
match/exclude terms (auto-derived in
`sikayetvar_matching.build_complaint_validation_terms` from sibling hotel
names -- no brand name is hard-coded anywhere in this pipeline). A
dedicated single-property page skips this step entirely and everything on
it is trusted (`requires_validation=False`).

Per-complaint outcomes: `COMPLAINT_MATCHED`, `COMPLAINT_REVIEW_REQUIRED`,
`COMPLAINT_EXCLUDED_OTHER_PROPERTY`. Only `COMPLAINT_MATCHED` rows get
reply rows written; all three statuses are kept in the raw complaints CSV
(nothing is silently dropped).

## Running it

```bash
# Smoke test: 10 hotels, 1 page each, 20 complaints max
python3 scripts/sikayetvar/run_all_hotels_pipeline.py --smoke-test

# Discovery + review report only, no complaint scraping (section 55: dry run)
python3 scripts/sikayetvar/run_all_hotels_pipeline.py --dry-run

# Full run
python3 scripts/sikayetvar/run_all_hotels_pipeline.py

# Individual stages, with resume
python3 scripts/sikayetvar/01_discover_hotels.py --resume
python3 scripts/sikayetvar/03_collect_complaint_links.py --max-pages 1
python3 scripts/sikayetvar/04_collect_complaint_details.py --limit 20
```

Requires Python >=3.10 (the project's `pyproject.toml` floor); this
machine's default `python3` is 3.9, so these scripts were developed and
run against `/opt/homebrew/bin/python3.11`.

## Resume / checkpoints

- `01`: `--resume` skips hotel_ids already in the mapping table.
- `03`: a source page already present in the links CSV is not re-crawled
  (pass `--force` to override).
- `04`: a `(hotel_id, canonical_complaint_url)` pair already in the raw
  complaints CSV is not re-evaluated; a complaint_url with nothing left to
  resolve is not re-fetched from the network at all.

Every stage appends/flushes as it goes, so a killed run keeps its progress.

## Anti-bot / safe stop

No CAPTCHA or block page was encountered scraping this site (verified via
plain `requests` across discovery + the full 3-hotel pilot + this all-hotel
run); `AntiBotBlock` exists as a guard in case that changes. On trigger,
every stage saves whatever it collected so far, marks the affected
hotel(s) `BLOCKED_SAFE_STOP` in the status CSV, and stops -- no proxy
rotation, no CAPTCHA solving, no stealth fingerprinting is attempted.

## Known limitations

- **Discovery only tries direct slug guesses + the noisy search fallback.**
  Chain umbrella accounts not already seeded in
  `sikayetvar_manual_aliases.json` and not guessable from the property name
  will show up `NOT_FOUND` even if a page exists under an unrelated slug.
  `reports/sikayetvar_all_hotels_manual_review.csv` is where a human closes
  that gap.
- **`support_count` is frequently empty.** Sikayetvar's "Destekle" button
  doesn't render a count in server-rendered HTML in the pages sampled;
  left blank (never fabricated as `0`) per section 24 of the spec.
- **`progress_exists` (the "Gelişme" / resolution-update feature) could not
  be verified against a live example** during development; the parser
  looks for it but its DOM structure is unconfirmed.
- **Complaint-level chain validation only knows about *sibling project
  hotels already in this 192-hotel dataset.*** A shared page could still
  carry complaints about a same-brand property entirely outside Bodrum
  (not a project hotel at all); those complaints fall to
  `COMPLAINT_REVIEW_REQUIRED` rather than being confidently excluded,
  since guessing at non-project brand names generically was judged too
  risky (better a human looks at it than a wrong auto-exclude).

## Data ethics

No email, phone, address, or member-profile URL is collected -- only the
complaint text, dates, view/support counts, and company responses as
published on the complaint page itself (section 76 of the spec). Only
public pages are fetched; nothing behind login is accessed (section 77).
