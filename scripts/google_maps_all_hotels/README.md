# Google Maps all-hotels pipeline

This isolated pipeline uses only public, rendered Google Maps pages. It does not use
private endpoints, login bypass, CAPTCHA solving, proxy rotation, or stealth methods.
A challenge page triggers `BLOCKED_SAFE_STOP`, saves checkpoints, and ends the run.

Run order:

```bash
python3 scripts/google_maps_all_hotels/run_all_hotels_reviews.py --targets-only
python3 scripts/google_maps_all_hotels/run_all_hotels_reviews.py --smoke-test --max-reviews-per-hotel 10
python3 scripts/google_maps_all_hotels/run_all_hotels_reviews.py --resume
```

Useful scoped runs:

```bash
python3 scripts/google_maps_all_hotels/run_all_hotels_reviews.py --dry-run --max-hotels 10
python3 scripts/google_maps_all_hotels/run_all_hotels_reviews.py --hotel-id BOD001 --resume
python3 scripts/google_maps_all_hotels/run_all_hotels_reviews.py --area Akyarlar --resume
python3 scripts/google_maps_all_hotels/run_all_hotels_reviews.py --process-only
```

The default cap is 75 reviews per hotel. Collection is sequential with a randomized
2–5 second delay. `--force` is false by default and creates timestamped backups before
replacing per-hotel checkpoints. The existing five-hotel case-study data and helpers
are read-only inputs and are never overwritten. If anonymous Maps exposes only its
limited view, those five existing Google Travel corpora may be imported into the new
schema with explicit `read_only_case_study_import` batch provenance. This fallback is
never presented as a newly scraped Maps sample and cannot make the 10-hotel smoke gate
pass for the rest of the master list.
