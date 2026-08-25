ŞİKAYETVAR SEVERITY / ESCALATION FEATURES

Source: notebook 21, built on notebook 12–14's canonical 236-row clean/NLP corpus (32 matched hotels).
What it measures: linguistic escalation intensity, not event severity or star rating — Şikayetvar has no rating, so a plain positive/negative sentiment score would not discriminate (every row is already a complaint).
Method: rule-based, explainable keyword lexicon (HIGH: legal/health/fraud/threat language; MEDIUM: explicit strong dissatisfaction), same word-boundary matching as the aspect detector. No black-box model, no new labeled ground truth.
Tiers: HIGH takes priority over MEDIUM when both keyword sets match; BASELINE = aspect complaint with no escalation language detected.
Distribution: HIGH 33/236 (%14.0), MEDIUM 132/236 (%55.9), BASELINE 71/236 (%30.1).
Company response relationship: HIGH-tier complaints get a company response slightly more often (%21.2) than MEDIUM (%15.2) or BASELINE (%16.9) — correlational, not causal.
Aspect relationship: SAFETY_SECURITY has the highest HIGH share (%30.8); NOISE has none (%0) — see reports/sikayetvar_severity_by_aspect.csv.
Not validated against independent human labels — same methodological caveat as the aspect taxonomy (TOPIC_MODEL_NOT_RELIABLE decision in notebook 14).
Not produced: hotel- or area-level severity breakdowns (sample too small to be reliable at that granularity); only aspect-level and corpus-level tables are reported.
Files: reports/sikayetvar_severity_distribution.csv, reports/sikayetvar_severity_by_aspect.csv, reports/sikayetvar_severity_dictionary.csv (keyword audit), data/processed/sikayetvar_complaints_severity.csv (complaint-level tier).
