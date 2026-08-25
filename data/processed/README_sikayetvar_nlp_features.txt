ŞİKAYETVAR NLP FEATURES

Corpus definition: Notebook 12 canonical-unique reliably matched clean complaints; 236 rows, 229 non-empty texts.
Preprocessing: NFKC, Turkish lowercase, whitespace/punctuation cleanup, URL/email/phone masking in derived NLP text; raw text unchanged.
Taxonomy: 18 explainable multi-label aspects; phrase-first, word-boundary and limited Turkish surface-form rules.
Mention rate: unique complaints matching aspect / group clean complaint count * 100. It is not a real customer problem rate.
Hotel aggregation: one row per 32 complaint-bearing hotels; sample reliability uses Notebook 13 HIGH/MEDIUM/LOW tiers.
Area aggregation: all 14 project areas retained; areas without clean complaint data have missing aspect rates, not zero.
Small-n handling: hotel n<5 and area n<10 excluded from main heatmaps but retained with flags.
Company response: existence/rate/time are operational descriptors and do not imply resolution.
Missing coverage policy: hotels without trusted complaint data must not receive aspect rate=0; use data-availability and mapping-status indicators downstream.
Limitations: self-selected platform, unequal mapping coverage, Turkish morphology, rule-based errors, multi-label overlap, incomplete official metadata, non-causal correlations.
Intended use: coverage-aware segmentation/opportunity features with reliability indicators; not hotel quality scoring.
