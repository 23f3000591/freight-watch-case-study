# Freight cost watchdog

Spots routes whose cost per tonne-km is rising in a way that looks unusual, then checks the
context notes before deciding whether the rise is **justified** or **unexplained**.

```
pip install -r requirements.txt
python -m freight_watch.pipeline          # -> outputs/output.csv (+ output_all_weeks.csv, cost_log.json)
python tests/eval_harness.py              # eval: adversarial cases, gold labels, injection recall
python repro_check.py                     # 3 untouched runs, compares SHA-256 of every output
python -m freight_watch.ask "why did Chennai-Bangalore get pricier in March 2025?"
```

## Outputs
- `outputs/output.csv` - every flagged or justified route-week (18 rows), exact columns of `sample_output_format_v2.csv`.
- `outputs/output_all_weeks.csv` - same columns for all 728 route-weeks, so every number can be inspected.
- `outputs/eval_report.txt`, `outputs/repro_report.txt`, `outputs/cost_log.json` - the three checks the brief asks for.

## How it works
1. **Weekly metric** (`metrics.py`). Group by route; week = Monday-Sunday, `week_of` = Monday.
   cost_per_tonne_km = sum(cost) / sum(tonnes x km) for the week's shipments.
   `route_type` is given; the code checks each route has exactly one.
2. **Baselines, exactly as specified.**
   *vs own history* = mean of the route's weekly values over the 8 weeks strictly before the current week
   (fewer if fewer exist; week 1 has none and shows `n/a`).
   *vs similar routes* = mean of the other routes' weekly values for the same `route_type` in the same week,
   excluding the route itself. Each route counts once (simple mean of route-level weekly values).
   These reproduce the numbers in the sample file (e.g. Delhi-Jaipur 2024-11-11: +35.5% / +21.0%).
3. **Flag** (`detect.py`). A week is flagged if cost is above its own 8-week average **and** one lens is both
   unusual (robust z >= 3 against that route's own *prior* deviations, no look-ahead) and material
   (>= +5% vs own history, or >= +10% vs peers). Judging each route against its own normal matters:
   Mumbai-Pune normally sits ~12% above its Short-route peer, Delhi-Jaipur ~10% below.
   Weeks need 8 prior deviations before they can be flagged (about the first 10 weeks are warm-up).
4. **Retrieve + verify** (`notes.py`, `explain.py`). Notes are indexed with TF-IDF and ranked per route.
   Ranking only proposes candidates. A note justifies a rise only if it passes all of: names this route or
   "All Routes"; says costs rose (notes saying "not affected", "stable", "returned to normal", "not part of
   this dataset", "absorbed" are rejected); its period overlaps the week; and, for all-routes notes, the route
   is not also unusually high vs its peers (a nationwide shock lifts the peers too) and the rise fits the
   stated size (N003: 5-7% plus one noise-sigma). Periods are read from the note text ("from Feb 24 to Mar 8",
   "festival week", "starting this week" = 8 weeks, after which the baseline has absorbed the step).
5. **Explain.** The reason is assembled from the verdict plus the cited note's own text, so it cannot invent a cause.
   Unexplained rows list nearby notes and why each was rejected.

## Results on the supplied data
18 rows: 3 justified (N002 Ahmedabad-Mumbai 2025-01-20; N001 Chennai-Bangalore 2025-02-24 and 2025-03-03),
15 unexplained: Chennai-Bangalore 03-10 and 03-17 (still +22%/+14% after the flood window ended and N009
says roads normal), Delhi-Jaipur Nov 2024 (no note; N007 is from May and is good news), Mumbai-Delhi
2025-05-12 (+11.6%, more than N003's 5-7%), and the Mumbai-Pune creep from Sep 2025 (N006/N010 say no effect).

## How I convinced myself it is right (`tests/eval_harness.py`)
- **A. 20 adversarial cases** labeled by hand from the notes: wrong route, week before/after a window, "not
  affected" notes, good-news notes, too-large rise, peer-only anomaly, diesel 7 months later. All pass.
- **B. Gold labels** for all 18 real output rows (flag set, verdict and cited note), plus an independent
  hand-written route/window audit of every justified row. All pass.
- **C. Injection test:** add a known one-week cost jump to untouched weeks (7 routes x 3 weeks). Recall: +10% 62%,
  +15% 90%, +20% 100%, +30% 100%. Small jumps are missed by design (threshold trades recall for fewer false alarms).
- **D. Threshold sweep:** z>=2.5/3/3.5/4 gives 23/18/15/14 flags; the core events are stable across it.
- Caveat: gold labels were written by me after reading the notes and the output, so B shows the logic matches my
  reading, not that my reading is the grader's.

## Reproducibility
No randomness anywhere (no LLM, TF-IDF is deterministic, ties broken by note id, fixed number formatting).
`repro_check.py` runs the pipeline 3 times; all output hashes are identical (`outputs/repro_report.txt`).

## Cost of one full run
LLM calls 0, input tokens 0, output tokens 0, cost $0 (`outputs/cost_log.json`). Runtime ~0.2 s on a laptop.
I chose not to use an LLM for the verdict because the guardrails reward never inventing a reason and require
identical reruns; a deterministic gate gives both for free. The "AI" here is retrieval plus grounded generation
from retrieved text; the Q&A (`ask.py`) is also rule-based and answers only from the output file.

## Limitations and judgment calls
- With 10 notes, TF-IDF retrieval is almost trivially small; the scope/date/effect checks do the real work. For
  messier notes, swap in an embedding model + vector DB and an LLM at temperature 0 for window/effect extraction,
  keeping the same verify step as the gate.
- Note-period and effect extraction is regex-based and tuned to the wording of these 10 notes. Unknown wording
  defaults to the conservative choice (one-week window, no cost effect).
- Mumbai-Delhi 2025-05-12 is arguably diesel plus noise; I left it unexplained rather than over-credit N003.
- A slow creep hides behind a trailing baseline: Mumbai-Pune is flagged in 10 of its ~16 weeks of drift, mostly via the
  peer lens. A longer-horizon baseline would catch more but is outside the brief's definitions.
- Routes are direction-specific (Mumbai-Delhi is not Delhi-Mumbai). The zip contains `sample_output_format_v2.csv`;
  the brief names `sample_output_format.csv`. I followed v2.
