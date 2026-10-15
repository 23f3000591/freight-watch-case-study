# Freight Cost Watchdog

Watches weekly freight cost per tonne-km for each route, flags rises that look unusual, and checks the
context notes before deciding whether each rise is **justified** or **unexplained**. A rise is only called
justified when a note genuinely applies to that route and that week. Otherwise it is left unexplained for a
human to review.

## TL;DR

| | |
|---|---|
| Data scored | 2,940 shipments, 7 routes, 104 weeks = 728 route-weeks |
| Rows in `output.csv` | 18 (3 justified, 15 unexplained) |
| Justified | Ahmedabad-Mumbai 2025-01-20 (N002); Chennai-Bangalore 2025-02-24 and 2025-03-03 (N001) |
| LLM calls / tokens / cost | 0 / 0 / $0 for a full run (runtime about 0.2 s) |
| Reproducibility | 3 runs give byte-identical outputs (SHA-256 compared) |
| Eval | 20 adversarial cases + 18 gold rows + injection test, all passing |

## Quick start

```bash
pip install -r requirements.txt          # pandas, numpy, scikit-learn
python -m freight_watch.pipeline         # writes outputs/output.csv and friends
python tests/eval_harness.py             # eval harness (exit code 1 on any failure)
python repro_check.py                    # 3 runs, compares hashes of every output
python -m freight_watch.ask "why did Chennai-Bangalore get pricier in March 2025?"
```

## Where each requirement of the brief is covered

| Requirement | Where |
|---|---|
| Group by route and route_type | `metrics.py` (route_type is given; the code checks each route has exactly one) |
| Weekly cost per tonne-km, Mon-Sun weeks | `metrics.py` |
| Baseline 1: trailing 8 prior weeks | `metrics.py` (fewer weeks if fewer exist; week 1 shows `n/a`) |
| Baseline 2: same-week average of other routes of the same type | `metrics.py` |
| Flag rising and unusual | `detect.py` |
| Check notes before flagging (RAG) | `notes.py` (TF-IDF retrieval) + `explain.py` (verification) |
| Plain-English reason grounded in notes | `explain.py` |
| Hallucination guardrail | A note must pass 5 checks (below); otherwise the row stays "Yes" with no note id |
| Eval harness | `tests/eval_harness.py`, results in `outputs/eval_report.txt` |
| Reproducibility | `repro_check.py`, results in `outputs/repro_report.txt` |
| Token / cost log | `outputs/cost_log.json` |
| Q&A over findings (stretch) | `freight_watch/ask.py` |
| Output CSV in sample format | `outputs/output.csv` (header identical to `sample_output_format_v2.csv`) |

## Outputs

- `outputs/output.csv`: every flagged or justified route-week, in the sample format.
- `outputs/output_all_weeks.csv`: the same columns for all 728 route-weeks, so any number can be inspected.
- `outputs/eval_report.txt`, `outputs/repro_report.txt`, `outputs/cost_log.json`.

Example row (Delhi-Jaipur, a real unexplained flag):

```
Delhi-Jaipur,2024-11-11,4.17,+35.5% vs this route's past average,+21.0% vs similar-length routes this week,Yes,,"Unexplained. Cost is +35.5% ... No matching note found for this route or date range. ..."
```

## How it works

```
shipments.csv --> weekly metrics --> anomaly flag --+--> not flagged: "No"
                  (2 baselines)      (rising + unusual)|
                                                      +--> flagged: rank notes (TF-IDF)
context_notes.csv --> parse scope / period / effect -->|        |
                                                       |   5 verification checks
                                                       |        |
                                                       |   pass: "No (justified)" + note id
                                                       |   fail: "Yes" + why nearby notes were rejected
```

### 1. Weekly metric
`cost_per_tonne_km = sum(freight cost) / sum(tonnes x km)` over the week's shipments for a route.
Weeks run Monday to Sunday and `week_of` is the Monday. A full weekly calendar is used, so a week with no
shipments is a gap, not "the previous row".

### 2. The two baselines (as specified)
- **vs own history:** mean of the route's weekly values over up to 8 weeks strictly before the current week.
  If fewer than 8 prior weeks exist, whatever is available is used. Week 1 has no history and shows `n/a`.
- **vs similar routes:** mean of the *other* routes' weekly values for the same `route_type` in the same week,
  excluding the route itself. Each route counts once.

These reproduce the numbers in the sample file (for example Delhi-Jaipur 2024-11-11: +35.5% / +21.0%).

### 3. When a week is flagged (`detect.py`)
Cost must be above the route's own 8-week average **and** at least one lens must be both unusual and material:

| Lens | Unusual | Material |
|---|---|---|
| Own history | robust z >= 3 against that route's *prior* weekly deviations | >= +5% |
| Similar routes | robust z >= 3 against that route's *prior* peer gaps | >= +10% |

Why judge each route against its own normal: Mumbai-Pune usually sits about 12% above its Short-route peer and
Delhi-Jaipur about 10% below, so a single fixed peer threshold would be wrong for both. Only past weeks are used
(no look-ahead). A week needs 8 prior deviations before it can be flagged, so roughly the first 10 weeks are
warm-up. Thresholds are in `config.py`.

### 4. Retrieve and verify (`notes.py`, `explain.py`)
Notes are indexed with TF-IDF and ranked per route. Ranking only *proposes* candidates. A note can justify a
rise only if **all five** checks pass:

1. **Scope:** it names this route, or "All Routes" (routes are direction-specific).
2. **Effect:** its text says costs rose. Notes saying "not affected", "stable", "no major disruptions", "returned
   to normal", "improved", "not part of this dataset" or "absorbed ... without a rate change" are rejected.
3. **Period:** its period overlaps this Monday-Sunday week. Periods are read from the note text ("from Feb 24 to
   Mar 8", "festival week", "about a week", "starting this week").
4. **Peers:** an all-routes note cannot explain a route that is also unusually high versus its peers, because a
   nationwide shock lifts the peers too.
5. **Size:** if the note states a size (N003: 5-7%), the observed rise must fit it plus one noise-sigma of tolerance.
   A "starting this week" note stops explaining after 8 weeks, because the 8-week baseline has absorbed the step.

If no note passes, the verdict is "Yes" (unexplained), `matched_note_id` is blank, and the reason lists the nearby
notes (within 28 days) and exactly why each was rejected. If none are nearby it says "No matching note found for
this route or date range."

### 5. Explanation
The reason text is assembled from the verdict plus the cited note's own wording, so it cannot invent a cause.

## Results on the supplied data

| Route | Week(s) | Verdict | Why |
|---|---|---|---|
| Ahmedabad-Mumbai | 2025-01-20 | Justified, N002 | Festival-week surcharge, same week |
| Chennai-Bangalore | 2025-02-24, 2025-03-03 | Justified, N001 | Flood window Feb 24 to Mar 8 |
| Chennai-Bangalore | 2025-03-10, 2025-03-17 | Unexplained | Still +22% / +14% after the window ended; N009 says roads normal |
| Delhi-Jaipur | 2024-11-11, 2024-11-18 | Unexplained | No note in range; N007 is from May and is good news |
| Mumbai-Delhi | 2025-05-12 | Unexplained | +11.6% exceeds N003's 5-7% |
| Mumbai-Pune | 10 weeks from 2025-09-15 | Unexplained | Steady creep vs peers; N006 and N010 report no cost effect |

## Evaluation (`tests/eval_harness.py`)

- **A. Adversarial verdict cases (20):** wrong route, week before and after a window, "not affected" notes,
  good-news notes, a rise far above a note's stated size, peer-only anomalies, a diesel note 7 months later.
- **B. Gold labels:** the flagged set, verdict and cited note for all 18 real output rows, plus an independent
  hand-written route and window audit of every justified row.
- **C. Injection test:** add a known one-week cost jump to untouched weeks (7 routes x 3 weeks) and measure recall.

  | Jump | +10% | +15% | +20% | +30% |
  |---|---|---|---|---|
  | Recall | 62% | 90% | 100% | 100% |

- **D. Threshold sweep:** z >= 2.5 / 3 / 3.5 / 4 gives 23 / 18 / 15 / 14 flagged weeks. The core events are stable.

Caveat: the gold labels were written by hand from the notes and the output, so B shows the logic matches that
reading, not that the reading is the only possible one.

## Reproducibility

There is no randomness anywhere: no LLM, deterministic TF-IDF, ties broken by note id, fixed number formatting.
`repro_check.py` runs the pipeline three times and compares SHA-256 hashes of every output. They are identical
(`outputs/repro_report.txt`). Because no LLM is used, there is no temperature to report.

## Cost of one full run

LLM calls: 0. Input tokens: 0. Output tokens: 0. Estimated cost: $0 (`outputs/cost_log.json`).
I chose not to use an LLM for the verdict because the guardrails reward never inventing a reason and require
identical reruns, and a deterministic gate gives both. The AI component is retrieval plus grounded generation
from retrieved text. The Q&A tool is also rule-based and answers only from the output file.

## Limitations and judgment calls

- With 10 notes, retrieval is nearly trivial. The scope, date and effect checks do the real work.
- Period and effect extraction is regex-based and tuned to the wording of these notes. Unknown wording defaults
  to the conservative choice (one-week window, no cost effect). For messier notes, use an embedding model plus
  vector DB and an LLM at temperature 0 for extraction, keeping the same verification step as the gate.
- Mumbai-Delhi 2025-05-12 is arguably diesel plus noise. I left it unexplained rather than over-credit N003.
- A slow creep hides behind a trailing baseline. Mumbai-Pune is flagged in 10 of the 15 weeks from 2025-09-15,
  mostly via the peer lens. A longer-horizon baseline would catch more but is outside the brief's definitions.
- Small jumps (about +10%) are missed by design, to keep false alarms low.
- The zip's sample file is `sample_output_format_v2.csv` while the brief names `sample_output_format.csv`.
  I followed v2. Its row 3 has an unquoted comma, so it does not parse with a standard CSV reader; my output
  quotes fields properly.

## Project layout

```
freight_watch/
  config.py      thresholds and horizons
  metrics.py     weekly cost per tonne-km, both baselines
  detect.py      rising and unusual (robust z-scores, prior weeks only)
  notes.py       note parsing (scope, period, effect, size) and TF-IDF retrieval
  explain.py     5-check verification and reason text
  pipeline.py    end-to-end run, writes CSVs and the cost log
  ask.py         question answering over the output
tests/eval_harness.py
repro_check.py
data/            the three input files
outputs/         results, eval report, reproducibility report, cost log
```
