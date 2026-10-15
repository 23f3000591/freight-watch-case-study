import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from freight_watch import detect                                   
from freight_watch.explain import FlagContext, decide               
from freight_watch.metrics import load_shipments, weekly_metrics 
from freight_watch.notes import NoteRetriever, load_notes
from freight_watch.pipeline import build_output

SHIP, NOTES = ROOT / "data" / "shipment_records.csv", ROOT / "data" / "context_notes.csv"
notes = load_notes(NOTES)
retriever = NoteRetriever(notes)
failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def verdict(route, week, own, peer, own_lens=True, peer_lens=False, sigma=2.5):
    ctx = FlagContext(route, pd.Timestamp(week), own, peer, sigma, own_lens, peer_lens)
    return decide(ctx, retriever.rank(route))


# ---------------------------------------------------------------- Part A
print("Part A: adversarial verdict cases (expected values labeled by hand from the notes)")
CASES = [
    # (label, route, week, own%, peer%, own_lens, peer_lens, expected_note_id or "")
    ("festival week itself -> N002",          "Ahmedabad-Mumbai", "2025-01-20", 29.5, 22.5, True, True, "N002"),
    ("festival note, week AFTER -> none",     "Ahmedabad-Mumbai", "2025-01-27", 20.0, 15.0, True, True, ""),
    ("festival note, wrong route -> none",    "Mumbai-Pune",      "2025-01-20", 20.0, 15.0, True, True, ""),
    ("flood, first week -> N001",             "Chennai-Bangalore", "2025-02-24", 31.5, 38.7, True, True, "N001"),
    ("flood, week ending inside window",      "Chennai-Bangalore", "2025-03-03", 22.8, 32.0, True, True, "N001"),
    ("flood, week after window -> none",      "Chennai-Bangalore", "2025-03-10", 22.1, 33.8, True, True, ""),
    ("flood, 'back to normal' week -> none",  "Chennai-Bangalore", "2025-03-17", 13.7, 33.5, True, True, ""),
    ("flood, week BEFORE window -> none",     "Chennai-Bangalore", "2025-02-17", 15.0, 20.0, True, True, ""),
    ("flood note, wrong route -> none",       "Delhi-Jaipur",     "2025-03-03", 20.0, 20.0, True, True, ""),
    ("maintenance 'not affected' -> none",    "Mumbai-Delhi",     "2024-07-29", 10.0, 8.0, True, False, ""),
    ("road resurfacing is good news -> none", "Delhi-Jaipur",     "2024-05-20", 12.0, 12.0, True, True, ""),
    ("'no disruptions' note + big rise -> none", "Kolkata-Bhubaneswar", "2025-06-09", 12.0, 9.0, True, False, ""),
    ("toll note not in dataset -> none",      "Mumbai-Pune",      "2024-03-11", 12.0, 12.0, True, False, ""),
    ("diesel, small own-lens rise -> N003",   "Mumbai-Delhi",     "2025-05-12", 6.5, 3.0, True, False, "N003"),
    ("diesel, rise far above 5-7% -> none",   "Mumbai-Delhi",     "2025-05-12", 11.6, 6.7, True, False, ""),
    ("diesel but peer-lens flag -> none",     "Mumbai-Pune",      "2025-05-19", 6.0, 25.0, False, True, ""),
    ("diesel, 7 months later -> none",        "Mumbai-Pune",      "2025-12-01", 6.0, 5.0, True, False, ""),
    ("diesel, week BEFORE note -> none",      "Mumbai-Delhi",     "2025-04-28", 6.0, 3.0, True, False, ""),
    ("stable-demand note -> none",            "Mumbai-Pune",      "2025-09-15", 9.2, 23.6, False, True, ""),
    ("no note anywhere near -> none",         "Delhi-Jaipur",     "2024-11-11", 35.5, 21.0, True, True, ""),
]
for label, route, week, own, peer, ol, pl, expected in CASES:
    d = verdict(route, week, own, peer, ol, pl)
    check(label, d.note_id == expected and d.justified == bool(expected),
          f"-> {d.note_id or 'unexplained'}")

# ---------------------------------------------------------------- Part B
print("\nPart B: gold labels on the real output + independent citation audit")
metrics = detect.add_flags(weekly_metrics(load_shipments(SHIP)))
out = build_output(metrics, retriever)
flagged = out[out["flagged"] != "No"].set_index(["route", "week_of"])

GOLD = {  # (route, week_of): expected matched_note_id ("" = unexplained)
    ("Ahmedabad-Mumbai", "2025-01-20"): "N002",
    ("Chennai-Bangalore", "2025-02-24"): "N001", ("Chennai-Bangalore", "2025-03-03"): "N001",
    ("Chennai-Bangalore", "2025-03-10"): "", ("Chennai-Bangalore", "2025-03-17"): "",
    ("Delhi-Jaipur", "2024-11-11"): "", ("Delhi-Jaipur", "2024-11-18"): "",
    ("Mumbai-Delhi", "2025-05-12"): "",
    **{("Mumbai-Pune", w): "" for w in ["2025-09-15", "2025-10-06", "2025-10-20", "2025-10-27",
                                       "2025-11-03", "2025-11-17", "2025-12-01", "2025-12-08",
                                       "2025-12-15", "2025-12-22"]},
}
got = {k: (v if isinstance(v, str) else "") for k, v in flagged["matched_note_id"].fillna("").items()}
check("flagged rows are exactly the gold set", set(got) == set(GOLD),
      f"(got {len(got)}, gold {len(GOLD)}; extra={set(got) - set(GOLD)}, missing={set(GOLD) - set(got)})")
wrong = {k: (got[k], GOLD[k]) for k in GOLD if k in got and got[k] != GOLD[k]}
check("every verdict / cited note matches gold", not wrong, str(wrong))

# Independent oracle for the only notes that justify anything: hand-written windows.
ORACLE = {"N001": ("Chennai-Bangalore", "2025-02-24", "2025-03-08"),
          "N002": ("Ahmedabad-Mumbai", "2025-01-20", "2025-01-26")}
bad = []
for (route, week), row in flagged.iterrows():
    if row["flagged"] == "No (justified)":
        nid = row["matched_note_id"]
        r, s, e = ORACLE[nid]
        wk = pd.Timestamp(week)
        if not (route == r and wk <= pd.Timestamp(e) and wk + pd.Timedelta(days=6) >= pd.Timestamp(s)):
            bad.append((route, week, nid))
check("every justified row passes the independent route+window audit", not bad, str(bad))
check("unexplained rows never carry a note id",
      (flagged[flagged["flagged"] == "Yes"]["matched_note_id"] == "").all())
justified_text_ok = all(
    flagged.loc[k, "reason"].count(flagged.loc[k, "matched_note_id"]) >= 1
    for k in flagged.index if flagged.loc[k, "flagged"] == "No (justified)")
check("justified reasons name the cited note id", justified_text_ok)

# ---------------------------------------------------------------- Part C
print("\nPart C: detector sensitivity (inject a one-week cost jump into untouched weeks)")
raw = load_shipments(SHIP)
raw["freight_cost_inr"] = raw["freight_cost_inr"].astype(float)
weeks = sorted(raw["week_of"].unique())
INJECT_IDX = [20, 32, 40]            # all well before any real event and after the warm-up
results = {}
for pct in (10, 15, 20, 30):
    hit = total = 0
    for route in sorted(raw["route"].unique()):
        for idx in INJECT_IDX:
            df = raw.copy()
            mask = (df["route"] == route) & (df["week_of"] == weeks[idx])
            df.loc[mask, "freight_cost_inr"] *= 1 + pct / 100
            m = detect.add_flags(weekly_metrics(df))
            row = m[(m["route"] == route) & (m["week_of"] == weeks[idx])].iloc[0]
            hit += bool(row["anomaly"]); total += 1
    results[pct] = hit / total
    print(f"  +{pct}% jump: detected {hit}/{total} ({hit / total:.0%})")
check("recall >= 95% for a +20% jump", results[20] >= 0.95)
check("recall == 100% for a +30% jump", results[30] == 1.0)

# ---------------------------------------------------------------- Part D
print("\nPart D: threshold sensitivity (untouched data)")
orig = detect.Z_THRESHOLD
for z in (2.5, 3.0, 3.5, 4.0):
    detect.Z_THRESHOLD = z
    n = int(detect.add_flags(weekly_metrics(raw))["anomaly"].sum())
    print(f"  z >= {z}: {n} flagged route-weeks")
detect.Z_THRESHOLD = orig

print("\nRESULT:", "ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
