"""End-to-end run: shipments + notes -> output CSVs + cost log.

    python -m freight_watch.pipeline --out outputs/run1
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .detect import add_flags
from .explain import FlagContext, decide
from .metrics import load_shipments, weekly_metrics
from .notes import NoteRetriever, load_notes

COLUMNS = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes",
           "flagged", "matched_note_id", "reason"]
ROOT = Path(__file__).resolve().parent.parent


def _pct(value, suffix: str) -> str:
    return "n/a (not enough data)" if pd.isna(value) else f"{value:+.1f}% {suffix}"


def build_output(metrics: pd.DataFrame, retriever: NoteRetriever) -> pd.DataFrame:
    ranked = {r: retriever.rank(r) for r in metrics["route"].unique()}
    rows = []
    for r in metrics.itertuples(index=False):
        if r.anomaly:
            ctx = FlagContext(r.route, r.week_of, r.own_pct, r.peer_pct, r.sigma_own,
                              bool(r.own_lens), bool(r.peer_lens))
            d = decide(ctx, ranked[r.route])
            flagged, note_id, reason = ("No (justified)" if d.justified else "Yes"), d.note_id, d.reason
        else:
            flagged, note_id = "No", ""
            reason = ("Not enough prior history to judge." if r.prior_weeks < 1 else
                      "No anomaly: cost is not rising in an unusual way versus own history or similar routes.")
        rows.append({
            "route": r.route, "week_of": r.week_of.strftime("%Y-%m-%d"),
            "cost_per_tonne_km": f"{r.cost_per_tonne_km:.2f}",
            "vs_own_history": _pct(r.own_pct, "vs this route's past average"),
            "vs_similar_routes": _pct(r.peer_pct, "vs similar-length routes this week"),
            "flagged": flagged, "matched_note_id": note_id, "reason": reason})
    return pd.DataFrame(rows, columns=COLUMNS)


def run(shipments, notes, out_dir) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = add_flags(weekly_metrics(load_shipments(shipments)))
    notes_list = load_notes(notes)
    full = build_output(metrics, NoteRetriever(notes_list))

    anomalies = full[full["flagged"] != "No"]
    anomalies.to_csv(out_dir / "output.csv", index=False, lineterminator="\n")
    full.to_csv(out_dir / "output_all_weeks.csv", index=False, lineterminator="\n")
    # No LLM is used anywhere: verdicts and wording come from rules + retrieved note text.
    (out_dir / "cost_log.json").write_text(json.dumps({
        "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0,
        "retrieval": "scikit-learn TF-IDF (local, free)", "route_weeks_scored": len(full),
        "rows_flagged_or_justified": len(anomalies)}, indent=2) + "\n")
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipments", default=ROOT / "data" / "shipment_records.csv")
    ap.add_argument("--notes", default=ROOT / "data" / "context_notes.csv")
    ap.add_argument("--out", default=ROOT / "outputs")
    a = ap.parse_args()
    t = time.time()
    full = run(a.shipments, a.notes, a.out)
    print(full["flagged"].value_counts().to_string())
    print(f"done in {time.time() - t:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
