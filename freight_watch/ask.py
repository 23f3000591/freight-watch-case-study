import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]


def _find_routes(q: str, routes):
    q = q.lower()
    exact = [r for r in routes if r.lower() in q]
    if exact:
        return exact
    cities = [r for r in routes if all(part.lower() in q for part in r.split("-"))]
    return cities


def answer(question: str, table: pd.DataFrame) -> str:
    q = question.lower()
    routes = sorted(table["route"].unique())
    found = _find_routes(q, routes)
    if not found:
        return "I couldn't tell which route you mean. Routes I know: " + ", ".join(routes) + "."
    month = next((i for i, m in enumerate(MONTHS, 1) if re.search(rf"\b{m[:3]}\w*\b", q)), None)
    year = re.search(r"\b(20\d\d)\b", q)

    t = table[table["route"].isin(found)].copy()
    t["date"] = pd.to_datetime(t["week_of"])
    if month:
        t = t[t["date"].dt.month == month]
    if year:
        t = t[t["date"].dt.year == int(year.group(1))]
    if t.empty:
        return "I have no weekly data for that route and period."

    hits = t[t["flagged"] != "No"]
    if hits.empty:
        lo, hi = t["cost_per_tonne_km"].astype(float).min(), t["cost_per_tonne_km"].astype(float).max()
        return (f"No week in that period was flagged for {', '.join(found)}. Cost per tonne-km ranged "
                f"{lo:.2f} to {hi:.2f}, with no unusual rise versus its own history or similar routes.")
    lines = []
    for r in hits.itertuples():
        verdict = "justified" if r.flagged == "No (justified)" else "UNEXPLAINED (needs review)"
        cite = f" [note {r.matched_note_id}]" if r.matched_note_id else ""
        lines.append(f"- {r.route}, week of {r.week_of}: {r.cost_per_tonne_km} per tonne-km, "
                     f"{r.vs_own_history}; {verdict}{cite}. {r.reason}")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        sys.exit('usage: python -m freight_watch.ask "why did Mumbai-Pune get pricier in October 2025?"')
    table = pd.read_csv(ROOT / "outputs" / "output_all_weeks.csv", dtype=str, keep_default_na=False)
    print(answer(" ".join(sys.argv[1:]), table))


if __name__ == "__main__":
    main()
