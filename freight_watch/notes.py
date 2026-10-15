"""Load context notes, extract each note's scope / period / cost effect, and rank them by TF-IDF.

Retrieval (TF-IDF cosine) proposes which notes look relevant to a route; structured checks on
scope, dates and stated effect then decide whether a note may actually justify a cost rise.
Everything is rule-based and deterministic. Unknown wording defaults to the conservative choice
(one-week window, no cost effect), so a note can only justify a rise on positive evidence.
"""
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .config import STEP_NOTE_HORIZON_WEEKS

MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}

# Wording that says the note does NOT describe a cost rise. Checked first, so it wins.
NO_EFFECT_PATTERNS = [
    r"not (significantly )?affected", r"no (significant|major) (disruption|impact)",
    r"without a rate change", r"not part of this dataset", r"remained (stable|normal)",
    r"returned to normal", r"improved", r"\bstable\b",
]
# Wording that says costs went up.
RAISES_COST_PATTERNS = [
    r"surcharge", r"higher (trip )?costs?", r"pushing up", r"costs? (rose|increased)",
    r"prices rose", r"detours?",
]


@dataclass(frozen=True)
class Note:
    note_id: str
    date: pd.Timestamp
    applies_to: str
    text: str
    effect: str                      # "raises_cost" | "no_effect" | "unknown"
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    window_kind: str                 # "range" | "step" | "quarter" | "week"
    magnitude: Optional[tuple]       # stated size of the effect in %, e.g. (5.0, 7.0)

    @property
    def is_all_routes(self) -> bool:
        return self.applies_to.strip().lower() == "all routes"

    def covers_route(self, route: str) -> bool:
        return self.is_all_routes or self.applies_to.strip().lower() == route.lower()


def _classify_effect(text: str) -> str:
    t = text.lower()
    if any(re.search(p, t) for p in NO_EFFECT_PATTERNS):
        return "no_effect"
    if any(re.search(p, t) for p in RAISES_COST_PATTERNS):
        return "raises_cost"
    return "unknown"


def _extract_window(text: str, date: pd.Timestamp):
    m = re.search(r"from ([A-Z][a-z]{2}) (\d{1,2}) to ([A-Z][a-z]{2}) (\d{1,2})", text)
    if m and m.group(1) in MONTHS and m.group(3) in MONTHS:
        start = pd.Timestamp(date.year, MONTHS[m.group(1)], int(m.group(2)))
        end = pd.Timestamp(date.year, MONTHS[m.group(3)], int(m.group(4)))
        if end < start:
            end = end.replace(year=end.year + 1)
        return start, end, "range"
    if re.search(r"starting (this week|from)", text, re.I):
        return date, date + timedelta(weeks=STEP_NOTE_HORIZON_WEEKS) - timedelta(days=1), "step"
    if re.search(r"this quarter", text, re.I):
        return date - timedelta(days=91), date, "quarter"
    return date, date + timedelta(days=6), "week"   # "festival week", "about a week", default


def _extract_magnitude(text: str):
    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*%", text)
    return (float(m.group(1)), float(m.group(2))) if m else None


def load_notes(path) -> list:
    df = pd.read_csv(path)
    notes = []
    for r in df.itertuples(index=False):
        date = pd.Timestamp(r.date)
        start, end, kind = _extract_window(r.note, date)
        notes.append(Note(r.note_id, date, r.applies_to, r.note, _classify_effect(r.note),
                          start, end, kind, _extract_magnitude(r.note)))
    return sorted(notes, key=lambda n: n.note_id)


class NoteRetriever:
    """TF-IDF index over the notes. Drop-in swappable for an embedding model + vector DB."""

    QUERY_TERMS = "cost increase surcharge disruption fuel diesel toll flood weather festival delay"

    def __init__(self, notes):
        self.notes = notes
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform([f"{n.applies_to} {n.text}" for n in notes])

    def rank(self, route: str):
        """All notes ordered by similarity to the route + cost-rise vocabulary (ties by note_id)."""
        q = self.vec.transform([f"{route.replace('-', ' ')} {self.QUERY_TERMS}"])
        scores = linear_kernel(q, self.matrix).ravel()
        return sorted(zip(self.notes, scores), key=lambda x: (-round(x[1], 12), x[0].note_id))
