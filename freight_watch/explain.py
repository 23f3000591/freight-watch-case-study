from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import pandas as pd

from .config import NOTE_PROXIMITY_DAYS


@dataclass(frozen=True)
class FlagContext:
    route: str
    week_of: pd.Timestamp
    own_pct: float
    peer_pct: float
    sigma_own: float
    own_lens: bool
    peer_lens: bool


@dataclass(frozen=True)
class Check:
    note: object
    score: float
    status: str      # "applies" or a failure code
    detail: str


@dataclass(frozen=True)
class Decision:
    justified: bool
    note_id: str
    reason: str


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _first_sentence(text: str) -> str:
    return text.split(". ")[0].rstrip(".") + "."


def check_note(note, score, ctx: FlagContext) -> Optional[Check]:
    """Return None if the note is out of scope (never mentioned), else a Check."""
    if not note.covers_route(ctx.route):
        return None
    if note.effect != "raises_cost":
        return Check(note, score, "no_cost_effect",
                     "does not describe a cost rise on this route, so it cannot justify one")
    week_end = ctx.week_of + timedelta(days=6)
    if note.window_end < ctx.week_of or note.window_start > week_end:
        return Check(note, score, "outside_window",
                     f"covers {_d(note.window_start)} to {_d(note.window_end)}, which does not "
                     f"include the week of {_d(ctx.week_of)}")
    if note.is_all_routes and ctx.peer_lens:
        return Check(note, score, "peer_gap",
                     "is nationwide, so it also lifts the peer routes; it cannot explain this "
                     "route being unusually high versus its peers")
    if note.magnitude:
        lo, hi = note.magnitude
        cap = hi + ctx.sigma_own
        if ctx.own_pct > cap:
            return Check(note, score, "too_large",
                         f"accounts for roughly {lo:g}-{hi:g}%, but the observed rise is "
                         f"{ctx.own_pct:+.1f}% (allowed up to {cap:.1f}%), so most of it is unexplained")
    return Check(note, score, "applies", "")


def _lens_text(ctx: FlagContext) -> str:
    lenses = [name for name, on in (("its own history", ctx.own_lens),
                                    ("similar routes", ctx.peer_lens)) if on]
    return (f"Cost is {ctx.own_pct:+.1f}% vs its own 8-week average and {ctx.peer_pct:+.1f}% vs "
            f"similar routes, which is unusual against {' and '.join(lenses)}.")


def decide(ctx: FlagContext, ranked_notes) -> Decision:
    checks = [c for n, s in ranked_notes if (c := check_note(n, s, ctx))]
    applying = [c for c in checks if c.status == "applies"]
    if applying:
        # Route-specific notes beat all-routes notes; then retrieval score; then id (deterministic).
        best = sorted(applying, key=lambda c: (c.note.is_all_routes, -round(c.score, 12),
                                               c.note.note_id))[0]
        n = best.note
        size = (f"The observed rise is within the note's stated {n.magnitude[0]:g}-{n.magnitude[1]:g}% "
                f"range (plus noise tolerance)." if n.magnitude else
                "The note gives no size, so the size of the rise is not checked against it.")
        reason = (f"Matches note {n.note_id} dated {_d(n.date)} (applies to {n.applies_to}): "
                  f"\"{n.text}\" The week of {_d(ctx.week_of)} overlaps the note's period "
                  f"({_d(n.window_start)} to {_d(n.window_end)}) and the note describes a "
                  f"cost-raising cause. {size} {_lens_text(ctx)}")
        return Decision(True, n.note_id, reason)

    near = []
    for c in checks:
        gap = max((c.note.window_start - (ctx.week_of + timedelta(days=6))).days,
                  (ctx.week_of - c.note.window_end).days, 0)
        if gap <= NOTE_PROXIMITY_DAYS:
            near.append(c)
    if near:
        parts = [f"{c.note.note_id} ({_d(c.note.date)}, \"{_first_sentence(c.note.text)}\") "
                 f"{c.detail}" for c in near[:3]]
        tail = ("Closest notes: " + "; ".join(parts) + ". No genuine justification found; "
                "flagged for review.")
    else:
        tail = ("No matching note found for this route or date range. Cost rise looks unexplained "
                "and worth a human review.")
    return Decision(False, "", f"Unexplained. {_lens_text(ctx)} {tail}")
