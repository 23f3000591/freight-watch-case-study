"""Decide which route-weeks are 'rising and out of the ordinary'.

A week is flagged when cost is above the route's own 8-week average AND at least one lens
is both statistically unusual and materially large:
  * own lens : % above own history is a robust z >= 3 versus that route's PRIOR weekly deviations
               and is at least 5%
  * peer lens: % above same-type peers is a robust z >= 3 versus that route's PRIOR peer gaps
               and is at least 10%
Using the route's own prior deviations as the yardstick matters: Mumbai-Pune normally sits
~12% above its Short-route peer while Delhi-Jaipur sits ~10% below, so one fixed peer
threshold would be wrong for both. Only past weeks are used (no look-ahead).
"""
import numpy as np
import pandas as pd

from .config import (MATERIAL_OWN_PCT, MATERIAL_PEER_PCT, MIN_DEV_HISTORY, SIGMA_FLOOR_PCT,
                     Z_THRESHOLD)


def _robust_z(values: np.ndarray):
    """z-score of each point against the median/MAD of strictly earlier points."""
    z = np.full(len(values), np.nan)
    sigma = np.full(len(values), np.nan)
    for i in range(len(values)):
        hist = values[:i]
        hist = hist[~np.isnan(hist)]
        if len(hist) < MIN_DEV_HISTORY or np.isnan(values[i]):
            continue
        med = np.median(hist)
        s = max(1.4826 * np.median(np.abs(hist - med)), SIGMA_FLOOR_PCT)
        z[i], sigma[i] = (values[i] - med) / s, s
    return z, sigma


def add_flags(metrics: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, grp in metrics.groupby("route", sort=True):
        grp = grp.sort_values("week_of").copy()
        grp["z_own"], grp["sigma_own"] = _robust_z(grp["own_pct"].to_numpy())
        grp["z_peer"], _ = _robust_z(grp["peer_pct"].to_numpy())
        out.append(grp)
    m = pd.concat(out, ignore_index=True)
    m["own_lens"] = (m["z_own"] >= Z_THRESHOLD) & (m["own_pct"] >= MATERIAL_OWN_PCT)
    m["peer_lens"] = (m["z_peer"] >= Z_THRESHOLD) & (m["peer_pct"] >= MATERIAL_PEER_PCT)
    m["anomaly"] = (m["own_pct"] > 0) & (m["own_lens"] | m["peer_lens"])
    return m
