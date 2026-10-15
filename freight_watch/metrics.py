"""Weekly cost per tonne-km, plus the two baselines the brief defines exactly."""
import numpy as np
import pandas as pd

from .config import BASELINE_WEEKS


def load_shipments(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["shipment_date"])
    df["route"] = df["origin"].str.strip() + "-" + df["destination"].str.strip()
    # Weeks run Monday-Sunday; week_of is that week's Monday.
    day = df["shipment_date"].dt.normalize()
    df["week_of"] = day - pd.to_timedelta(day.dt.weekday, unit="D")
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    return df


def weekly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    
    types = df.groupby("route")["route_type"].nunique()
    if (types > 1).any():
        raise ValueError(f"Routes with more than one route_type: {list(types[types > 1].index)}")
    route_type = df.groupby("route")["route_type"].first()

    g = df.groupby(["route", "week_of"], as_index=False).agg(
        cost=("freight_cost_inr", "sum"), tonne_km=("tonne_km", "sum"), shipments=("shipment_id", "count"))
    g["cost_per_tonne_km"] = g["cost"] / g["tonne_km"]

    # Full weekly calendar so a week with no shipments counts as a gap, not as "the previous row".
    wide = g.pivot(index="week_of", columns="route", values="cost_per_tonne_km")
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="7D"))
    wide.index.name = "week_of"

    prior = wide.shift(1).rolling(BASELINE_WEEKS, min_periods=1)
    base, n_prior = prior.mean(), prior.count()

    peer = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    for rtype, routes in route_type.groupby(route_type).groups.items():
        vals = wide[list(routes)]
        total, count = vals.sum(axis=1), vals.notna().sum(axis=1)
        for r in routes:
            denom = count - vals[r].notna().astype(int)
            peer[r] = (total - vals[r].fillna(0)) / denom.where(denom > 0)

    long = pd.concat({"cost_per_tonne_km": wide, "own_baseline": base, "prior_weeks": n_prior,
                      "peer_avg": peer}, axis=1)
    long = long.stack(future_stack=True).reset_index()
    long = long.dropna(subset=["cost_per_tonne_km"]).rename(columns={"level_1": "route"})
    long["route_type"] = long["route"].map(route_type)
    long["own_pct"] = (long["cost_per_tonne_km"] / long["own_baseline"] - 1) * 100
    long["peer_pct"] = (long["cost_per_tonne_km"] / long["peer_avg"] - 1) * 100
    return long.sort_values(["route", "week_of"]).reset_index(drop=True)
