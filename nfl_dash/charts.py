import altair as alt
import pandas as pd
from .utils import ORDER_LABELS, week_sort_key

def _x_week_sorted():
    return alt.X("week_label:N", sort=ORDER_LABELS, title=None)

def bankroll_line(df: pd.DataFrame, height: int = 260):
    d = week_sort_key(df)
    if "bankroll" not in d.columns:
        d["bankroll"] = 0.0
    y_min = float(d["bankroll"].min()) if len(d) else 0.0
    y_max = float(d["bankroll"].max()) if len(d) else 1.0
    pad = max(1.0, (y_max - y_min) * 0.06)
    return (
        alt.Chart(d)
        .mark_line(point=True)
        .encode(
            x=_x_week_sorted(),
            y=alt.Y("bankroll:Q", title="Bankroll ($)", scale=alt.Scale(domain=[y_min - pad, y_max + pad])),
            tooltip=["week_label","bankroll"]
        )
        .properties(height=height)
    )

def weekly_profit_bar(df: pd.DataFrame, height: int = 260):
    d = week_sort_key(df)
    if "profit" not in d.columns:
        d["profit"] = 0.0
    return (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=_x_week_sorted(),
            y=alt.Y("profit:Q", title="Weekly Profit ($)"),
            tooltip=["week_label","profit"]
        )
        .properties(height=height)
    )

def cum_profit_area(df: pd.DataFrame, height: int = 260):
    d = week_sort_key(df).copy()
    if "profit" not in d.columns:
        d["profit"] = 0.0
    d["cum_profit"] = d["profit"].cumsum()
    y_min = float(d["cum_profit"].min()) if len(d) else 0.0
    y_max = float(d["cum_profit"].max()) if len(d) else 1.0
    pad = max(1.0, (y_max - y_min) * 0.06)
    return (
        alt.Chart(d)
        .mark_area(opacity=0.4)
        .encode(
            x=_x_week_sorted(),
            y=alt.Y("cum_profit:Q", title="Cumulative Profit ($)", scale=alt.Scale(domain=[y_min - pad, y_max + pad])),
            tooltip=["week_label","cum_profit"]
        )
        .properties(height=height)
    )

def yield_line(df: pd.DataFrame, height: int = 260):
    d = week_sort_key(df).copy()
    if "stake" not in d.columns:
        d["stake"] = 0.0
    if "profit" not in d.columns:
        d["profit"] = 0.0
    d["yield_pct"] = d.apply(lambda r: (r["profit"] / r["stake"] * 100.0) if r["stake"] and r["stake"] != 0 else 0.0, axis=1)
    return (
        alt.Chart(d)
        .mark_line(point=True)
        .encode(
            x=_x_week_sorted(),
            y=alt.Y("yield_pct:Q", title="Weekly Yield (%)"),
            tooltip=["week_label","yield_pct"]
        )
        .properties(height=height)
    )
