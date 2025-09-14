import altair as alt
import pandas as pd
from .utils import ORDER_LABELS, ORDER_INDEX, add_week_order

def _week_sort():
    return alt.Sort(domain=ORDER_LABELS)

def chart_bankroll(df: pd.DataFrame, height: int = 220):
    x = alt.X("week_label:N", sort=_week_sort(), title=None)
    y = alt.Y("bankroll:Q", title="Bankroll ($)")
    return alt.Chart(df).mark_line(point=True).encode(x=x, y=y, tooltip=["week_label","bankroll"]).properties(height=height)

def chart_profit_bars(df: pd.DataFrame, height: int = 220):
    x = alt.X("week_label:N", sort=_week_sort(), title=None)
    y = alt.Y("profit:Q", title="Weekly Profit ($)")
    color = alt.condition(alt.datum.profit >= 0, alt.value("#1a7f37"), alt.value("#c92a2a"))
    return alt.Chart(df).mark_bar().encode(x=x, y=y, color=color, tooltip=["week_label","profit","stake"]).properties(height=height)

def chart_stake_bars(df: pd.DataFrame, height: int = 220):
    x = alt.X("week_label:N", sort=_week_sort(), title=None)
    y = alt.Y("stake:Q", title="Stake ($)")
    return alt.Chart(df).mark_bar().encode(x=x, y=y, tooltip=["week_label","stake"]).properties(height=height)

def chart_drawdown_area(df: pd.DataFrame, height: int = 220):
    # drawdown relativo al máximo previo
    ser = pd.to_numeric(df["bankroll"], errors="coerce").fillna(method="ffill")
    roll_max = ser.cummax()
    dd = (ser - roll_max)
    dd_df = df[["week_label"]].copy()
    dd_df["drawdown"] = dd
    x = alt.X("week_label:N", sort=_week_sort(), title=None)
    y = alt.Y("drawdown:Q", title="Drawdown ($)")
    return alt.Chart(dd_df).mark_area(opacity=0.5).encode(x=x, y=y, tooltip=["week_label","drawdown"]).properties(height=height)

def chart_sparkline_cumprofit(df: pd.DataFrame, height: int = 200):
    # df: columns ["week_label","cum_profit"]
    x = alt.X("week_label:N", sort=_week_sort(), title=None)
    y = alt.Y("cum_profit:Q", title="Cumulative Profit ($)")
    return alt.Chart(df).mark_line(point=True).encode(x=x, y=y, tooltip=["week_label","cum_profit"]).properties(height=height)

def chart_last8_profit(pnl: pd.DataFrame, profits: pd.Series, last: int = 8, height: int = 200):
    tmp = pnl[["week_label"]].copy()
    tmp["profit"] = profits.values
    tmp = add_week_order(tmp).sort_values("week_order")
    tail = tmp.tail(last)
    x = alt.X("week_label:N", sort=list(tail["week_label"]), title=None)
    y = alt.Y("profit:Q", title=f"Last {len(tail)} Weeks ($)")
    color = alt.condition(alt.datum.profit >= 0, alt.value("#1a7f37"), alt.value("#c92a2a"))
    return alt.Chart(tail).mark_bar().encode(x=x, y=y, color=color, tooltip=["week_label","profit"]).properties(height=height)
