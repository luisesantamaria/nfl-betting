import altair as alt
import pandas as pd

def _pad_domain(series, min_pad, pct=0.06):
    ymin, ymax = float(series.min()), float(series.max())
    pad = max(min_pad, (ymax - ymin) * pct)
    return [ymin - pad, ymax + pad]

def chart_bankroll(df: pd.DataFrame, height=220):
    dom = _pad_domain(df["bankroll"], 10.0)
    return alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("week_label:N", sort=None, title=""),
        y=alt.Y("bankroll:Q", title="Bankroll ($)", scale=alt.Scale(domain=dom, zero=False, nice=False)),
        tooltip=[alt.Tooltip("week_label:N", title="Week"), alt.Tooltip("bankroll:Q", title="Bankroll", format="$.2f")],
    ).properties(height=height, width="container")

def chart_profit_bars(df: pd.DataFrame, height=220):
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("week_label:N", sort=None, title=""),
        y=alt.Y("profit:Q", title="Profit ($)", scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("week_label:N", title="Week"),
                 alt.Tooltip("profit:Q", title="Profit", format="$.2f"),
                 alt.Tooltip("stake:Q", title="Stake", format="$.2f")],
    ).properties(height=height, width="container")

def chart_drawdown_area(bank_df: pd.DataFrame, height=220):
    if bank_df.empty:
        return alt.Chart(pd.DataFrame({"week_label":[],"drawdown_%":[]})).mark_area().encode(
            x="week_label:N", y="drawdown_%:Q").properties(height=height)
    dd_df = bank_df.copy()
    dd_df["rolling_max"] = dd_df["bankroll"].cummax()
    dd_df["drawdown_%"]  = (dd_df["bankroll"] / dd_df["rolling_max"] - 1.0) * 100.0
    dom = _pad_domain(dd_df["drawdown_%"], 1.0)
    dom[1] = max(0, dom[1])  # techo al menos 0
    return alt.Chart(dd_df).mark_area(opacity=0.3).encode(
        x=alt.X("week_label:N", sort=None, title=""),
        y=alt.Y("drawdown_%:Q", title="Max Drawdown (%)", scale=alt.Scale(domain=dom, zero=True, nice=True)),
        tooltip=[alt.Tooltip("week_label:N", title="Week"),
                 alt.Tooltip("drawdown_%:Q", title="Drawdown", format=".1f")],
    ).properties(height=height, width="container")

def chart_stake_bars(df: pd.DataFrame, height=220):
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("week_label:N", sort=None, title=""),
        y=alt.Y("stake:Q", title="Stake ($)", scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("week_label:N", title="Week"), alt.Tooltip("stake:Q", title="Stake", format="$.2f")],
    ).properties(height=height, width="container")

def chart_sparkline_cumprofit(cum_df: pd.DataFrame, height=200):
    dom = _pad_domain(cum_df["cum_profit"], 5.0)
    area = alt.Chart(cum_df).mark_area(opacity=0.25).encode(
        x=alt.X("week_label:N", sort=None, title=""),
        y=alt.Y("cum_profit:Q", title="Cumulative Profit ($)", scale=alt.Scale(domain=dom, zero=False, nice=False)),
        tooltip=[alt.Tooltip("week_label:N", title="Week"), alt.Tooltip("cum_profit:Q", title="Cum Profit", format="$.2f")],
    ).properties(height=height, width="container")
    line = alt.Chart(cum_df).mark_line().encode(x=alt.X("week_label:N", sort=None, title=""), y="cum_profit:Q")
    return area + line

def chart_last8_profit(pnl_df: pd.DataFrame, profits: pd.Series, last=8, height=200):
    last8 = pd.DataFrame({"week_label": pnl_df["week_label"], "profit": profits})
    last8 = last8.tail(last) if len(last8) > last else last8
    return alt.Chart(last8).mark_bar().encode(
        x=alt.X("week_label:N", sort=None, title=""),
        y=alt.Y("profit:Q", title="Last Weeks Profit ($)", scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("week_label:N", title="Week"), alt.Tooltip("profit:Q", title="Profit", format="$.2f")],
    ).properties(height=height, width="container")
