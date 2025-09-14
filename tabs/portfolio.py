import streamlit as st
import pandas as pd
import altair as alt
from nfl_dash.utils import load_pnl, ORDER_INDEX

def chart_bankroll(df: pd.DataFrame, height: int) -> alt.Chart:
    d = df.copy()
    d["week_order"] = d["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    d = d.sort_values("week_order")
    y_min = max(0.0, d["bankroll"].min() - 40) if "bankroll" in d.columns else None
    y_max = d["bankroll"].max() + 40 if "bankroll" in d.columns else None
    return alt.Chart(d).mark_line(point=True).encode(
        x=alt.X("week_label:N", sort=list(ORDER_INDEX.keys()), title=""),
        y=alt.Y("bankroll:Q", title="Bankroll ($)", scale=alt.Scale(domain=(y_min, y_max)) if y_min and y_max else alt.Undefined),
        tooltip=["week_label", "bankroll"]
    ).properties(height=260)

def chart_weekly_profit(df: pd.DataFrame, height: int) -> alt.Chart:
    d = df.copy()
    d["week_order"] = d["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    d = d.sort_values("week_order")
    return alt.Chart(d).mark_bar().encode(
        x=alt.X("week_label:N", sort=list(ORDER_INDEX.keys()), title=""),
        y=alt.Y("profit:Q", title="Weekly Profit ($)"),
        tooltip=["week_label", "profit"]
    ).properties(height=260)

def chart_cumprofit_alt(df: pd.DataFrame, height: int) -> alt.Chart:
    d = df.copy()
    d["week_order"] = d["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    d = d.sort_values("week_order")
    if "profit" in d.columns:
        d["cum_profit"] = d["profit"].cumsum()
    else:
        d["cum_profit"] = 0.0
    return alt.Chart(d).mark_area(opacity=0.35).encode(
        x=alt.X("week_label:N", sort=list(ORDER_INDEX.keys()), title=""),
        y=alt.Y("cum_profit:Q", title="Cumulative Profit ($)"),
        tooltip=["week_label", "cum_profit"]
    ).properties(height=260)

def chart_stake(df: pd.DataFrame, height: int) -> alt.Chart:
    d = df.copy()
    d["week_order"] = d["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    d = d.sort_values("week_order")
    return alt.Chart(d).mark_bar().encode(
        x=alt.X("week_label:N", sort=list(ORDER_INDEX.keys()), title=""),
        y=alt.Y("stake:Q", title="Weekly Stake ($)"),
        tooltip=["week_label", "stake"]
    ).properties(height=260)

def render(season: int):
    st.header("Portfolio")
    pnl = load_pnl(season)
    if pnl.empty:
        st.info("No portfolio data.")
        return

    g1, g2 = st.columns(2)
    with g1:
        st.altair_chart(chart_bankroll(pnl, 260), use_container_width=True)
    with g2:
        st.altair_chart(chart_weekly_profit(pnl, 260), use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        st.altair_chart(chart_cumprofit_alt(pnl, 260), use_container_width=True)
    with g4:
        st.altair_chart(chart_stake(pnl, 260), use_container_width=True)
