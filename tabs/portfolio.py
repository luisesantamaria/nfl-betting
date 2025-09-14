import pandas as pd
import streamlit as st
from nfl_dash.data_io import load_pnl_weekly
from nfl_dash.utils import kpis_from_pnl, add_week_order
from nfl_dash.charts import (
    chart_bankroll,
    chart_profit_bars,
    chart_drawdown_area,
    chart_stake_bars,
)

def render(season: int):
    st.subheader("Portfolio")

    pnl = load_pnl_weekly(season)
    if pnl.empty:
        st.caption("No `pnl_weekly_{year}.csv` found for this season.")
        return

    initial_bankroll, final_bankroll, total_profit, total_stake, yield_pct, profits, stakes = kpis_from_pnl(pnl)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Initial", f"${initial_bankroll:,.2f}")
    m2.metric("Final",   f"${final_bankroll:,.2f}", f"{(final_bankroll-initial_bankroll):,.2f}")
    m3.metric("Total Profit", f"${total_profit:,.2f}")
    m4.metric("Yield",        f"{yield_pct:.2f}%")

    bank_df  = add_week_order(pnl[["week_label", "bankroll"]].dropna())
    prof_df  = add_week_order(pd.DataFrame({"week_label": pnl["week_label"], "profit": profits, "stake": stakes}))
    stake_df = add_week_order(pd.DataFrame({"week_label": pnl["week_label"], "stake": stakes}))

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    with c1: st.altair_chart(chart_bankroll(bank_df, height=220), use_container_width=True)
    with c2: st.altair_chart(chart_profit_bars(prof_df, height=220), use_container_width=True)
    with c3: st.altair_chart(chart_drawdown_area(bank_df, height=220), use_container_width=True)
    with c4: st.altair_chart(chart_stake_bars(stake_df, height=220), use_container_width=True)
