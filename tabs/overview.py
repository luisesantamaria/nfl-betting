import pandas as pd
import streamlit as st
from nfl_dash.data_io import load_pnl_weekly, load_bets_this_week
from nfl_dash.utils import kpis_from_pnl, add_week_order
from nfl_dash.charts import chart_sparkline_cumprofit, chart_last8_profit
from nfl_dash.utils import season_stage

def render(season: int):
    st.subheader("Overview")

    pnl = load_pnl_weekly(season)
    stage = season_stage(season, pnl)
    bets_week = load_bets_this_week(season) if stage == "in_season" else pd.DataFrame()

    # Bets de esta semana (si hay)
    if not bets_week.empty:
        st.markdown("**This Week’s Bets**")
        cols = [c for c in ["week","week_label","schedule_date","side","team","opponent","ml","decimal_odds","model_prob","edge","ev","stake"] if c in bets_week.columns]
        st.dataframe(bets_week[cols] if cols else bets_week, use_container_width=True)
        st.divider()

    st.markdown("**Season Overview**")
    if pnl.empty:
        st.caption("No `pnl_weekly_{year}.csv` found for this season.")
        return

    initial_bankroll, final_bankroll, total_profit, total_stake, yield_pct, profits, stakes = kpis_from_pnl(pnl)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Initial", f"${initial_bankroll:,.2f}")
    k2.metric("Final",   f"${final_bankroll:,.2f}", f"{(final_bankroll-initial_bankroll):,.2f}")
    k3.metric("Total Profit", f"${total_profit:,.2f}")
    k4.metric("Yield",        f"{yield_pct:.2f}%")

    # Altura dinámica si no hay bets arriba
    H_MAIN = 280 if bets_week.empty else 200
    H_MINI = 230 if bets_week.empty else 200

    cum_df = add_week_order(pd.DataFrame({"week_label": pnl["week_label"], "cum_profit": profits.cumsum()}))
    cA, cB = st.columns(2)
    with cA:
        st.altair_chart(chart_sparkline_cumprofit(cum_df, height=H_MAIN), use_container_width=True)
    with cB:
        st.altair_chart(chart_last8_profit(pnl, profits, last=8, height=H_MINI), use_container_width=True)
