import pandas as pd
import streamlit as st
from nfl_dash.data_io import load_pnl_weekly, load_bets_this_week
from nfl_dash.utils import add_week_order, season_stage
from nfl_dash.charts import chart_sparkline_cumprofit, chart_last8_profit

INITIAL_BANKROLL = 1000.0

def _profits_and_stakes(pnl: pd.DataFrame):
    if pnl.empty:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")
    profits = pd.to_numeric(pnl.get("profit"), errors="coerce") if "profit" in pnl.columns else None
    if profits is None or profits.isna().all():
        bk = pd.to_numeric(pnl.get("bankroll"), errors="coerce")
        profits = bk.diff().fillna(0.0)
    stakes = pd.to_numeric(pnl.get("stake"), errors="coerce") if "stake" in pnl.columns else pd.Series([0.0]*len(pnl))
    profits = profits.fillna(0.0)
    stakes = stakes.fillna(0.0)
    return profits, stakes

def render(season: int):
    st.subheader("Overview")

    pnl = load_pnl_weekly(season)
    stage = season_stage(season, pnl)
    bets_week = load_bets_this_week(season) if stage == "in_season" else pd.DataFrame()

    if not bets_week.empty:
        st.markdown("**This Week’s Bets**")
        cols = [c for c in ["week","week_label","schedule_date","side","team","opponent","ml","decimal_odds","model_prob","edge","ev","stake"] if c in bets_week.columns]
        st.dataframe(bets_week[cols] if cols else bets_week, use_container_width=True)
        st.divider()

    st.markdown("**Season Overview**")
    if pnl.empty:
        st.caption("No `pnl.csv` found for this season.")
        return

    profits, stakes = _profits_and_stakes(pnl)
    total_profit = float(profits.sum())
    total_stake  = float(stakes.sum()) if len(stakes) else 0.0
    yield_pct    = (total_profit / total_stake * 100.0) if total_stake > 0 else 0.0
    final_bankroll = INITIAL_BANKROLL + total_profit

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Initial", f"${INITIAL_BANKROLL:,.2f}")
    k2.metric("Final",   f"${final_bankroll:,.2f}", f"{(final_bankroll-INITIAL_BANKROLL):,.2f}")
    k3.metric("Total Profit", f"${total_profit:,.2f}")
    k4.metric("Yield",        f"{yield_pct:.2f}%")

    H_MAIN = 280 if bets_week.empty else 200
    H_MINI = 230 if bets_week.empty else 200

    cum_df = add_week_order(pd.DataFrame({"week_label": pnl["week_label"].astype(str), "cum_profit": profits.cumsum()}))
    cA, cB = st.columns(2)
    with cA:
        st.altair_chart(chart_sparkline_cumprofit(cum_df, height=H_MAIN), use_container_width=True)
    with cB:
        st.altair_chart(chart_last8_profit(pnl, profits, last=8, height=H_MINI), use_container_width=True)
