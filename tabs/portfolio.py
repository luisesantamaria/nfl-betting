import pandas as pd
import streamlit as st
from nfl_dash.data_io import load_pnl_weekly
from nfl_dash.utils import add_week_order
from nfl_dash.charts import (
    chart_bankroll,
    chart_profit_bars,
    chart_drawdown_area,
    chart_stake_bars,
)

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
    st.subheader("Portfolio")

    pnl = load_pnl_weekly(season)
    if pnl.empty:
        st.caption("No `pnl.csv` found for this season.")
        return

    profits, stakes = _profits_and_stakes(pnl)
    total_profit = float(profits.sum())
    total_stake  = float(stakes.sum()) if len(stakes) else 0.0
    yield_pct    = (total_profit / total_stake * 100.0) if total_stake > 0 else 0.0
    final_bankroll = INITIAL_BANKROLL + total_profit

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Initial", f"${INITIAL_BANKROLL:,.2f}")
    m2.metric("Final",   f"${final_bankroll:,.2f}", f"{(final_bankroll-INITIAL_BANKROLL):,.2f}")
    m3.metric("Total Profit", f"${total_profit:,.2f}")
    m4.metric("Yield",        f"{yield_pct:.2f}%")

    bank_df  = add_week_order(pd.DataFrame({"week_label": pnl["week_label"].astype(str), "bankroll": INITIAL_BANKROLL + profits.cumsum()}))
    prof_df  = add_week_order(pd.DataFrame({"week_label": pnl["week_label"].astype(str), "profit": profits, "stake": stakes}))
    stake_df = add_week_order(pd.DataFrame({"week_label": pnl["week_label"].astype(str), "stake":  stakes}))

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    with c1: st.altair_chart(chart_bankroll(bank_df, height=220, auto_domain=True), use_container_width=True)
    with c2: st.altair_chart(chart_profit_bars(prof_df, height=220), use_container_width=True)
    with c3: st.altair_chart(chart_drawdown_area(bank_df, height=220), use_container_width=True)
    with c4: st.altair_chart(chart_stake_bars(stake_df, height=220), use_container_width=True)
