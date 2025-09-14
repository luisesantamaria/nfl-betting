import streamlit as st
import pandas as pd
from nfl_dash.data_io import load_pnl
from nfl_dash.charts import bankroll_line, weekly_profit_bar, cum_profit_area, yield_line

def _metrics(pnl: pd.DataFrame):
    final_bk = float(pnl["bankroll"].iloc[-1]) if "bankroll" in pnl.columns and len(pnl) else 0.0
    total_profit = float(pnl["profit"].sum()) if "profit" in pnl.columns else 0.0
    n_bets = int((pnl["n_bets"].sum()) if "n_bets" in pnl.columns else 0)
    st.metric("Final Bankroll", f"${final_bk:,.2f}")
    st.metric("Total Profit", f"${total_profit:,.2f}")
    st.metric("Bets (sum)", f"{n_bets}")

def render(season: int, stage: str):
    pnl = load_pnl(season)
    if pnl.empty:
        st.info("No PnL data available for this season.")
        return

    m1, m2, m3 = st.columns(3)
    with m1: _metrics(pnl)
    with m2: st.write("")
    with m3: st.write(f"**Stage:** {stage}")

    st.markdown("---")

    H = 260
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.altair_chart(bankroll_line(pnl, height=H), use_container_width=True)
    with r1c2:
        st.altair_chart(weekly_profit_bar(pnl, height=H), use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.altair_chart(cum_profit_area(pnl, height=H), use_container_width=True)
    with r2c2:
        st.altair_chart(yield_line(pnl, height=H), use_container_width=True)
