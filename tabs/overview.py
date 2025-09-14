import streamlit as st
import pandas as pd
from nfl_dash.data_io import load_pnl, load_bets_this_week
from nfl_dash.charts import bankroll_line, weekly_profit_bar
from nfl_dash.utils import team_logo

def _bet_card(df_row: pd.Series):
    home = df_row.get("team", "")
    away = df_row.get("opponent", "")
    odds = df_row.get("ml", df_row.get("decimal_odds", ""))
    wl   = df_row.get("won", None)
    stake = df_row.get("stake", None)
    prof  = df_row.get("profit", None)
    side  = str(df_row.get("side","")).upper()
    wl_badge = ""
    if wl in (0,1):
        wl_badge = f"<span style='padding:2px 8px;border-radius:10px;background:{'#0bb965' if wl==1 else '#e15241'};color:white;font-size:12px;'>{'WIN' if wl==1 else 'LOSS'}</span>"

    logo_l = team_logo(home)
    logo_r = team_logo(away)
    logos_html = ""
    if logo_l and logo_r:
        logos_html = f"""
        <div style="display:flex;align-items:center;gap:12px;justify-content:center;margin-top:4px;">
          <img src="{logo_l}" style="height:42px;width:auto;" />
          <span style="font-weight:700;font-size:18px;">vs</span>
          <img src="{logo_r}" style="height:42px;width:auto;" />
        </div>
        """

    stake_html = f"<div style='font-size:12px;color:#666;'>Stake: {stake:.2f}</div>" if stake is not None else ""
    profit_html = f"<div style='font-size:12px;color:{'#0bb965' if (prof or 0)>=0 else '#e15241'};'>Profit: {prof:.2f}</div>" if prof is not None else ""
    odds_html = f"<div style='font-size:12px;color:#666;'>Moneyline: {odds}</div>" if pd.notna(odds) and odds != "" else ""

    st.markdown(f"""
    <div style="border:1px solid #EEE;border-radius:12px;padding:12px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div style="font-weight:700;">{side} • {home} vs {away}</div>
        {wl_badge}
      </div>
      {logos_html}
      <div style="display:flex;gap:14px;justify-content:center;margin-top:6px;">
        {odds_html}{stake_html}{profit_html}
      </div>
    </div>
    """, unsafe_allow_html=True)

def render(season: int, stage: str):
    bets_wk = load_bets_this_week(season)
    show_bets = not bets_wk.empty

    if show_bets:
        st.markdown(f"**This Week’s Bets** • *{stage}*")
        cols = st.columns(3)
        for i, (_, r) in enumerate(bets_wk.iterrows()):
            with cols[i % 3]:
                _bet_card(r)
        st.markdown("---")

    pnl = load_pnl(season)
    if pnl.empty:
        st.info("No PnL data available for this season.")
        return

    H_BANK = 360 if not show_bets else 280
    H_PROF = 360 if not show_bets else 260

    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(bankroll_line(pnl, height=H_BANK), use_container_width=True)
    with c2:
        st.altair_chart(weekly_profit_bar(pnl, height=H_PROF), use_container_width=True)
