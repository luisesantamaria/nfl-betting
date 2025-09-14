import streamlit as st
import pandas as pd
from nfl_dash.utils import load_bets, load_scores_for_bets, week_sort_key, ORDER_INDEX, team_logo

def render(season: int):
    st.header("Bets")
    bets = load_bets(season)
    if bets.empty:
        st.info("No bets.")
        return

    scores = load_scores_for_bets(season)
    if not scores.empty:
        bets = _merge_scores(bets, scores)

    bets = week_sort_key(bets).sort_values(["week_order", "schedule_date"]).reset_index(drop=True)
    _render_bet_cards_grid(bets)

def _merge_scores(df_bets: pd.DataFrame, df_scores: pd.DataFrame) -> pd.DataFrame:
    l = df_bets.copy()
    s = df_scores.copy()
    l["pair"] = l.apply(lambda r: "_".join(sorted([str(r.get("team","")).upper(), str(r.get("opponent","")).upper()])), axis=1)
    s["pair"] = s.apply(lambda r: "_".join(sorted([str(r.get("home_team","")).upper(), str(r.get("away_team","")).upper()])), axis=1)
    keep = ["pair", "score_home", "score_away"]
    l = l.merge(s[keep], on="pair", how="left")
    return l.drop(columns=["pair"])

def _render_bet_cards_grid(df: pd.DataFrame):
    ncols = 2
    cols = st.columns(ncols)
    for i, (_, r) in enumerate(df.iterrows()):
        with cols[i % ncols]:
            _bet_card(r)

def _bet_card(r: pd.Series):
    team = str(r.get("team","")).upper()
    opp = str(r.get("opponent","")).upper()
    wl = r.get("week_label", "")
    sd = r.get("schedule_date", "")
    ml = r.get("ml", r.get("decimal_odds", ""))
    stake = r.get("stake", "")
    profit = r.get("profit", "")
    won = r.get("won", "")
    score_h = r.get("score_home", "")
    score_a = r.get("score_away", "")

    win_str = "" if pd.isna(won) or str(won)=="" else ("WIN" if int(won)==1 else "LOSS")
    win_color = "#0a7a0a" if win_str=="WIN" else ("#b00020" if win_str=="LOSS" else "#999")

    st.markdown(
        f"""
<div style="border:1px solid #EEE;border-radius:14px;padding:12px;margin-bottom:10px;">
  <div style="display:flex;align-items:center;gap:12px;justify-content:space-between;">
    <div style="display:flex;align-items:center;gap:12px;">
      <img src="{team_logo(team)}" width="36"/>
      <div style="font-weight:600;">{team} vs {opp}</div>
    </div>
    <div style="font-weight:700;color:{win_color};">{win_str}</div>
  </div>
  <div style="margin-top:6px;display:flex;align-items:center;justify-content:space-between;">
    <div style="font-size:24px;font-weight:700;line-height:1;">
      {str(score_h) if pd.notna(score_h) and score_h!='' else '-'} : {str(score_a) if pd.notna(score_a) and score_a!='' else '-'}
    </div>
    <div style="text-align:right;opacity:0.8;">{wl}</div>
  </div>
  <div style="margin-top:8px;display:flex;align-items:center;justify-content:space-between;opacity:0.9;">
    <div>Moneyline: {'' if pd.isna(ml) else ml}</div>
    <div>Stake: {'' if pd.isna(stake) else f"${float(stake):.2f}"}</div>
  </div>
  <div style="margin-top:4px;display:flex;align-items:center;justify-content:space-between;opacity:0.9;">
    <div>Date: {sd}</div>
    <div>Profit: {'' if pd.isna(profit) else f"${float(profit):.2f}"}</div>
  </div>
</div>
""",
        unsafe_allow_html=True
    )
