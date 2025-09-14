import streamlit as st
import pandas as pd
from nfl_dash.data_io import load_bets, load_scores_for_bets
from nfl_dash.utils import team_logo, week_sort_key

def _merge_scores(bets: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if bets.empty or scores.empty:
        return bets
    key_b = ["season","week","team","opponent"]
    bb = bets.copy()
    bb["pair"] = bb.apply(lambda r: "_".join(sorted([str(r.get("team","")), str(r.get("opponent",""))])), axis=1)
    ss = scores.copy()
    ss["pair"] = ss.apply(lambda r: "_".join(sorted([str(r.get("home_team","")), str(r.get("away_team",""))])), axis=1)
    merged = bb.merge(ss[["season","week","pair","score_home","score_away","home_team","away_team"]],
                      on=["season","week","pair"], how="left")
    def pick_score(row):
        if str(row.get("team","")).upper() == str(row.get("home_team","")).upper():
            return int(row["score_home"]) if pd.notna(row.get("score_home")) else None, int(row["score_away"]) if pd.notna(row.get("score_away")) else None
        if str(row.get("team","")).upper() == str(row.get("away_team","")).upper():
            return int(row["score_away"]) if pd.notna(row.get("score_away")) else None, int(row["score_home"]) if pd.notna(row.get("score_home")) else None
        return None, None
    th, to = [], []
    for _, r in merged.iterrows():
        a, b = pick_score(r)
        th.append(a); to.append(b)
    merged["team_score"] = th
    merged["opp_score"]  = to
    return merged

def _bet_card(row: pd.Series):
    team  = row.get("team","")
    opp   = row.get("opponent","")
    wl    = row.get("won", None)
    ml    = row.get("ml", row.get("decimal_odds",""))
    stake = row.get("stake", None)
    prof  = row.get("profit", None)
    wlab  = row.get("week_label","")
    sc_t  = row.get("team_score", None)
    sc_o  = row.get("opp_score", None)

    badge = ""
    if wl in (0,1):
        badge = f"<span style='padding:2px 10px;border-radius:12px;background:{'#0bb965' if wl==1 else '#e15241'};color:white;font-size:12px;'>{'WIN' if wl==1 else 'LOSS'}</span>"

    logo_l = team_logo(team)
    logo_r = team_logo(opp)
    score_html = ""
    if logo_l and logo_r and (sc_t is not None or sc_o is not None):
        score_html = f"""
        <div style="display:flex;align-items:center;gap:14px;justify-content:center;margin-top:6px;">
          <img src="{logo_l}" style="height:54px;width:auto;" />
          <div style="font-weight:800;font-size:22px;">{sc_t if sc_t is not None else '-' } – {sc_o if sc_o is not None else '-'}</div>
          <img src="{logo_r}" style="height:54px;width:auto;" />
        </div>
        """

    odds_html  = f"<div style='font-size:12px;color:#666;'>Moneyline: {ml}</div>" if pd.notna(ml) and ml != "" else ""
    stake_html = f"<div style='font-size:12px;color:#666;'>Stake: {stake:.2f}</div>" if stake is not None else ""
    profit_html= f"<div style='font-size:12px;color:{'#0bb965' if (prof or 0)>=0 else '#e15241'};'>Profit: {prof:.2f}</div>" if prof is not None else ""

    st.markdown(f"""
    <div style="border:1px solid #EEE;border-radius:12px;padding:12px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div style="font-weight:700;">{wlab} • {team} vs {opp}</div>
        {badge}
      </div>
      {score_html}
      <div style="display:flex;gap:16px;justify-content:center;margin-top:8px;">
        {odds_html}{stake_html}{profit_html}
      </div>
    </div>
    """, unsafe_allow_html=True)

def render(season: int, stage: str):
    bets = load_bets(season)
    if bets.empty:
        st.info("No bets for this season.")
        return

    scores = load_scores_for_bets(season)
    bets = _merge_scores(week_sort_key(bets), scores)
    bets = bets.sort_values(["week_order","schedule_date"]).reset_index(drop=True)

    cols = st.columns(3)
    for i, (_, r) in enumerate(bets.iterrows()):
        with cols[i % 3]:
            _bet_card(r)
