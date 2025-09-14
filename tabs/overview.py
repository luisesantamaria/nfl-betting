import streamlit as st
import pandas as pd
import altair as alt
from nfl_dash.utils import load_pnl, load_bets, load_scores_for_bets, week_sort_key, ORDER_INDEX, team_logo

def chart_bankroll(df: pd.DataFrame, height: int) -> alt.Chart:
    d = df.copy()
    d = d.dropna(subset=["week_label"])
    d["week_order"] = d["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    d = d.sort_values("week_order")
    y_min = max(0.0, d["bankroll"].min() - 40) if "bankroll" in d.columns else None
    y_max = d["bankroll"].max() + 40 if "bankroll" in d.columns else None
    return alt.Chart(d).mark_line(point=True).encode(
        x=alt.X("week_label:N", sort=list(ORDER_INDEX.keys()), title=""),
        y=alt.Y("bankroll:Q", title="Bankroll ($)", scale=alt.Scale(domain=(y_min, y_max)) if y_min and y_max else alt.Undefined),
        tooltip=["week_label", "bankroll"]
    ).properties(height=height)

def chart_weekly_profit(df: pd.DataFrame, height: int) -> alt.Chart:
    d = df.copy()
    d = d.dropna(subset=["week_label"])
    d["week_order"] = d["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    d = d.sort_values("week_order")
    return alt.Chart(d).mark_bar().encode(
        x=alt.X("week_label:N", sort=list(ORDER_INDEX.keys()), title=""),
        y=alt.Y("profit:Q", title="Weekly Profit ($)"),
        tooltip=["week_label", "profit"]
    ).properties(height=height)

def _bets_this_week(df_bets: pd.DataFrame) -> pd.DataFrame:
    if df_bets.empty or "schedule_date" not in df_bets.columns:
        return pd.DataFrame()
    b = df_bets.copy()
    b = b[b.get("won").isna() | (b["won"].astype(str) == "")]
    if "week_label" in b.columns:
        b = week_sort_key(b)
        wk = b["week_order"].min() if len(b) else None
        if wk is not None:
            b = b[b["week_order"] == wk].copy()
    return b

def _merge_scores(df_bets: pd.DataFrame, df_scores: pd.DataFrame) -> pd.DataFrame:
    if df_bets.empty or df_scores.empty:
        return df_bets
    l = df_bets.copy()
    l["pair"] = l.apply(lambda r: "_".join(sorted([str(r.get("team","")).upper(), str(r.get("opponent","")).upper()])), axis=1)
    s = df_scores.copy()
    s["pair"] = s.apply(lambda r: "_".join(sorted([str(r.get("home_team","")).upper(), str(r.get("away_team","")).upper()])), axis=1)
    keep = ["pair", "score_home", "score_away"]
    l = l.merge(s[keep], on="pair", how="left")
    return l.drop(columns=["pair"])

def render(season: int):
    st.header("Overview")
    pnl = load_pnl(season)
    bets = load_bets(season)
    scores = load_scores_for_bets(season)
    bets_open = _bets_this_week(bets)
    bets_open = _merge_scores(bets_open, scores)

    show_bets = not bets_open.empty
    H_BANK = 320 if show_bets else 380
    H_PROF = 320 if show_bets else 380

    col1, col2 = st.columns(2)
    with col1:
        if not pnl.empty and "bankroll" in pnl.columns:
            st.altair_chart(chart_bankroll(pnl, height=H_BANK), use_container_width=True)
        else:
            st.info("No portfolio data.")
    with col2:
        if not pnl.empty and "profit" in pnl.columns:
            st.altair_chart(chart_weekly_profit(pnl, height=H_PROF), use_container_width=True)
        else:
            st.info("No weekly profit data.")

    if show_bets:
        st.subheader("This Week's Bets")
        _render_bet_cards(bets_open)

def _render_bet_cards(df: pd.DataFrame):
    if df.empty:
        return
    df = df.copy()
    n = len(df)
    cols = st.columns(2) if n > 1 else [st.container()]
    for i, (_, r) in enumerate(df.iterrows()):
        slot = cols[i % len(cols)]
        with slot:
            _bet_card(r)

def _bet_card(r: pd.Series):
    team = str(r.get("team","")).upper()
    opp = str(r.get("opponent","")).upper()
    wl = r.get("week_label", "")
    sd = r.get("schedule_date", "")
    ml = r.get("ml", r.get("decimal_odds", ""))
    stake = r.get("stake", "")
    prob = r.get("model_prob", "")
    score_h = r.get("score_home", "")
    score_a = r.get("score_away", "")
    st.markdown(
        f"""
<div style="border:1px solid #EEE;border-radius:14px;padding:12px;margin-bottom:10px;">
  <div style="display:flex;align-items:center;gap:12px;justify-content:space-between;">
    <div style="display:flex;align-items:center;gap:12px;">
      <img src="{team_logo(team)}" width="36"/>
      <div style="font-weight:600;">{team} vs {opp}</div>
    </div>
    <div style="opacity:0.7;">{wl}</div>
  </div>
  <div style="margin-top:6px;display:flex;align-items:center;justify-content:space-between;">
    <div style="font-size:24px;font-weight:700;line-height:1;">
      {str(score_h) if pd.notna(score_h) and score_h!='' else '-'} : {str(score_a) if pd.notna(score_a) and score_a!='' else '-'}
    </div>
    <div style="text-align:right;">
      <div style="font-size:12px;opacity:0.7;">Model Prob</div>
      <div style="font-weight:600;">{'' if pd.isna(prob) else f"{float(prob)*100:.1f}%"} </div>
    </div>
  </div>
  <div style="margin-top:8px;display:flex;align-items:center;justify-content:space-between;opacity:0.85;">
    <div>Moneyline: {'' if pd.isna(ml) else ml}</div>
    <div>Stake: {'' if pd.isna(stake) else f"${float(stake):.2f}"}</div>
  </div>
  <div style="margin-top:4px;opacity:0.6;font-size:12px;">{sd}</div>
</div>
""",
        unsafe_allow_html=True
    )
