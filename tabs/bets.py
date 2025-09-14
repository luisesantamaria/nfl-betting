from __future__ import annotations
import pandas as pd
import streamlit as st

from nfl_dash.data_io import load_ledger
from nfl_dash.utils import (
    week_sort_key,
    week_label_from_num,
    team_logo,
    american_to_decimal,
    norm_abbr,
    load_scores_for_bets,
)

CARD_CSS = """
<style>
.bet-card {border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:14px; margin-bottom:12px;}
.bet-head {display:flex; justify-content:space-between; align-items:center; font-size:0.9rem; opacity:0.85;}
.bet-score {display:flex; justify-content:center; align-items:center; gap:16px; margin:8px 0 6px 0;}
.bet-score .team {display:flex; align-items:center; gap:10px;}
.bet-score img {width:40px; height:40px;}
.bet-score .nums {font-size:1.4rem; font-weight:700;}
.bet-foot {display:flex; justify-content:space-between; align-items:center; font-size:0.9rem; opacity:0.9;}
.badge-win {color:#21c55d; font-weight:700;}
.badge-loss {color:#ef4444; font-weight:700;}
</style>
"""

def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # week & label
    if "week" not in out.columns and "week_num" in out.columns:
        out["week"] = out["week_num"]
    if "week_label" not in out.columns:
        if "week" in out.columns:
            out["week_label"] = out["week"].apply(week_label_from_num)
        else:
            out["week_label"] = "Week 999"

    # team/opponent
    if "team" not in out.columns:
        if {"home_team","away_team"}.issubset(out.columns):
            out["team"] = out["home_team"]
            out["opponent"] = out["away_team"]
    if "opponent" not in out.columns and "opp" in out.columns:
        out["opponent"] = out["opp"]

    for c in ("team","opponent","home_team","away_team"):
        if c in out.columns:
            out[c] = out[c].astype(str).map(norm_abbr)

    # odds & money
    if "decimal_odds" not in out.columns and "ml" in out.columns:
        out["decimal_odds"] = out["ml"].apply(american_to_decimal)

    for c in ("ml","stake","profit","decimal_odds"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # result flag if available
    if "profit" in out.columns:
        out["result_flag"] = out["profit"].apply(lambda x: "WIN" if pd.notna(x) and x > 0 else ("LOSS" if pd.notna(x) and x < 0 else ""))

    return out

def _merge_scores(bets: pd.DataFrame, year: int) -> pd.DataFrame:
    scores = load_scores_for_bets(year)
    if scores.empty:
        return bets

    df = bets.copy()

    # prefer event_id join
    if "event_id" in df.columns and "event_id" in scores.columns:
        df = df.merge(
            scores[["event_id","score_home","score_away","home_team","away_team"]],
            on="event_id", how="left"
        )
    # fallback by teams
    elif {"home_team","away_team"}.issubset(df.columns) and {"home_team","away_team"}.issubset(scores.columns):
        df = df.merge(
            scores[["home_team","away_team","score_home","score_away"]],
            on=["home_team","away_team"], how="left"
        )
    # final fallback by normalized team/opponent
    elif {"team","opponent"}.issubset(df.columns) and {"home_team","away_team"}.issubset(scores.columns):
        m = scores.rename(columns={"home_team":"team","away_team":"opponent"})
        df = df.merge(m[["team","opponent","score_home","score_away"]], on=["team","opponent"], how="left")

    return df

def _render_card(row: pd.Series):
    team = row.get("team") or row.get("home_team") or ""
    opp  = row.get("opponent") or row.get("away_team") or ""
    wl   = str(row.get("result_flag") or "").upper()
    wl_badge = f'<span class="badge-win">{wl}</span>' if wl == "WIN" else (f'<span class="badge-loss">{wl}</span>' if wl == "LOSS" else "")
    wk   = row.get("week_label", "")
    ml   = row.get("ml")
    stake = row.get("stake")
    profit = row.get("profit")
    s_h = row.get("score_home")
    s_a = row.get("score_away")

    logo_team = team_logo(team, size=40) or ""
    logo_opp  = team_logo(opp, size=40) or ""

    score_html = ""
    if pd.notna(s_h) and pd.notna(s_a):
        score_html = f"""
        <div class="bet-score">
          <div class="team">{f'<img src="{logo_team}"/>' if logo_team else ''}<div>{team}</div></div>
          <div class="nums">{int(s_h)}&nbsp;–&nbsp;{int(s_a)}</div>
          <div class="team"><div>{opp}</div>{f'<img src="{logo_opp}"/>' if logo_opp else ''}</div>
        </div>
        """
    else:
        score_html = f"""
        <div class="bet-score">
          <div class="team">{f'<img src="{logo_team}"/>' if logo_team else ''}<div>{team}</div></div>
          <div style="opacity:.65">vs</div>
          <div class="team"><div>{opp}</div>{f'<img src="{logo_opp}"/>' if logo_opp else ''}</div>
        </div>
        """

    ml_txt = f"Moneyline {int(ml) if pd.notna(ml) else '—'}"
    foot_l = ml_txt
    foot_r = []
    if pd.notna(stake):  foot_r.append(f"Stake ${stake:,.2f}")
    if pd.notna(profit): foot_r.append(f"Profit ${profit:,.2f}")
    foot_r_txt = " · ".join(foot_r) if foot_r else ""

    st.markdown(
        f"""
        <div class="bet-card">
          <div class="bet-head"><div>{wk}</div><div>{wl_badge}</div></div>
          {score_html}
          <div class="bet-foot"><div>{foot_l}</div><div>{foot_r_txt}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render(year: int):
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    df = load_ledger(year)
    if df.empty:
        st.info("No bets for this season.")
        return

    df = _standardize(df)
    df = _merge_scores(df, year)
    df = week_sort_key(df).sort_values("week_order", kind="stable")

    # grid responsiva: 3 columnas en desktop
    cols_per_row = 3
    rows = [df.iloc[i:i+cols_per_row] for i in range(0, len(df), cols_per_row)]
    for chunk in rows:
        cols = st.columns(len(chunk))
        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                _render_card(row)
