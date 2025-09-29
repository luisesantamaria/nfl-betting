# tabs/overview.py
import pandas as pd
import streamlit as st

from nfl_dash.data_io import load_pnl_weekly, load_bets_this_week
from nfl_dash.utils import (
    kpis_from_pnl,
    add_week_order,
    season_stage,
    week_label_to_num,
    norm_abbr,
)
from nfl_dash.charts import chart_sparkline_cumprofit, chart_last8_profit
from nfl_dash.live_scores import fetch_espn_scoreboard_df
from nfl_dash.components import bet_card as render_bet_card


# =========================
# Enriquecer bets con scoreboard ESPN (scores + estado)
# =========================
def _enrich_with_espn(bets_df: pd.DataFrame, season: int) -> pd.DataFrame:
    if bets_df.empty:
        return bets_df

    # Determinar semana de las bets mostradas
    wk = None
    if "week" in bets_df.columns:
        wk = (
            pd.to_numeric(bets_df["week"], errors="coerce")
            .dropna()
            .astype(int)
            .max()
        )
    if wk is None and "week_label" in bets_df.columns:
        nums = [week_label_to_num(x) for x in bets_df["week_label"].dropna().astype(str).unique()]
        nums = [n for n in nums if n is not None]
        wk = max(nums) if nums else None

    if wk is None:
        return bets_df

    sb = fetch_espn_scoreboard_df(season=int(season), week=int(wk))
    if sb is None or sb.empty:
        return bets_df

    # Normalizar a abreviaciones y crear índice de búsqueda por (home, away) y (away, home)
    sb = sb.copy()
    sb["home_abbr"] = sb["home_team"].astype(str).map(norm_abbr)
    sb["away_abbr"] = sb["away_team"].astype(str).map(norm_abbr)

    idx = {}
    for r in sb.itertuples(index=False):
        key1 = (r.home_abbr, r.away_abbr)
        key2 = (r.away_abbr, r.home_abbr)
        idx[key1] = r
        idx[key2] = r

    out = bets_df.copy()
    for c in ("score_home", "score_away", "short", "status"):
        if c not in out.columns:
            out[c] = pd.NA

    for i, r in out.iterrows():
        team = norm_abbr(str(r.get("team", "")))
        opp  = norm_abbr(str(r.get("opponent", "")))
        home = norm_abbr(str(r.get("home_team", "")))
        away = norm_abbr(str(r.get("away_team", "")))
        side = str(r.get("side", "")).lower()

        # Resolver quién es home/away de la apuesta
        h = a = ""
        if home and away:
            h, a = home, away
        elif team and opp:
            if side == "home":
                h, a = team, opp
            elif side == "away":
                h, a = opp, team
            else:
                # Desconocido: igual funciona porque indexamos ambas direcciones
                h, a = team, opp

        srow = idx.get((h, a))
        if srow:
            out.at[i, "score_home"] = srow.home_score
            out.at[i, "score_away"] = srow.away_score
            out.at[i, "short"]       = srow.short
            out.at[i, "status"]      = str(srow.state).upper()

    return out


def render(season: int):
    st.subheader("Overview")

    # Datos base
    pnl = load_pnl_weekly(season)
    stage = season_stage(season, pnl)
    bets_week = load_bets_this_week(season) if stage == "in_season" else pd.DataFrame()

    # Enriquecer bets de esta semana con ESPN (scores / estado)
    if not bets_week.empty:
        bets_week = _enrich_with_espn(bets_week, season)

    # Bets de esta semana (si hay)
    if not bets_week.empty:
        st.markdown("**This Week’s Bets**")
        cards = list(bets_week.itertuples(index=False))
        idx = 0
        cols_per_row = 4
        rows = (len(cards) + cols_per_row - 1) // cols_per_row
        for _ in range(rows):
            col_objs = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if idx < len(cards):
                    with col_objs[j]:
                        render_bet_card(pd.Series(cards[idx]._asdict()))
                    idx += 1
        st.divider()

    # Overview de temporada
    st.markdown("**Season Overview**")
    if pnl.empty:
        st.caption("No `pnl_weekly_{year}.csv` found for this season.")
        return

    # KPIs: initial fijo en 1000
    _, final_bankroll, total_profit, total_stake, yield_pct, profits, stakes = kpis_from_pnl(pnl)
    initial_bankroll_fixed = 1000.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Initial", f"${initial_bankroll_fixed:,.2f}")
    k2.metric("Final",   f"${final_bankroll:,.2f}", f"{(final_bankroll-initial_bankroll_fixed):,.2f}")
    k3.metric("Total Profit", f"${total_profit:,.2f}")
    k4.metric("Yield",        f"{yield_pct:.2f}%")

    # Alturas dinámicas: si NO hay bets arriba, agrandamos ambos charts por igual
    if bets_week.empty:
        H_BANK = 380
        H_PROF = 380
    else:
        H_BANK = 200
        H_PROF = 200

    # Cumulative profit (usa el orden de semana interno)
    cum_df = add_week_order(pd.DataFrame({
        "week_label": pnl["week_label"].astype(str),
        "cum_profit": profits.cumsum()
    }))

    cA, cB = st.columns(2)
    with cA:
        st.altair_chart(chart_sparkline_cumprofit(cum_df, height=H_BANK), use_container_width=True)
    with cB:
        st.altair_chart(chart_last8_profit(pnl, profits, last=8, height=H_PROF), use_container_width=True)
