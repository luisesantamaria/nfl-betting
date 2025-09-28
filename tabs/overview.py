# tabs/overview.py
import math
import numpy as np
import pandas as pd
import streamlit as st

from nfl_dash.data_io import load_pnl_weekly, load_bets_this_week
from nfl_dash.utils import (
    kpis_from_pnl, add_week_order, season_stage,
    ORDER_INDEX, norm_abbr
)
from nfl_dash.charts import chart_sparkline_cumprofit, chart_last8_profit
from nfl_dash.live_scores import fetch_espn_scoreboard_df


def _espn_scores_for_week(season: int, week: int) -> pd.DataFrame:
    """
    Igual que Live: trae el scoreboard de ESPN para esa (season, week)
    y lo normaliza a abrevs con norm_abbr.
    Devuelve: home_abbr, away_abbr, home_score, away_score, start_time, state
    """
    sb = fetch_espn_scoreboard_df(season=int(season), week=int(week))
    if sb.empty:
        return pd.DataFrame(columns=[
            "home_abbr","away_abbr","home_score","away_score","start_time","state"
        ])

    out = sb.copy()
    out["home_abbr"]  = out["home_team"].astype(str).map(norm_abbr)
    out["away_abbr"]  = out["away_team"].astype(str).map(norm_abbr)
    out["start_time"] = pd.to_datetime(out["start_time"], errors="coerce", utc=True)
    out["state"]      = out["state"].astype(str).str.lower()
    out = out[out["home_abbr"].notna() & out["away_abbr"].notna()].copy()
    return out[["home_abbr","away_abbr","home_score","away_score","start_time","state"]]


def _enrich_bets_with_espn(view: pd.DataFrame, debug: bool=False) -> pd.DataFrame:
    """
    Inyecta marcadores (live/final) en las bets usando solo ESPN.
    IMPORTANTe: escribe todas las variantes de columnas que el componente puede usar:
      - score_home / score_away
      - home_score / away_score (alias)
      - home_team / away_team
      - team_score / opponent_score
    """
    if view.empty:
        return view

    v = view.copy()

    # Tipos básicos
    if "schedule_date" in v.columns:
        v["schedule_date"] = pd.to_datetime(v["schedule_date"], errors="coerce", utc=True)
    for c in ("season","week"):
        if c in v.columns:
            v[c] = pd.to_numeric(v[c], errors="coerce")

    # Asegurar abrevs consistentes en bets
    for c in ("team","opponent"):
        if c in v.columns:
            v[c] = v[c].astype(str).map(norm_abbr)

    # Inicializa columnas esperadas por la card
    for col in ("score_home","score_away","home_score","away_score",
                "team_score","opponent_score","home_team","away_team"):
        if col not in v.columns:
            v[col] = np.nan

    grp_cols = [c for c in ("season","week") if c in v.columns]
    if not grp_cols:
        return v

    v = v.reset_index().rename(columns={"index": "bet_idx"})

    # Debug UI
    dbg_container = st.expander("Debug · ESPN matching (Overview)", expanded=False) if debug else None

    for (ssn, wk), idxs in v.groupby(grp_cols).groups.items():
        if pd.isna(ssn) or pd.isna(wk):
            continue

        sb = _espn_scores_for_week(int(ssn), int(wk))
        if debug and dbg_container is not None:
            with dbg_container:
                st.markdown(f"**(season={int(ssn)}, week={int(wk)}) — ESPN scoreboard**")
                st.dataframe(sb, use_container_width=True)

        if sb.empty:
            continue

        sub = v.loc[idxs].copy()

        # Emparejar por par de equipos (sin importar local/visitante)
        cand = sub.merge(sb, how="cross")
        same_pair = (
            ((cand["team"] == cand["home_abbr"]) & (cand["opponent"] == cand["away_abbr"])) |
            ((cand["team"] == cand["away_abbr"]) & (cand["opponent"] == cand["home_abbr"]))
        )
        cand = cand.loc[same_pair].copy()
        if debug and dbg_container is not None:
            with dbg_container:
                st.markdown("**Candidatos (emparejados por equipo)**")
                st.dataframe(cand, use_container_width=True)

        if cand.empty:
            continue

        # Kickoff más cercano a schedule_date
        if "schedule_date" in cand.columns:
            sd = pd.to_datetime(cand["schedule_date"], errors="coerce", utc=True)
            cand["abs_diff"] = (cand["start_time"] - sd).abs().dt.total_seconds()
        else:
            cand["abs_diff"] = 0

        pick = (cand.sort_values(["bet_idx","abs_diff"])
                    .groupby("bet_idx", as_index=False).first())

        # Escribir TODO en la bet:
        for _, row in pick.iterrows():
            i = int(row["bet_idx"])
            # equipos para el card (algunas UIs usan esto para logos)
            v.at[i, "home_team"] = row["home_abbr"]
            v.at[i, "away_team"] = row["away_abbr"]

            # scores en los dos formatos
            v.at[i, "score_home"] = row["home_score"]
            v.at[i, "score_away"] = row["away_score"]
            v.at[i, "home_score"] = row["home_score"]
            v.at[i, "away_score"] = row["away_score"]

            # derivadas por lado del pick
            is_team_home = (v.at[i, "team"] == row["home_abbr"])
            v.at[i, "team_score"] = row["home_score"] if is_team_home else row["away_score"]
            v.at[i, "opponent_score"] = row["away_score"] if is_team_home else row["home_score"]

    return v.drop(columns=["bet_idx"], errors="ignore")


def render(season: int):
    st.subheader("Overview")

    # Toggle de debug (no afecta producción si no lo activas)
    debug = st.checkbox("Debug ESPN matching (Overview)", value=False)

    pnl = load_pnl_weekly(season)
    stage = season_stage(season, pnl)

    # ---------- Bets de esta semana (scores vía ESPN) ----------
    bets_week = load_bets_this_week(season) if stage == "in_season" else pd.DataFrame()
    if not bets_week.empty:
        st.markdown("**This Week’s Bets**")

        view = bets_week.copy()
        try:
            view = _enrich_bets_with_espn(view, debug=debug)
        except Exception:
            if debug:
                st.exception("Error enriqueciendo bets con ESPN")
            # seguimos mostrando sin scores

        # Orden y tipos mínimos para las cards
        for c in ("score_home","score_away","home_score","away_score",
                  "team_score","opponent_score","profit","stake","decimal_odds"):
            if c in view.columns:
                view[c] = pd.to_numeric(view[c], errors="coerce")

        if "schedule_date" in view.columns:
            view["schedule_date"] = pd.to_datetime(view["schedule_date"], errors="coerce")

        if "week_label" in view.columns:
            view["week_label"] = view["week_label"].astype(str)
            view["__order"] = view["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
            sort_cols = ["__order"]
            if "schedule_date" in view.columns:
                sort_cols.append("schedule_date")
            view = view.sort_values(sort_cols).drop(columns="__order", errors="ignore")

        # Render cards (mismo componente que Bets)
        from nfl_dash.components import bet_card as render_bet_card
        cards = list(view.itertuples(index=False))
        idx = 0
        cols_per_row = 4
        rows = math.ceil(len(cards) / cols_per_row)
        for _ in range(rows):
            col_objs = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if idx < len(cards):
                    with col_objs[j]:
                        render_bet_card(pd.Series(cards[idx]._asdict()))
                    idx += 1

        st.divider()

    # ---------- Season Overview ----------
    st.markdown("**Season Overview**")
    if pnl.empty:
        st.caption("No `pnl.csv` found for this season.")
        return

    initial_bankroll, final_bankroll, total_profit, total_stake, yield_pct, profits, stakes = kpis_from_pnl(pnl)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Initial", f"${initial_bankroll:,.2f}")
    k2.metric("Final",   f"${final_bankroll:,.2f}", f"{(final_bankroll-initial_bankroll):,.2f}")
    k3.metric("Total Profit", f"${total_profit:,.2f}")
    k4.metric("Yield",        f"{yield_pct:.2f}%")

    H_BANK, H_PROF = (380, 380) if bets_week.empty else (200, 200)

    cum_df = add_week_order(pd.DataFrame({
        "week_label": pnl["week_label"].astype(str),
        "cum_profit": profits.cumsum()
    }))

    cA, cB = st.columns(2)
    with cA:
        st.altair_chart(chart_sparkline_cumprofit(cum_df, height=H_BANK), use_container_width=True)
    with cB:
        st.altair_chart(chart_last8_profit(pnl, profits, last=8, height=H_PROF), use_container_width=True)
