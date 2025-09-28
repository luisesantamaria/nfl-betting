# tabs/overview.py
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from nfl_dash.data_io import load_pnl_weekly, load_bets_this_week
from nfl_dash.utils import (
    kpis_from_pnl, add_week_order, season_stage, ORDER_INDEX, norm_abbr
)
from nfl_dash.charts import chart_sparkline_cumprofit, chart_last8_profit
from nfl_dash.live_scores import fetch_espn_scoreboard_df

LIVE_ODDS = Path("data/live/odds.csv")


def _enrich_scores_for_display(view: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas para que las tarjetas muestren marcador:
      - home_team, away_team, home_score, away_score
      - team_score, opponent_score (por si el componente los usa)
    1) Intenta odds.csv (más confiable para 'FINAL')
    2) Fallback ESPN para LIVE (state='in')
    """
    if view.empty:
        return view

    v = view.copy()

    # --- 1) Enriquecer con odds.csv ---
    if LIVE_ODDS.exists():
        odds = pd.read_csv(LIVE_ODDS, low_memory=False)

        need = [
            "season", "week", "week_label", "schedule_date",
            "home_team", "away_team", "score_home", "score_away", "home_win"
        ]
        if all(c in odds.columns for c in need):
            # tipos/fechas
            for c in ("season", "week"):
                v[c] = pd.to_numeric(v[c], errors="coerce")
                odds[c] = pd.to_numeric(odds[c], errors="coerce")
            v["schedule_date"] = pd.to_datetime(v.get("schedule_date"), errors="coerce", utc=True)
            odds["schedule_date"] = pd.to_datetime(odds.get("schedule_date"), errors="coerce", utc=True)

            # normaliza abrevs
            for c in ("team", "opponent"):
                v[c] = v[c].astype(str).map(norm_abbr)
            for c in ("home_team", "away_team"):
                odds[c] = odds[c].astype(str).map(norm_abbr)

            tmp = v.reset_index().rename(columns={"index": "bet_idx"}).merge(
                odds[need],
                on=["season", "week", "week_label"],
                how="left",
                suffixes=("", "_o"),
            )

            same_pair = (
                ((tmp["team"] == tmp["home_team"]) & (tmp["opponent"] == tmp["away_team"])) |
                ((tmp["team"] == tmp["away_team"]) & (tmp["opponent"] == tmp["home_team"]))
            )
            tmp = tmp.loc[same_pair].copy()

            tmp["abs_time_diff"] = (tmp["schedule_date_o"] - tmp["schedule_date"]).abs().dt.total_seconds()
            tmp = tmp.sort_values(["bet_idx", "abs_time_diff"]).groupby("bet_idx", as_index=False).first()

            # copia a v
            v = v.reset_index().rename(columns={"index": "bet_idx"}).merge(
                tmp[["bet_idx", "home_team", "away_team", "score_home", "score_away", "home_win"]],
                on="bet_idx", how="left"
            ).drop(columns=["bet_idx"])

            # Deriva team/opponent scores por si el componente los usa
            # Identifica si el pick coincide con el home del evento
            is_team_home = v["home_team"].notna() & (v["team"] == v["home_team"])
            v["team_score"] = np.where(is_team_home, v["score_home"], v["score_away"])
            v["opponent_score"] = np.where(is_team_home, v["score_away"], v["score_home"])

    # --- 2) Fallback ESPN para LIVE ---
    # Si faltan scores pero el juego ya arrancó, intenta ESPN
    now_utc = datetime.now(timezone.utc)
    mask_need = (
        v["home_team"].notna() & v["away_team"].notna() &
        (v["score_home"].isna() | v["score_away"].isna()) &
        pd.to_datetime(v.get("schedule_date"), errors="coerce", utc=True).le(now_utc)
    )
    if mask_need.any():
        for wk in sorted(v.loc[mask_need, "week"].dropna().unique().tolist()):
            try:
                sb = fetch_espn_scoreboard_df(int(v["season"].iloc[0]), int(wk))
            except Exception:
                sb = pd.DataFrame()

            if sb.empty:
                continue

            # Normaliza a abrevs
            for c in ("home_team", "away_team"):
                sb[c] = sb[c].astype(str).map(norm_abbr)

            # quedarnos solo con juegos in/post
            sb = sb[sb["state"].isin(["in", "post"])].copy()

            # merge por par de equipos (sin orden)
            sub = v[(v["week"] == wk) & mask_need].reset_index().rename(columns={"index": "bet_idx"})
            cand = sub.merge(
                sb[["home_team", "away_team", "home_score", "away_score", "start_time", "state"]],
                on=[], how="cross"
            )

            # filtramos coincidencias de par
            same_pair = (
                ((cand["team"] == cand["home_team_y"]) & (cand["opponent"] == cand["away_team_y"])) |
                ((cand["team"] == cand["away_team_y"]) & (cand["opponent"] == cand["home_team_y"]))
            )
            cand = cand.loc[same_pair].copy()

            # kickoff más cercano
            cand["abs_time_diff"] = (pd.to_datetime(cand["start_time"], utc=True) -
                                     pd.to_datetime(cand["schedule_date"], utc=True)).abs().dt.total_seconds()
            cand = cand.sort_values(["bet_idx", "abs_time_diff"]).groupby("bet_idx", as_index=False).first()

            # aplicar updates: solo rellenamos donde falte
            if not cand.empty:
                upd = cand[["bet_idx", "home_team_y", "away_team_y", "home_score", "away_score", "state"]].copy()
                upd = upd.rename(columns={
                    "home_team_y": "home_team",
                    "away_team_y": "away_team",
                })
                v = sub.merge(upd, on="bet_idx", how="left", suffixes=("", "_espn")).set_index("bet_idx")

                for col in ("home_team", "away_team", "score_home", "score_away"):
                    espn_col = col if col in upd.columns else col.replace("score_", "")  # score_home->home_score
                    if espn_col in v.columns:
                        v[col] = v[col].combine_first(v[espn_col])

                # Recalcula team/opponent score cuando llenamos desde ESPN
                is_team_home = v["home_team"].notna() & (v["team"] == v["home_team"])
                v["team_score"] = v["team_score"].combine_first(np.where(is_team_home, v["score_home"], v["score_away"]))
                v["opponent_score"] = v["opponent_score"].combine_first(np.where(is_team_home, v["score_away"], v["score_home"]))

                # Regresa al DF original
                v = v.reset_index(drop=True)

                # Escribe de vuelta en el original por índices
                idxs = sub["bet_idx"].values
                base = view.index.take(idxs)
                view.loc[base, :] = v.loc[:, view.columns].values

                # y sigue con el v que venimos regresando
                v = view

    return v


def render(season: int):
    st.subheader("Overview")

    # --- Datos base
    pnl = load_pnl_weekly(season)
    stage = season_stage(season, pnl)

    # --- Bets de esta semana (solo leer + enriquecer PARA MOSTRAR)
    bets_week = load_bets_this_week(season) if stage == "in_season" else pd.DataFrame()
    if not bets_week.empty:
        st.markdown("**This Week’s Bets**")

        view = bets_week.copy()

        # Enriquecer scores para la UI (odds -> ESPN)
        try:
            view = _enrich_scores_for_display(view)
        except Exception:
            pass

        # Normaliza tipos mínimos (sin cálculos de negocio)
        for c in ("team_score", "opponent_score", "score_home", "score_away", "profit", "stake", "decimal_odds"):
            if c in view.columns:
                view[c] = pd.to_numeric(view[c], errors="coerce")
        if "schedule_date" in view.columns:
            view["schedule_date"] = pd.to_datetime(view["schedule_date"], errors="coerce")
        if "status" in view.columns:
            view["status"] = view["status"].astype(str).str.upper()
        if "week_label" in view.columns:
            view["week_label"] = view["week_label"].astype(str)

        # Orden como en Bets
        if "week_label" in view.columns:
            view["__order"] = view["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
            sort_cols = ["__order"]
            if "schedule_date" in view.columns:
                sort_cols.append("schedule_date")
            view = view.sort_values(sort_cols).drop(columns="__order", errors="ignore")

        # Render con el mismo componente de tarjetas
        from nfl_dash.components import bet_card as render_bet_card  # import perezoso
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

    # --- Overview de temporada (leyendo pnl.csv ya generado por el workflow)
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

    # Cumulative profit
    cum_df = add_week_order(pd.DataFrame({
        "week_label": pnl["week_label"].astype(str),
        "cum_profit": profits.cumsum()
    }))

    cA, cB = st.columns(2)
    with cA:
        st.altair_chart(chart_sparkline_cumprofit(cum_df, height=H_BANK), use_container_width=True)
    with cB:
        st.altair_chart(chart_last8_profit(pnl, profits, last=8, height=H_PROF), use_container_width=True)
