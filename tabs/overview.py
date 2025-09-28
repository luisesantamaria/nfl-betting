# tabs/overview.py
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from nfl_dash.data_io import load_pnl_weekly, load_bets_this_week
from nfl_dash.utils import kpis_from_pnl, add_week_order, season_stage, ORDER_INDEX
from nfl_dash.charts import chart_sparkline_cumprofit, chart_last8_profit


LIVE_ODDS = Path("data/live/odds.csv")


def _fallback_enrich_with_live_odds(bets_week: pd.DataFrame) -> pd.DataFrame:
    """
    Enriquecimiento mínimo usando data/live/odds.csv cuando enrich_bets_with_scores
    no aporta scores/estado. Empareja por (season, week, week_label) + par de equipos,
    y el kickoff más cercano.
    """
    if not LIVE_ODDS.exists() or bets_week.empty:
        return bets_week

    odds = pd.read_csv(LIVE_ODDS, low_memory=False)
    need_cols = [
        "season", "week", "week_label", "schedule_date",
        "home_team", "away_team", "score_home", "score_away", "home_win"
    ]
    if not all(c in odds.columns for c in need_cols):
        return bets_week

    # Tipos/fechas
    bets = bets_week.copy()
    for c in ("season", "week"):
        if c in bets.columns:
            bets[c] = pd.to_numeric(bets[c], errors="coerce")
    bets["schedule_date"] = pd.to_datetime(bets.get("schedule_date"), errors="coerce", utc=True)

    odds["schedule_date"] = pd.to_datetime(odds.get("schedule_date"), errors="coerce", utc=True)
    for c in ("season", "week"):
        odds[c] = pd.to_numeric(odds[c], errors="coerce")

    # Índice para escoger mejor match
    bets = bets.reset_index().rename(columns={"index": "bet_idx"})

    cand = bets.merge(
        odds[need_cols],
        on=["season", "week", "week_label"],
        how="left",
        suffixes=("", "_o"),
    )

    same_pair = (
        ((cand["team"] == cand["home_team"]) & (cand["opponent"] == cand["away_team"])) |
        ((cand["team"] == cand["away_team"]) & (cand["opponent"] == cand["home_team"]))
    )
    cand = cand.loc[same_pair].copy()

    cand["abs_time_diff"] = (cand["schedule_date_o"] - cand["schedule_date"]).abs().dt.total_seconds().abs()
    cand = cand.sort_values(["bet_idx", "abs_time_diff"]).groupby("bet_idx", as_index=False).first()

    # team_score/opponent_score
    cand["team_is_home"] = (cand["team"] == cand["home_team"])
    cand["team_score"] = np.where(cand["team_is_home"], cand["score_home"], cand["score_away"])
    cand["opponent_score"] = np.where(cand["team_is_home"], cand["score_away"], cand["score_home"])

    # status: final/live/open
    now_utc = datetime.now(timezone.utc)
    has_scores = cand["team_score"].notna() & cand["opponent_score"].notna()
    is_final = cand["home_win"].notna()
    is_live = has_scores & ~is_final & (cand["schedule_date"] <= now_utc)
    is_open = ~has_scores & (cand["schedule_date"] > now_utc)

    cand["status"] = np.where(is_final, "FINAL", np.where(is_live, "LIVE", "OPEN"))

    # result + profit solo para FINAL
    def _result(row):
        if row["status"] != "FINAL":
            return None
        if math.isclose(float(row["team_score"]), float(row["opponent_score"]), rel_tol=0.0, abs_tol=1e-9):
            return "PUSH"
        return "WIN" if row["team_score"] > row["opponent_score"] else "LOSS"

    cand["result"] = cand.apply(_result, axis=1)

    def _profit(row):
        if row.get("status") != "FINAL":
            return np.nan
        stake = float(row.get("stake", 0.0) or 0.0)
        odds_dec = float(row.get("decimal_odds", np.nan))
        res = row.get("result")
        if not np.isfinite(odds_dec):
            return np.nan
        if res == "WIN":
            return stake * (odds_dec - 1.0)
        if res == "LOSS":
            return -stake
        return 0.0  # PUSH/VOID
    cand["profit"] = cand.apply(_profit, axis=1)

    # Mantén columnas originales + las nuevas
    keep_new = ["team_score", "opponent_score", "status", "result", "profit"]
    out = bets.merge(cand[["bet_idx"] + keep_new], on="bet_idx", how="left")
    out = out.drop(columns=["bet_idx"], errors="ignore")
    return out


def render(season: int):
    st.subheader("Overview")

    # Datos base
    pnl = load_pnl_weekly(season)
    stage = season_stage(season, pnl)

    # Bets de esta semana (tarjetas)
    bets_week = load_bets_this_week(season) if stage == "in_season" else pd.DataFrame()
    if not bets_week.empty:
        st.markdown("**This Week’s Bets**")

        view = bets_week.copy()

        # 1) Intentar enrich oficial (si existe)
        try:
            from nfl_dash.odds_scores import enrich_bets_with_scores
            view = enrich_bets_with_scores(view, season)
        except Exception:
            pass

        # 2) Fallback: si no hay columnas de score/estado, tratar de enriquecer con live/odds.csv
        if not {"team_score", "opponent_score"}.issubset(set(view.columns)):
            view = _fallback_enrich_with_live_odds(view)

        # 3) Asegurar status/resultado/profit coherentes
        now_utc = datetime.now(timezone.utc)
        if "status" not in view.columns:
            # Derivar status mínimo: FINAL si tiene home_win, LIVE si tiene score y ya arrancó, si no OPEN
            hw = view.get("home_win")
            has_hw = (hw.notna()) if hw is not None else pd.Series(False, index=view.index)
            has_scores = view.get("team_score").notna() & view.get("opponent_score").notna() if \
                {"team_score", "opponent_score"}.issubset(view.columns) else pd.Series(False, index=view.index)
            sched = pd.to_datetime(view.get("schedule_date"), errors="coerce", utc=True)
            started = sched.notna() & (sched <= now_utc)
            view["status"] = np.where(has_hw, "FINAL", np.where(has_scores & started, "LIVE", "OPEN"))

        # Profit solo si FINAL
        if "profit" not in view.columns:
            def _profit_row(r):
                if str(r.get("status", "")).upper() != "FINAL":
                    return np.nan
                stake = float(r.get("stake", 0.0) or 0.0)
                odds_dec = float(r.get("decimal_odds", np.nan))
                ts, oscore = r.get("team_score"), r.get("opponent_score")
                if pd.isna(odds_dec) or pd.isna(ts) or pd.isna(oscore):
                    return np.nan
                if math.isclose(float(ts), float(oscore), rel_tol=0.0, abs_tol=1e-9):
                    return 0.0
                return stake * (odds_dec - 1.0) if ts > oscore else -stake
            view["profit"] = view.apply(_profit_row, axis=1)

        # Ordenar como en Bets
        if "week_label" in view.columns:
            view["week_label"] = view["week_label"].astype(str)
            view["__order"] = view["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
            sort_cols = ["__order"]
            if "schedule_date" in view.columns:
                view["schedule_date"] = pd.to_datetime(view["schedule_date"], errors="coerce")
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

    # Overview de temporada
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

    # Alturas dinámicas: si NO hay bets arriba, agrandamos ambos charts por igual
    if bets_week.empty:
        H_BANK = 380
        H_PROF = 380
    else:
        H_BANK = 200
        H_PROF = 200

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
