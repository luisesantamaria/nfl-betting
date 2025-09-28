# tabs/overview.py
import math
import pandas as pd
import streamlit as st

from nfl_dash.data_io import load_pnl_weekly, load_bets_this_week
from nfl_dash.utils import kpis_from_pnl, add_week_order, season_stage, norm_abbr
from nfl_dash.charts import chart_sparkline_cumprofit, chart_last8_profit
from nfl_dash.live_scores import fetch_espn_scoreboard_df


def _espn_scores_for_week(season: int, week: int) -> pd.DataFrame:
    sb = fetch_espn_scoreboard_df(season=int(season), week=int(week))
    if sb.empty:
        return pd.DataFrame(columns=[
            "home_abbr","away_abbr","home_score","away_score","start_time","state","short"
        ])
    out = sb.copy()
    out["home_abbr"]  = out["home_team"].astype(str).map(norm_abbr)
    out["away_abbr"]  = out["away_team"].astype(str).map(norm_abbr)
    out["start_time"] = pd.to_datetime(out["start_time"], errors="coerce", utc=True)
    out["state"]      = out["state"].astype(str).str.lower()
    out["short"]      = out["short"].astype(str)
    out = out[out["home_abbr"].notna() & out["away_abbr"].notna()].copy()
    return out[["home_abbr","away_abbr","home_score","away_score","start_time","state","short"]]


def _enrich_bets_with_espn(view: pd.DataFrame, debug: bool=False) -> pd.DataFrame:
    """
    Empareja cada bet (team/opponent) con el juego de ESPN de la misma week.
    Añade: score_home, score_away, state, status_short.
    """
    v = view.copy()
    if v.empty or "week_label" not in v.columns:
        return v

    # Numeric 'week' si no existe
    if "week" not in v.columns:
        v["week"] = (
            v["week_label"].astype(str).str.replace("Week ", "", regex=False)
            .where(~v["week_label"].astype(str).str.contains("Week "), None)
        )
        v["week"] = pd.to_numeric(v["week"], errors="coerce")

    # Agrupa por season/week
    grp_cols = ["season", "week"]
    if "season" not in v.columns:
        v["season"] = pd.to_numeric(v.get("year", 0), errors="coerce").fillna(0).astype(int)

    # Prepara columnas destino
    for col in ["score_home","score_away","home_team","away_team","state","status_short"]:
        if col not in v.columns:
            v[col] = pd.NA

    for (ssn, wk), idxs in v.groupby(grp_cols).groups.items():
        try:
            sb = _espn_scores_for_week(int(ssn), int(wk))
            if sb.empty:
                continue

            sub = v.loc[idxs, ["team","opponent","week","season"]].copy()
            sub["team_abbr"]     = sub["team"].astype(str).map(norm_abbr)
            sub["opponent_abbr"] = sub["opponent"].astype(str).map(norm_abbr)
            sub["bet_idx"]       = sub.index

            cand = sub.merge(sb, how="cross", suffixes=("_bet","_sb"))

            def _is_match(r):
                t, o = r["team_abbr"], r["opponent_abbr"]
                h, a = r["home_abbr"], r["away_abbr"]
                return (t == h and o == a) or (t == a and o == h)

            cand = cand[_is_match]

            if cand.empty:
                continue

            cand["abs_diff"] = 0
            pick = (cand.sort_values(["bet_idx","abs_diff"])
                        .groupby("bet_idx", as_index=False).first())

            for _, row in pick.iterrows():
                i = int(row["bet_idx"])

                v.loc[i, "home_team"] = row["home_abbr"]
                v.loc[i, "away_team"] = row["away_abbr"]

                hs = row["home_score"]; aw = row["away_score"]
                v.loc[i, "score_home"] = hs
                v.loc[i, "score_away"] = aw

                # Nuevo: estado + short detail
                v.loc[i, "state"]        = str(row.get("state","")).lower()
                v.loc[i, "status_short"] = str(row.get("short",""))
        except Exception:
            continue

    return v


def render(season: int):
    st.subheader("Overview")

    # Datos base
    pnl = load_pnl_weekly(season)
    stage = season_stage(season, pnl)
    bets_week = load_bets_this_week(season) if stage == "in_season" else pd.DataFrame()

    # Bets de esta semana (como cards)
    if not bets_week.empty:
        # Enriquecer con ESPN para scores/estado
        view = _enrich_bets_with_espn(bets_week, debug=False)

        st.markdown("**This Week’s Bets**")
        from nfl_dash.components import bet_card  # import perezoso

        cards = list(view.itertuples(index=False))
        idx = 0
        cols_per_row = 4
        rows = math.ceil(len(cards) / cols_per_row)
        for _ in range(rows):
            col_objs = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if idx < len(cards):
                    with col_objs[j]:
                        bet_card(pd.Series(cards[idx]._asdict()))
                    idx += 1
        st.divider()

    # Overview de temporada
    st.markdown("**Season Overview**")
    if pnl.empty:
        st.caption("No `pnl.csv` found for this season yet.")
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
