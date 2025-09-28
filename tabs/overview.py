# tabs/overview.py
from __future__ import annotations
import math
import pandas as pd
import streamlit as st

from nfl_dash.data_io import load_pnl_weekly, load_bets_this_week
from nfl_dash.utils import kpis_from_pnl, add_week_order, season_stage, norm_abbr
from nfl_dash.charts import chart_sparkline_cumprofit, chart_last8_profit
from nfl_dash.live_scores import fetch_espn_scoreboard_df
from nfl_dash.components import bet_card as render_bet_card


def _week_for_overview(pnl: pd.DataFrame, bets_week: pd.DataFrame) -> int | None:
    """
    Escogemos la semana a consultar en ESPN:
    - Si hay bets de esta semana, usamos su week (o week_order máximo).
    - Si no, intentamos con pnl (última semana).
    """
    w = None
    if bets_week is not None and not bets_week.empty:
        if "week" in bets_week.columns and pd.to_numeric(bets_week["week"], errors="coerce").notna().any():
            w = int(pd.to_numeric(bets_week["week"], errors="coerce").max())
        elif "week_order" in bets_week.columns and pd.to_numeric(bets_week["week_order"], errors="coerce").notna().any():
            w = int(pd.to_numeric(bets_week["week_order"], errors="coerce").max())
    if w is None and pnl is not None and not pnl.empty and "week" in pnl.columns:
        if pd.to_numeric(pnl["week"], errors="coerce").notna().any():
            w = int(pd.to_numeric(pnl["week"], errors="coerce").max())
    return w


def _espn_df(season: int, week: int) -> pd.DataFrame:
    """Trae el scoreboard de ESPN y crea columnas con abreviaturas."""
    try:
        df = fetch_espn_scoreboard_df(season=int(season), week=int(week)).copy()
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        return df

    # Normalizamos nombres -> abreviaturas
    df["home_abbr"] = df["home_team"].astype(str).map(norm_abbr)
    df["away_abbr"] = df["away_team"].astype(str).map(norm_abbr)

    # Aseguramos tipos/fechas
    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)

    # Campos que usaremos en el render
    keep = [
        "home_team", "away_team",
        "home_abbr", "away_abbr",
        "home_score", "away_score",
        "state", "short", "start_time",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA
    return df[keep]


def _enrich_bets_with_espn(bets_week: pd.DataFrame, espn: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Empareja cada bet con un juego del scoreboard ESPN, tolerando el orden (home/away flip).
    Devuelve el mismo DF + columnas:
      home_team, away_team, home_score, away_score, state, status_short
    """
    v = bets_week.copy()

    # Normalizamos team/opponent y calculamos "home/away reales" del juego
    v["team_abbr"] = v.get("team", "").astype(str).map(norm_abbr)
    v["opp_abbr"]  = v.get("opponent", "").astype(str).map(norm_abbr)

    def _bet_home(r):
        s = str(r.get("side", "")).lower()
        if s == "home":
            return r["team_abbr"]
        elif s == "away":
            return r["opp_abbr"]
        # Sin side confiable: dejamos vacío para evitar match incorrecto
        return ""

    def _bet_away(r):
        s = str(r.get("side", "")).lower()
        if s == "home":
            return r["opp_abbr"]
        elif s == "away":
            return r["team_abbr"]
        return ""

    v["bet_home_abbr"] = v.apply(_bet_home, axis=1)
    v["bet_away_abbr"] = v.apply(_bet_away, axis=1)

    # Campos que el card necesita
    v["home_team"] = v["bet_home_abbr"]
    v["away_team"] = v["bet_away_abbr"]
    v["home_score"] = pd.NA
    v["away_score"] = pd.NA
    v["state"] = pd.NA
    v["status_short"] = pd.NA

    # Tabla de candidatos para debug
    cand_rows = []

    # Indexamos ESPN por pares (home,away) y (away,home)
    if espn is None or espn.empty:
        espn = pd.DataFrame(columns=["home_abbr", "away_abbr"])

    # Loop simple (pocas bets por semana): facilita debug claro
    for i, row in v.iterrows():
        bh = str(row.get("bet_home_abbr", "") or "")
        ba = str(row.get("bet_away_abbr", "") or "")

        # Candidatos en orden natural y volteado
        m_nat = espn[(espn["home_abbr"] == bh) & (espn["away_abbr"] == ba)]
        m_flp = espn[(espn["home_abbr"] == ba) & (espn["away_abbr"] == bh)]

        match = None
        if len(m_nat) == 1:
            match = m_nat.iloc[0]
        elif len(m_flp) == 1:
            match = m_flp.iloc[0]
        elif len(m_nat) >= 1:
            # si hay múltiples (raro), nos quedamos con el primero por hora más reciente
            match = m_nat.sort_values("start_time").iloc[-1] if "start_time" in m_nat.columns else m_nat.iloc[0]
        elif len(m_flp) >= 1:
            match = m_flp.sort_values("start_time").iloc[-1] if "start_time" in m_flp.columns else m_flp.iloc[0]

        # Guardamos debug
        cand_rows.append({
            "bet_idx": i,
            "bet_home": bh,
            "bet_away": ba,
            "m_home": match["home_abbr"] if match is not None else None,
            "m_away": match["away_abbr"] if match is not None else None,
            "home_score": match["home_score"] if match is not None else None,
            "away_score": match["away_score"] if match is not None else None,
            "state": match["state"] if match is not None else None,
            "short": match["short"] if match is not None else None,
            "start_time": match["start_time"] if match is not None else None,
        })

        # volcamos al dataframe de bets
        if match is not None:
            v.at[i, "home_team"] = bh
            v.at[i, "away_team"] = ba
            v.at[i, "home_score"] = match["home_score"]
            v.at[i, "away_score"] = match["away_score"]
            v.at[i, "state"] = match["state"]
            v.at[i, "status_short"] = match["short"]

    if debug:
        with st.expander("Debug · ESPN matching (Overview)", expanded=True):
            st.markdown(f"**Scoreboard ESPN** (season=**, week=**)")
            cols = ["home_abbr", "away_abbr", "home_score", "away_score", "start_time", "state", "short"]
            st.dataframe(espn[cols].sort_values("start_time", na_position="last").reset_index(drop=True), use_container_width=True)
            st.markdown("**Candidatos (emparejados por bet)**")
            dbg = pd.DataFrame(cand_rows)
            st.dataframe(dbg, use_container_width=True)

    return v


def render(season: int):
    st.subheader("Overview")

    # Datos base
    pnl = load_pnl_weekly(season)
    stage = season_stage(season, pnl)
    bets_week = load_bets_this_week(season) if stage == "in_season" else pd.DataFrame()

    # Toggle de debug
    debug = st.checkbox("Mostrar debug de emparejamiento ESPN (Overview)", value=False)

    # En Overview pedías los bet-cards (no tabla)
    if not bets_week.empty:
        # === Enriquecemos con ESPN ===
        wk = _week_for_overview(pnl, bets_week)
        espn = _espn_df(season, wk) if wk is not None else pd.DataFrame()
        view = _enrich_bets_with_espn(bets_week, espn, debug=debug)

        st.markdown("**This Week’s Bets**")

        # orden aproximado: por kickoff si lo tenemos
        sort_cols = []
        if "schedule_date" in view.columns:
            view["schedule_date"] = pd.to_datetime(view["schedule_date"], errors="coerce", utc=True)
            sort_cols.append("schedule_date")
        elif "start_time" in view.columns:
            view["start_time"] = pd.to_datetime(view["start_time"], errors="coerce", utc=True)
            sort_cols.append("start_time")

        if len(sort_cols):
            view = view.sort_values(sort_cols)

        # Render cards (mismo layout que Bets)
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

    # === Overview de temporada (gráficas) ===
    st.markdown("**Season Overview**")
    if pnl.empty:
        st.caption("No `pnl_weekly_{year}.csv` found for this season.")
        return

    initial_bankroll, final_bankroll, total_profit, total_stake, yield_pct, profits, stakes = kpis_from_pnl(pnl)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Initial", f"${initial_bankroll:,.2f}")
    k2.metric("Final",   f"${final_bankroll:,.2f}", f"{(final_bankroll-initial_bankroll):,.2f}")
    k3.metric("Total Profit", f"${total_profit:,.2f}")
    k4.metric("Yield",        f"{yield_pct:.2f}%")

    # Alturas dinámicas de charts
    H_BANK = 200 if not bets_week.empty else 380
    H_PROF = 200 if not bets_week.empty else 380

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
