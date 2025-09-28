# tabs/overview.py
import math
import pandas as pd
import streamlit as st

from nfl_dash.data_io import load_pnl_weekly, load_bets_this_week
from nfl_dash.utils import kpis_from_pnl, add_week_order, season_stage, ORDER_INDEX
from nfl_dash.charts import chart_sparkline_cumprofit, chart_last8_profit


def render(season: int):
    st.subheader("Overview")

    # --- Datos base
    pnl = load_pnl_weekly(season)
    stage = season_stage(season, pnl)

    # --- Bets de esta semana (solo leer lo que ya está anotado en data/live/bets.csv)
    bets_week = load_bets_this_week(season) if stage == "in_season" else pd.DataFrame()
    if not bets_week.empty:
        st.markdown("**This Week’s Bets**")

        view = bets_week.copy()

        # Normaliza tipos mínimos para el componente (sin cálculos)
        for c in ("team_score", "opponent_score", "profit", "stake", "decimal_odds"):
            if c in view.columns:
                view[c] = pd.to_numeric(view[c], errors="coerce")
        if "schedule_date" in view.columns:
            view["schedule_date"] = pd.to_datetime(view["schedule_date"], errors="coerce")
        if "status" in view.columns:
            view["status"] = view["status"].astype(str).str.upper()
        if "week_label" in view.columns:
            view["week_label"] = view["week_label"].astype(str)

        # Orden como en Bets: por orden de semana y kickoff
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

    # Alturas dinámicas
    H_BANK, H_PROF = (380, 380) if bets_week.empty else (200, 200)

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
