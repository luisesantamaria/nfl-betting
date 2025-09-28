import math
import pandas as pd
import streamlit as st
from importlib import reload

from nfl_dash.data_io import load_ledger
from nfl_dash.odds_scores import enrich_bets_with_scores
from nfl_dash.utils import ORDER_INDEX
import nfl_dash.components as _comps

# Fuerza a usar la versión actualizada del componente en caliente (evita módulos cacheados)
_comps = reload(_comps)
render_bet_card = _comps.bet_card

def render(season: int):
    st.subheader("Bets")

    bets_all_raw = load_ledger(season)
    bets_all = enrich_bets_with_scores(bets_all_raw, season) if not bets_all_raw.empty else bets_all_raw

    if bets_all.empty:
        st.caption("No bets file for this season.")
        return

    # Orden y week_label
    view = bets_all.copy()
    if "week_label" in view.columns:
        view["week_label"] = view["week_label"].astype(str)
        view["__order"] = view["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
        sort_cols = ["__order"]
        if "schedule_date" in view.columns:
            sort_cols.append("schedule_date")
        view = view.sort_values(sort_cols).drop(columns="__order")

    # Agrupar por semana con headers
    if "week_label" in view.columns:
        for wlab, dfw in view.groupby("week_label", sort=False):
            st.markdown(f"### {wlab}")
            cards = list(dfw.itertuples(index=False))
            idx = 0
            cols_per_row = 4
            rows = math.ceil(len(cards) / cols_per_row)
            for _ in range(rows):
                col_objs = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if idx < len(cards):
                        with col_objs[j]:
                            # NO usar st.write con el retorno; el componente imprime y devuelve None
                            render_bet_card(pd.Series(cards[idx]._asdict()))
                        idx += 1
            st.divider()
    else:
        # fallback sin week_label
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
