import math
import pandas as pd
import streamlit as st

from nfl_dash.data_io import load_ledger
from nfl_dash.odds_scores import enrich_bets_with_scores
from nfl_dash.utils import ORDER_INDEX
from nfl_dash.components import bet_card as render_bet_card

def render(season: int):
    st.subheader("Bets")

    bets_all_raw = load_ledger(season)
    bets_all = enrich_bets_with_scores(bets_all_raw, season) if not bets_all_raw.empty else bets_all_raw

    if bets_all.empty:
        st.caption("No bets file for this season.")
        return

    view = bets_all.copy()
    if "week_label" in view.columns:
        view["week_label"] = view["week_label"].astype(str)
        view["__order"] = view["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
        sort_cols = ["__order"]
        if "schedule_date" in view.columns:
            sort_cols.append("schedule_date")
        view = view.sort_values(sort_cols).drop(columns="__order")

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
                            render_bet_card(pd.Series(cards[idx]._asdict()))
                        idx += 1
            st.divider()
    else:
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
