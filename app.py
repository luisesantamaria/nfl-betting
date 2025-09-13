import math
import pandas as pd
import streamlit as st

from nfl_dash.data_io import list_available_seasons, load_pnl_weekly
from nfl_dash.utils import season_stage

# importa las "tabs" (módulos, no multipage)
from tabs.overview import render as render_overview
from tabs.portfolio import render as render_portfolio
from tabs.bets import render as render_bets

st.set_page_config(
    page_title="NFL EV Betting — Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Título + selector de temporada arriba (sin sidebar)
st.markdown("<h1 style='margin-bottom:0.2rem'>NFL EV Betting — Dashboard</h1>", unsafe_allow_html=True)

seasons = list_available_seasons()
if not seasons:
    st.error("No seasons found in data/processed/portfolio.")
    st.stop()

default_season = max(seasons)
col_title, col_season = st.columns([1, 1/3], gap="large")
with col_season:
    season = st.selectbox("Season", options=seasons, index=seasons.index(default_season), label_visibility="visible")

st.session_state["season"] = season

# Estado (caption)
pnl = load_pnl_weekly(season)
stage = season_stage(season, pnl)
status_map = {"locked":"Locked","preseason":"Preseason","in_season":"In Season","ended":"Season Ended","unknown":"Unknown"}
st.caption(f"Status: **{status_map.get(stage, 'Unknown')}**")

# Tabs visuales (dentro de un solo app.py)
tab_overview, tab_portfolio, tab_bets = st.tabs(["Overview", "Portfolio", "Bets"])

with tab_overview:
    render_overview(season)

with tab_portfolio:
    render_portfolio(season)

with tab_bets:
    render_bets(season)
