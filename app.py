import streamlit as st
import pandas as pd

from nfl_dash.data_io import list_available_seasons, load_pnl_weekly
from nfl_dash.utils import season_stage

# importa las funciones de cada tab (no multipage)
from tabs.overview import render as render_overview
from tabs.portfolio import render as render_portfolio
from tabs.bets import render as render_bets

st.set_page_config(
    page_title="NFL EV Betting — Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Oculta sidebar y reduce padding superior del contenedor
st.markdown(
    """
    <style>
    [data-testid="stSidebar"]{display:none!important;}
    [data-testid="collapsedControl"]{display:none!important;}
    .block-container{padding-top:0.8rem !important;}
    /* compacta el select */
    .compact-select > div[data-baseweb="select"] { min-height: 38px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===================== HEADER (título + season en la misma fila) =====================
seasons = list_available_seasons()
if not seasons:
    st.error("No seasons found in data/processed/portfolio.")
    st.stop()

default_season = max(seasons)

# fila header
h_left, h_right = st.columns([0.66, 0.34], gap="large")
with h_left:
    st.markdown(
        "<h1 style='margin:0'>NFL EV Betting — Dashboard</h1>",
        unsafe_allow_html=True,
    )
with h_right:
    # pequeño spacer para alinear verticalmente con el título
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    season = st.selectbox(
        "Season",
        options=seasons,
        index=seasons.index(default_season),
        label_visibility="visible",
        key="season_select",
    )

st.session_state["season"] = season

# estado justo debajo del título, a la izquierda
pnl = load_pnl_weekly(season)
stage = season_stage(season, pnl)
status_map = {
    "locked":"Locked",
    "preseason":"Preseason",
    "in_season":"In Season",
    "ended":"Season Ended",
    "unknown":"Unknown"
}
st.caption(f"Status: **{status_map.get(stage, 'Unknown')}**")

# ===================== TABS =====================
tab_overview, tab_portfolio, tab_bets = st.tabs(["Overview", "Portfolio", "Bets"])

with tab_overview:
    render_overview(season)

with tab_portfolio:
    render_portfolio(season)

with tab_bets:
    render_bets(season)
