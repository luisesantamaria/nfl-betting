import streamlit as st
import pandas as pd

from nfl_dash.data_io import list_available_seasons, load_pnl_weekly
from nfl_dash.utils import season_stage

from tabs.overview import render as render_overview
from tabs.portfolio import render as render_portfolio
from tabs.bets import render as render_bets
from tabs.live import render as render_live

st.set_page_config(
    page_title="NFL EV Betting — Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"]{display:none!important;}
    [data-testid="collapsedControl"]{display:none!important;}
    .block-container{padding-top:0.8rem !important;}
    .compact-select > div[data-baseweb="select"] { min-height: 38px; }
    </style>
    """,
    unsafe_allow_html=True,
)

seasons = list_available_seasons()
if not seasons:
    st.error("No seasons found.")
    st.stop()
default_season = max(seasons)
SPACER_PX = 25

h_left, h_right = st.columns([0.66, 0.34], gap="large")
with h_left:
    st.markdown("<h1 style='margin:0'>NFL EV Betting — Dashboard</h1>", unsafe_allow_html=True)
with h_right:
    st.markdown(f"<div style='height:{SPACER_PX}px'></div>", unsafe_allow_html=True)
    season = st.selectbox(
        "Season",
        options=seasons,
        index=seasons.index(default_season),
        label_visibility="visible",
        key="season_select",
    )

st.session_state["season"] = season

pnl_df = load_pnl_weekly(season)
stage = season_stage(season, pnl_df)
status_map = {"locked":"Locked","preseason":"Preseason","in_season":"In Season","ended":"Season Ended","unknown":"Unknown"}
st.caption(f"Status: **{status_map.get(stage, 'Unknown')}**")

tab_overview, tab_portfolio, tab_bets, tab_live = st.tabs(["Overview", "Portfolio", "Bets", "Live"])

with tab_overview:
    render_overview(season)
with tab_portfolio:
    render_portfolio(season)
with tab_bets:
    render_bets(season)
with tab_live:
    render_live(season)
