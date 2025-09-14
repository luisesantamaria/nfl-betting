import streamlit as st
from nfl_dash.data_io import list_available_seasons, load_pnl
from nfl_dash.utils import season_stage
from tabs.overview import render as render_overview
from tabs.portfolio import render as render_portfolio
from tabs.bets import render as render_bets

st.set_page_config(page_title="NFL EV Betting — Dashboard", layout="wide")

seasons = list_available_seasons()
default_year = max(seasons) if seasons else 2024

c1, c2 = st.columns([0.72, 0.28], vertical_alignment="center")
with c1:
    st.markdown("## NFL EV Betting — Dashboard")
with c2:
    season = st.selectbox("Season", seasons, index=seasons.index(default_year) if seasons else 0)

pnl_df = load_pnl(season)
stage = season_stage(season, pnl_df)

tabs = st.tabs(["Overview", "Portfolio", "Bets"])

with tabs[0]:
    render_overview(season, stage)

with tabs[1]:
    render_portfolio(season, stage)

with tabs[2]:
    render_bets(season, stage)
