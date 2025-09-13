import streamlit as st
import pandas as pd
from nfl_dash.config import SEASON_RULES
from nfl_dash.data_io import list_available_seasons, load_pnl_weekly
from nfl_dash.utils import season_stage

st.set_page_config(page_title="NFL EV Betting — Dashboard", layout="wide")

# Temporada seleccionada (sidebar)
seasons = list_available_seasons()
if not seasons:
    st.error("No seasons found in data/processed/portfolio.")
    st.stop()

default_season = max(seasons)
season = st.sidebar.selectbox("Season", options=seasons, index=seasons.index(default_season))
st.session_state["season"] = season

# Estado (etiqueta)
pnl = load_pnl_weekly(season)
stage = season_stage(season, pnl)
status_map = {"locked":"Locked","preseason":"Preseason","in_season":"In Season","ended":"Season Ended","unknown":"Unknown"}
st.sidebar.caption(f"Status: **{status_map.get(stage, 'Unknown')}**")

st.title("NFL EV Betting — Dashboard")
st.write("Use the top-left **Pages** to navigate: Overview • Portfolio • Bets.")
