import pandas as pd
import streamlit as st

from nfl_dash.live_scores import fetch_espn_scoreboard_df
from nfl_dash.components import game_card

def _detect_default_week(season: int) -> int:
    try:
        from pathlib import Path
        f = Path(__file__).resolve().parents[1] / "data" / "live" / "odds.csv"
        if f.exists():
            df = pd.read_csv(f, low_memory=False)
            df = df[pd.to_numeric(df.get("season", pd.Series([])), errors="coerce").astype("Int64").eq(season)]
            if not df.empty and "week" in df.columns:
                w = pd.to_numeric(df["week"], errors="coerce").dropna()
                if not w.empty:
                    return int(w.max())
    except Exception:
        pass
    return 1

def render(season: int):
    st.subheader("Live")

    col_sel, col_ref = st.columns([0.8, 0.2])
    with col_sel:
        week = st.number_input("Week", min_value=1, max_value=22, value=_detect_default_week(season), step=1)
    with col_ref:
        refresh_secs = st.selectbox("Auto-refresh", options=[0, 30, 60, 120], index=2)

    # Fetch con manejo de errores (no crashea la app)
    try:
        df = fetch_espn_scoreboard_df(season=int(season), week=int(week))
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        st.info("Live feed unavailable or no games for this week right now.")
        return

    # Auto-refresh opcional usando streamlit-extras si está instalado
    if (df["state"].astype(str).str.lower().eq("in")).any() and refresh_secs > 0:
        try:
            from streamlit_extras.st_autorefresh import st_autorefresh
            st_autorefresh(interval=refresh_secs*1000, key=f"live_autorefresh_{season}_{week}")
        except Exception:
            st.caption("Tip: add `streamlit-extras` to enable auto-refresh widget.")

    st.caption("All games for selected week. LIVE games are highlighted.")

    cards = list(df.itertuples(index=False))
    idx = 0
    cols_per_row = 3
    rows = (len(cards) + cols_per_row - 1) // cols_per_row
    for _ in range(rows):
        col_objs = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if idx < len(cards):
                with col_objs[j]:
                    game_card(pd.Series(cards[idx]._asdict()))
                idx += 1
