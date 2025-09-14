import math
import pandas as pd
import streamlit as st

from nfl_dash.live_scores import fetch_espn_scoreboard_df
from nfl_dash.config import SEASON_RULES
from nfl_dash.utils import norm_abbr
from nfl_dash.logos import get_logo_url

DAY_ORDER = ["Thursday", "Saturday", "Sunday", "Monday"]


def _guess_current_week(season: int) -> int:
    """Calcula la semana de NFL a partir del regular_start de SEASON_RULES."""
    rules = SEASON_RULES.get(int(season), {})
    start = rules.get("regular_start", None)
    if not start:
        return 1
    start_ts = pd.Timestamp(start, tz="UTC")
    now = pd.Timestamp.now(tz="UTC")
    if now < start_ts:
        return 1
    # Semana 1 comienza en regular_start; clamp 1..22 (incluye postemporada si hiciera falta)
    weeks = int((now - start_ts).days // 7) + 1
    return max(1, min(22, weeks))


def _pick_week_and_fetch(season: int):
    """Toma la semana estimada; si viene vacía, prueba semana-1 y semana+1."""
    w = _guess_current_week(season)
    for cand in [w, max(1, w - 1), min(22, w + 1)]:
        df = fetch_espn_scoreboard_df(int(season), int(cand))
        if not df.empty:
            return cand, df
    return w, pd.DataFrame()


def _bucket_day(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "TBD"
    local = ts.tz_convert("US/Eastern")
    return local.strftime("%A")  # Thursday/Saturday/Sunday/Monday


def _kickoff_str(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    local = ts.tz_convert("US/Eastern")
    # Formato: 9/14 • 1:00 PM ET
    try:
        return local.strftime("%-m/%-d • %-I:%M %p ET")
    except Exception:
        # En Windows, usar sin '-'
        return local.strftime("%m/%d • %I:%M %p ET")


def _live_badge(state: str) -> str:
    s = (state or "").lower()
    if s == "in":
        return "<span style='color:#d90429;font-weight:800;'>● LIVE</span>"
    if s == "post":
        return "<span style='opacity:.75;font-weight:700;'>Final</span>"
    return "<span style='opacity:.6;'>Scheduled</span>"


def _team_logo(name: str, bg="#222") -> str:
    abbr = norm_abbr(name)
    url = get_logo_url(abbr)
    if url:
        return f"<img src='{url}' width='44' height='44' style='object-fit:contain;'/>"
    t = (abbr or name or 'NA')[:3].upper()
    return (
        f"<div style='width:44px;height:44px;border-radius:50%;background:{bg};"
        f"color:#fff;display:flex;align-items:center;justify-content:center;font-weight:800'>{t}</div>"
    )


def _game_card(row: pd.Series):
    left_logo = _team_logo(row.get("home_team", ""), "#1f2937")
    right_logo = _team_logo(row.get("away_team", ""), "#374151")
    state = str(row.get("state", ""))
    live_html = _live_badge(state)

    hs = row.get("home_score", None)
    as_ = row.get("away_score", None)
    has_score = pd.notna(hs) and pd.notna(as_)
    score_html = (
        f"<div style='font-weight:900;font-size:26px;letter-spacing:.5px;'>{int(hs)} — {int(as_)}</div>"
        if has_score else "<div style='opacity:.55;font-weight:700;'>TBD</div>"
    )

    kick = _kickoff_str(row.get("start_time"))
    home = str(row.get("home_team", ""))
    away = str(row.get("away_team", ""))

    st.markdown(
        f"""
    <div style="border:1px solid #e9e9e9;border-radius:12px;padding:12px;margin-bottom:10px;
                background:linear-gradient(180deg,rgba(0,0,0,0.02),rgba(0,0,0,0));font-size:12.5px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div style="font-weight:600;">{kick}</div>
        <div>{live_html}</div>
      </div>

      <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:10px;">
        <div style="width:44px;display:flex;align-items:center;justify-content:center;">{left_logo}</div>
        <div style="flex:1;text-align:center;">{score_html}</div>
        <div style="width:44px;display:flex;align-items:center;justify-content:center;">{right_logo}</div>
      </div>

      <div style="display:flex;align-items:center;justify-content:space-between;margin-top:6px;">
        <div style="font-size:12px;opacity:.8;font-weight:600;">{home}</div>
        <div style="font-size:12px;opacity:.5;font-weight:700;">vs</div>
        <div style="font-size:12px;opacity:.8;font-weight:600;text-align:right;">{away}</div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render(season: int):
    st.subheader("Live")

    week, df = _pick_week_and_fetch(season)
    if df.empty:
        st.caption("No games found for this week.")
        return

    df = df.copy()
    df["bucket"] = df["start_time"].apply(_bucket_day)

    # Orden fijo de días; si falta alguno, se omite
    cat = pd.Categorical(df["bucket"], categories=DAY_ORDER, ordered=True)
    df["bucket"] = cat

    # Orden dentro de cada día por hora; el jueves (aunque ya haya pasado) queda arriba por el bucket
    df = df.sort_values(["bucket", "start_time"], ascending=[True, True])

    # Render por día en secciones
    for day in [d for d in DAY_ORDER if (df["bucket"] == d).any()]:
        st.markdown(f"### {day}")
        rows = df[df["bucket"] == day].reset_index(drop=True)

        # Grid 3 columnas (ajústalo si prefieres 2 o 4)
        cols_per_row = 3
        total = len(rows)
        it = 0
        while it < total:
            col_objs = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if it >= total:
                    break
                with col_objs[j]:
                    _game_card(rows.loc[it])
                it += 1
