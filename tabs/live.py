import math
import pandas as pd
import streamlit as st

from nfl_dash.live_scores import fetch_espn_scoreboard_df
from nfl_dash.config import SEASON_RULES
from nfl_dash.utils import norm_abbr
from nfl_dash.logos import get_logo_url

DAY_ORDER = ["Thursday", "Saturday", "Sunday", "Monday"]
SUNDAY_ORDER = ["Sun 1:00 PM", "Sun 4:05 PM", "Sun 4:25 PM", "Sun 8:20 PM", "Sun Other"]


def _wednesday_anchor_et(ts_et: pd.Timestamp) -> pd.Timestamp:
    """Devuelve el miércoles 00:00 ET de la semana de ts_et (o el miércoles anterior si el juego es jueves)."""
    # Asegurar zona horaria
    if ts_et.tz is None:
        ts_et = ts_et.tz_localize("US/Eastern")
    else:
        ts_et = ts_et.tz_convert("US/Eastern")
    # weekday: Mon=0 ... Sun=6; Wednesday=2
    days_since_wed = (ts_et.weekday() - 2) % 7
    anchor = (ts_et - pd.Timedelta(days=days_since_wed)).normalize()
    return anchor


def _guess_current_week(season: int) -> int:
    """Calcula la semana NFL usando corte de miércoles 00:00 ET respecto a regular_start."""
    rules = SEASON_RULES.get(int(season), {})
    regular_start = rules.get("regular_start", None)
    if not regular_start:
        return 1
    # regular_start es kickoff del TNF semana 1; anclar al miércoles de esa semana
    rs = pd.Timestamp(regular_start)
    if rs.tz is None:
        rs = rs.tz_localize("UTC")
    rs_et = rs.tz_convert("US/Eastern")
    anchor_wed = _wednesday_anchor_et(rs_et)

    now_et = pd.Timestamp.now(tz="US/Eastern")
    if now_et < anchor_wed:
        return 1

    delta_days = (now_et - anchor_wed).days
    week = int(delta_days // 7) + 1
    return max(1, min(22, week))


def _pick_week_and_fetch(season: int):
    """Usa la semana estimada; si vacío, prueba semana-1 y semana+1."""
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


def _sunday_subbucket(ts: pd.Timestamp) -> str:
    """Clasifica ventanas del domingo: 1:00, 4:05, 4:25, 8:20 (SNF); otros -> 'Sun Other'."""
    if pd.isna(ts):
        return "Sun Other"
    local = ts.tz_convert("US/Eastern")
    if local.weekday() != 6:  # Sunday=6
        return ""
    h, m = local.hour, local.minute
    if h == 13:
        return "Sun 1:00 PM"
    if h == 16 and m == 5:
        return "Sun 4:05 PM"
    if h == 16 and m == 25:
        return "Sun 4:25 PM"
    if h == 20 and m == 20:
        return "Sun 8:20 PM"
    return "Sun Other"


def _kickoff_str(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    local = ts.tz_convert("US/Eastern")
    try:
        return local.strftime("%-m/%-d • %-I:%M %p ET")
    except Exception:
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
    # Día (Thursday/Saturday/Sunday/Monday)
    df["bucket"] = df["start_time"].apply(_bucket_day)
    df["bucket"] = pd.Categorical(df["bucket"], categories=DAY_ORDER, ordered=True)

    # Sub-bucket solo para Sunday (ventanas)
    df["sub"] = df["start_time"].apply(_sunday_subbucket)
    df["sub"] = pd.Categorical(df["sub"], categories=SUNDAY_ORDER, ordered=True)

    # Orden general: día -> hora
    df = df.sort_values(["bucket", "start_time"], ascending=[True, True])

    for day in [d for d in DAY_ORDER if (df["bucket"] == d).any()]:
        st.markdown(f"### {day}")
        rows = df[df["bucket"] == day].reset_index(drop=True)

        if day != "Sunday":
            # Render simple en grid
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
        else:
            # Sunday: sub-secciones por ventana
            for sub in [s for s in SUNDAY_ORDER if (rows["sub"] == s).any()]:
                sub_rows = rows[rows["sub"] == sub].reset_index(drop=True)
                if sub_rows.empty:
                    continue
                st.markdown(f"#### {sub}")
                cols_per_row = 3
                total = len(sub_rows)
                it = 0
                while it < total:
                    col_objs = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if it >= total:
                            break
                        with col_objs[j]:
                            _game_card(sub_rows.loc[it])
                        it += 1
