# tabs/live.py
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from nfl_dash.live_scores import fetch_espn_scoreboard_df
from nfl_dash.components import game_card

ET = ZoneInfo("America/New_York")

def _labor_day_et(year: int) -> datetime:
    d = datetime(year, 9, 1, tzinfo=ET)
    while d.weekday() != 0:  # Monday
        d += timedelta(days=1)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)

def _week1_tnf_et(year: int) -> datetime:
    labor = _labor_day_et(year)          # Mon of Labor Day week
    thu   = labor + timedelta(days=3)    # Thu of that week
    return thu.replace(hour=20, minute=20)

def _tuesday_anchor_et(year: int) -> datetime:
    tnf = _week1_tnf_et(year)
    week_monday = (tnf - timedelta(days=tnf.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return week_monday + timedelta(days=1)  # Tue 00:00 ET

def _autodetect_week(season: int, now_utc: datetime | None = None) -> int:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    now_et  = now_utc.astimezone(ET)
    anchor  = _tuesday_anchor_et(season)
    wk = 1 if now_et < anchor else int(((now_et - anchor).days // 7) + 1)
    return max(1, min(22, wk))

def _day_bucket(et_ts: pd.Timestamp) -> str:
    d = et_ts.tz_convert(ET)
    return {3:"Thursday", 5:"Saturday", 6:"Sunday", 0:"Monday"}.get(d.weekday(), d.strftime("%A"))

def _sun_slot(et_ts: pd.Timestamp) -> tuple[int, str]:
    d = et_ts.tz_convert(ET)
    # Hora sin cero a la izquierda (portable)
    label = d.strftime("%I:%M %p").lstrip("0")
    return d.hour * 60 + d.minute, f"Sun {label}"

def render(season: int):
    # Week autodetect 100% calendario
    week = _autodetect_week(int(season))

    # Header “Live · Week X”
    st.markdown(
        f"""
        <div style="
            display:flex;align-items:center;gap:.6rem;
            margin:.25rem 0 1rem 0;
        ">
          <div style="font-size:1.25rem;font-weight:800;letter-spacing:.2px;">Live</div>
          <div style="
              padding:.15rem .65rem;border-radius:999px;
              border:1px solid rgba(255,255,255,.12);
              background:rgba(255,255,255,.04);
              font-weight:800;letter-spacing:.4px;font-size:.95rem;
          ">Week {week}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Traer tablero ESPN
    try:
        df = fetch_espn_scoreboard_df(season=int(season), week=int(week))
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        st.info("No games found for this week (yet).")
        return

    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    df["start_et"]   = df["start_time"].dt.tz_convert(ET)
    df["state"]      = df["state"].astype(str).str.lower()   # 'pre' | 'in' | 'post'
    df["day_bucket"] = df["start_et"].apply(_day_bucket)
    df["is_live"]    = df["state"].eq("in")

    # Auto-refresh solo si hay juegos en vivo
    if df["is_live"].any():
        try:
            from streamlit_extras.st_autorefresh import st_autorefresh
            st_autorefresh(interval=60_000, key=f"live_autorefresh_{season}_{week}")
        except Exception:
            pass  # si no está instalado, seguimos sin auto-refresh

    # Orden: día (Thu, Sat, Sun, Mon) → en vivo primero → hora
    day_order = {"Thursday": 1, "Saturday": 2, "Sunday": 3, "Monday": 4}
    df["__day_ord"]  = df["day_bucket"].map(day_order).fillna(99)
    df["__kick_ord"] = df["start_et"].apply(lambda x: x.hour * 60 + x.minute)
    df["__live_ord"] = (~df["is_live"]).astype(int)  # live primero
    df = df.sort_values(["__day_ord", "__live_ord", "start_et"])

    # Render por secciones
    for day_name in ["Thursday", "Saturday", "Sunday", "Monday"]:
        day_df = df[df["day_bucket"].eq(day_name)]
        if day_df.empty:
            continue

        st.markdown(f"### {day_name}")

        if day_name == "Sunday":
            dd = day_df.copy()
            slots = dd["start_et"].apply(_sun_slot)
            dd["__slot_ord"] = [s[0] for s in slots]
            dd["__slot_lab"] = [s[1] for s in slots]
            for slot in sorted(dd["__slot_ord"].unique()):
                g = dd[dd["__slot_ord"].eq(slot)]
                st.markdown(f"**{g.iloc[0]['__slot_lab']}**")
                cards = list(g.itertuples(index=False))
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
                st.divider()
        else:
            cards = list(day_df.itertuples(index=False))
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
