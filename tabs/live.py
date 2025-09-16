# tabs/live.py
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from nfl_dash.live_scores import fetch_espn_scoreboard_df
from nfl_dash.components import game_card

ET = ZoneInfo("America/New_York")

def _labor_day_et(year: int) -> datetime:
    # Primer lunes de septiembre (00:00 ET)
    d = datetime(year, 9, 1, tzinfo=ET)
    while d.weekday() != 0:  # 0 = Monday
        d += timedelta(days=1)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)

def _week1_tnf_et(year: int) -> datetime:
    # TNF de Week 1: jueves de esa semana, 8:20 PM ET
    labor = _labor_day_et(year)                 # Lunes
    thu   = labor + timedelta(days=3)           # Jueves
    return thu.replace(hour=20, minute=20)

def _tuesday_anchor_et(year: int) -> datetime:
    # Martes 00:00 ET de la semana del TNF (ancla de cortes semanales)
    tnf = _week1_tnf_et(year)
    week_monday = (tnf - timedelta(days=tnf.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return week_monday + timedelta(days=2)  # Tuesday 00:00 ET

def _autodetect_week(season: int, now_utc: datetime | None = None) -> tuple[int, datetime, str]:
    """
    Devuelve (week, tue_anchor_et, source_label).
    Regla general: semana = floor((now_et - anchor_et)/7) + 1, acotada [1,22].
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET)
    anchor = _tuesday_anchor_et(season)

    if now_et < anchor:
        wk = 1
    else:
        wk = int(((now_et - anchor).days // 7) + 1)
    wk = max(1, min(22, wk))
    return wk, anchor, "auto:labor_day_rule"

def _fmt_time_et(dt: pd.Timestamp | datetime) -> str:
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    try:
        return dt.astimezone(ET).strftime("%-I:%M %p")
    except Exception:
        return dt.astimezone(ET).strftime("%I:%M %p").lstrip("0")

def _day_bucket(et_dt: pd.Timestamp) -> str:
    d = et_dt.tz_convert(ET)
    wd = d.weekday()  # 0=Mon ... 6=Sun
    return {3:"Thursday", 5:"Saturday", 6:"Sunday", 0:"Monday"}.get(wd, d.strftime("%A"))

def _sun_slot(et_dt: pd.Timestamp) -> tuple[int, str]:
    """Para Sunday, agrupa por horario (ej. 'Sun 1:00 PM'). Devuelve (orden, etiqueta)."""
    d = et_dt.tz_convert(ET)
    label = f"Sun {_fmt_time_et(d)}"
    order = d.hour * 60 + d.minute
    return order, label

def render(season: int):
    st.subheader("Live")

    # Semana autodetectada por calendario (sin depender de config ni odds.csv)
    now_utc = datetime.now(timezone.utc)
    week, anchor_et, src = _autodetect_week(int(season), now_utc)

    now_et_txt = now_utc.astimezone(ET).strftime("%a %b %d, %I:%M %p ET")
    anchor_txt = anchor_et.strftime("%b %d")
    st.caption(f"Auto-detected **Week {week}** • Now: {now_et_txt} • Tue cutoff anchor: **{anchor_txt}** • Source: {src}")

    # Trae el scoreboard de ESPN para esa semana
    try:
        df = fetch_espn_scoreboard_df(season=int(season), week=int(week))
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        st.info("No live data for this week yet.")
        return

    # Normaliza fechas a ET y orden básico
    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    df["start_et"]   = df["start_time"].dt.tz_convert(ET)
    df["state"]      = df["state"].astype(str).str.lower()   # 'pre' | 'in' | 'post'
    df["day_bucket"] = df["start_et"].apply(lambda x: _day_bucket(x))
    df["is_live"]    = df["state"].eq("in")

    # Orden de días: Thu → Sat → Sun → Mon
    day_order = {"Thursday": 1, "Saturday": 2, "Sunday": 3, "Monday": 4}
    df["__day_ord"] = df["day_bucket"].map(day_order).fillna(99)

    # Dentro de cada día, LIVE primero, luego por kickoff
    df["__kick_ord"] = df["start_et"].apply(lambda x: x.hour * 60 + x.minute)
    df["__live_ord"] = (~df["is_live"]).astype(int)  # live (True) → 0

    df = df.sort_values(["__day_ord", "__live_ord", "start_et"])

    # Render por secciones
    for day_name in ["Thursday", "Saturday", "Sunday", "Monday"]:
        day_df = df[df["day_bucket"].eq(day_name)]
        if day_df.empty:
            continue

        st.markdown(f"### {day_name}")

        if day_name == "Sunday":
            # Agrupa por slot horario dentro del domingo
            day_df = day_df.copy()
            tmp = day_df["start_et"].apply(_sun_slot)
            day_df["__slot_ord"] = [t[0] for t in tmp]
            day_df["__slot_lab"] = [t[1] for t in tmp]
            for slot_ord in sorted(day_df["__slot_ord"].unique()):
                g = day_df[day_df["__slot_ord"].eq(slot_ord)]
                if g.empty:
                    continue
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
