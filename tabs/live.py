# tabs/live.py
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import requests

from nfl_dash.live_scores import fetch_espn_scoreboard_df
from nfl_dash.components import game_card

ET = ZoneInfo("America/New_York")

# =========================
# ESPN: obtener semana "oficial"
# =========================
_ESPN_ENDPOINTS = [
    "https://site.web.api.espn.com/apis/v2/sports/football/nfl/scoreboard",
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
]
_ESPN_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
    "Connection": "keep-alive",
}

def _get_api_week_espn(season: int, timeout: int = 12) -> int | None:
    params = {"seasontype": 2, "dates": int(season)}
    for base in _ESPN_ENDPOINTS:
        try:
            r = requests.get(base, params=params, headers=_ESPN_HEADERS, timeout=timeout)
            if r.status_code != 200:
                continue
            data = r.json()
            wk = (data.get("week") or {}).get("number")
            if wk is not None:
                return int(wk)
        except Exception:
            continue
    return None

# =========================
# Corte diferido: miércoles 02:00 ET
# =========================
CUTOVER_DOW_ET = 2   # 0=Mon, 1=Tue, 2=Wed
CUTOVER_HOUR_ET = 2  # 02:00 ET

def _after_cutover_et(dt_utc: datetime) -> bool:
    now_et = dt_utc.astimezone(ET)
    if now_et.weekday() > CUTOVER_DOW_ET:
        return True
    if now_et.weekday() < CUTOVER_DOW_ET:
        return False
    return now_et.hour >= CUTOVER_HOUR_ET

# =========================
# Fallback calendario
# =========================
def _labor_day_et(year: int) -> datetime:
    d = datetime(year, 9, 1, tzinfo=ET)
    while d.weekday() != 0:  # Monday
        d += timedelta(days=1)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)

def _week1_tnf_et(year: int) -> datetime:
    labor = _labor_day_et(year)
    thu   = labor + timedelta(days=3)
    return thu.replace(hour=20, minute=20)

def _tuesday_anchor_et(year: int) -> datetime:
    tnf = _week1_tnf_et(year)
    week_monday = (tnf - timedelta(days=tnf.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return week_monday + timedelta(days=1)

def _calendar_week_nat(season: int, now_utc: datetime) -> int:
    now_et  = now_utc.astimezone(ET)
    anchor  = _tuesday_anchor_et(season)
    if now_et < anchor:
        wk = 1
    else:
        wk = int(((now_et - anchor).days // 7) + 1)
    return max(1, min(22, wk))

# =========================
# Semana efectiva
# =========================
def _effective_week(season: int, now_utc: datetime | None = None) -> int:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    api_week = _get_api_week_espn(season)
    if api_week is None:
        api_week = _calendar_week_nat(season, now_utc)

    passed = _after_cutover_et(now_utc)
    week_eff = api_week if passed else max(1, api_week - 1)
    return max(1, min(22, week_eff))

# =========================
# Helpers de presentación
# =========================
def _fmt_time_et(dt: pd.Timestamp | datetime) -> str:
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    return dt.astimezone(ET).strftime("%-I:%M %p")

def _day_bucket(et_dt: pd.Timestamp) -> str:
    d = et_dt.tz_convert(ET)
    wd = d.weekday()
    return {3:"Thursday", 5:"Saturday", 6:"Sunday", 0:"Monday"}.get(wd, d.strftime("%A"))

def _sun_slot(et_dt: pd.Timestamp) -> tuple[int, str]:
    d = et_dt.tz_convert(ET)
    label = f"Sun {_fmt_time_et(d)}"
    return d.hour * 60 + d.minute, label

# =========================
# Render principal
# =========================
def render(season: int):
    week = _effective_week(int(season))

    # Header “Live · Week X” con circulito
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

    try:
        df = fetch_espn_scoreboard_df(season=int(season), week=week)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        st.info("No games found for this week yet.")
        return

    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    df["start_et"]   = df["start_time"].dt.tz_convert(ET)
    df["state"]      = df["state"].astype(str).str.lower()
    df["day_bucket"] = df["start_et"].apply(_day_bucket)
    df["is_live"]    = df["state"].eq("in")

    day_order = {"Thursday": 1, "Saturday": 2, "Sunday": 3, "Monday": 4}
    df["__day_ord"]  = df["day_bucket"].map(day_order).fillna(99)
    df["__kick_ord"] = df["start_et"].apply(lambda x: x.hour * 60 + x.minute)
    df["__live_ord"] = (~df["is_live"]).astype(int)
    df = df.sort_values(["__day_ord", "__live_ord", "start_et"])

    for day_name in ["Thursday", "Saturday", "Sunday", "Monday"]:
        day_df = df[df["day_bucket"].eq(day_name)]
        if day_df.empty:
            continue

        st.markdown(f"### {day_name}")

        if day_name == "Sunday":
            day_df = day_df.copy()
            slots = day_df["start_et"].apply(_sun_slot)
            day_df["__slot_ord"] = [s[0] for s in slots]
            day_df["__slot_lab"] = [s[1] for s in slots]
            for slot in sorted(day_df["__slot_ord"].unique()):
                g = day_df[day_df["__slot_ord"].eq(slot)]
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
