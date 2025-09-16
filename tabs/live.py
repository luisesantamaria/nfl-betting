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
# ESPN helpers (para leer api_week "oficial")
# =========================
_ESPN_ENDPOINTS = [
    "https://site.web.api.espn.com/apis/v2/sports/football/nfl/scoreboard",
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
]

_ESPN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Connection": "keep-alive",
}

def _get_api_week_espn(season: int, timeout: int = 12) -> int | None:
    """
    Pide el scoreboard SIN 'week' para que ESPN devuelva data.week.number (la 'semana oficial' actual).
    """
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
# Regla de corte diferido (tu requerimiento)
# =========================
CUTOVER_DOW_ET = 2   # 0=Mon, 1=Tue, 2=Wed
CUTOVER_HOUR_ET = 2  # 02:00 ET (~03:00 BRT)

def _after_cutover_et(dt_utc: datetime) -> bool:
    """True si ya pasamos el corte semanal (miércoles 02:00 ET)."""
    now_et = dt_utc.astimezone(ET)
    if now_et.weekday() > CUTOVER_DOW_ET:
        return True
    if now_et.weekday() < CUTOVER_DOW_ET:
        return False
    return now_et.hour >= CUTOVER_HOUR_ET

# =========================
# Fallback por calendario (solo si ESPN falla)
# =========================
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
    """
    Martes 00:00 ET de la semana del TNF. ¡OJO!: debe ser +1 día desde el lunes de esa semana.
    """
    tnf = _week1_tnf_et(year)
    week_monday = (tnf - timedelta(days=tnf.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return week_monday + timedelta(days=1)  # Tue 00:00 ET  (NO CAMBIAR A +2)

def _calendar_week_nat(season: int, now_utc: datetime) -> int:
    now_et  = now_utc.astimezone(ET)
    anchor  = _tuesday_anchor_et(season)
    if now_et < anchor:
        wk = 1
    else:
        wk = int(((now_et - anchor).days // 7) + 1)
    return max(1, min(22, wk))

# =========================
# Semana efectiva (ESPN -> corte diferido -> clamp)
# =========================
def _effective_week(season: int, now_utc: datetime | None = None) -> tuple[int, int | None, bool]:
    """
    Retorna (week_mostrar, api_week_espn, passed_cutover).
    Lógica:
      - api_week = ESPN.week.number (si falla, usar _calendar_week_nat)
      - si NO pasamos el corte (miércoles 02:00 ET), week_mostrar = max(1, api_week - 1)
      - si pasamos el corte, week_mostrar = api_week
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    api_week = _get_api_week_espn(season)
    if api_week is None:
        api_week = _calendar_week_nat(season, now_utc)

    passed = _after_cutover_et(now_utc)
    week_eff = api_week if passed else max(1, api_week - 1)
    week_eff = max(1, min(22, week_eff))
    return week_eff, api_week, passed

# =========================
# Presentación
# =========================
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
    d = et_dt.tz_convert(ET)
    label = f"Sun {_fmt_time_et(d)}"
    order = d.hour * 60 + d.minute
    return order, label

# =========================
# Render principal
# =========================
def render(season: int):
    st.subheader("Live")

    now_utc = datetime.now(timezone.utc)
    week, api_week, passed = _effective_week(int(season), now_utc)

    # ===== LABEL DE DEPURACIÓN QUE PEDISTE =====
    now_et = now_utc.astimezone(ET).strftime("%a %b %d, %I:%M %p ET")
    st.caption(
        f"ESPN api_week: **{api_week}** · Week mostrada (con corte Mié 02:00 ET): **{week}** · "
        f"after_cutover: **{passed}** · now: {now_et}"
    )

    # Trae el scoreboard de ESPN para la semana efectiva
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
