# tabs/live.py
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from nfl_dash.live_scores import fetch_espn_scoreboard_df
from nfl_dash.components import game_card

# Usamos ET para toda la lógica del calendario NFL
ET = ZoneInfo("America/New_York")

# =========================
# Helpers de calendario
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
    Ancla 'histórica' del cálculo: martes 00:00 ET de la semana del TNF.
    Sirve para indexar las semanas 1..22 con un cálculo determinista.
    """
    tnf = _week1_tnf_et(year)
    week_monday = (tnf - timedelta(days=tnf.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return week_monday + timedelta(days=1)  # Tue 00:00 ET

# --- Corte diferido: MIÉRCOLES 02:00 ET (≈ 03:00 BRT)
CUTOVER_DOW_ET = 2   # 0=Mon,1=Tue,2=Wed
CUTOVER_HOUR_ET = 2  # 02:00 ET

def _after_cutover_et(dt_utc: datetime) -> bool:
    """
    True si ya pasamos el corte semanal (miércoles 02:00 ET).
    """
    now_et = dt_utc.astimezone(ET)
    if now_et.weekday() > CUTOVER_DOW_ET:
        return True
    if now_et.weekday() < CUTOVER_DOW_ET:
        return False
    return now_et.hour >= CUTOVER_HOUR_ET

def _autodetect_week_with_cutover(season: int, now_utc: datetime | None = None) -> int:
    """
    Semana 'efectiva' para el dashboard Live:
    - Calcula la semana natural con ancla del martes.
    - Si aún NO pasamos el corte del miércoles 02:00 ET, usa (semana natural - 1).
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    now_et  = now_utc.astimezone(ET)
    anchor  = _tuesday_anchor_et(season)

    # Semana natural basada en el ancla
    wk_nat = 1 if now_et < anchor else int(((now_et - anchor).days // 7) + 1)
    wk_nat = max(1, min(22, wk_nat))

    # Ajuste por corte diferido
    if not _after_cutover_et(now_utc):
        return max(1, wk_nat - 1)
    return wk_nat

# =========================
# Helpers de presentación
# =========================
def _day_bucket(et_ts: pd.Timestamp) -> str:
    d = et_ts.tz_convert(ET)
    return {3:"Thursday", 5:"Saturday", 6:"Sunday", 0:"Monday"}.get(d.weekday(), d.strftime("%A"))

def _sun_slot(et_ts: pd.Timestamp) -> tuple[int, str]:
    d = et_ts.tz_convert(ET)
    label = d.strftime("%I:%M %p").lstrip("0")  # sin cero a la izquierda
    return d.hour * 60 + d.minute, f"Sun {label}"

def _suggest_refresh_interval(df: pd.DataFrame, now_et: datetime) -> int | None:
    # 1) si hay juegos live → 60s
    if df["state"].astype(str).str.lower().eq("in").any():
        return 60

    # 2) si hay un kickoff en ≤ 120 min → 60s
    future = df[df["state"].astype(str).str.lower().eq("pre")]
    if not future.empty:
        mins_to = (future["start_et"] - pd.Timestamp(now_et)).dt.total_seconds() / 60
        mins_to = mins_to[mins_to >= 0]
        if not mins_to.empty and mins_to.min() <= 120:
            return 60

    # 3) ventanas típicas NFL → 120s
    wd, hr = now_et.weekday(), now_et.hour  # 0=Mon
    in_thu = (wd == 3 and 18 <= hr <= 23)   # Thu 6–11:59 PM ET
    in_mon = (wd == 0 and 18 <= hr <= 23)   # Mon 6–11:59 PM ET
    in_sat = (wd == 5 and 12 <= hr <= 23)   # Sat 12–11:59 PM ET
    in_sun = (wd == 6 and 12 <= hr <= 23)   # Sun 12–11:59 PM ET
    if in_thu or in_mon or in_sat or in_sun:
        return 120

    # 4) hay partidos en próximas 24h → 180s
    if not future.empty:
        mins_any = (future["start_et"] - pd.Timestamp(now_et)).dt.total_seconds() / 60
        if (mins_any >= 0).any() and mins_any.min() <= 24*60:
            return 180

    # 5) fuera de ventanas → sin refresh
    return None

# =========================
# Render principal
# =========================
def render(season: int):
    # >>>>>>>>> CORAZÓN DEL FIX: usamos la semana con corte diferido
    week = _autodetect_week_with_cutover(int(season))

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

    # Traer tablero ESPN (no crashea si falla)
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

    # Auto-refresh “inteligente”
    now_et = datetime.now(timezone.utc).astimezone(ET)
    interval = _suggest_refresh_interval(df, now_et)
    if interval:
        try:
            from streamlit_extras.st_autorefresh import st_autorefresh
            st_autorefresh(interval=interval * 1000, key=f"live_autorefresh_{season}_{week}_{interval}")
        except Exception:
            pass  # si no está instalado, no refresca automático

    # Orden: día → en vivo primero → hora
    day_order = {"Thursday": 1, "Saturday": 2, "Sunday": 3, "Monday": 4}
    df["__day_ord"]  = df["day_bucket"].map(day_order).fillna(99)
    df["__kick_ord"] = df["start_et"].apply(lambda x: x.hour * 60 + x.minute)
    df["__live_ord"] = (~df["is_live"]).astype(int)
    df = df.sort_values(["__day_ord", "__live_ord", "start_et"])

    # Render por secciones y slots del domingo
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
