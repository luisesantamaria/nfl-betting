# tabs/live.py
import requests
import pandas as pd
import streamlit as st

from nfl_dash.live_scores import fetch_espn_scoreboard_df
from nfl_dash.config import SEASON_RULES
from nfl_dash.utils import norm_abbr
from nfl_dash.logos import get_logo_url

DAY_ORDER = ["Thursday", "Saturday", "Sunday", "Monday"]
SUN_WINDOWS = ["Sun 1:00 PM", "Sun 4:05 PM", "Sun 4:25 PM", "Sun 8:20 PM", "Sun Other"]


# ---------------- Semana actual ----------------
def _espn_current_week(season: int) -> int | None:
    url = "https://site.api.espn.com/apis/v2/sports/football/nfl/scoreboard"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        wk = (r.json().get("week") or {}).get("number")
        if isinstance(wk, int) and 1 <= wk <= 22:
            return wk
    except Exception:
        pass
    return None


def _tuesday_anchor_et(ts_et: pd.Timestamp) -> pd.Timestamp:
    ts_et = ts_et.tz_localize("US/Eastern") if ts_et.tz is None else ts_et.tz_convert("US/Eastern")
    # Mon=0...Sun=6 ; Tuesday=1
    days_since_tue = (ts_et.weekday() - 1) % 7
    return (ts_et - pd.Timedelta(days=days_since_tue)).normalize()


def _guess_week_by_rules(season: int) -> int:
    rules = SEASON_RULES.get(int(season), {})
    rs = rules.get("regular_start")
    if not rs:
        return 1
    rs_utc = pd.Timestamp(rs)
    rs_utc = rs_utc.tz_localize("UTC") if rs_utc.tz is None else rs_utc
    week1_tue = _tuesday_anchor_et(rs_utc.tz_convert("US/Eastern"))
    now_et = pd.Timestamp.now(tz="US/Eastern")
    if now_et < week1_tue:
        return 1
    delta_days = (now_et - week1_tue).days
    return int(min(22, max(1, delta_days // 7 + 1)))


def _auto_week(season: int) -> int:
    return _espn_current_week(season) or _guess_week_by_rules(season)


def _pick_week_and_fetch(season: int):
    w = _auto_week(season)
    for cand in [w, min(22, w + 1), max(1, w - 1)]:
        df = fetch_espn_scoreboard_df(int(season), int(cand))
        if not df.empty:
            return cand, df
    return w, pd.DataFrame()


# ---------------- Render helpers ----------------
def _bucket_day(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "TBD"
    return ts.tz_convert("US/Eastern").strftime("%A")


def _sun_window(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "Sun Other"
    t = ts.tz_convert("US/Eastern")
    if t.weekday() != 6:  # Sunday
        return ""
    h, m = t.hour, t.minute
    if h == 13:
        return "Sun 1:00 PM"
    if h == 16 and m == 5:
        return "Sun 4:05 PM"
    if h == 16 and m == 25:
        return "Sun 4:25 PM"
    if h == 20 and m == 20:
        return "Sun 8:20 PM"
    return "Sun Other"


def _kick_txt(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    t = ts.tz_convert("US/Eastern")
    try:
        return t.strftime("%-m/%-d • %-I:%M %p ET")
    except Exception:
        return t.strftime("%m/%d • %I:%M %p ET")


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
    hs, as_ = row.get("home_score"), row.get("away_score")
    has_score = pd.notna(hs) and pd.notna(as_)
    score_html = (
        f"<div style='font-weight:900;font-size:26px;letter-spacing:.5px;'>{int(hs)} — {int(as_)}</div>"
        if has_score else "<div style='opacity:.55;font-weight:700;'>TBD</div>"
    )

    st.markdown(
        f"""
    <div style="border:1px solid #e9e9e9;border-radius:12px;padding:12px;margin-bottom:10px;
                background:linear-gradient(180deg,rgba(0,0,0,0.02),rgba(0,0,0,0));font-size:12.5px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div style="font-weight:600;">{_kick_txt(row.get("start_time"))}</div>
        <div>{_live_badge(row.get("state",""))}</div>
      </div>

      <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:10px;">
        <div style="width:44px;display:flex;align-items:center;justify-content:center;">{left_logo}</div>
        <div style="flex:1;text-align:center;">{score_html}</div>
        <div style="width:44px;display:flex;align-items:center;justify-content:center;">{right_logo}</div>
      </div>

      <div style="display:flex;align-items:center;justify-content:space-between;margin-top:6px;">
        <div style="font-size:12px;opacity:.8;font-weight:600;">{row.get("home_team","")}</div>
        <div style="font-size:12px;opacity:.5;font-weight:700;">vs</div>
        <div style="font-size:12px;opacity:.8;font-weight:600;text-align:right;">{row.get("away_team","")}</div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ---------------- Pestaña Live ----------------
def render(season: int):
    st.subheader("Live")

    week, df = _pick_week_and_fetch(season)
    if df.empty:
        st.caption("No games found for this week.")
        return

    # refresco amable si hay LIVE
    if (df["state"].astype(str).str.lower().eq("in")).any():
        try:
            from streamlit_extras.st_autorefresh import st_autorefresh
            st_autorefresh(interval=60_000, key=f"live_auto_{season}_{week}")
        except Exception:
            pass

    st.caption(f"Showing Week {week} (auto-detected, Tue 00:00 ET cutoff)")

    df = df.copy()
    df["bucket"] = df["start_time"].apply(_bucket_day)
    df["bucket"] = pd.Categorical(df["bucket"], categories=DAY_ORDER, ordered=True)
    df["sub"] = df["start_time"].apply(_sun_window)
    df["sub"] = pd.Categorical(df["sub"], categories=SUN_WINDOWS, ordered=True)
    df = df.sort_values(["bucket", "start_time"])

    for day in [d for d in DAY_ORDER if (df["bucket"] == d).any()]:
        st.markdown(f"### {day}")
        rows = df[df["bucket"] == day].reset_index(drop=True)

        if day != "Sunday":
            cols_per_row, i, n = 3, 0, len(rows)
            while i < n:
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i >= n: break
                    with cols[j]: _game_card(rows.loc[i])
                    i += 1
        else:
            for sub in [s for s in SUN_WINDOWS if (rows["sub"] == s).any()]:
                sub_rows = rows[rows["sub"] == sub].reset_index(drop=True)
                if sub_rows.empty: continue
                st.markdown(f"#### {sub}")
                cols_per_row, i, n = 3, 0, len(sub_rows)
                while i < n:
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i >= n: break
                        with cols[j]: _game_card(sub_rows.loc[i])
                        i += 1
