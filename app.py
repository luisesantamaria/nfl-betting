import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path
from datetime import datetime, timedelta, timezone
import math
import requests

st.set_page_config(page_title="NFL EV Betting - Dashboard", layout="wide")

# =========================
# Config de temporadas
# =========================
SEASON_RULES = {
    2024: {"season_start": "2024-09-05","season_end": "2025-02-12","activate_days_before": 7,"bets_open_week": 3},
    2025: {"season_start": "2025-09-04","season_end": "2026-02-11","activate_days_before": 7,"bets_open_week": 3},
}

# =========================
# Rutas
# =========================
def resolve_dir(*parts) -> Path:
    candidates = [
        Path(__file__).resolve().parent.joinpath(*parts),
        Path.cwd().joinpath(*parts),
        Path(__file__).resolve().parents[1].joinpath(*parts),
    ]
    for p in candidates:
        if p.exists(): return p
    return candidates[0]

PORTFOLIO_DIR = resolve_dir("data", "processed", "portfolio")
ARCHIVE_DIR   = resolve_dir("data", "archive")
BETSWEEK_DIR  = resolve_dir("data", "processed", "bets")   # opcional: this_week.csv

# Donde buscar odds/resultados (incluye tu 'bootstrap' en raíz y 'data/bootstrap')
ODDS_DIRS     = [
    resolve_dir("bootstrap"),                # <--- raíz/bootstrap
    resolve_dir("data", "bootstrap"),        # <--- data/bootstrap
    resolve_dir("data", "processed", "odds"),
    resolve_dir("data"),
]

LOGOS_DIR     = resolve_dir("assets", "logos", "nfl")      # cache local opcional

# =========================
# Helpers
# =========================
ORDER_LABELS = [f"Week {i}" for i in range(1, 19)] + ["Wild Card", "Divisional", "Conference", "Super Bowl"]
ORDER_INDEX  = {lab: i for i, lab in enumerate(ORDER_LABELS)}
PLAYOFF_LABEL_TO_NUM = {"Wild Card":19,"Divisional":20,"Conference":21,"Super Bowl":22}

TEAM_FIX = {  # normaliza siglas
    "STL":"LA","LAR":"LA","LA":"LA","SD":"LAC","SDG":"LAC","OAK":"LV","LVR":"LV","WSH":"WAS","JAC":"JAX",
    "GNB":"GB","KAN":"KC","NWE":"NE","NOR":"NO","SFO":"SF","TAM":"TB",
}

def norm_abbr(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip().upper()
    return TEAM_FIX.get(s, s)

def add_week_order(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["week_label"] = x["week_label"].astype(str)
    x["__order"] = x["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    x = x.sort_values("__order").drop(columns="__order")
    x["week_label"] = pd.Categorical(x["week_label"], categories=ORDER_LABELS, ordered=True)
    return x

def week_label_to_num(val) -> int:
    if pd.isna(val): return 999
    s = str(val)
    if s.startswith("Week "):
        try: return int(s.split(" ")[1])
        except: return 999
    return PLAYOFF_LABEL_TO_NUM.get(s, 999)

def make_pair(a: str, b: str) -> str:
    a, b = norm_abbr(a), norm_abbr(b)
    return f"{a}_{b}" if a < b else f"{b}_{a}"

def list_available_seasons():
    seasons = []
    for f in sorted(PORTFOLIO_DIR.glob("pnl_weekly_*.csv")):
        try: seasons.append(int(f.stem.split("_")[-1]))
        except: pass
    for y in SEASON_RULES: seasons.append(y)
    return sorted(set(seasons))

@st.cache_data
def load_pnl_weekly(year: int) -> pd.DataFrame:
    f = PORTFOLIO_DIR / f"pnl_weekly_{year}.csv"
    if not f.exists(): return pd.DataFrame()
    df = pd.read_csv(f)
    for col in ("week", "profit", "stake", "bankroll"):
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    if "week_label" in df.columns: df = add_week_order(df)
    else: df["week_label"] = "Week 999"
    return df

def american_to_decimal(v):
    try: v = float(v)
    except: return float("nan")
    return 1 + (100/abs(v) if v < 0 else v/100)

def decimal_to_american(d):
    try: d = float(d)
    except: return float("nan")
    return round((d - 1) * 100, 0) if d >= 2.0 else round(-100 / (d - 1), 0)

def find_ledger_path(year: int) -> Path | None:
    season_dir = ARCHIVE_DIR / f"season={year}"
    if not season_dir.exists(): return None
    candidates = [*season_dir.glob("bets_ledger*.csv"), season_dir / "ledger.csv", *season_dir.glob("*.csv")]
    for p in candidates:
        if p.exists(): return p
    return None

@st.cache_data
def load_ledger(year: int) -> pd.DataFrame:
    p = find_ledger_path(year)
    if not p: return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    for col in ("decimal_odds", "ml", "stake", "profit"):
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    if "decimal_odds" not in df.columns and "ml" in df.columns:
        df["decimal_odds"] = df["ml"].apply(american_to_decimal)
    # normaliza equipos base
    for c in ("team","opponent","home_team","away_team"):
        if c in df.columns: df[c] = df[c].astype(str).map(norm_abbr)
    return df

def load_bets_this_week(year: int) -> pd.DataFrame:
    candidates = [BETSWEEK_DIR / f"season={year}" / "this_week.csv", BETSWEEK_DIR / "this_week.csv"]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            for col in ("decimal_odds", "ml", "stake", "model_prob", "edge", "ev"):
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
            if "week_label" not in df.columns and "week" in df.columns:
                def week_label_from_num(n):
                    try: n = int(n)
                    except: return "Week 999"
                    if 1 <= n <= 18: return f"Week {n}"
                    return {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(n, f"Week {n}")
                df["week_label"] = df["week"].apply(week_label_from_num)
            if "week_label" in df.columns: df = add_week_order(df)
            for c in ("team","opponent"):
                if c in df.columns: df[c] = df[c].astype(str).map(norm_abbr)
            return df
    return pd.DataFrame()

def season_stage(year: int, pnl_df: pd.DataFrame) -> str:
    rule = SEASON_RULES.get(year, {})
    now = datetime.now(timezone.utc)
    if not pnl_df.empty:
        labels = set(map(str, pnl_df["week_label"].astype(str).unique()))
        if "Super Bowl" in labels or "Conference" in labels: return "ended"
    start = datetime.fromisoformat(rule.get("season_start", f"{year}-09-05")).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(rule.get("season_end",   f"{year+1}-02-12")).replace(tzinfo=timezone.utc)
    activate_from = start - timedelta(days=int(rule.get("activate_days_before", 0)))
    if now < activate_from: return "locked"
    if now < start: return "preseason"
    if now <= end: return "in_season"
    return "ended"

def kpis_from_pnl(df: pd.DataFrame):
    profits = pd.to_numeric(df.get("profit"), errors="coerce").fillna(0.0)
    stakes  = pd.to_numeric(df.get("stake"),  errors="coerce").fillna(0.0)
    banks   = pd.to_numeric(df.get("bankroll"), errors="coerce")
    first_bankroll   = float(banks.iloc[0]); first_profit = float(profits.iloc[0])
    initial_bankroll = float(first_bankroll - first_profit)
    final_bankroll   = float(banks.iloc[-1])
    total_profit     = float(profits.sum())
    total_stake      = float(stakes.sum())
    yield_pct        = (total_profit / total_stake * 100.0) if total_stake > 0 else 0.0
    return initial_bankroll, final_bankroll, total_profit, total_stake, yield_pct, profits, stakes

# --------- Carga de SCORES/RESULTADOS desde archivos de odds ---------
def _candidate_odds_files(year: int):
    names = [
        f"odds_season_{year}.csv",           # <--- tu nombre principal
        f"target_odds_{year}.csv",
        f"targets_odds_{year}.csv",
        "target_odds.csv",
        "targets_odds.csv",
        "historical_odds.csv",
    ]
    for d in ODDS_DIRS:
        # nombres explícitos
        for n in names:
            p = d / n
            if p.exists(): yield p
        # patrón genérico por si cambias nombres: incluye "odds" y año
        for p in d.glob(f"*odds*{year}*.csv"):
            yield p

@st.cache_data
def load_scores_table(year: int) -> pd.DataFrame:
    """
    Devuelve tabla con columnas: season, week, home_team, away_team, score_home, score_away, schedule_date, pair
    para el año solicitado, combinando el/los archivos disponibles.
    """
    frames = []
    seen = set()
    for p in _candidate_odds_files(year):
        key = p.resolve().as_posix()
        if key in seen: continue
        seen.add(key)
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        cols_needed_some = {"home_team","away_team","week","season"}
        if not cols_needed_some.issubset(set(df.columns)):
            continue

        df = df.copy()
        for c in ("home_team","away_team"):
            df[c] = df[c].astype(str).map(norm_abbr)
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
        df["week"]   = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
        if "schedule_date" in df.columns:
            try:
                df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")
            except Exception:
                pass

        df = df[df["season"].astype("Int64").eq(year)]
        if df.empty: continue

        df["pair"] = [make_pair(a,b) for a,b in zip(df["home_team"], df["away_team"])]

        keep = ["season","week","home_team","away_team","pair",
                "score_home","score_away","schedule_date"]
        keep = [c for c in keep if c in df.columns]
        frames.append(df[keep])

    if not frames:
        return pd.DataFrame(columns=["season","week","home_team","away_team","pair","score_home","score_away","schedule_date"])

    out = pd.concat(frames, ignore_index=True)
    out = (out.sort_values(["season","week","schedule_date"])
              .drop_duplicates(subset=["season","week","pair"], keep="last"))
    return out

def ensure_week_num_column(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "wk_num" not in x.columns:
        if "week" in x.columns and pd.api.types.is_numeric_dtype(x["week"]):
            x["wk_num"] = pd.to_numeric(x["week"], errors="coerce").astype("Int64")
        elif "week_label" in x.columns:
            x["wk_num"] = x["week_label"].apply(week_label_to_num).astype("Int64")
        else:
            x["wk_num"] = pd.Series([None]*len(x), dtype="Int64")
    return x

@st.cache_data
def enrich_bets_with_scores(bets_df: pd.DataFrame, year: int) -> pd.DataFrame:
    if bets_df.empty: return bets_df
    b = bets_df.copy()
    for c in ("team","opponent"):
        if c in b.columns:
            b[c] = b[c].astype(str).map(norm_abbr)
    b = ensure_week_num_column(b)
    b["pair"] = [make_pair(a,o) for a,o in zip(b.get("team",""), b.get("opponent",""))]

    scores = load_scores_table(year)
    if scores.empty: return b

    merged = b.merge(
        scores,
        left_on=["season","wk_num","pair"],
        right_on=["season","week","pair"],
        how="left",
        suffixes=("","_sc")
    )
    if "schedule_date" not in b.columns and "schedule_date_sc" in merged.columns:
        merged["schedule_date"] = merged["schedule_date_sc"]

    for c in ("home_team","away_team"):
        if c not in merged.columns and f"{c}_sc" in merged.columns:
            merged[c] = merged[f"{c}_sc"]

    drop_cols = [c for c in merged.columns if c.endswith("_sc")] + ["week_y"]
    merged = merged.rename(columns={"week_x":"week"}).drop(columns=[c for c in drop_cols if c in merged.columns], errors="ignore")
    return merged

# ---------- Logos (URL o local cache) ----------
ESPN_SLUG = {
    "ARI":"ari","ATL":"atl","BAL":"bal","BUF":"buf","CAR":"car","CHI":"chi","CIN":"cin","CLE":"cle",
    "DAL":"dal","DEN":"den","DET":"det","GB":"gb","HOU":"hou","IND":"ind","JAX":"jax","KC":"kc",
    "LA":"lar","LAR":"lar","LAC":"lac","LV":"lv","MIA":"mia","MIN":"min","NE":"ne","NO":"no",
    "NYG":"nyg","NYJ":"nyj","PHI":"phi","PIT":"pit","SEA":"sea","SF":"sf","TB":"tb","TEN":"ten",
    "WAS":"wsh","WSH":"wsh"
}

@st.cache_data(ttl=60*60*24)
def get_logo_url(abbr: str) -> str | None:
    if not abbr: return None
    a = abbr.upper().strip()
    local = LOGOS_DIR / f"{a}.png"
    if local.exists(): return local.as_posix()
    candidates = [
        f"https://static.www.nfl.com/t_q-best/league/api/clubs/logos/{a}",
        f"https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/{ESPN_SLUG.get(a, a.lower())}.png",
        f"https://a.espncdn.com/i/teamlogos/nfl/500/{ESPN_SLUG.get(a, a.lower())}.png",
    ]
    for url in candidates:
        try:
            r = requests.head(url, timeout=2, allow_redirects=True)
            if r.status_code == 200: return url
        except Exception:
            continue
    return None

# ============ UI ============
st.title("NFL EV Betting — Dashboard")

seasons = list_available_seasons()
if not seasons:
    st.warning("No seasons found."); st.stop()

season = st.selectbox("Season", options=seasons, index=seasons.index(max(seasons)))
pnl = load_pnl_weekly(season)
bets_all_raw = load_ledger(season)
bets_all = enrich_bets_with_scores(bets_all_raw, season) if not bets_all_raw.empty else bets_all_raw
stage = season_stage(season, pnl)

status_map = {"locked":"Locked","preseason":"Preseason","in_season":"In Season","ended":"Season Ended","unknown":"Unknown"}
st.caption(f"Status: **{status_map.get(stage, 'Unknown')}**")

tab_overview, tab_portfolio, tab_bets = st.tabs(["Overview", "Portfolio", "Bets"])

# ---------- OVERVIEW ----------
with tab_overview:
    if stage == "in_season":
        bets_week = load_bets_this_week(season)
        if not bets_week.empty:
            st.subheader("This Week’s Bets")
            cols = [c for c in ["week","week_label","schedule_date","side","team","opponent","ml","decimal_odds","model_prob","edge","ev","stake"] if c in bets_week.columns]
            st.dataframe(bets_week[cols] if cols else bets_week, use_container_width=True)
            st.divider()

    st.subheader("Season Overview")
    if pnl.empty:
        st.caption("No hay `pnl_weekly` para esta temporada.")
    else:
        initial_bankroll, final_bankroll, total_profit, total_stake, yield_pct, profits, stakes = kpis_from_pnl(pnl)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Initial", f"${initial_bankroll:,.2f}")
        k2.metric("Final",   f"${final_bankroll:,.2f}", f"{(final_bankroll-initial_bankroll):,.2f}")
        k3.metric("Total Profit", f"${total_profit:,.2f}")
        k4.metric("Yield",        f"{yield_pct:.2f}%")

        cum_df = add_week_order(pd.DataFrame({"week_label": pnl["week_label"], "cum_profit": profits.cumsum()}))
        y2min, y2max = float(cum_df["cum_profit"].min()), float(cum_df["cum_profit"].max())
        pad2 = max(5.0, (y2max - y2min) * 0.06)
        spark = (
            alt.Chart(cum_df).mark_area(opacity=0.25)
            .encode(
                x=alt.X("week_label:N", sort=None, title=""),
                y=alt.Y("cum_profit:Q", title="Cumulative Profit ($)", scale=alt.Scale(domain=[y2min - pad2, y2max + pad2], zero=False, nice=False)),
                tooltip=[alt.Tooltip("week_label:N", title="Week"), alt.Tooltip("cum_profit:Q", title="Cum Profit", format="$.2f")],
            ).properties(height=200, width="container")
        ) + alt.Chart(cum_df).mark_line().encode(x=alt.X("week_label:N", sort=None, title=""), y="cum_profit:Q")

        last8 = add_week_order(pd.DataFrame({"week_label": pnl["week_label"], "profit": profits}))
        if len(last8) > 8: last8 = last8.tail(8)
        mini_bars = (
            alt.Chart(last8).mark_bar()
            .encode(
                x=alt.X("week_label:N", sort=None, title=""),
                y=alt.Y("profit:Q", title="Last 8 Weeks Profit ($)", scale=alt.Scale(zero=True)),
                tooltip=[alt.Tooltip("week_label:N", title="Week"), alt.Tooltip("profit:Q", title="Profit", format="$.2f")],
            ).properties(height=200, width="container")
        )
        cA, cB = st.columns(2)
        with cA: st.altair_chart(spark, use_container_width=True)
        with cB: st.altair_chart(mini_bars, use_container_width=True)

# ---------- PORTFOLIO (grid 2×2) ----------
with tab_portfolio:
    st.subheader("Portfolio")
    if pnl.empty:
        st.caption("No hay `pnl_weekly` para esta temporada.")
    else:
        initial_bankroll, final_bankroll, total_profit, total_stake, yield_pct, profits, stakes = kpis_from_pnl(pnl)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Initial", f"${initial_bankroll:,.2f}")
        m2.metric("Final",   f"${final_bankroll:,.2f}", f"{(final_bankroll-initial_bankroll):,.2f}")
        m3.metric("Total Profit", f"${total_profit:,.2f}")
        m4.metric("Yield",        f"{yield_pct:.2f}%")

        bank_df = add_week_order(pnl[["week_label", "bankroll"]].dropna())
        prof_df = add_week_order(pd.DataFrame({"week_label": pnl["week_label"], "profit": profits, "stake": stakes}))
        cum_df  = add_week_order(pd.DataFrame({"week_label": pnl["week_label"], "cum_profit": profits.cumsum()}))
        stake_df= add_week_order(pd.DataFrame({"week_label": pnl["week_label"], "stake": stakes}))

        H = 220
        ymin, ymax = float(bank_df["bankroll"].min()), float(bank_df["bankroll"].max()); pad = max(10.0, (ymax - ymin) * 0.06)
        bank_chart = alt.Chart(bank_df).mark_line(point=True).encode(
            x=alt.X("week_label:N", sort=None, title=""),
            y=alt.Y("bankroll:Q", title="Bankroll ($)", scale=alt.Scale(domain=[ymin - pad, ymax + pad], zero=False, nice=False)),
            tooltip=[alt.Tooltip("week_label:N", title="Week"), alt.Tooltip("bankroll:Q", title="Bankroll", format="$.2f")],
        ).properties(height=H, width="container")

        profit_chart = alt.Chart(prof_df).mark_bar().encode(
            x=alt.X("week_label:N", sort=None, title=""),
            y=alt.Y("profit:Q", title="Profit ($)", scale=alt.Scale(zero=True)),
            tooltip=[alt.Tooltip("week_label:N", title="Week"), alt.Tooltip("profit:Q", title="Profit", format="$.2f"), alt.Tooltip("stake:Q", title="Stake", format="$.2f")],
        ).properties(height=H, width="container")

        y2min, y2max = float(cum_df["cum_profit"].min()), float(cum_df["cum_profit"].max()); pad2 = max(5.0, (y2max - y2min) * 0.06)
        cum_chart = alt.Chart(cum_df).mark_line(point=True).encode(
            x=alt.X("week_label:N", sort=None, title=""),
            y=alt.Y("cum_profit:Q", title="Cumulative Profit ($)", scale=alt.Scale(domain=[y2min - pad2, y2max + pad2], zero=False, nice=False)),
            tooltip=[alt.Tooltip("week_label:N", title="Week"), alt.Tooltip("cum_profit:Q", title="Cum Profit", format="$.2f")],
        ).properties(height=H, width="container")

        stake_chart = alt.Chart(stake_df).mark_bar().encode(
            x=alt.X("week_label:N", sort=None, title=""),
            y=alt.Y("stake:Q", title="Stake ($)", scale=alt.Scale(zero=True)),
            tooltip=[alt.Tooltip("week_label:N", title="Week"), alt.Tooltip("stake:Q", title="Stake", format="$.2f")],
        ).properties(height=H, width="container")

        c1, c2 = st.columns(2); c3, c4 = st.columns(2)
        with c1: st.altair_chart(bank_chart, use_container_width=True)
        with c2: st.altair_chart(profit_chart, use_container_width=True)
        with c3: st.altair_chart(cum_chart, use_container_width=True)
        with c4: st.altair_chart(stake_chart, use_container_width=True)

# ---------- BETS (tarjetas compactas con SCORE) ----------
ESPN_SLUG = ESPN_SLUG  # alias local

@st.cache_data(ttl=60*60*24)
def get_logo_url(abbr: str) -> str | None:
    if not abbr: return None
    a = abbr.upper().strip()
    local = LOGOS_DIR / f"{a}.png"
    if local.exists(): return local.as_posix()
    candidates = [
        f"https://static.www.nfl.com/t_q-best/league/api/clubs/logos/{a}",
        f"https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/{ESPN_SLUG.get(a, a.lower())}.png",
        f"https://a.espncdn.com/i/teamlogos/nfl/500/{ESPN_SLUG.get(a, a.lower())}.png",
    ]
    for url in candidates:
        try:
            r = requests.head(url, timeout=2, allow_redirects=True)
            if r.status_code == 200: return url
        except Exception:
            continue
    return None

def bet_card(row: pd.Series):
    team = norm_abbr(row.get("team", ""))
    opp  = norm_abbr(row.get("opponent", ""))
    side = str(row.get("side", "")).title() if pd.notna(row.get("side")) else ""
    wl_profit = pd.to_numeric(row.get("profit"), errors="coerce")
    stake = pd.to_numeric(row.get("stake"), errors="coerce")
    dec = pd.to_numeric(row.get("decimal_odds"), errors="coerce")
    ml  = pd.to_numeric(row.get("ml"), errors="coerce")
    if pd.isna(ml) and pd.notna(dec): ml = decimal_to_american(dec)

    if pd.isna(wl_profit): status_color, status_label = "#888888", "OPEN"
    elif wl_profit > 0:    status_color, status_label = "#1FA37C", "WIN"
    elif wl_profit < 0:    status_color, status_label = "#D64545", "LOSS"
    else:                  status_color, status_label = "#8884D8", "PUSH"

    wk = str(row.get("week_label", row.get("week", "")))
    date_txt = str(row.get("schedule_date", ""))[:10] if pd.notna(row.get("schedule_date")) else ""

    # SCORE desde odds_season_YEAR.csv u otros
    sh = row.get("score_home", None); sa = row.get("score_away", None)
    htm = norm_abbr(row.get("home_team","")); atm = norm_abbr(row.get("away_team",""))
    score_txt = ""
    if pd.notna(sh) and pd.notna(sa) and (htm or atm):
        try: score_txt = f"{htm or 'HOME'} {int(sh)} — {atm or 'AWAY'} {int(sa)}"
        except Exception: score_txt = ""

    team_logo = get_logo_url(team)
    opp_logo  = get_logo_url(opp)
    ml_txt    = f"{ml:+.0f}" if pd.notna(ml) else "—"
    stake_txt = f"${stake:,.2f}" if pd.notna(stake) else "—"
    prof_txt  = f"${wl_profit:,.2f}" if pd.notna(wl_profit) else "—"

    st.markdown(f"""
    <div style="
        border:1px solid #e9e9e9;border-radius:12px;padding:10px; margin-bottom:8px;
        background:linear-gradient(180deg, rgba(0,0,0,0.02), rgba(0,0,0,0.00));
        font-size:12.5px;
    ">
      <div style="display:flex; align-items:center; justify-content:space-between;">
        <div style="display:flex; align-items:center; gap:8px;">
          <div style="width:8px;height:8px;border-radius:50%;background:{status_color};"></div>
          <div style="font-weight:600;">{wk}</div>
          <div style="opacity:.7;">{date_txt}</div>
        </div>
        <div style="font-weight:700;color:{status_color}">{status_label}</div>
      </div>

      <div style="display:flex; align-items:center; gap:10px; margin-top:8px;">
        <div style="display:flex; align-items:center; gap:6px;">
          {"<img src='"+team_logo+"' width='22' />" if team_logo else f"<div style='width:22px;height:22px;border-radius:50%;background:#222;color:#fff;display:flex;align-items:center;justify-content:center;font-size:10px;'>{team[:3]}</div>"}
          <div style="font-weight:600;">{team}</div>
          <div style="opacity:.65;">({side})</div>
          <div style="opacity:.5;">vs</div>
          {"<img src='"+opp_logo+"' width='22' />" if opp_logo else f"<div style='width:22px;height:22px;border-radius:50%;background:#555;color:#fff;display:flex;align-items:center;justify-content:center;font-size:10px;'>{opp[:3]}</div>"}
          <div style="font-weight:600;">{opp}</div>
        </div>
      </div>

      {"<div style='margin-top:6px;opacity:.85;'>" + score_txt + "</div>" if score_txt else ""}

      <div style="display:flex; gap:14px; margin-top:8px; flex-wrap:wrap;">
        <div><span style="opacity:.6;">Moneyline:</span> <strong>{ml_txt}</strong></div>
        <div><span style="opacity:.6;">Stake:</span> <strong>{stake_txt}</strong></div>
        <div><span style="opacity:.6;">Profit:</span> <strong style="color:{status_color};">{prof_txt}</strong></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with tab_bets:
    st.subheader("Bets")
    if bets_all.empty:
        st.caption("No hay archivo de bets para esta temporada.")
    else:
        view = bets_all.copy()
        if "week_label" in view.columns:
            view["week_label"] = view["week_label"].astype(str)
            view["__order"] = view["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
            sort_cols = ["__order"]
            if "schedule_date" in view.columns: sort_cols.append("schedule_date")
            view = view.sort_values(sort_cols).drop(columns="__order")

        # Tarjetas 4 por fila (sin "ver tabla completa")
        cards = list(view.itertuples(index=False))
        idx = 0
        cols_per_row = 4
        rows = math.ceil(len(cards) / cols_per_row)
        for _ in range(rows):
            col_objs = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if idx < len(cards):
                    with col_objs[j]:
                        bet_card(pd.Series(cards[idx]._asdict()))
                    idx += 1
