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
    2024: {
        "season_start": "2024-09-05",
        "season_end":   "2025-02-12",
        "activate_days_before": 7,
        "bets_open_week": 3,
    },
    2025: {
        "season_start": "2025-09-04",
        "season_end":   "2026-02-11",
        "activate_days_before": 7,
        "bets_open_week": 3,
    },
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
        if p.exists():
            return p
    return candidates[0]

PORTFOLIO_DIR = resolve_dir("data", "processed", "portfolio")
ARCHIVE_DIR   = resolve_dir("data", "archive")
BETSWEEK_DIR  = resolve_dir("data", "processed", "bets")  # opcional: this_week.csv
LOGOS_DIR     = resolve_dir("assets", "logos", "nfl")     # si usas workflow para cachear

# =========================
# Helpers
# =========================
ORDER_LABELS = [f"Week {i}" for i in range(1, 19)] + ["Wild Card", "Divisional", "Conference", "Super Bowl"]
ORDER_INDEX  = {lab: i for i, lab in enumerate(ORDER_LABELS)}

def add_week_order(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["week_label"] = x["week_label"].astype(str)
    x["__order"] = x["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    x = x.sort_values("__order").drop(columns="__order")
    x["week_label"] = pd.Categorical(x["week_label"], categories=ORDER_LABELS, ordered=True)
    return x

def list_available_seasons():
    seasons = []
    for f in sorted(PORTFOLIO_DIR.glob("pnl_weekly_*.csv")):
        try:
            seasons.append(int(f.stem.split("_")[-1]))
        except:
            pass
    for y in SEASON_RULES:
        seasons.append(y)
    return sorted(set(seasons))

@st.cache_data
def load_pnl_weekly(year: int) -> pd.DataFrame:
    f = PORTFOLIO_DIR / f"pnl_weekly_{year}.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    for col in ("week", "profit", "stake", "bankroll"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "week_label" in df.columns:
        df = add_week_order(df)
    else:
        df["week_label"] = "Week 999"
    return df

def american_to_decimal(v):
    try:
        v = float(v)
    except Exception:
        return float("nan")
    return 1 + (100/abs(v) if v < 0 else v/100)

def decimal_to_american(d):
    try:
        d = float(d)
    except Exception:
        return float("nan")
    return round((d - 1) * 100, 0) if d >= 2.0 else round(-100 / (d - 1), 0)

def find_ledger_path(year: int) -> Path | None:
    season_dir = ARCHIVE_DIR / f"season={year}"
    if not season_dir.exists():
        return None
    candidates = [
        *season_dir.glob("bets_ledger*.csv"),
        season_dir / "ledger.csv",
        *season_dir.glob("*.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

@st.cache_data
def load_ledger(year: int) -> pd.DataFrame:
    p = find_ledger_path(year)
    if not p:
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    for col in ("decimal_odds", "ml", "stake", "profit"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "decimal_odds" not in df.columns and "ml" in df.columns:
        df["decimal_odds"] = df["ml"].apply(american_to_decimal)
    return df

def load_bets_this_week(year: int) -> pd.DataFrame:
    candidates = [
        BETSWEEK_DIR / f"season={year}" / "this_week.csv",
        BETSWEEK_DIR / "this_week.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            for col in ("decimal_odds", "ml", "stake", "model_prob", "edge", "ev"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "week_label" not in df.columns and "week" in df.columns:
                def week_label_from_num(n):
                    try:
                        n = int(n)
                    except:
                        return "Week 999"
                    if 1 <= n <= 18: return f"Week {n}"
                    return {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(n, f"Week {n}")
                df["week_label"] = df["week"].apply(week_label_from_num)
            if "week_label" in df.columns:
                df = add_week_order(df)
            return df
    return pd.DataFrame()

def season_stage(year: int, pnl_df: pd.DataFrame) -> str:
    rule = SEASON_RULES.get(year, {})
    now = datetime.now(timezone.utc)

    if not pnl_df.empty:
        labels = set(map(str, pnl_df["week_label"].astype(str).unique()))
        if "Super Bowl" in labels or "Conference" in labels:
            return "ended"

    start = datetime.fromisoformat(rule.get("season_start", f"{year}-09-05")).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(rule.get("season_end",   f"{year+1}-02-12")).replace(tzinfo=timezone.utc)
    activate_from = start - timedelta(days=int(rule.get("activate_days_before", 0)))

    if now < activate_from:
        return "locked"
    if now < start:
        return "preseason"
    if now <= end:
        return "in_season"
    return "ended"

def kpis_from_pnl(df: pd.DataFrame):
    profits = pd.to_numeric(df.get("profit"), errors="coerce").fillna(0.0)
    stakes  = pd.to_numeric(df.get("stake"),  errors="coerce").fillna(0.0)
    banks   = pd.to_numeric(df.get("bankroll"), errors="coerce")
    first_bankroll   = float(banks.iloc[0])
    first_profit     = float(profits.iloc[0])
    initial_bankroll = float(first_bankroll - first_profit)
    final_bankroll   = float(banks.iloc[-1])
    total_profit     = float(profits.sum())
    total_stake      = float(stakes.sum())
    yield_pct        = (total_profit / total_stake * 100.0) if total_stake > 0 else 0.0
    return initial_bankroll, final_bankroll, total_profit, total_stake, yield_pct, profits, stakes

# ============ UI ============
st.title("NFL EV Betting — Dashboard")

seasons = list_available_seasons()
if not seasons:
    st.warning("No seasons found.")
    st.stop()

season = st.selectbox("Season", options=seasons, index=seasons.index(max(seasons)))
pnl = load_pnl_weekly(season)
bets_all = load_ledger(season)
stage = season_stage(season, pnl)

status_map = {
    "locked": "Locked",
    "preseason": "Preseason",
    "in_season": "In Season",
    "ended": "Season Ended",
    "unknown": "Unknown",
}
st.caption(f"Status: **{status_map.get(stage, 'Unknown')}**")

tab_overview, tab_portfolio, tab_bets = st.tabs(["Overview", "Portfolio", "Bets"])

# ---------- OVERVIEW ----------
with tab_overview:
    if stage == "in_season":
        bets_week = load_bets_this_week(season)
        if not bets_week.empty:
            st.subheader("This Week’s Bets")
            cols = [c for c in [
                "week","week_label","schedule_date","side","team","opponent",
                "ml","decimal_odds","model_prob","edge","ev","stake"
            ] if c in bets_week.columns]
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

        cum_df = pd.DataFrame({"week_label": pnl["week_label"], "cum_profit": profits.cumsum()})
        cum_df = add_week_order(cum_df)
        y2min, y2max = float(cum_df["cum_profit"].min()), float(cum_df["cum_profit"].max())
        pad2 = max(5.0, (y2max - y2min) * 0.06)

        spark = (
            alt.Chart(cum_df)
            .mark_area(opacity=0.25)
            .encode(
                x=alt.X("week_label:N", sort=None, title=""),
                y=alt.Y("cum_profit:Q",
                        title="Cumulative Profit ($)",
                        scale=alt.Scale(domain=[y2min - pad2, y2max + pad2], zero=False, nice=False)),
                tooltip=[alt.Tooltip("week_label:N", title="Week"),
                         alt.Tooltip("cum_profit:Q", title="Cum Profit", format="$.2f")],
            )
            .properties(height=200, width="container")
        ) + (
            alt.Chart(cum_df)
            .mark_line()
            .encode(x=alt.X("week_label:N", sort=None, title=""), y="cum_profit:Q")
        )

        last8 = pd.DataFrame({"week_label": pnl["week_label"], "profit": profits})
        last8 = add_week_order(last8)
        if len(last8) > 8:
            last8 = last8.tail(8)
        mini_bars = (
            alt.Chart(last8)
            .mark_bar()
            .encode(
                x=alt.X("week_label:N", sort=None, title=""),
                y=alt.Y("profit:Q", title="Last 8 Weeks Profit ($)", scale=alt.Scale(zero=True)),
                tooltip=[alt.Tooltip("week_label:N", title="Week"),
                         alt.Tooltip("profit:Q", title="Profit", format="$.2f")],
            )
            .properties(height=200, width="container")
        )

        cA, cB = st.columns(2)
        with cA:
            st.altair_chart(spark, use_container_width=True)
        with cB:
            st.altair_chart(mini_bars, use_container_width=True)

# ---------- PORTFOLIO (grid 2×2, gráficos compactos) ----------
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

        # Compact heights
        H = 220

        ymin, ymax = float(bank_df["bankroll"].min()), float(bank_df["bankroll"].max())
        pad = max(10.0, (ymax - ymin) * 0.06)
        bank_chart = (
            alt.Chart(bank_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("week_label:N", sort=None, title=""),
                y=alt.Y("bankroll:Q", title="Bankroll ($)",
                        scale=alt.Scale(domain=[ymin - pad, ymax + pad], zero=False, nice=False)),
                tooltip=[alt.Tooltip("week_label:N", title="Week"),
                         alt.Tooltip("bankroll:Q", title="Bankroll", format="$.2f")],
            )
            .properties(height=H, width="container")
        )

        profit_chart = (
            alt.Chart(prof_df)
            .mark_bar()
            .encode(
                x=alt.X("week_label:N", sort=None, title=""),
                y=alt.Y("profit:Q", title="Profit ($)", scale=alt.Scale(zero=True)),
                tooltip=[alt.Tooltip("week_label:N", title="Week"),
                         alt.Tooltip("profit:Q", title="Profit", format="$.2f"),
                         alt.Tooltip("stake:Q", title="Stake", format="$.2f")],
            )
            .properties(height=H, width="container")
        )

        y2min, y2max = float(cum_df["cum_profit"].min()), float(cum_df["cum_profit"].max())
        pad2 = max(5.0, (y2max - y2min) * 0.06)
        cum_chart = (
            alt.Chart(cum_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("week_label:N", sort=None, title=""),
                y=alt.Y("cum_profit:Q", title="Cumulative Profit ($)",
                        scale=alt.Scale(domain=[y2min - pad2, y2max + pad2], zero=False, nice=False)),
                tooltip=[alt.Tooltip("week_label:N", title="Week"),
                         alt.Tooltip("cum_profit:Q", title="Cum Profit", format="$.2f")],
            )
            .properties(height=H, width="container")
        )

        stake_chart = (
            alt.Chart(stake_df)
            .mark_bar()
            .encode(
                x=alt.X("week_label:N", sort=None, title=""),
                y=alt.Y("stake:Q", title="Stake ($)", scale=alt.Scale(zero=True)),
                tooltip=[alt.Tooltip("week_label:N", title="Week"),
                         alt.Tooltip("stake:Q", title="Stake", format="$.2f")],
            )
            .properties(height=H, width="container")
        )

        # Grid 2×2 (compacto)
        c1, c2 = st.columns(2)
        with c1: st.altair_chart(bank_chart, use_container_width=True)
        with c2: st.altair_chart(profit_chart, use_container_width=True)
        c3, c4 = st.columns(2)
        with c3: st.altair_chart(cum_chart, use_container_width=True)
        with c4: st.altair_chart(stake_chart, use_container_width=True)

        with st.expander("Ver tabla semanal (pnl_weekly)"):
            st.dataframe(pnl, use_container_width=True)

# ---------- Logos automáticos (URL) ----------
# Mapeo de slugs ESPN en casos especiales
ESPN_SLUG = {
    "ARI":"ari","ATL":"atl","BAL":"bal","BUF":"buf","CAR":"car","CHI":"chi","CIN":"cin","CLE":"cle",
    "DAL":"dal","DEN":"den","DET":"det","GB":"gb","HOU":"hou","IND":"ind","JAX":"jax","KC":"kc",
    "LA":"lar","LAR":"lar","LAC":"lac","LV":"lv","MIA":"mia","MIN":"min","NE":"ne","NO":"no",
    "NYG":"nyg","NYJ":"nyj","PHI":"phi","PIT":"pit","SEA":"sea","SF":"sf","TB":"tb","TEN":"ten",
    "WAS":"wsh","WSH":"wsh"
}

@st.cache_data(ttl=60*60*24)
def get_logo_url(abbr: str) -> str | None:
    """Intenta NFL CDN y ESPN. Devuelve la primera URL que responda 200."""
    if not abbr:
        return None
    a = abbr.upper().strip()

    # 1) Si existe archivo local cacheado por workflow
    local = LOGOS_DIR / f"{a}.png"
    if local.exists():
        return local.as_posix()

    candidates = []

    # 2) NFL CDN (muchas veces funciona con la ABBR tal cual)
    candidates.append(f"https://static.www.nfl.com/t_q-best/league/api/clubs/logos/{a}")

    # 3) ESPN (scoreboard set) con slug especial (wsh, lar, lac, jax, etc.)
    slug = ESPN_SLUG.get(a, a.lower())
    candidates.append(f"https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/{slug}.png")

    # 4) ESPN genérico (fallback)
    candidates.append(f"https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png")

    for url in candidates:
        try:
            r = requests.head(url, timeout=2, allow_redirects=True)
            if r.status_code == 200:
                return url
        except Exception:
            continue
    return None

# ---------- BETS (tarjetas compactas) ----------
def bet_card(row: pd.Series):
    team = str(row.get("team", "")).upper()
    opp  = str(row.get("opponent", "")).upper()
    side = str(row.get("side", "")).title() if pd.notna(row.get("side")) else ""
    wl_profit = pd.to_numeric(row.get("profit"), errors="coerce")
    stake = pd.to_numeric(row.get("stake"), errors="coerce")
    dec = pd.to_numeric(row.get("decimal_odds"), errors="coerce")
    ml  = pd.to_numeric(row.get("ml"), errors="coerce")

    # Moneyline mostrado siempre que se pueda
    if pd.isna(ml) and pd.notna(dec):
        ml = decimal_to_american(dec)

    if pd.isna(wl_profit):
        status_color = "#888888"
        status_label = "OPEN"
    elif wl_profit > 0:
        status_color = "#1FA37C"  # verde
        status_label = "WIN"
    elif wl_profit < 0:
        status_color = "#D64545"  # rojo
        status_label = "LOSS"
    else:
        status_color = "#8884D8"  # neutro/push
        status_label = "PUSH"

    wk = str(row.get("week_label", row.get("week", "")))
    date_txt = str(row.get("schedule_date", ""))[:10] if pd.notna(row.get("schedule_date")) else ""

    # Scores si existen
    score_txt = ""
    sh = row.get("score_home", None)
    sa = row.get("score_away", None)
    htm = str(row.get("home_team", "")) or (team if side.lower()=="home" else "")
    atm = str(row.get("away_team", "")) or (opp  if side.lower()=="away" else "")
    if pd.notna(sh) and pd.notna(sa) and (str(htm) or str(atm)):
        try:
            score_txt = f"{htm or 'HOME'} {int(sh)} — {atm or 'AWAY'} {int(sa)}"
        except Exception:
            score_txt = ""

    # Logos (URL auto) o pastillas
    team_logo = get_logo_url(team)
    opp_logo  = get_logo_url(opp)

    # Textos moneyline/stake/profit
    ml_txt    = f"{ml:+.0f}" if pd.notna(ml) else "—"
    stake_txt = f"${stake:,.2f}" if pd.notna(stake) else "—"
    prof_txt  = f"${wl_profit:,.2f}" if pd.notna(wl_profit) else "—"

    # Card compacta (menos padding, tipografía chica, imagen pequeña)
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

      {"<div style='margin-top:6px;opacity:.80;'>" + score_txt + "</div>" if score_txt else ""}

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
            if "schedule_date" in view.columns:
                sort_cols.append("schedule_date")
            view = view.sort_values(sort_cols).drop(columns="__order")

        # Tarjetas compactas en grilla 4 por fila
        for wk, grp in view.groupby("week_label", sort=False):
            st.markdown(f"### {wk}")
            cards = list(grp.itertuples(index=False))
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

        with st.expander("Ver tabla completa"):
            cols = [c for c in [
                "season","week","week_label","schedule_date","side","team","opponent",
                "home_team","away_team","score_home","score_away",
                "ml","decimal_odds","stake","profit","status","result","won"
            ] if c in view.columns]
            st.dataframe(view[cols] if cols else view, use_container_width=True)
