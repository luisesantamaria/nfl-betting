import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path
from datetime import datetime, timedelta, timezone

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
    # Ordenar SIEMPRE por etiqueta con índice numérico
    if "week_label" in df.columns:
        df = add_week_order(df)
    else:
        # fallback mínimo
        df["week_label"] = "Week 999"
    return df

def american_to_decimal(v):
    try:
        v = float(v)
    except Exception:
        return float("nan")
    return 1 + (100/abs(v) if v < 0 else v/100)

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
            # Asegura week_label si viene sólo 'week'
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
    """locked | preseason | in_season | ended"""
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

# =========================
# UI
# =========================
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

# =========================
# OVERVIEW (sin repetir Bankroll, y con orden X correcto)
# =========================
with tab_overview:
    # Sección de bets de esta semana: sólo si hay archivo y estamos in_season
    if stage == "in_season":
        bets_week = load_bets_this_week(season)
        if not bets_week.empty:
            st.subheader("This Week’s Bets")
            cols = [c for c in [
                "week","week_label","schedule_date","side","team","opponent",
                "decimal_odds","ml","model_prob","edge","ev","stake"
            ] if c in bets_week.columns]
            st.dataframe(bets_week[cols] if cols else bets_week, use_container_width=True)
            st.divider()

    # KPIs + 2 mini visuales diferentes a Portfolio
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

        # (A) Cumulative Profit — sparkline (área+línea), con ORDEN CORRECTO
        cum_df = pd.DataFrame({
            "week_label": pnl["week_label"],
            "cum_profit": profits.cumsum()
        })
        cum_df = add_week_order(cum_df)
        y2min, y2max = float(cum_df["cum_profit"].min()), float(cum_df["cum_profit"].max())
        pad2 = max(5.0, (y2max - y2min) * 0.06)

        spark = (
            alt.Chart(cum_df)
            .mark_area(opacity=0.25)
            .encode(
                x=alt.X("week_label:N", sort=None, title=""),  # sort=None para respetar orden del DF
                y=alt.Y("cum_profit:Q",
                        title="Cumulative Profit ($)",
                        scale=alt.Scale(domain=[y2min - pad2, y2max + pad2], zero=False, nice=False)),
                tooltip=[alt.Tooltip("week_label:N", title="Week"),
                         alt.Tooltip("cum_profit:Q", title="Cum Profit", format="$.2f")],
            )
            .properties(height=220, width="container")
        ) + (
            alt.Chart(cum_df)
            .mark_line()
            .encode(x=alt.X("week_label:N", sort=None, title=""), y="cum_profit:Q")
        )

        # (B) Last 8 Weeks Profit — barras pequeñas (orden correcto)
        last8 = pd.DataFrame({
            "week_label": pnl["week_label"],
            "profit": profits,
        })
        last8 = add_week_order(last8)
        if len(last8) > 8:
            last8 = last8.tail(8)  # últimas 8 semanas en orden correcto

        mini_bars = (
            alt.Chart(last8)
            .mark_bar()
            .encode(
                x=alt.X("week_label:N", sort=None, title=""),
                y=alt.Y("profit:Q", title="Last 8 Weeks Profit ($)", scale=alt.Scale(zero=True)),
                tooltip=[alt.Tooltip("week_label:N", title="Week"),
                         alt.Tooltip("profit:Q", title="Profit", format="$.2f")],
            )
            .properties(height=220, width="container")
        )

        cA, cB = st.columns(2)
        with cA:
            st.altair_chart(spark, use_container_width=True)
        with cB:
            st.altair_chart(mini_bars, use_container_width=True)

# =========================
# PORTFOLIO (detalle con varias vistas)
# =========================
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

        # (1) Bankroll (detalle)
        bank = pnl[["week_label", "bankroll"]].dropna()
        bank = add_week_order(bank)
        ymin, ymax = float(bank["bankroll"].min()), float(bank["bankroll"].max())
        pad = max(10.0, (ymax - ymin) * 0.06)
        bank_chart = (
            alt.Chart(bank)
            .mark_line(point=True)
            .encode(
                x=alt.X("week_label:N", sort=None, title=""),
                y=alt.Y("bankroll:Q",
                        title="Bankroll ($)",
                        scale=alt.Scale(domain=[ymin - pad, ymax + pad], zero=False, nice=False)),
                tooltip=[alt.Tooltip("week_label:N", title="Week"),
                         alt.Tooltip("bankroll:Q", title="Bankroll", format="$.2f")],
            )
            .properties(height=320, width="container")
        )
        st.subheader("Bankroll")
        st.altair_chart(bank_chart, use_container_width=True)

        # (2) Weekly Profit
        prof_df = pd.DataFrame({"week_label": pnl["week_label"], "profit": profits, "stake": stakes})
        prof_df = add_week_order(prof_df)
        profit_chart = (
            alt.Chart(prof_df)
            .mark_bar()
            .encode(
                x=alt.X("week_label:N", sort=None, title=""),
                y=alt.Y("profit:Q", title="Profit ($)", scale=alt.Scale(zero=True)),
                tooltip=[
                    alt.Tooltip("week_label:N", title="Week"),
                    alt.Tooltip("profit:Q", title="Profit", format="$.2f"),
                    alt.Tooltip("stake:Q",  title="Stake",  format="$.2f"),
                ],
            )
            .properties(height=260, width="container")
        )
        st.subheader("Weekly Profit")
        st.altair_chart(profit_chart, use_container_width=True)

        # (3) Cumulative Profit (detalle)
        cum_df = pd.DataFrame({"week_label": pnl["week_label"], "cum_profit": profits.cumsum()})
        cum_df = add_week_order(cum_df)
        y2min, y2max = float(cum_df["cum_profit"].min()), float(cum_df["cum_profit"].max())
        pad2 = max(5.0, (y2max - y2min) * 0.06)
        cum_chart = (
            alt.Chart(cum_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("week_label:N", sort=None, title=""),
                y=alt.Y("cum_profit:Q",
                        title="Cumulative Profit ($)",
                        scale=alt.Scale(domain=[y2min - pad2, y2max + pad2], zero=False, nice=False)),
                tooltip=[alt.Tooltip("week_label:N", title="Week"),
                         alt.Tooltip("cum_profit:Q", title="Cum Profit", format="$.2f")],
            )
            .properties(height=260, width="container")
        )
        st.subheader("Cumulative Profit")
        st.altair_chart(cum_chart, use_container_width=True)

        # (4) Stake por semana
        stake_df = pd.DataFrame({"week_label": pnl["week_label"], "stake": stakes})
        stake_df = add_week_order(stake_df)
        stake_chart = (
            alt.Chart(stake_df)
            .mark_bar()
            .encode(
                x=alt.X("week_label:N", sort=None, title=""),
                y=alt.Y("stake:Q", title="Stake ($)", scale=alt.Scale(zero=True)),
                tooltip=[alt.Tooltip("week_label:N", title="Week"),
                         alt.Tooltip("stake:Q", title="Stake", format="$.2f")],
            )
            .properties(height=220, width="container")
        )
        st.subheader("Stake per Week")
        st.altair_chart(stake_chart, use_container_width=True)

        # (5) Win rate por semana (si hay ledger)
        ledger = load_ledger(season)
        if not ledger.empty and "profit" in ledger.columns:
            tmp = ledger.copy()
            tmp["won_flag"] = pd.to_numeric(tmp["profit"], errors="coerce").fillna(0.0) > 0
            if "week_label" not in tmp.columns:
                if "week" in tmp.columns:
                    def week_label_from_num(n):
                        try:
                            n = int(n)
                        except:
                            return "Week 999"
                        if 1 <= n <= 18: return f"Week {n}"
                        return {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(n, f"Week {n}")
                    tmp["week_label"] = tmp["week"].apply(week_label_from_num)
                else:
                    tmp["week_label"] = "Week 999"
            g = (tmp.groupby("week_label", as_index=False)
                     .agg(win_rate=("won_flag","mean"), n_bets=("won_flag","size")))
            g = add_week_order(g)
            g["win_rate_pct"] = g["win_rate"] * 100.0
            win_chart = (
                alt.Chart(g)
                .mark_bar()
                .encode(
                    x=alt.X("week_label:N", sort=None, title=""),
                    y=alt.Y("win_rate_pct:Q", title="Win rate (%)", scale=alt.Scale(zero=True)),
                    tooltip=[
                        alt.Tooltip("week_label:N", title="Week"),
                        alt.Tooltip("win_rate_pct:Q", title="Win rate", format=".1f"),
                        alt.Tooltip("n_bets:Q", title="# Bets"),
                    ],
                )
                .properties(height=220, width="container")
            )
            st.subheader("Win Rate per Week")
            st.altair_chart(win_chart, use_container_width=True)

        with st.expander("Ver tabla semanal (pnl_weekly)"):
            st.dataframe(pnl, use_container_width=True)

# =========================
# BETS (sólo tabla)
# =========================
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
        cols = [c for c in [
            "season","week","week_label","schedule_date","side","team","opponent",
            "decimal_odds","ml","stake","profit","status","result","won"
        ] if c in view.columns]
        st.dataframe(view[cols] if cols else view, use_container_width=True)
