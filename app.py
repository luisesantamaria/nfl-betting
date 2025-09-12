import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="NFL EV Betting - Dashboard", layout="wide")

# =========================
# Config de temporadas (ajústalo cuando quieras)
# =========================
SEASON_RULES = {
    2024: {
        "season_start": "2024-09-05",
        "season_end":   "2025-02-12",   # ~post Super Bowl
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
# Rutas de datos
# =========================
def resolve_dir(*parts) -> Path:
    """Intento varias raíces (compatible con Streamlit Cloud)"""
    candidates = [
        Path(__file__).resolve().parent.joinpath(*parts),      # app.py en raíz
        Path.cwd().joinpath(*parts),                           # CWD del runtime
        Path(__file__).resolve().parents[1].joinpath(*parts),  # por si lo mueves a /app
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

PORTFOLIO_DIR = resolve_dir("data", "processed", "portfolio")
ARCHIVE_DIR   = resolve_dir("data", "archive")
BETSWEEK_DIR  = resolve_dir("data", "processed", "bets")  # opcional: bets de esta semana

# =========================
# Helpers
# =========================
ORDER_LABELS = [f"Week {i}" for i in range(1, 19)] + ["Wild Card", "Divisional", "Conference", "Super Bowl"]
ORDER_INDEX  = {lab: i for i, lab in enumerate(ORDER_LABELS)}

def list_available_seasons():
    seasons = []
    for f in sorted(PORTFOLIO_DIR.glob("pnl_weekly_*.csv")):
        try:
            seasons.append(int(f.stem.split("_")[-1]))
        except:
            pass
    # añade seasons del config aunque aún no tengan CSV
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
    # ordenar SIEMPRE por etiqueta
    df["week_label"] = df["week_label"].astype(str)
    df["__order"] = df["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    df = df.sort_values("__order").drop(columns="__order")
    df["week_label"] = pd.Categorical(df["week_label"], categories=ORDER_LABELS, ordered=True)
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
    """
    Busca bets de esta semana para la season actual (si decides exportarlas).
    Rutas soportadas:
      - data/processed/bets/season=YYYY/this_week.csv
      - data/processed/bets/this_week.csv
    """
    candidates = [
        BETSWEEK_DIR / f"season={year}" / "this_week.csv",
        BETSWEEK_DIR / "this_week.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            # normaliza numéricos más comunes
            for col in ("decimal_odds", "ml", "stake", "model_prob", "edge", "ev"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
    return pd.DataFrame()

def season_stage(year: int, pnl_df: pd.DataFrame) -> str:
    """
    'locked'     -> antes de activar
    'preseason'  -> activado pero antes del kickoff
    'in_season'  -> entre start y end
    'ended'      -> después del end (o si el CSV llegó a 'Super Bowl')
    """
    rule = SEASON_RULES.get(year, {})
    now = datetime.now(timezone.utc)

    # si el CSV ya tiene 'Super Bowl', damos por terminada
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

# =========================
# UI
# =========================
st.title("NFL EV Betting — Dashboard")

seasons = list_available_seasons()
if not seasons:
    st.warning("No seasons found.")
    st.stop()

# Selector de temporada (una sola UI con 3 pestañas)
season = st.selectbox("Season", options=seasons, index=seasons.index(max(seasons)))

pnl = load_pnl_weekly(season)
bets_all = load_ledger(season)
stage = season_stage(season, pnl)

# Header de estado
status_map = {
    "locked": "Locked",
    "preseason": "Preseason",
    "in_season": "In Season",
    "ended": "Season Ended",
    "unknown": "Unknown",
}
st.caption(f"Status: **{status_map.get(stage, 'Unknown')}**")

# Pestañas: Overview | Portfolio | Bets
tab_overview, tab_portfolio, tab_bets = st.tabs(["Overview", "Portfolio", "Bets"])

# =========================
# OVERVIEW
# =========================
with tab_overview:
    # Sección: Bets de esta semana (arriba)
    st.subheader("This Week’s Bets")

    if stage == "ended":
        st.info("✅ Season Ended — no upcoming bets.")
    elif stage in ("locked", "preseason"):
        rule = SEASON_RULES.get(season, {})
        start = rule.get("season_start", "?")
        st.info(f"🟡 No bets yet. Season activates before kickoff ({start}); las apuestas aparecen desde **Week {rule.get('bets_open_week', 3)}**.")
    else:
        # temporada en curso
        bets_week = load_bets_this_week(season)
        if bets_week.empty:
            st.caption("No hay archivo de 'this_week.csv' todavía. Cuando lo exportes, aparecerá aquí automáticamente.")
        else:
            # tabla compacta típica
            cols = [c for c in [
                "week","week_label","schedule_date","side","team","opponent",
                "decimal_odds","ml","model_prob","edge","ev","stake"
            ] if c in bets_week.columns]
            st.dataframe(bets_week[cols] if cols else bets_week, use_container_width=True)

    st.divider()

    # Sección: gráfica simple del portafolio actual (solo bankroll)
    st.subheader("Portfolio (Bankroll)")
    if pnl.empty:
        st.caption("No hay `pnl_weekly` para esta temporada.")
    else:
        # KPIs básicos para contexto
        profits = pd.to_numeric(pnl.get("profit"), errors="coerce").fillna(0.0)
        banks   = pd.to_numeric(pnl.get("bankroll"), errors="coerce")
        initial_bankroll = float(banks.iloc[0] - profits.iloc[0])
        final_bankroll   = float(banks.iloc[-1])
        k1, k2 = st.columns(2)
        k1.metric("Initial", f"${initial_bankroll:,.2f}")
        k2.metric("Final",   f"${final_bankroll:,.2f}", f"{(final_bankroll-initial_bankroll):,.2f}")

        bank = pnl[["week_label", "bankroll"]].dropna()
        ymin, ymax = float(bank["bankroll"].min()), float(bank["bankroll"].max())
        pad = max(10.0, (ymax - ymin) * 0.06)

        chart = (
            alt.Chart(bank)
            .mark_line(point=True)
            .encode(
                x=alt.X("week_label:N", sort=list(ORDER_LABELS), title=""),
                y=alt.Y("bankroll:Q",
                        title="Bankroll ($)",
                        scale=alt.Scale(domain=[ymin - pad, ymax + pad], zero=False, nice=False)),
                tooltip=[alt.Tooltip("week_label:N", title="Week"),
                         alt.Tooltip("bankroll:Q", title="Bankroll", format="$.2f")],
            )
            .properties(height=320, width="container")
        )
        st.altair_chart(chart, use_container_width=True)

# =========================
# PORTFOLIO (dos gráficas + métricas)
# =========================
with tab_portfolio:
    st.subheader("Portfolio")
    if pnl.empty:
        st.caption("No hay `pnl_weekly` para esta temporada.")
    else:
        profits = pd.to_numeric(pnl.get("profit"), errors="coerce").fillna(0.0)
        stakes  = pd.to_numeric(pnl.get("stake"),  errors="coerce").fillna(0.0)
        banks   = pd.to_numeric(pnl.get("bankroll"), errors="coerce")

        first_bankroll   = float(banks.iloc[0])
        first_profit     = float(profits.iloc[0])
        initial_bankroll = float(first_bankroll - first_profit)
        final_bankroll   = float(banks.iloc[-1])
        total_profit     = float(profits.sum())
        total_stake      = float(stakes.sum())
        yield_pct        = (total_profit / total_stake * 100.0) if total_stake > 0 else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Initial", f"${initial_bankroll:,.2f}")
        m2.metric("Final",   f"${final_bankroll:,.2f}", f"{(final_bankroll-initial_bankroll):,.2f}")
        m3.metric("Total Profit", f"${total_profit:,.2f}")
        m4.metric("Yield",        f"{yield_pct:.2f}%")

        # Bankroll (sin baseline 0)
        bank = pnl[["week_label", "bankroll"]].dropna()
        ymin, ymax = float(bank["bankroll"].min()), float(bank["bankroll"].max())
        pad = max(10.0, (ymax - ymin) * 0.06)

        bank_chart = (
            alt.Chart(bank)
            .mark_line(point=True)
            .encode(
                x=alt.X("week_label:N", sort=list(ORDER_LABELS), title=""),
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

        # Weekly Profit (barras)
        prof_df = pd.DataFrame({
            "week_label": pnl["week_label"],
            "profit": profits,
            "stake": stakes,
        })
        profit_chart = (
            alt.Chart(prof_df)
            .mark_bar()
            .encode(
                x=alt.X("week_label:N", sort=list(ORDER_LABELS), title=""),
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

        with st.expander("Ver tabla semanal"):
            st.dataframe(pnl, use_container_width=True)

# =========================
# BETS (solo tabla, sin filtros)
# =========================
with tab_bets:
    st.subheader("Bets")
    if bets_all.empty:
        st.caption("No hay archivo de bets para esta temporada.")
    else:
        # Orden por week_label y fecha si existe
        view = bets_all.copy()
        if "week_label" in view.columns:
            view["week_label"] = view["week_label"].astype(str)
            view["__order"] = view["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
            view = view.sort_values(["__order"] + ([ "schedule_date"] if "schedule_date" in view.columns else [])).drop(columns="__order")
        cols = [c for c in [
            "season","week","week_label","schedule_date","side","team","opponent",
            "decimal_odds","ml","stake","profit","status","result","won"
        ] if c in view.columns]
        st.dataframe(view[cols] if cols else view, use_container_width=True)
