import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="NFL EV Betting - Dashboard", layout="wide")

# =========================
# Config de temporadas (ajústalo cuando tengas fechas reales)
# =========================
SEASON_RULES = {
    # Ejemplo realista 2024 (ajusta si quieres)
    2024: {
        "season_start": "2024-09-05",   # Kickoff (aprox)
        "activate_days_before": 7,      # mostrar "Preseason/Coming soon" 7 días antes
        "bets_open_week": 3,            # bets desde Week 3
    },
    # Prepara 2025 (puedes ajustar cuando haya fechas oficiales)
    2025: {
        "season_start": "2025-09-04",
        "activate_days_before": 7,
        "bets_open_week": 3,
    },
}

# =========================
# Rutas de datos
# =========================
def resolve_dir(*parts) -> Path:
    """Intenta varias raíces (compatible con Streamlit Cloud)"""
    candidates = [
        Path(__file__).resolve().parent.joinpath(*parts),          # app.py en raíz
        Path.cwd().joinpath(*parts),                               # CWD del runtime
        Path(__file__).resolve().parents[1].joinpath(*parts),      # por si lo metes en /app
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

PORTFOLIO_DIR = resolve_dir("data", "processed", "portfolio")
ARCHIVE_DIR   = resolve_dir("data", "archive")

# =========================
# Helpers comunes
# =========================
ORDER_LABELS = [f"Week {i}" for i in range(1, 19)] + ["Wild Card", "Divisional", "Conference", "Super Bowl"]
ORDER_INDEX  = {lab: i for i, lab in enumerate(ORDER_LABELS)}

def list_available_seasons():
    seasons = []
    for f in sorted(PORTFOLIO_DIR.glob("pnl_weekly_*.csv")):
        try:
            seasons.append(int(f.stem.split("_")[-1]))
        except:  # pragma: no cover
            pass
    # Asegura también seasons del config, aunque todavía no existan CSVs
    for y in SEASON_RULES:
        if y not in seasons:
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
    # ordenamos SIEMPRE por etiqueta
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

def season_stage(year: int) -> str:
    """Devuelve: 'locked', 'preseason', 'in_season' según fechas del config."""
    rule = SEASON_RULES.get(year)
    if not rule:
        return "unknown"
    start = datetime.fromisoformat(rule["season_start"]).replace(tzinfo=timezone.utc)
    now   = datetime.now(timezone.utc)
    activate_from = start - timedelta(days=int(rule.get("activate_days_before", 0)))
    if now < activate_from:
        return "locked"
    if now < start:
        return "preseason"
    return "in_season"

# =========================
# UI principal
# =========================
st.title("NFL EV Betting — Dashboard")

seasons = list_available_seasons()
if not seasons:
    st.warning("No seasons found.")
    st.stop()

# Tabs por temporada
season_tabs = st.tabs([str(y) for y in seasons])

for tab, year in zip(season_tabs, seasons):
    with tab:
        stage = season_stage(year)
        pnl = load_pnl_weekly(year)

        # Header contextual
        subcol1, subcol2 = st.columns([3, 1])
        with subcol1:
            st.subheader(f"Season {year}")
        with subcol2:
            st.caption(f"Stage: **{stage.replace('_',' ').title()}**")

        # Mensajes según etapa
        if stage == "locked":
            start = SEASON_RULES.get(year, {}).get("season_start", "?")
            st.info(f"🚧 **Season {year}** aún no está activa. Se habilita automáticamente ~{SEASON_RULES[year]['activate_days_before']} días antes del kickoff ({start}).")
            continue
        elif stage == "preseason":
            st.warning("🟡 Preseason: preparando datos. Las apuestas abren a partir de **Week 3**.")

        # Sub-tabs dentro de cada season
        sub1, sub2 = st.tabs(["Portfolio", "Bets"])

        # ---------- PORTFOLIO ----------
        with sub1:
            if pnl.empty:
                st.info("No hay `pnl_weekly` para esta temporada todavía.")
            else:
                # KPIs (reconstruye initial desde primera fila)
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

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Initial", f"${initial_bankroll:,.2f}")
                k2.metric("Final",   f"${final_bankroll:,.2f}", f"{(final_bankroll-initial_bankroll):,.2f}")
                k3.metric("Total Profit", f"${total_profit:,.2f}")
                k4.metric("Yield",        f"{yield_pct:.2f}%")

                # Gráfica de bankroll (sin baseline 0)
                bank = pnl[["week_label", "bankroll"]].dropna()
                ymin = float(bank["bankroll"].min())
                ymax = float(bank["bankroll"].max())
                pad  = max(10.0, (ymax - ymin) * 0.06)

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

                # Profit semanal
                prof = pd.DataFrame({
                    "week_label": pnl["week_label"],
                    "profit": profits,
                    "stake": stakes,
                })
                profit_chart = (
                    alt.Chart(prof)
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

        # ---------- BETS (antes “Ledger”) ----------
        with sub2:
            bets = load_ledger(year)
            # Regla de apertura de bets (Weeks 1–2 se muestra mensaje)
            rule = SEASON_RULES.get(year, {})
            open_wk = int(rule.get("bets_open_week", 3))
            # Si ya hay pnl, detecta la semana máxima; si no, solo aplica regla básica
            has_weeks = "week_label" in pnl.columns and len(pnl) > 0
            max_wk_label = str(pnl["week_label"].max()) if has_weeks else None

            if stage == "in_season" and has_weeks:
                # Si estamos antes de Week 3, mostrar aviso
                # (El mapping a número real es aproximado, según ORDER_INDEX)
                current_index = ORDER_INDEX.get(max_wk_label, 0)
                need_index    = ORDER_INDEX.get(f"Week {open_wk}", 0)
                if current_index < need_index:
                    st.info(f"📈 Weeks 1–2: acumulando features. **Bets** se habilita a partir de **Week {open_wk}**.")
            elif stage != "in_season":
                st.info("Las apuestas se mostrarán una vez iniciada la temporada.")

            if bets.empty:
                st.caption("No hay archivo de bets para esta temporada todavía.")
            else:
                st.subheader("Bets")
                # Filtros rápidos
                colA, colB = st.columns(2)
                teams = sorted(pd.unique(pd.concat([
                    bets.get("team", pd.Series(dtype=str)).astype(str),
                    bets.get("opponent", pd.Series(dtype=str)).astype(str)
                ]).dropna()))
                sel_teams = colA.multiselect("Team (either side)", options=teams, default=teams[:])
                sides = ["home", "away"] if "side" in bets.columns else []
                sel_sides = colB.multiselect("Side", options=sides, default=sides)

                mask = pd.Series(True, index=bets.index)
                if sel_teams:
                    m_team = bets.get("team", pd.Series(index=bets.index, dtype=str)).astype(str).isin(sel_teams)
                    m_opp  = bets.get("opponent", pd.Series(index=bets.index, dtype=str)).astype(str).isin(sel_teams)
                    mask &= (m_team | m_opp)
                if sel_sides and "side" in bets.columns:
                    mask &= bets["side"].astype(str).str.lower().isin(sel_sides)

                view = bets[mask].copy()
                # Algunas columnas útiles si existen
                show_cols = [c for c in [
                    "season","week","week_label","schedule_date","side","team","opponent",
                    "decimal_odds","ml","stake","profit","status","result","won"
                ] if c in view.columns]
                st.dataframe(view[show_cols] if show_cols else view, use_container_width=True)
