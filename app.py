import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="NFL EV Betting - Portfolio", layout="wide")

# ---------- Localización robusta de la carpeta de datos ----------
def resolve_data_dir():
    candidates = [
        Path(__file__).resolve().parent / "data" / "processed" / "portfolio",          # app.py en raíz
        Path.cwd() / "data" / "processed" / "portfolio",                               # CWD del runtime
        Path(__file__).resolve().parents[1] / "data" / "processed" / "portfolio",      # por si lo mueves a app/
    ]
    for p in candidates:
        if p.exists() and any(p.glob("pnl_weekly_*.csv")):
            return p
    return candidates[0]

DATA_DIR = resolve_data_dir()

# ---------- Helpers ----------
@st.cache_data
def list_season_files(data_dir: Path):
    files = sorted(data_dir.glob("pnl_weekly_*.csv"))
    out = {}
    for f in files:
        try:
            season = int(f.stem.split("_")[-1])  # pnl_weekly_2024.csv -> 2024
            out[season] = f
        except Exception:
            pass
    return out

@st.cache_data
def load_weekly_pnl(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("week", "profit", "stake", "bankroll"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# Orden correcto para el eje X
ORDER_LABELS = [f"Week {i}" for i in range(1, 19)] + ["Wild Card", "Divisional", "Conference", "Super Bowl"]
ORDER_INDEX  = {lab: i for i, lab in enumerate(ORDER_LABELS)}

# ---------- UI ----------
st.title("NFL EV Betting — Portfolio")

season_files = list_season_files(DATA_DIR)
if not season_files:
    st.error("No encontré archivos `pnl_weekly_*.csv`.")
    st.caption(f"Ruta buscada: `{DATA_DIR}`")
    st.stop()

default_season = max(season_files.keys())
season = st.selectbox(
    "Season",
    options=sorted(season_files.keys()),
    index=sorted(season_files.keys()).index(default_season),
)

df = load_weekly_pnl(season_files[season])

# ---------- Ordenar por semana / etiqueta ----------
if "week" in df.columns and df["week"].notna().any():
    df = df.sort_values("week")
else:
    df["week_label"] = df["week_label"].astype(str)
    df["__order"] = df["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    df = df.sort_values("__order").drop(columns="__order")
df["week_label"] = pd.Categorical(df["week_label"], categories=ORDER_LABELS, ordered=True)

# ---------- Métricas ----------
if len(df):
    initial_bankroll = float(df["bankroll"].iloc[0] - df["profit"].iloc[0])  # aprox. inicial
    final_bankroll   = float(df["bankroll"].iloc[-1])
    total_profit     = float(df["profit"].sum())
    total_stake      = float(df["stake"].sum())
    yield_pct        = (total_profit / total_stake * 100.0) if total_stake > 0 else 0.0
else:
    initial_bankroll = final_bankroll = 1000.0
    total_profit = total_stake = yield_pct = 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Initial", f"${initial_bankroll:,.2f}")
c2.metric("Final",   f"${final_bankroll:,.2f}", f"{final_bankroll-initial_bankroll:,.2f}")
c3.metric("Total Profit", f"${total_profit:,.2f}")
c4.metric("Yield",        f"{yield_pct:.2f}%")

# ---------- Gráficas ----------
st.subheader(f"Bankroll — {season}")
st.line_chart(df.set_index("week_label")["bankroll"])

st.subheader("Weekly Profit")
st.bar_chart(df.set_index("week_label")["profit"])

with st.expander("Ver tabla semanal"):
    st.dataframe(df, use_container_width=True)
