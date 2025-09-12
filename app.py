import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="NFL EV Betting - Portfolio", layout="wide")

# --- Resolver ruta de datos de forma robusta ---
def resolve_data_dir():
    candidates = [
        Path(__file__).resolve().parent / "data" / "processed" / "portfolio",  # app.py en raíz
        Path.cwd() / "data" / "processed" / "portfolio",                       # cwd del runtime
        Path(__file__).resolve().parents[1] / "data" / "processed" / "portfolio",  # por si lo mueves a app/
    ]
    for p in candidates:
        files = list((p).glob("pnl_weekly_*.csv"))
        if p.exists() and files:
            return p
    # si ninguna tiene archivos, devolvemos la más probable (raíz) para mensaje de error
    return candidates[0]

DATA_DIR = resolve_data_dir()

@st.cache_data
def list_season_files():
    files = sorted(DATA_DIR.glob("pnl_weekly_*.csv"))
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
    for col in ("week","profit","stake","bankroll"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

st.title("NFL EV Betting — Portfolio")

season_files = list_season_files()
if not season_files:
    st.error("No encontré archivos `pnl_weekly_*.csv` en tu repo.")
    st.caption(f"Busqué en: `{DATA_DIR}`")
    st.stop()

default_season = max(season_files.keys())
season = st.selectbox(
    "Season",
    options=sorted(season_files.keys()),
    index=sorted(season_files.keys()).index(default_season),
)

df = load_weekly_pnl(season_files[season])

col1, col2, col3, col4 = st.columns(4)
initial_bankroll = float(df["bankroll"].iloc[0] - df["profit"].iloc[0]) if len(df) else 1000.0
final_bankroll = float(df["bankroll"].iloc[-1]) if len(df) else initial_bankroll
total_profit = float(df["profit"].sum()) if len(df) else 0.0
total_stake = float(df["stake"].sum()) if len(df) else 0.0
yield_pct = (total_profit / total_stake * 100.0) if total_stake > 0 else 0.0

col1.metric("Initial", f"${initial_bankroll:,.2f}")
col2.metric("Final", f"${final_bankroll:,.2f}", f"{final_bankroll-initial_bankroll:,.2f}")
col3.metric("Total Profit", f"${total_profit:,.2f}")
col4.metric("Yield", f"{yield_pct:.2f}%")

st.subheader(f"Bankroll — {season}")
st.line_chart(df.set_index("week_label")["bankroll"])

st.subheader("Weekly Profit")
st.bar_chart(df.set_index("week_label")["profit"])

with st.expander("Ver tabla semanal"):
    st.dataframe(df, use_container_width=True)
