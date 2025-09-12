import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="NFL EV Betting - Portfolio", layout="wide")

# Ruta a los CSV semanales
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "portfolio"

@st.cache_data
def list_season_files():
    if not DATA_DIR.exists():
        return {}
    files = sorted(DATA_DIR.glob("pnl_weekly_*.csv"))
    out = {}
    for f in files:
        # pnl_weekly_2024.csv -> 2024
        try:
            season = int(f.stem.split("_")[-1])
            out[season] = f
        except Exception:
            pass
    return out

@st.cache_data
def load_weekly_pnl(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Orden y tipos
    if "week" in df.columns:
        df["week"] = pd.to_numeric(df["week"], errors="coerce")
    if "profit" in df.columns:
        df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0.0)
    if "stake" in df.columns:
        df["stake"] = pd.to_numeric(df["stake"], errors="coerce").fillna(0.0)
    if "bankroll" in df.columns:
        df["bankroll"] = pd.to_numeric(df["bankroll"], errors="coerce")
    return df

st.title("NFL EV Betting — Portfolio")

season_files = list_season_files()
if not season_files:
    st.warning("No encontré archivos `pnl_weekly_*.csv` en `data/processed/portfolio/`.")
    st.stop()

default_season = max(season_files.keys())
season = st.selectbox("Season", options=sorted(season_files.keys()), index=sorted(season_files.keys()).index(default_season))

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
