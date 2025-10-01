# scripts/build_pnl_live.py
import os, sys, math
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# ---------- Config ----------
BETS_CANDIDATES = [
    os.getenv("BPNL_BETS_CSV"),
    "data/live/bets.csv",
    "data/bets.csv",
]
ODDS_CANDIDATES = [
    os.getenv("BPNL_ODDS_CSV"),
    "data/live/odds.csv",
    "data/odds.csv",
]

bets_path = next((p for p in BETS_CANDIDATES if p and os.path.exists(p)), None)
odds_path = next((p for p in ODDS_CANDIDATES if p and os.path.exists(p)), None)

if not bets_path:
    print("ERROR: bets.csv no encontrado (busqué en data/live/ y data/).", file=sys.stderr)
    sys.exit(1)
if not odds_path:
    print("ERROR: odds.csv no encontrado (busqué en data/live/ y data/).", file=sys.stderr)
    sys.exit(1)

BANKROLL_INITIAL = float(os.getenv("BANKROLL_INITIAL", "1000"))
OUT_DIR = "data/live"
os.makedirs(OUT_DIR, exist_ok=True)
PNL_PATH = os.path.join(OUT_DIR, "pnl.csv")

# ---------- Helpers ----------
TEAM_FIX = {
    "STL":"LA","LAR":"LA","SD":"LAC","SDG":"LAC","OAK":"LV","LVR":"LV","WSH":"WAS","JAC":"JAX",
    "GNB":"GB","KAN":"KC","NWE":"NE","NOR":"NO","SFO":"SF","TAM":"TB"
}
def _norm_team(x: str) -> str:
    s = str(x).upper().strip()
    return TEAM_FIX.get(s, s)

def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def _week_label_from_num(n: int) -> str:
    if pd.isna(n): return pd.NA
    try: n = int(n)
    except Exception: return pd.NA
    if 1 <= n <= 18: return f"Week {n}"
    return {
        19: "Wild Card",
        20: "Divisional",
        21: "Conference",
        22: "Super Bowl",
    }.get(n, f"Week {n}")

def _normalize_week_order(df: pd.DataFrame) -> pd.Series:
    # Preferir week > week_label > week_order
    if "week" in df.columns:
        wk = _safe_num(df["week"])
        if wk.notna().any():
            return wk.fillna(999).astype(int)
    if "week_label" in df.columns:
        wk = df["week_label"].astype(str).str.extract(r"(\d+)")[0]
        wk = _safe_num(wk)
        if wk.notna().any():
            return wk.fillna(999).astype(int)
    if "week_order" in df.columns:
        wk = _safe_num(df["week_order"])
        return wk.fillna(999).astype(int)
    return pd.Series([999]*len(df), index=df.index, dtype=int)

# ---------- Carga ----------
bets = pd.read_csv(bets_path, low_memory=False)
odds = pd.read_csv(odds_path, low_memory=False)

# Tipos mínimos y normalizaciones
for c in ("season","week"):
    if c in bets.columns:
        bets[c] = _safe_num(bets[c]).astype("Int64")
if "week_label" not in bets.columns or bets["week_label"].isna().any():
    bets["week_label"] = bets["week"].apply(_week_label_from_num)
bets["week_order"] = _normalize_week_order(bets)
bets["schedule_date"] = pd.to_datetime(bets.get("schedule_date"), errors="coerce", utc=True)
bets["stake"] = _safe_num(bets.get("stake")).fillna(0.0)
bets["profit"] = _safe_num(bets.get("profit"))

# Determinar si la bet está liquidada
settled = bets["profit"].notna()
bets["settled"] = settled

# ---------- Agregado semanal SOLO de semanas finalizadas ----------
group_cols = ["season","week","week_label","week_order"]

# Una semana está finalizada si TODAS sus bets están liquidadas (profit notna)
week_done = (
    bets.groupby(group_cols, dropna=False)["settled"]
        .transform("all")
)

final_weeks = bets.loc[week_done].copy()

if final_weeks.empty:
    # Si no hay semanas finalizadas, escribir un CSV vacío con headers útiles
    empty = pd.DataFrame(columns=["season","week","week_label","profit","stake","yield_%","bankroll","status_week","updated_at_utc"])
    empty.to_csv(PNL_PATH, index=False)
    print(f"PNL escrito en {PNL_PATH} (sin semanas finalizadas aún)")
    sys.exit(0)

agg = (
    final_weeks.groupby(group_cols, dropna=False)
               .agg(stake=("stake","sum"),
                    profit=("profit","sum"),
                    n_bets=("stake","size"))
               .reset_index()
)

def safe_yield(row):
    s = row["stake"]
    return (row["profit"]/s*100.0) if s and not math.isclose(s, 0.0) else 0.0

agg["yield_%"] = agg.apply(safe_yield, axis=1)
agg["status_week"] = "final"

# Orden cronológico
agg = agg.sort_values(["season","week_order","week","week_label"]).reset_index(drop=True)

# Bankroll acumulado desde BANKROLL_INITIAL
running = BANKROLL_INITIAL
bankroll = []
for _, r in agg.iterrows():
    running += r["profit"]
    bankroll.append(running)
agg["bankroll"] = bankroll

# Marca de tiempo
now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
agg["updated_at_utc"] = now_iso

# Salida
out_cols = ["season","week","week_label","profit","stake","yield_%","bankroll","status_week","updated_at_utc"]
agg[out_cols].to_csv(PNL_PATH, index=False)

print(f"PNL escrito en {PNL_PATH}")
