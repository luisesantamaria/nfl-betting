# scripts/build_pnl_live.py
import os, sys, math, json
from datetime import datetime, timezone
import pandas as pd

# -------- Config --------
CANDIDATES = [
    os.getenv("BPNL_BETS_CSV"),
    "data/live/bets.csv",
    "data/bets.csv",
]
bets_path = next((p for p in CANDIDATES if p and os.path.exists(p)), None)
if not bets_path:
    print("ERROR: bets.csv no encontrado (busqué en data/live/ y data/).", file=sys.stderr)
    sys.exit(1)

BANKROLL_INITIAL = float(os.getenv("BANKROLL_INITIAL", "1000"))  # Week 3 = 1000
OUT_DIR = "data/live"
os.makedirs(OUT_DIR, exist_ok=True)
PNL_PATH = os.path.join(OUT_DIR, "pnl.csv")
ANCHOR_PATH = os.path.join(OUT_DIR, "bankroll_anchor.json")

# -------- Carga --------
df = pd.read_csv(bets_path)

needed = ["season","week","week_label","week_order","decimal_odds","stake"]
miss = [c for c in needed if c not in df.columns]
if miss:
    print(f"ERROR: Faltan columnas en bets.csv: {miss}", file=sys.stderr)
    sys.exit(1)

df["season"] = df["season"].astype(int)
df["week"] = df["week"].astype(int)
df["week_order"] = df["week_order"].astype(int)
df["stake"] = pd.to_numeric(df["stake"], errors="coerce").fillna(0.0)
df["schedule_date"] = pd.to_datetime(df.get("schedule_date"), errors="coerce", utc=True)

# -------- Inferir resultado --------
def infer_result(row):
    r = row.get("result", None)
    if isinstance(r, str) and r.strip().lower() in {"win","loss","push","void"}:
        return r.strip().lower()
    ts = row.get("team_score", None)
    oscore = row.get("opponent_score", None)
    if pd.notna(ts) and pd.notna(oscore):
        try:
            ts = float(ts); oscore = float(oscore)
        except:
            return None
        if math.isclose(ts, oscore):
            return "push"
        return "win" if ts > oscore else "loss"
    return None  # abierta

df["result_inferred"] = df.apply(infer_result, axis=1)
df["settled"] = df["result_inferred"].isin(["win","loss","push","void"])

def profit_calc(row):
    if not row["settled"]:
        return 0.0
    stake = float(row["stake"])
    odds = float(row["decimal_odds"])
    r = row["result_inferred"]
    if r == "win":
        return stake * (odds - 1.0)
    if r == "loss":
        return -stake
    # push/void = 0
    return 0.0

df["profit_calc"] = df.apply(profit_calc, axis=1)

# -------- Agregar por semana --------
group_cols = ["season","week","week_label","week_order"]
agg = (
    df.groupby(group_cols, dropna=False)
      .apply(lambda g: pd.Series({
          "stake": g.loc[g["settled"], "stake"].sum(),     # solo liquidadas
          "profit": g.loc[g["settled"], "profit_calc"].sum(),
          "n_bets": len(g),
          "n_settled": int(g["settled"].sum()),
          "max_kickoff": g["schedule_date"].max()
      }))
      .reset_index()
      .sort_values(["season","week_order","week"])
      .reset_index(drop=True)
)

def safe_yield(row):
    s = row["stake"]
    return (row["profit"]/s*100.0) if s and not math.isclose(s,0.0) else 0.0

agg["yield_%"] = agg.apply(safe_yield, axis=1)
agg["status_week"] = agg.apply(lambda r: "final" if r["n_settled"] == r["n_bets"] else "open", axis=1)

# Bankroll acumulado desde BANKROLL_INITIAL
running = BANKROLL_INITIAL
bankroll = []
for _, r in agg.iterrows():
    running += r["profit"]
    bankroll.append(running)
agg["bankroll"] = bankroll

now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
agg["updated_at_utc"] = now_iso

out_cols = ["season","week","week_label","profit","stake","yield_%","bankroll","status_week","updated_at_utc"]
agg[out_cols].to_csv(PNL_PATH, index=False)
print(f"PNL escrito en {PNL_PATH}")

# -------- Anchor (última semana 'final') --------
final_rows = agg[agg["status_week"] == "final"]
if not final_rows.empty:
    last_final = final_rows.iloc[-1]
    anchor = {
        "season": int(last_final["season"]),
        "week": int(last_final["week"]),
        "bankroll_final": float(last_final["bankroll"]),
        "updated_at": now_iso,
    }
else:
    # Si no hay semanas finalizadas aún, ancla = inicial
    anchor = {
        "season": int(df["season"].min()) if len(df) else 2025,
        "week": int(df["week"].min()) if len(df) else 3,
        "bankroll_final": float(BANKROLL_INITIAL),
        "updated_at": now_iso,
    }

with open(ANCHOR_PATH, "w") as f:
    json.dump(anchor, f)
print(f"Anchor escrito en {ANCHOR_PATH}: {anchor}")
