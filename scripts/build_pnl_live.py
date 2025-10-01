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

OUT_DIR = "data/live"
os.makedirs(OUT_DIR, exist_ok=True)
PNL_PATH = os.path.join(OUT_DIR, "pnl.csv")

# ---------- Carga ----------
bets = pd.read_csv(bets_path)
odds = pd.read_csv(odds_path)

# Columnas necesarias
bets_needed = [
    "season","week","week_label","week_order","schedule_date",
    "side","team","opponent","decimal_odds","stake","game_id"
]
odds_needed = [
    "season","week","week_label","schedule_date",
    "home_team","away_team","score_home","score_away"
]

missing_b = [c for c in bets_needed if c not in bets.columns]
missing_o = [c for c in odds_needed if c not in odds.columns]
if missing_b:
    print(f"ERROR: Faltan columnas en bets.csv: {missing_b}", file=sys.stderr)
    sys.exit(1)
if missing_o:
    print(f"ERROR: Faltan columnas en odds.csv: {missing_o}", file=sys.stderr)
    sys.exit(1)

# Tipos
bets["season"] = pd.to_numeric(bets["season"], errors="coerce").astype(int)
bets["week"] = pd.to_numeric(bets["week"], errors="coerce").astype(int)
bets["week_order"] = pd.to_numeric(bets["week_order"], errors="coerce").astype(int)
bets["stake"] = pd.to_numeric(bets["stake"], errors="coerce").fillna(0.0)
bets["decimal_odds"] = pd.to_numeric(bets["decimal_odds"], errors="coerce")

# Fechas a datetime (UTC)
bets["schedule_date"] = pd.to_datetime(bets["schedule_date"], errors="coerce", utc=True)
odds["schedule_date"] = pd.to_datetime(odds["schedule_date"], errors="coerce", utc=True)

# ---------- Join robusto Bets x Odds ----------
# 1) Merge por (season, week, week_label) para crear candidatos
bets = bets.reset_index().rename(columns={"index":"bet_idx"})
cand = bets.merge(
    odds[["season","week","week_label","schedule_date","home_team","away_team","score_home","score_away"]],
    on=["season","week","week_label"],
    how="left",
    suffixes=("","_o")
)

# 2) Filtra a los que tengan mismos equipos (sin importar orden)
same_pair = (
    ((cand["team"] == cand["home_team"]) & (cand["opponent"] == cand["away_team"])) |
    ((cand["team"] == cand["away_team"]) & (cand["opponent"] == cand["home_team"]))
)
cand = cand.loc[same_pair].copy()

# 3) Si hay varios candidatos (por diferencias mínimas de hora), escoge el de kickoff más cercano
cand["abs_time_diff"] = (cand["schedule_date_o"] - cand["schedule_date"]).abs().dt.total_seconds().abs()
cand.sort_values(["bet_idx","abs_time_diff"], inplace=True)
cand = cand.groupby("bet_idx", as_index=False).first()

# 4) Casting de scores
m = cand.copy()
m["score_home"] = pd.to_numeric(m["score_home"], errors="coerce")
m["score_away"] = pd.to_numeric(m["score_away"], errors="coerce")

# 5) Scores por equipo elegido
m["team_is_home"] = (m["team"] == m["home_team"])
m["team_score"] = np.where(m["team_is_home"], m["score_home"], m["score_away"])
m["opp_score"]  = np.where(m["team_is_home"], m["score_away"], m["score_home"])

# 6) Resultado y liquidación
m["settled"] = m["team_score"].notna() & m["opp_score"].notna()

def result_of(row):
    if not row["settled"]:
        return None
    if math.isclose(float(row["team_score"]), float(row["opp_score"]), rel_tol=0.0, abs_tol=1e-9):
        return "push"
    return "win" if row["team_score"] > row["opp_score"] else "loss"

m["result"] = m.apply(result_of, axis=1)

# 7) Profit por apuesta (solo liquidadas)
def profit_calc(row):
    if not row["settled"]:
        return 0.0
    stake = float(row["stake"])
    odds_v = float(row["decimal_odds"])
    r = row["result"]
    if r == "win":
        return stake * (odds_v - 1.0)
    if r == "loss":
        return -stake
    return 0.0  # push/void

m["profit_calc"] = m.apply(profit_calc, axis=1)

# ---------- Agregado semanal ----------
group_cols = ["season","week","week_label","week_order"]
agg = (
    m.groupby(group_cols, dropna=False)
     .apply(lambda g: pd.Series({
         "stake": g.loc[g["settled"], "stake"].sum(),        # solo liquidadas
         "profit": g.loc[g["settled"], "profit_calc"].sum(),
         "n_bets": len(g),
         "n_settled": int(g["settled"].sum()),
     }))
     .reset_index()
)

def safe_yield(row):
    s = row["stake"]
    return (row["profit"]/s*100.0) if s and not math.isclose(s, 0.0) else 0.0

agg["yield_%"] = agg.apply(safe_yield, axis=1)
agg["status_week"] = agg.apply(lambda r: "final" if r["n_settled"] == r["n_bets"] and r["n_bets"] > 0 else "open", axis=1)

# Orden cronológico
agg = agg.sort_values(["season","week_order","week"]).reset_index(drop=True)

# ---------- Bankroll: SOLO desde bets (cuando la semana esté finalizada) ----------
# Tomamos el bankroll de la semana desde bets.csv -> columna 'bankroll_week_final'
# Si no existe/está vacío, dejamos NaN y lo reportamos.
if "bankroll_week_final" in bets.columns:
    # Convertir a numérico y quedarnos con el último valor no nulo por semana
    bets["_bk_final"] = pd.to_numeric(bets["bankroll_week_final"], errors="coerce")
    wk_bk = (
        bets.dropna(subset=["_bk_final"])
            .sort_values(["season","week_order","week","schedule_date"])
            .groupby(["season","week","week_label","week_order"], as_index=False)["_bk_final"]
            .last()
            .rename(columns={"_bk_final":"bankroll"})
    )
else:
    wk_bk = pd.DataFrame(columns=["season","week","week_label","week_order","bankroll"])

# Mantener SOLO semanas finalizadas y anexar bankroll (desde bets)
final_weeks = agg[agg["status_week"].eq("final")].copy()
pnl = final_weeks.merge(
    wk_bk,
    on=["season","week","week_label","week_order"],
    how="left"
)

# Marca de tiempo
now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
pnl["updated_at_utc"] = now_iso

# Salida (solo semanas finalizadas)
out_cols = ["season","week","week_label","profit","stake","yield_%","bankroll","status_week","updated_at_utc"]
pnl[out_cols].to_csv(PNL_PATH, index=False)

# Logs útiles
not_matched = set(bets["bet_idx"]) - set(m["bet_idx"])
if not_matched:
    gids = bets.loc[bets["bet_idx"].isin(not_matched), "game_id"].tolist()
    print(f"AVISO: {len(not_matched)} apuestas no se pudieron emparejar con odds por equipos/fecha. game_id: {gids}")

# Avisar si hay semanas finalizadas sin bankroll en bets
missing_bk = pnl[pnl["bankroll"].isna()]
if len(missing_bk):
    weeks_txt = missing_bk[["season","week","week_label"]].to_dict(orient="records")
    print(f"AVISO: {len(missing_bk)} semana(s) FINALIZADAS no tienen 'bankroll_week_final' en bets.csv -> {weeks_txt}")

print(f"PNL escrito en {PNL_PATH} (semanas incluidas: {len(pnl)})")
