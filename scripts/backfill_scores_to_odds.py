#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
from nfl_data_py import import_schedules

POST_WK_MAP = {1: 19, 2: 20, 3: 21, 4: 22}

_ABBR = {
    "ARZ":"ARI","ARI":"ARI","CARDINALS":"ARI",
    "ATL":"ATL","FALCONS":"ATL",
    "BAL":"BAL","RAVENS":"BAL",
    "BUF":"BUF","BILLS":"BUF",
    "CAR":"CAR","PANTHERS":"CAR",
    "CHI":"CHI","BEARS":"CHI",
    "CIN":"CIN","BENGALS":"CIN",
    "CLE":"CLE","BROWNS":"CLE",
    "DAL":"DAL","COWBOYS":"DAL",
    "DEN":"DEN","BRONCOS":"DEN",
    "DET":"DET","LIONS":"DET",
    "GB":"GB","GNB":"GB","PACKERS":"GB",
    "HOU":"HOU","TEXANS":"HOU",
    "IND":"IND","COLTS":"IND",
    "JAX":"JAX","JAC":"JAX","JAGUARS":"JAX",
    "KC":"KC","KAN":"KC","CHIEFS":"KC",
    "LV":"LV","LVR":"LV","RAIDERS":"LV","OAK":"LV",
    "LAC":"LAC","SD":"LAC","CHARGERS":"LAC",
    "LAR":"LAR","LA":"LAR","RAMS":"LAR","STL":"LAR",
    "MIA":"MIA","DOLPHINS":"MIA",
    "MIN":"MIN","VIKINGS":"MIN",
    "NE":"NE","NWE":"NE","PATRIOTS":"NE",
    "NO":"NO","NOR":"NO","SAINTS":"NO",
    "NYG":"NYG","GIANTS":"NYG",
    "NYJ":"NYJ","JETS":"NYJ",
    "PHI":"PHI","EAGLES":"PHI",
    "PIT":"PIT","STEELERS":"PIT",
    "SEA":"SEA","SEAHAWKS":"SEA",
    "SF":"SF","SFO":"SF","49ERS":"SF",
    "TB":"TB","TAM":"TB","BUCCANEERS":"TB",
    "TEN":"TEN","TITANS":"TEN",
    "WAS":"WSH","WSH":"WSH","WFT":"WSH","COMMANDERS":"WSH",
}

def norm_abbr(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    if s in _ABBR:
        return _ABBR[s]
    parts = s.replace(".", "").replace("-", " ").split()
    joined = "".join(parts)
    if joined in _ABBR:
        return _ABBR[joined]
    if parts and parts[-1] in _ABBR:
        return _ABBR[parts[-1]]
    return s

def ensure_week_num(df: pd.DataFrame) -> pd.Series:
    if "week" in df.columns and pd.api.types.is_numeric_dtype(df["week"]):
        return pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    if "week_label" in df.columns:
        m = {f"Week {i}": i for i in range(1, 19)}
        m.update({"Wild Card": 19, "Divisional": 20, "Conference": 21, "Super Bowl": 22})
        return df["week_label"].astype(str).map(m).astype("Int64")
    return pd.Series(pd.array([None]*len(df), dtype="Int64"))

def pair_key(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.Series([f"{x}_{y}" if x < y else f"{y}_{x}" for x, y in zip(a, b)])

def main(year: int):
    repo = Path(__file__).resolve().parent.parent
    season_dir = repo / "data" / "archive" / f"season={year}"
    season_dir.mkdir(parents=True, exist_ok=True)

    odds_path = season_dir / "odds.csv"
    if not odds_path.exists():
        raise FileNotFoundError(f"{odds_path} no existe.")

    odds = pd.read_csv(odds_path, low_memory=False)
    for c in ("home_team","away_team","team","opponent"):
        if c in odds.columns:
            odds[c] = odds[c].astype(str).map(norm_abbr)
    if "home_team" not in odds.columns and "team" in odds.columns:
        odds["home_team"] = odds["team"]
    if "away_team" not in odds.columns and "opponent" in odds.columns:
        odds["away_team"] = odds["opponent"]

    odds["season"] = pd.to_numeric(odds.get("season", year), errors="coerce").fillna(year).astype(int)
    odds["wk_num"]  = ensure_week_num(odds)
    odds["pair"]    = pair_key(odds["home_team"], odds["away_team"])

    sched = import_schedules([year])
    sched.columns = [c.lower() for c in sched.columns]
    ht = "home_team" if "home_team" in sched.columns else "home"
    at = "away_team" if "away_team" in sched.columns else "away"
    wk = "week"
    stype = "season_type" if "season_type" in sched.columns else ("game_type" if "game_type" in sched.columns else None)
    hsc = "home_score" if "home_score" in sched.columns else ("score_home" if "score_home" in sched.columns else None)
    asc = "away_score" if "away_score" in sched.columns else ("score_away" if "score_away" in sched.columns else None)
    date_col = next((c for c in ["gameday","game_date","start_time","game_time","start_time_eastern","start_time_utc"] if c in sched.columns), None)

    keep = [c for c in [ht, at, wk, stype, hsc, asc, date_col] if c]
    sched = sched[keep].copy()
    sched[ht] = sched[ht].astype(str).map(norm_abbr)
    sched[at] = sched[at].astype(str).map(norm_abbr)
    sched["wk_num"] = pd.to_numeric(sched[wk], errors="coerce").astype("Int64")
    if stype and stype in sched.columns:
        is_post = sched[stype].astype(str).str.upper().isin(["POST","POSTSEASON","PLAYOFFS"])
        sched.loc[is_post, "wk_num"] = sched.loc[is_post, wk].map(POST_WK_MAP).astype("Int64")
    if date_col:
        sched["schedule_date"] = pd.to_datetime(sched[date_col], errors="coerce", utc=True)

    if hsc is None or asc is None:
        raise RuntimeError("Schedule sin columnas de score.")

    sched = sched.rename(columns={hsc: "score_home", asc: "score_away", ht: "home_team", at: "away_team"})
    sched["pair"] = pair_key(sched["home_team"], sched["away_team"])
    sched = sched[["wk_num","pair","score_home","score_away","schedule_date"]].drop_duplicates()

    merged = odds.merge(sched, on=["wk_num","pair"], how="left", validate="m:1")

    col_order = [
        "season","wk_num","week","week_label","schedule_date",
        "home_team","away_team",
        "ml_home","ml_away","decimal_home","decimal_away",
        "spread_home","spread_away","over_under_line",
        "score_home","score_away","event_id"
    ]
    cols = [c for c in col_order if c in merged.columns]
    rest = [c for c in merged.columns if c not in cols]
    out = merged[cols + rest]

    out.to_csv(odds_path, index=False)
    print(f"updated: {odds_path} rows={len(out)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    main(args.year)
