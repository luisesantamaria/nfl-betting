#!/usr/bin/env python3
from pathlib import Path
import sys
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Asegura import local (nfl_dash/*)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nfl_dash.live_scores import fetch_espn_scoreboard_df
from nfl_dash.utils import norm_abbr

ET = ZoneInfo("America/New_York")

def _ensure_score_cols(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "score_home" not in x.columns: x["score_home"] = pd.NA
    if "score_away" not in x.columns: x["score_away"] = pd.NA
    return x

def _pair(a: str, b: str) -> str:
    return f"{a}_{b}" if a < b else f"{b}_{a}"

def _normalize_teams(df: pd.DataFrame, home_col: str, away_col: str) -> pd.DataFrame:
    x = df.copy()
    x[home_col] = x[home_col].astype(str).map(norm_abbr)
    x[away_col] = x[away_col].astype(str).map(norm_abbr)
    x["__pair"]  = [_pair(h, a) for h, a in zip(x[home_col], x[away_col])]
    return x

def _slot_complete_mask(score_df: pd.DataFrame) -> pd.Series:
    # “Slot” = mismos minutos ET de kickoff (ej. todos los de 1:00 PM)
    # Un slot se considera COMPLETO solo si TODOS sus juegos están en 'post'
    g = score_df.groupby("start_et_min")["state"].apply(lambda s: (s.str.lower() == "post").all())
    return score_df["start_et_min"].map(g.to_dict()).astype(bool)

def _persist_for_week(odds_week: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    if odds_week.empty:
        return odds_week

    try:
        sb = fetch_espn_scoreboard_df(season=int(season), week=int(week))
    except Exception:
        return odds_week

    if sb.empty:
        return odds_week

    sb = sb.copy()
    sb["start_time"] = pd.to_datetime(sb["start_time"], errors="coerce", utc=True)
    sb["start_et"]   = sb["start_time"].dt.tz_convert(ET)
    sb["start_et_min"] = sb["start_et"].dt.floor("min")
    sb["state"] = sb["state"].astype(str).str.lower()
    sb["short"] = sb["short"].astype(str)  # suele traer "Final", "Final/OT", etc.

    # Normaliza equipos y crea pair
    sb = _normalize_teams(sb, "home_team", "away_team")
    sb = sb.rename(columns={"home_score":"__home_score", "away_score":"__away_score"})

    # GUARDA DE ORO:
    # 1) Solo slots donde TODOS están en 'post'
    # 2) Y además el propio juego está 'post' y el short contiene "Final"
    slot_done_mask = _slot_complete_mask(sb)
    final_mask = sb["state"].eq("post") & sb["short"].str.contains("final", case=False, na=False)
    sb = sb[slot_done_mask & final_mask]
    if sb.empty:
        return odds_week

    # Prepara odds semana
    ow = _normalize_teams(odds_week, "home_team", "away_team")

    # Merge por pair; (semana y temporada están implícitas en el fetch)
    m = ow.merge(
        sb[["__pair","__home_score","__away_score"]],
        on="__pair", how="left"
    )

    # Asigna solo si hay score FINAL del slot completo (no toca NaN)
    set_home = m["__home_score"].notna()
    set_away = m["__away_score"].notna()

    # Asegura columnas
    if "score_home" not in m.columns: m["score_home"] = pd.NA
    if "score_away" not in m.columns: m["score_away"] = pd.NA

    m.loc[set_home, "score_home"] = pd.to_numeric(m.loc[set_home, "__home_score"], errors="coerce")
    m.loc[set_away, "score_away"] = pd.to_numeric(m.loc[set_away, "__away_score"], errors="coerce")

    # Limpia auxiliares
    m = m.drop(columns=[c for c in m.columns if c.startswith("__")], errors="ignore")
    return m

def main():
    odds_path = REPO_ROOT / "data" / "live" / "odds.csv"
    if not odds_path.exists():
        print("[persist] no odds.csv, nothing to do")
        return

    odds = pd.read_csv(odds_path, low_memory=False)
    if odds.empty:
        print("[persist] odds.csv empty, nothing to do")
        return

    odds = _ensure_score_cols(odds)

    if "season" not in odds.columns or "week" not in odds.columns:
        print("[persist] odds missing 'season' or 'week'")
        return

    season = int(pd.to_numeric(odds["season"], errors="coerce").dropna().max())
    weeks = sorted(pd.to_numeric(odds["week"], errors="coerce").dropna().unique().astype(int).tolist())
    if not weeks:
        print("[persist] no weeks found in odds")
        return

    before = odds.copy()
    for w in weeks:
        part_mask = pd.to_numeric(odds["week"], errors="coerce").eq(w)
        part = odds[part_mask].copy()
        rest = odds[~part_mask].copy()

        updated = _persist_for_week(part, season=season, week=w)
        odds = pd.concat([rest, updated], ignore_index=True)

    # Solo escribe si hubo cambios
    if before.equals(odds):
        print("[persist] no score changes to persist")
        return

    # Mantén orden y agrega score_* al final si no estaban
    cols = list(before.columns)
    for c in ["score_home","score_away"]:
        if c not in cols:
            cols.append(c)
    for c in odds.columns:
        if c not in cols:
            cols.append(c)
    odds = odds.reindex(columns=cols)

    odds.to_csv(odds_path, index=False)
    print(f"[persist] wrote {odds_path} | rows={len(odds)} at {datetime.now(timezone.utc).isoformat()}")

if __name__ == "__main__":
    main()
