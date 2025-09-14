import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

# Usa el fetch del módulo live_scores (ya lo tienes en nfl_dash/live_scores.py)
from nfl_dash.live_scores import fetch_espn_scoreboard_df

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
ODDS_PATH = ROOT / "data" / "live" / "odds.csv"

# --------------------- helpers de calendario (auto) --------------------- #
def labor_day_et(year: int) -> datetime:
    d = datetime(year, 9, 1, tzinfo=ET)
    while d.weekday() != 0:  # 0 = Monday
        d += timedelta(days=1)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)

def week1_tnf_et(year: int) -> datetime:
    # TNF de Week 1 ≈ Thu after Labor Day, 8:20 PM ET
    return labor_day_et(year) + timedelta(days=3, hours=20, minutes=20)

def current_season_year(now_utc: datetime | None = None) -> int:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET)
    yr = now_et.year
    return yr if now_et >= week1_tnf_et(yr) else yr - 1

def tuesday_anchor_et(year: int) -> datetime:
    # Semana NFL/ESPN: Tue 00:00 ET → Mon 23:59 ET
    start = week1_tnf_et(year)
    anchor = (start + timedelta(days=(1 - start.weekday()) % 7)).replace(hour=0, minute=0, second=0, microsecond=0)
    if anchor <= start:
        anchor += timedelta(days=7)
    return anchor

def autodetect_week(season: int, now_utc: datetime | None = None) -> int:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET)
    anchor = tuesday_anchor_et(season)
    if now_et < anchor:
        return 1
    delta = now_et - anchor
    return int(delta.days // 7 + 2)  # +2 porque anchor está entre W1 y W2

# --------------------- persistencia de scores en odds.csv --------------------- #
def _ensure_score_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Crea score_home / score_away si no existen (como Float64 para poder NaN)."""
    out = df.copy()
    for col in ("score_home", "score_away"):
        if col not in out.columns:
            out[col] = pd.Series([pd.NA] * len(out), dtype="Float64")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

def merge_scores_into_odds(season: int, week: int) -> int:
    if not ODDS_PATH.exists():
        print(f"[persist] {ODDS_PATH} not found; nothing to update.")
        return 0

    try:
        odds = pd.read_csv(ODDS_PATH, low_memory=False)
    except Exception as e:
        print(f"[persist] failed to read odds.csv: {e}")
        return 0

    # Normaliza tipos y columnas básicas
    for c in ("season", "week"):
        if c in odds.columns:
            odds[c] = pd.to_numeric(odds[c], errors="coerce")
    if "home_team" not in odds.columns or "away_team" not in odds.columns:
        print("[persist] odds.csv is missing home_team/away_team columns; cannot merge.")
        return 0

    odds = _ensure_score_cols(odds)

    # Filtra temporada/semana
    mask = (odds.get("season") == season) & (odds.get("week") == week)
    idx = odds.index[mask].tolist()
    if not idx:
        print(f"[persist] no odds rows for season={season} week={week}")
        return 0
    view = odds.loc[idx].copy()

    # Scoreboard ESPN
    try:
        sb = fetch_espn_scoreboard_df(season=season, week=week)
    except Exception as e:
        print(f"[persist] fetch scoreboard failed: {e}")
        return 0

    if sb.empty:
        print("[persist] scoreboard empty")
        return 0

    # Solo finales; renombramos a nuestros nombres destino
    finals = sb[sb["state"].astype(str).str.lower().eq("post")].copy()
    if finals.empty:
        print("[persist] no final games to merge")
        return 0

    finals = finals.rename(columns={
        "home_team":  "home_team",
        "away_team":  "away_team",
        "home_score": "score_home",
        "away_score": "score_away",
    })

    keep = ["home_team", "away_team", "score_home", "score_away"]
    finals = finals[keep].copy()

    # Merge exacto por nombres de equipo (los nombres ya coinciden en tu pipeline)
    merged = view.merge(finals, on=["home_team", "away_team"], how="left", suffixes=("", "_sb"))

    # Determina qué filas tienen actualización
    got = merged["score_home_sb"].notna() | merged["score_away_sb"].notna()
    n_updates = int(got.sum())
    if n_updates == 0:
        print("[persist] nothing to update (no finals matched).")
        return 0

    # Aplica actualización a la porción filtrada
    updated = view.copy()
    for col in ("score_home", "score_away"):
        sc = f"{col}_sb"
        if sc in merged.columns:
            updated[col] = merged[col].where(merged[sc].isna(), merged[sc])

    # Reinyecta y guarda
    odds.loc[idx, ["score_home", "score_away"]] = updated[["score_home", "score_away"]].values
    odds.to_csv(ODDS_PATH, index=False)
    print(f"[persist] wrote {ODDS_PATH} | updated rows={n_updates}")
    return n_updates

def main():
    # Permite override manual: python scripts/persist_scores.py 2025 2
    season = None
    week = None
    if len(sys.argv) >= 2:
        try: season = int(sys.argv[1])
        except: season = None
    if len(sys.argv) >= 3:
        try: week = int(sys.argv[2])
        except: week = None

    if season is None:
        season = current_season_year()
    if week is None:
        week = autodetect_week(season)

    print(f"[persist] now → season={season}, week={week}")
    merge_scores_into_odds(season, week)

if __name__ == "__main__":
    main()
