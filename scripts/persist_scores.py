#!/usr/bin/env python
# scripts/persist_scores.py
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Permite imports locales del repo (nfl_dash/*)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nfl_dash.live_scores import fetch_espn_scoreboard_df
from nfl_dash.utils import norm_abbr

ET = ZoneInfo("America/New_York")
ODDS_PATH = os.path.join("data", "live", "odds.csv")

# -------------------------
# Mapeo a abreviaturas NFL
# -------------------------
TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

def normalize_team(s: str) -> str:
    """
    Normaliza nombres que vienen de ESPN (a veces completos) a abreviatura.
    - Si ya es abreviatura, la respeta.
    - Si es nombre completo, mapea -> abbr.
    - Luego pasa por norm_abbr para unificar variantes (e.g., STL->LA, SD->LAC).
    """
    if not isinstance(s, str):
        return s
    raw = s.strip()
    # Si ya viene abreviado (por ejemplo KC, LAC, LA...), respétalo
    up = raw.upper()
    if up in TEAM_NAME_TO_ABBR.values():
        return norm_abbr(up)
    # Si viene con nombre completo, mapea a abbr
    abbr = TEAM_NAME_TO_ABBR.get(raw, up)
    return norm_abbr(abbr)

# =========================
# Helpers de calendario
# =========================
def _labor_day_et(year: int) -> datetime:
    d = datetime(year, 9, 1, tzinfo=ET)
    while d.weekday() != 0:
        d += timedelta(days=1)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)

def _week1_tnf_et(year: int) -> datetime:
    labor = _labor_day_et(year)
    thu = labor + timedelta(days=3)
    return thu.replace(hour=20, minute=20)

def _tuesday_anchor_et(year: int) -> datetime:
    tnf = _week1_tnf_et(year)
    week_monday = (tnf - timedelta(days=tnf.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return week_monday + timedelta(days=1)  # Tue 00:00 ET

# Corte diferido al miércoles 02:00 ET
CUTOVER_DOW_ET = 2   # 0=Mon,1=Tue,2=Wed
CUTOVER_HOUR_ET = 2  # 02:00 ET

def _after_cutover_et(dt_utc: datetime) -> bool:
    now_et = dt_utc.astimezone(ET)
    if now_et.weekday() > CUTOVER_DOW_ET:
        return True
    if now_et.weekday() < CUTOVER_DOW_ET:
        return False
    return now_et.hour >= CUTOVER_HOUR_ET

def autodetect_week(season: int, now_utc: datetime | None = None) -> int:
    """
    Semana efectiva: antes del corte (miércoles 02:00 ET) restamos 1 semana.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET)
    anchor = _tuesday_anchor_et(season)
    if now_et < anchor:
        wk_nat = 1
    else:
        wk_nat = int(((now_et - anchor).days // 7) + 1)
    wk_nat = max(1, min(22, wk_nat))
    return wk_nat if _after_cutover_et(now_utc) else max(1, wk_nat - 1)

# =========================
# Utils
# =========================
def _make_pair(home: pd.Series, away: pd.Series) -> pd.Series:
    ha = home.astype(str).map(norm_abbr)
    aa = away.astype(str).map(norm_abbr)
    return np.where(ha < aa, ha + "_" + aa, aa + "_" + ha)

def _fetch_finals_two_weeks(season: int, week_eff: int) -> pd.DataFrame:
    """
    'Cinturón y tirantes': trae finales de la semana efectiva y de la semana previa.
    Así, aunque cambie la semana por el corte, seguimos capturando el MNF.
    """
    frames = []
    for wk in {week_eff, max(1, week_eff - 1)}:
        try:
            sb = fetch_espn_scoreboard_df(season=season, week=wk)
        except Exception:
            sb = pd.DataFrame()
        if not sb.empty:
            sb = sb.copy()
            sb["state"] = sb["state"].astype(str).str.lower()
            sb = sb[sb["state"].eq("post")]
            frames.append(sb)

    if not frames:
        return pd.DataFrame(columns=[
            "season","week","start_time","state","short",
            "home_team","home_score","away_team","away_score"
        ])

    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    return out

# =========================
# Main
# =========================
def main():
    if not os.path.exists(ODDS_PATH):
        print(f"[persist] SKIP: {ODDS_PATH} no existe.")
        return

    odds = pd.read_csv(ODDS_PATH, low_memory=False)

    # Asegura columnas de scores y home_win presentes (NA por default)
    for c in ("score_home", "score_away", "home_win"):
        if c not in odds.columns:
            odds[c] = pd.NA

    # Tipos básicos
    odds["season"] = pd.to_numeric(odds.get("season"), errors="coerce").astype("Int64")
    odds["week"]   = pd.to_numeric(odds.get("week"), errors="coerce").astype("Int64")
    for c in ("home_team", "away_team"):
        if c in odds.columns:
            odds[c] = odds[c].astype(str)

    # Temporada efectiva
    season_vals = odds["season"].dropna().unique()
    if len(season_vals) == 0:
        print("[persist] SKIP: season vacío en odds.")
        return
    season = int(season_vals[0])

    # Semana efectiva con corte diferido
    week_eff = autodetect_week(season)

    # Trae finales de semana efectiva y semana previa
    finals = _fetch_finals_two_weeks(season=season, week_eff=week_eff)
    if finals.empty:
        print("[persist] No hay juegos Final para persistir todavía.")
        return

    # (Opcional) Filtro de seguridad: últimos 36h por si la API marca "post" con delay
    try:
        finals["start_time"] = pd.to_datetime(finals["start_time"], errors="coerce", utc=True)
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=36)
        finals = finals[finals["start_time"] >= cutoff]
    except Exception:
        pass

    if finals.empty:
        print("[persist] No hay finales dentro de la ventana de 36h.")
        return

    # Normaliza equipos de ESPN a abreviaturas antes de merge
    finals["home_team"] = finals["home_team"].astype(str).map(normalize_team)
    finals["away_team"] = finals["away_team"].astype(str).map(normalize_team)

    # Pairs con abreviaturas (mismo criterio que odds)
    finals["pair"] = _make_pair(finals["home_team"], finals["away_team"])
    odds["pair"] = _make_pair(odds["home_team"], odds["away_team"])

    m = odds.merge(
        finals[["season", "week", "pair", "home_team", "away_team", "home_score", "away_score"]],
        on=["season", "week", "pair"],
        how="left",
        suffixes=("", "_sb"),
        validate="m:1"
    )

    # Mapear correctamente scores al "home" de odds aun si ESPN invierte el orden
    same_home = m["home_team"].astype(str).map(norm_abbr).eq(m["home_team_sb"].astype(str).map(norm_abbr))
    sh = np.where(same_home, m["home_score"], m["away_score"])
    sa = np.where(same_home, m["away_score"], m["home_score"])
    sh = pd.to_numeric(sh, errors="coerce")
    sa = pd.to_numeric(sa, errors="coerce")

    # Actualiza scores si hay valores nuevos (no sobreescribe con NA)
    m["score_home"] = np.where(pd.notna(sh), sh, m["score_home"])
    m["score_away"] = np.where(pd.notna(sa), sa, m["score_away"])

    # === home_win basado en scores ===
    def _compute_home_win(sr_home, sr_away):
        h = pd.to_numeric(sr_home, errors="coerce")
        a = pd.to_numeric(sr_away, errors="coerce")
        out = pd.Series(pd.NA, index=sr_home.index, dtype="Int64")
        both = h.notna() & a.notna()
        out.loc[both & (h > a)] = 1
        out.loc[both & (h < a)] = 0
        # si empatan (raro), dejamos NA
        return out

    hw_new = _compute_home_win(m["score_home"], m["score_away"])
    if "home_win" in m.columns:
        is_set = hw_new.notna()
        m.loc[is_set, "home_win"] = hw_new[is_set]
        m["home_win"] = pd.to_numeric(m["home_win"], errors="coerce").astype("Int64")
    else:
        m["home_win"] = pd.to_numeric(hw_new, errors="coerce").astype("Int64")

    # Limpia columnas auxiliares del merge
    drop_cols = [c for c in m.columns if c.endswith("_sb")] + ["home_score", "away_score", "pair"]
    m = m.drop(columns=[c for c in drop_cols if c in m.columns], errors="ignore")

    # Mantén exactamente el orden del odds original; si home_win no existía, lo agregamos al final
    base_cols = [c for c in odds.columns if c in m.columns]
    if "home_win" not in base_cols:
        base_cols = base_cols + ["home_win"]
    out = m[base_cols]

    out.to_csv(ODDS_PATH, index=False)
    print(f"[persist] wrote {ODDS_PATH} | rows={len(out)} at {datetime.now(timezone.utc).isoformat()}")

if __name__ == "__main__":
    main()
