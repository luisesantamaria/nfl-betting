#!/usr/bin/env python
# scripts/build_live_ytd_from_nflverse.py
import os, io, gzip, sys, requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# ---------- Config ----------
OUT_PATH = os.path.join("live", "stats.csv")
TIMEOUT_PARQUET = 120
TIMEOUT_CSVGZ  = 180

# Semana “efectiva”: mantenemos la semana anterior hasta Mié 02:00 ET (~03:00 BRT)
CUTOVER_DOW_ET  = 2   # 0=Mon,1=Tue,2=Wed
CUTOVER_HOUR_ET = 2   # 02:00 ET

def _after_cutover_et(dt_utc: datetime) -> bool:
    now_et = dt_utc.astimezone(ET)
    if now_et.weekday() > CUTOVER_DOW_ET:  # Thu..Sun
        return True
    if now_et.weekday() < CUTOVER_DOW_ET:  # Mon/Tue
        return False
    return now_et.hour >= CUTOVER_HOUR_ET  # Wed cutoff hour

def _labor_day_et(year: int) -> datetime:
    d = datetime(year, 9, 1, tzinfo=ET)
    while d.weekday() != 0:  # Monday
        d += timedelta(days=1)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)

def _week1_tnf_et(year: int) -> datetime:
    labor = _labor_day_et(year)
    thu   = labor + timedelta(days=3)
    return thu.replace(hour=20, minute=20)

def _tuesday_anchor_et(year: int) -> datetime:
    tnf = _week1_tnf_et(year)
    week_monday = (tnf - timedelta(days=tnf.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return week_monday + timedelta(days=1)  # Tue 00:00 ET

def _calendar_week_nat(season: int, now_utc: datetime) -> int:
    now_et  = now_utc.astimezone(ET)
    anchor  = _tuesday_anchor_et(season)
    if now_et < anchor:
        wk = 1
    else:
        wk = int(((now_et - anchor).days // 7) + 1)
    return max(1, min(22, wk))

def effective_week(season: int, now_utc: datetime | None = None) -> int:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    wk_nat = _calendar_week_nat(season, now_utc)
    return wk_nat if _after_cutover_et(now_utc) else max(1, wk_nat - 1)

# ---------- nflverse fetch ----------
def fetch_year_any(year: int) -> pd.DataFrame:
    base = "https://github.com/nflverse/nflverse-data/releases/download/pbp"
    parq = f"{base}/play_by_play_{year}.parquet"
    csvgz = f"{base}/play_by_play_{year}.csv.gz"
    try:
        r = requests.get(parq, timeout=TIMEOUT_PARQUET); r.raise_for_status()
        return pd.read_parquet(io.BytesIO(r.content))
    except Exception:
        r = requests.get(csvgz, timeout=TIMEOUT_CSVGZ); r.raise_for_status()
        with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
            return pd.read_csv(gz, low_memory=False)

# ---------- ETL / Métricas ----------
NEED_COLS = [
    "game_id","game_date","season","week","season_type",
    "home_team","away_team","posteam","defteam",
    "play_type","epa","yards_gained","down","first_down",
    "air_yards","yards_after_catch"
]

TEAM_FIX = {"STL":"LA","LAR":"LA","SD":"LAC","SDG":"LAC","OAK":"LV","LVR":"LV","WSH":"WAS","JAC":"JAX"}

def normalize_week(season_type, w):
    st = str(season_type).upper(); w = int(w)
    if "PRE" in st:
        return None  # descartamos preseason
    if "POST" in st:
        # map a 19..22 por compatibilidad
        if 19 <= w <= 22: return w
        return {1:19, 2:20, 3:21, 4:22, 5:22, 0:19}.get(w, 19)
    # REG
    return 1 if w < 1 else (18 if w > 18 else w)

def build_live_ytd(target_season: int) -> pd.DataFrame:
    # 1) Descargar solo la temporada objetivo
    pbp = fetch_year_any(target_season)

    # 2) Reducir columnas y limpiar
    pbp = pbp[[c for c in NEED_COLS if c in pbp.columns]].copy()
    pbp["season_type"] = pbp.get("season_type", "REG").astype(str).str.upper().str.strip()
    pbp = pbp[pbp["season_type"] != "PRE"].copy()
    pbp["week"] = pbp["week"].astype(int)
    pbp["week"] = [normalize_week(st, w) for st, w in zip(pbp["season_type"], pbp["week"])]
    pbp = pbp[pbp["week"].notna()].copy()
    pbp = pbp[pbp["play_type"].isin(["pass","run"]) & pbp["epa"].notna()].copy()
    pbp["game_date"] = pd.to_datetime(pbp.get("game_date", pd.NaT), errors="coerce")

    # 3) Cortar hasta la semana efectiva
    wk_eff = effective_week(target_season)
    pbp = pbp[(pbp["season"] == target_season) & (pbp["week"] <= wk_eff)].copy()

    # 4) Features por jugada
    is_pass = pbp["play_type"].eq("pass")
    is_run  = pbp["play_type"].eq("run")
    explosive = (is_pass & (pbp["yards_gained"] >= 20)) | (is_run & (pbp["yards_gained"] >= 10))

    pbp = pbp.assign(
        success=(pbp["epa"] > 0).astype(int),
        explosive=explosive.astype(int),
        is_third=(pbp["down"] == 3).astype(int),
        is_third_conv=((pbp["down"] == 3) & pbp["first_down"].fillna(False)).astype(int),
    )

    # 5) Normalización de equipos
    for col in ["posteam", "defteam", "home_team", "away_team"]:
        if col in pbp.columns:
            pbp[col] = pbp[col].astype(str).str.upper().str.strip().map(lambda x: TEAM_FIX.get(x, x))

    # 6) Agregados YTD ofensivos (por plays)
    off_all = (pbp.groupby("posteam")
        .agg(
            off_plays=("epa","size"),
            off_epa_per_play=("epa","mean"),
            off_yards_per_play=("yards_gained","mean"),
            off_success_rate=("success","mean"),
            off_explosive_rate=("explosive","mean"),
            off_third_down_rate=("is_third","mean"),
            off_third_down_conv_rate=("is_third_conv","mean"),
        )
        .rename_axis("team")
        .reset_index()
    )

    off_pass = (pbp[is_pass].groupby("posteam")
        .agg(
            off_pass_plays=("epa","size"),
            off_pass_epa=("epa","mean"),
            off_pass_yards_per_att=("yards_gained","mean"),
            off_pass_success_rate=("success","mean"),
            off_pass_explosive_rate=("explosive","mean"),
            off_air_yards_per_att=("air_yards", lambda x: x.dropna().mean() if x.dropna().size else np.nan),
            off_yac_per_rec=("yards_after_catch", lambda x: x.dropna().mean() if x.dropna().size else np.nan),
        )
        .rename_axis("team")
        .reset_index()
        .rename(columns={"posteam":"team"})
    )

    off_run = (pbp[is_run].groupby("posteam")
        .agg(
            off_rush_plays=("epa","size"),
            off_rush_epa=("epa","mean"),
            off_rush_yards_per_att=("yards_gained","mean"),
            off_rush_success_rate=("success","mean"),
            off_rush_explosive_rate=("explosive","mean"),
        )
        .rename_axis("team")
        .reset_index()
        .rename(columns={"posteam":"team"})
    )

    # 7) Agregados YTD defensivos (por plays)
    def_all = (pbp.groupby("defteam")
        .agg(
            def_plays=("epa","size"),
            def_epa_allowed=("epa","mean"),
            def_yards_per_play_allowed=("yards_gained","mean"),
            def_success_rate_allowed=("success","mean"),
            def_explosive_rate_allowed=("explosive","mean"),
            def_third_down_rate_allowed=("is_third","mean"),
            def_third_down_conv_rate_allowed=("is_third_conv","mean"),
        )
        .rename_axis("team")
        .reset_index()
        .rename(columns={"defteam":"team"})
    )

    def_pass = (pbp[is_pass].groupby("defteam")
        .agg(
            def_pass_plays=("epa","size"),
            def_pass_epa_allowed=("epa","mean"),
            def_pass_yards_per_att_allowed=("yards_gained","mean"),
            def_pass_success_rate_allowed=("success","mean"),
            def_pass_explosive_rate_allowed=("explosive","mean"),
            def_air_yards_per_att_allowed=("air_yards", lambda x: x.dropna().mean() if x.dropna().size else np.nan),
            def_yac_per_rec_allowed=("yards_after_catch", lambda x: x.dropna().mean() if x.dropna().size else np.nan),
        )
        .rename_axis("team")
        .reset_index()
        .rename(columns={"defteam":"team"})
    )

    def_run = (pbp[is_run].groupby("defteam")
        .agg(
            def_rush_plays=("epa","size"),
            def_rush_epa_allowed=("epa","mean"),
            def_rush_yards_per_att_allowed=("yards_gained","mean"),
            def_rush_success_rate_allowed=("success","mean"),
            def_rush_explosive_rate_allowed=("explosive","mean"),
        )
        .rename_axis("team")
        .reset_index()
        .rename(columns={"defteam":"team"})
    )

    # 8) Juegos disputados YTD (para contexto)
    off_games = pbp.groupby("posteam")["game_id"].nunique().rename("off_gp").reset_index().rename(columns={"posteam":"team"})
    def_games = pbp.groupby("defteam")["game_id"].nunique().rename("def_gp").reset_index().rename(columns={"defteam":"team"})
    gp = off_games.merge(def_games, on="team", how="outer")
    gp["gp"] = gp[["off_gp","def_gp"]].max(axis=1)
    gp = gp[["team","gp"]]

    # 9) Merge final
    ytd = (off_all
           .merge(off_pass, on="team", how="left")
           .merge(off_run,  on="team", how="left")
           .merge(def_all,  on="team", how="left")
           .merge(def_pass, on="team", how="left")
           .merge(def_run,  on="team", how="left")
           .merge(gp,       on="team", how="left")
    )

    # Orden y tipos
    order_cols = [
        "team","gp",
        "off_plays","off_epa_per_play","off_yards_per_play","off_success_rate","off_explosive_rate",
        "off_third_down_rate","off_third_down_conv_rate",
        "off_pass_plays","off_pass_epa","off_pass_yards_per_att","off_pass_success_rate","off_pass_explosive_rate",
        "off_air_yards_per_att","off_yac_per_rec",
        "off_rush_plays","off_rush_epa","off_rush_yards_per_att","off_rush_success_rate","off_rush_explosive_rate",
        "def_plays","def_epa_allowed","def_yards_per_play_allowed","def_success_rate_allowed","def_explosive_rate_allowed",
        "def_third_down_rate_allowed","def_third_down_conv_rate_allowed",
        "def_pass_plays","def_pass_epa_allowed","def_pass_yards_per_att_allowed","def_pass_success_rate_allowed","def_pass_explosive_rate_allowed",
        "def_air_yards_per_att_allowed","def_yac_per_rec_allowed",
        "def_rush_plays","def_rush_epa_allowed","def_rush_yards_per_att_allowed","def_rush_success_rate_allowed","def_rush_explosive_rate_allowed",
    ]
    exist_cols = [c for c in order_cols if c in ytd.columns]
    ytd = ytd[exist_cols].copy()

    # Opcional: ordenar por eficiencia ofensiva
    if "off_epa_per_play" in ytd.columns:
        ytd = ytd.sort_values("off_epa_per_play", ascending=False).reset_index(drop=True)

    return ytd

def main():
    # Temporada objetivo: env TARGET_SEASON o año “ET” si no se pasa
    env_season = os.environ.get("TARGET_SEASON")
    if env_season:
        season = int(env_season)
    else:
        season = datetime.now(ET).year

    ytd = build_live_ytd(season)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    ytd.to_csv(OUT_PATH, index=False)
    print(f"[live-ytd] wrote {OUT_PATH} (season={season}) rows={len(ytd)} at {datetime.now(timezone.utc).isoformat()}")

if __name__ == "__main__":
    main()
