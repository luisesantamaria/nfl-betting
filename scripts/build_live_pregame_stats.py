#!/usr/bin/env python
# scripts/build_live_pregame_stats.py
import os, io, gzip, requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# ---------------------------------
# Config
# ---------------------------------
ET = ZoneInfo("America/New_York")
OUT_PATH = os.path.join("data", "live", "stats.csv")
TIMEOUT_PARQUET = 120
TIMEOUT_CSVGZ  = 180

# Corte efectivo: mantener semana anterior hasta Mié 02:00 ET
CUTOVER_DOW_ET  = 2   # 0=Mon,1=Tue,2=Wed
CUTOVER_HOUR_ET = 2   # 02:00 ET
EWM_ALPHA = 0.35      # igual que tu notebook

TEAM_FIX = {
    "STL":"LA","LAR":"LA","SD":"LAC","SDG":"LAC","OAK":"LV","LVR":"LV","WSH":"WAS","JAC":"JAX",
    "GNB":"GB","KAN":"KC","NWE":"NE","NOR":"NO","SFO":"SF","TAM":"TB"
}

NEED = [
    "game_id","game_date","season","week","season_type",
    "home_team","away_team","posteam","defteam",
    "play_type","epa","yards_gained","down","first_down",
    "air_yards","yards_after_catch"
]

# ---------------------------------
# Utilidades de semana
# ---------------------------------
def _after_cutover_et(dt_utc: datetime) -> bool:
    now_et = dt_utc.astimezone(ET)
    if now_et.weekday() > CUTOVER_DOW_ET:  # Thu..Sun
        return True
    if now_et.weekday() < CUTOVER_DOW_ET:  # Mon/Tue
        return False
    return now_et.hour >= CUTOVER_HOUR_ET  # Mié 02:00 ET

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

def normalize_week(season_type, w):
    st = str(season_type).upper(); w = int(w)
    if "PRE" in st:
        return None
    if "POST" in st:
        if 19 <= w <= 22: return w
        return {1:19, 2:20, 3:21, 4:22, 5:22, 0:19}.get(w, 19)
    return 1 if w < 1 else (18 if w > 18 else w)

# ---------------------------------
# Descarga nflverse
# ---------------------------------
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

# ---------------------------------
# Construcción de dataset por juego (team-game)
# ---------------------------------
def build_team_game_df(target_season: int) -> pd.DataFrame:
    pbp = fetch_year_any(target_season)

    pbp = pbp[[c for c in NEED if c in pbp.columns]].copy()
    pbp["season_type"] = pbp.get("season_type", "REG").astype(str).str.upper().str.strip()
    pbp = pbp[pbp["season_type"] != "PRE"].copy()
    pbp["week"] = pbp["week"].astype(int)
    pbp["week"] = [normalize_week(st, w) for st, w in zip(pbp["season_type"], pbp["week"])]
    pbp = pbp[pbp["week"].notna()].copy()

    # <<< Asegurar UTC (tz-aware) desde aquí >>>
    pbp["game_date"] = pd.to_datetime(pbp.get("game_date", pd.NaT), errors="coerce", utc=True)

    # Limitar hasta la semana efectiva
    wk_eff = effective_week(target_season)
    pbp = pbp[(pbp["season"] == target_season) & (pbp["week"] <= wk_eff)].copy()

    # Sólo plays de run/pass con EPA
    pbp = pbp[pbp["play_type"].isin(["pass","run"]) & pbp["epa"].notna()].copy()

    # Normalizar equipos
    for col in ["posteam", "defteam", "home_team", "away_team"]:
        if col in pbp.columns:
            pbp[col] = (pbp[col].astype(str).str.upper().str.strip()
                        .map(lambda x: TEAM_FIX.get(x, x)))

    # Flags por jugada
    is_pass = pbp["play_type"].eq("pass")
    is_run  = pbp["play_type"].eq("run")
    explosive = (is_pass & (pbp["yards_gained"] >= 20)) | (is_run & (pbp["yards_gained"] >= 10))
    pbp = pbp.assign(
        success=(pbp["epa"] > 0).astype(int),
        explosive=explosive.astype(int),
        third_down=(pbp["down"] == 3).astype(int),
        third_down_conv=((pbp["down"] == 3) & pbp["first_down"].fillna(False)).astype(int),
    )

    gb_off = ["game_id","season","week","posteam"]
    gb_def = ["game_id","season","week","defteam"]

    off = (pbp.groupby(gb_off, as_index=False)
           .agg(
               off_epa_per_play=("epa","mean"),
               off_plays=("epa","size"),
               off_yards_per_play=("yards_gained","mean"),
               off_success_rate=("success","mean"),
               off_explosive_rate=("explosive","mean"),
               off_third_down_rate=("third_down","mean"),
               off_third_down_conv_rate=("third_down_conv","mean"),
           ).rename(columns={"posteam":"team"}))

    defn = (pbp.groupby(gb_def, as_index=False)
            .agg(
                def_epa_allowed=("epa","mean"),
                def_plays=("epa","size"),
                def_yards_per_play_allowed=("yards_gained","mean"),
                def_success_rate_allowed=("success","mean"),
                def_explosive_rate_allowed=("explosive","mean"),
                def_third_down_rate_allowed=("third_down","mean"),
                def_third_down_conv_rate_allowed=("third_down_conv","mean"),
            ).rename(columns={"defteam":"team"}))

    off_pass = (pbp[is_pass].groupby(gb_off, as_index=False)
        .agg(
            off_pass_epa=("epa","mean"),
            off_pass_plays=("epa","size"),
            off_pass_yards_per_att=("yards_gained","mean"),
            off_pass_success_rate=("success","mean"),
            off_pass_explosive_rate=("explosive","mean"),
            off_air_yards_per_att=("air_yards", lambda x: x.dropna().mean() if x.dropna().size else np.nan),
            off_yac_per_rec=("yards_after_catch", lambda x: x.dropna().mean() if x.dropna().size else np.nan),
        ).rename(columns={"posteam":"team"}))

    off_rush = (pbp[is_run].groupby(gb_off, as_index=False)
        .agg(
            off_rush_epa=("epa","mean"),
            off_rush_plays=("epa","size"),
            off_rush_yards_per_att=("yards_gained","mean"),
            off_rush_success_rate=("success","mean"),
            off_rush_explosive_rate=("explosive","mean"),
        ).rename(columns={"posteam":"team"}))

    def_pass = (pbp[is_pass].groupby(gb_def, as_index=False)
        .agg(
            def_pass_epa_allowed=("epa","mean"),
            def_pass_plays=("epa","size"),
            def_pass_yards_per_att_allowed=("yards_gained","mean"),
            def_pass_success_rate_allowed=("success","mean"),
            def_pass_explosive_rate_allowed=("explosive","mean"),
            def_air_yards_per_att_allowed=("air_yards", lambda x: x.dropna().mean() if x.dropna().size else np.nan),
            def_yac_per_rec_allowed=("yards_after_catch", lambda x: x.dropna().mean() if x.dropna().size else np.nan),
        ).rename(columns={"defteam":"team"}))

    def_rush = (pbp[is_run].groupby(gb_def, as_index=False)
        .agg(
            def_rush_epa_allowed=("epa","mean"),
            def_rush_plays=("epa","size"),
            def_rush_yards_per_att_allowed=("yards_gained","mean"),
            def_rush_success_rate_allowed=("success","mean"),
            def_rush_explosive_rate_allowed=("explosive","mean"),
        ).rename(columns={"defteam":"team"}))

    team_game = (off.merge(off_pass, on=["game_id","season","week","team"], how="left")
                    .merge(off_rush, on=["game_id","season","week","team"], how="left")
                    .merge(defn,     on=["game_id","season","week","team"], how="left")
                    .merge(def_pass, on=["game_id","season","week","team"], how="left")
                    .merge(def_rush, on=["game_id","season","week","team"], how="left"))

    game_dates = pbp[["game_id","game_date","home_team","away_team"]].drop_duplicates("game_id")
    team_game = team_game.merge(game_dates, on="game_id", how="left")

    team_game = team_game[(team_game["off_plays"]>=10) & (team_game["def_plays"]>=10)].copy()

    team_game["team"] = team_game["team"].astype(str).str.upper().str.strip().map(lambda x: TEAM_FIX.get(x, x))
    team_game = team_game.sort_values(["team","season","game_date","game_id"]).reset_index(drop=True)

    def _rebuild_gid(row):
        try:
            season = int(row["season"]); week = int(row["week"])
            home = str(row.get("home_team") or "").upper()
            away = str(row.get("away_team") or "").upper()
            home = TEAM_FIX.get(home, home); away = TEAM_FIX.get(away, away)
            if home and away:
                return f"{season}_{week:02d}_{away}_{home}"
        except Exception:
            pass
    # Si no podemos reconstruir, deja el original
        return row["game_id"]

    team_game["game_id"] = team_game.apply(_rebuild_gid, axis=1)
    return team_game

# ---------------------------------
# Rolling/expanding pregame (YTD) — armado en bloque para evitar fragmentación
# ---------------------------------
def build_pregame_one_season(df_team_season: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    base = df_team_season[["game_id","season","week","team","game_date"]].copy()
    out_parts = {"game_id": base["game_id"], "season": base["season"],
                 "week": base["week"], "team": base["team"], "game_date": base["game_date"]}
    for c in metric_cols:
        s_prev = df_team_season[c].shift(1)
        out_parts[f"{c}_pre_ytd"] = s_prev.expanding(min_periods=1).mean()
        out_parts[f"{c}_pre_ewm"] = s_prev.ewm(alpha=EWM_ALPHA, adjust=False).mean()
        out_parts[f"{c}_pre_l8"]  = s_prev.rolling(window=8, min_periods=1).mean()
    return pd.DataFrame(out_parts)

def _rebuild_gid_from_teams(season: int, week: int, away: str, home: str) -> str:
    return f"{season}_{int(week):02d}_{away}_{home}"

def _pad_next_week_from_odds(pregame: pd.DataFrame, season: int, out_order: list[str]) -> pd.DataFrame:
    """
    Crea filas para la semana siguiente (wk_eff+1) usando el calendario de data/live/odds.csv
    y copia las métricas *_pre_* del último registro disponible por equipo (hasta wk_eff).
    """
    LIVE_ODDS_PATH = os.path.join("data", "live", "odds.csv")
    if not os.path.exists(LIVE_ODDS_PATH):
        return pregame

    if pregame["week"].dropna().empty:
        return pregame

    wk_eff = int(pregame["week"].max())
    next_wk = wk_eff + 1
    if next_wk > 22:
        return pregame

    odds = pd.read_csv(LIVE_ODDS_PATH, low_memory=False)

    # Tipos consistentes
    for c in ("season","week"):
        if c in odds.columns:
            odds[c] = pd.to_numeric(odds[c], errors="coerce").astype("Int64")

    for c in ("home_team","away_team"):
        if c in odds.columns:
            odds[c] = (odds[c].astype(str).str.upper().str.strip()
                       .map(lambda x: TEAM_FIX.get(x, x)))

    if "schedule_date" in odds.columns:
        odds["schedule_date"] = pd.to_datetime(odds["schedule_date"], errors="coerce", utc=True)

    mask = (odds.get("season", pd.Series(dtype="Int64")).eq(season)) & (odds.get("week", pd.Series(dtype="Int64")).eq(next_wk))
    cols_cal = [c for c in ["season","week","home_team","away_team","schedule_date"] if c in odds.columns]
    cal = odds.loc[mask.fillna(False), cols_cal].dropna(subset=["home_team","away_team"])
    if cal.empty:
        return pregame

    # Último snapshot por equipo (hasta wk_eff)
    last_by_team = (
        pregame.sort_values(["team","week","game_date"], kind="mergesort")
               .groupby("team", as_index=False, sort=False)
               .tail(1)
               .set_index("team")
    )

    # Armar filas home/away de la próxima semana
    rows = []
    for _, r in cal.iterrows():
        season_i, week_i = int(r["season"]), int(r["week"])
        home, away = str(r["home_team"]), str(r["away_team"])
        gd = pd.to_datetime(r.get("schedule_date", pd.NaT), utc=True)

        base_h = last_by_team.loc[home] if home in last_by_team.index else None
        base_a = last_by_team.loc[away] if away in last_by_team.index else None

        def make_row(team, base_row):
            row = {c: np.nan for c in out_order}
            row["season"] = season_i
            row["week"]   = week_i
            row["team"]   = team
            row["game_date"] = gd
            row["week_label"] = f"Week {week_i}"
            row["game_id"] = _rebuild_gid_from_teams(season_i, week_i, away=away, home=home)
            if base_row is not None:
                for c in base_row.index:
                    if "_pre_" in c and c in out_order:
                        row[c] = base_row[c]
            return row

        rows.append(make_row(home, base_h))
        rows.append(make_row(away, base_a))

    pad_df = pd.DataFrame(rows)
    for c in out_order:
        if c not in pad_df.columns:
            pad_df[c] = np.nan
    pad_df = pad_df[out_order]

    # Tipos robustos antes de ordenar
    merged = pd.concat([pregame, pad_df], ignore_index=True)
    merged["week"] = pd.to_numeric(merged["week"], errors="coerce")
    merged["game_date"] = pd.to_datetime(merged["game_date"], errors="coerce", utc=True)
    merged["team"] = merged["team"].astype(str)
    merged["game_id"] = merged["game_id"].astype(str)

    merged = merged.sort_values(["week","game_date","team","game_id"], kind="mergesort")
    merged = merged.drop_duplicates(["season","week","team","game_id"], keep="first")
    return merged

def build_pregame_current_season(target_season: int) -> pd.DataFrame:
    team_game = build_team_game_df(target_season)

    metric_cols = [c for c in team_game.columns if c not in
                   ["game_id","season","week","team","game_date","home_team","away_team"]]

    cols_for_apply = ["game_id","season","week","team","game_date"] + metric_cols
    pregame = (team_game.groupby(["team","season"], group_keys=False, sort=False)[cols_for_apply]
               .apply(lambda df: build_pregame_one_season(df, metric_cols))
               .reset_index(drop=True))

    pregame["week_label"] = pregame["week"].apply(lambda w: f"Week {int(w)}" if pd.notna(w) else None)

    ordered = [
        "game_id","season","week","team","game_date",
        "off_epa_per_play_pre_ytd","off_epa_per_play_pre_ewm","off_epa_per_play_pre_l8",
        "off_plays_pre_ytd","off_plays_pre_ewm","off_plays_pre_l8",
        "off_yards_per_play_pre_ytd","off_yards_per_play_pre_ewm","off_yards_per_play_pre_l8",
        "off_success_rate_pre_ytd","off_success_rate_pre_ewm","off_success_rate_pre_l8",
        "off_explosive_rate_pre_ytd","off_explosive_rate_pre_ewm","off_explosive_rate_pre_l8",
        "off_third_down_rate_pre_ytd","off_third_down_rate_pre_ewm","off_third_down_rate_pre_l8",
        "off_third_down_conv_rate_pre_ytd","off_third_down_conv_rate_pre_ewm","off_third_down_conv_rate_pre_l8",
        "off_pass_epa_pre_ytd","off_pass_epa_pre_ewm","off_pass_epa_pre_l8",
        "off_pass_plays_pre_ytd","off_pass_plays_pre_ewm","off_pass_plays_pre_l8",
        "off_pass_yards_per_att_pre_ytd","off_pass_yards_per_att_pre_ewm","off_pass_yards_per_att_pre_l8",
        "off_pass_success_rate_pre_ytd","off_pass_success_rate_pre_ewm","off_pass_success_rate_pre_l8",
        "off_pass_explosive_rate_pre_ytd","off_pass_explosive_rate_pre_ewm","off_pass_explosive_rate_pre_l8",
        "off_air_yards_per_att_pre_ytd","off_air_yards_per_att_pre_ewm","off_air_yards_per_att_pre_l8",
        "off_yac_per_rec_pre_ytd","off_yac_per_rec_pre_ewm","off_yac_per_rec_pre_l8",
        "off_rush_epa_pre_ytd","off_rush_epa_pre_ewm","off_rush_epa_pre_l8",
        "off_rush_plays_pre_ytd","off_rush_plays_pre_ewm","off_rush_plays_pre_l8",
        "off_rush_yards_per_att_pre_ytd","off_rush_yards_per_att_pre_ewm","off_rush_yards_per_att_pre_l8",
        "off_rush_success_rate_pre_ytd","off_rush_success_rate_pre_ewm","off_rush_success_rate_pre_l8",
        "off_rush_explosive_rate_pre_ytd","off_rush_explosive_rate_pre_ewm","off_rush_explosive_rate_pre_l8",
        "def_epa_allowed_pre_ytd","def_epa_allowed_pre_ewm","def_epa_allowed_pre_l8",
        "def_plays_pre_ytd","def_plays_pre_ewm","def_plays_pre_l8",
        "def_yards_per_play_allowed_pre_ytd","def_yards_per_play_allowed_pre_ewm","def_yards_per_play_allowed_pre_l8",
        "def_success_rate_allowed_pre_ytd","def_success_rate_allowed_pre_ewm","def_success_rate_allowed_pre_l8",
        "def_explosive_rate_allowed_pre_ytd","def_explosive_rate_allowed_pre_ewm","def_explosive_rate_allowed_pre_l8",
        "def_third_down_rate_allowed_pre_ytd","def_third_down_rate_allowed_pre_ewm","def_third_down_rate_allowed_pre_l8",
        "def_third_down_conv_rate_allowed_pre_ytd","def_third_down_conv_rate_allowed_pre_ewm","def_third_down_conv_rate_allowed_pre_l8",
        "def_pass_epa_allowed_pre_ytd","def_pass_epa_allowed_pre_ewm","def_pass_epa_allowed_pre_l8",
        "def_pass_plays_pre_ytd","def_pass_plays_pre_ewm","def_pass_plays_pre_l8",
        "def_pass_yards_per_att_allowed_pre_ytd","def_pass_yards_per_att_allowed_pre_ewm","def_pass_yards_per_att_allowed_pre_l8",
        "def_pass_success_rate_allowed_pre_ytd","def_pass_success_rate_allowed_pre_ewm","def_pass_success_rate_allowed_pre_l8",
        "def_pass_explosive_rate_allowed_pre_ytd","def_pass_explosive_rate_allowed_pre_ewm","def_pass_explosive_rate_allowed_pre_l8",
        "def_air_yards_per_att_allowed_pre_ytd","def_air_yards_per_att_allowed_pre_ewm","def_air_yards_per_att_allowed_pre_l8",
        "def_yac_per_rec_allowed_pre_ytd","def_yac_per_rec_allowed_pre_ewm","def_yac_per_rec_allowed_pre_l8",
        "def_rush_epa_allowed_pre_ytd","def_rush_epa_allowed_pre_ewm","def_rush_epa_allowed_pre_l8",
        "def_rush_plays_pre_ytd","def_rush_plays_pre_ewm","def_rush_plays_pre_l8",
        "def_rush_yards_per_att_allowed_pre_ytd","def_rush_yards_per_att_allowed_pre_ewm","def_rush_yards_per_att_allowed_pre_l8",
        "def_rush_success_rate_allowed_pre_ytd","def_rush_success_rate_allowed_pre_ewm","def_rush_success_rate_allowed_pre_l8",
        "def_rush_explosive_rate_allowed_pre_ytd","def_rush_explosive_rate_allowed_pre_ewm","def_rush_explosive_rate_allowed_pre_l8",
        "week_label",
    ]

    for c in ordered:
        if c not in pregame.columns:
            pregame[c] = np.nan

    pregame = pregame[ordered].copy()

    # Orden consistente: Week asc., luego fecha, luego team y game_id (todo tipado)
    pregame["week"] = pd.to_numeric(pregame["week"], errors="coerce")
    pregame["game_date"] = pd.to_datetime(pregame["game_date"], errors="coerce", utc=True)
    pregame["team"] = pregame["team"].astype(str)
    pregame["game_id"] = pregame["game_id"].astype(str)

    pregame = pregame.sort_values(["week","game_date","team","game_id"], kind="mergesort").reset_index(drop=True)

    # Acolchona la próxima semana con calendario de odds (si existe)
    pregame = _pad_next_week_from_odds(pregame, season=target_season, out_order=ordered)

    return pregame

# ---------------------------------
# Main
# ---------------------------------
def main():
    env_season = os.environ.get("TARGET_SEASON")
    season = int(env_season) if env_season else datetime.now(ET).year

    df = build_pregame_current_season(season)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[live-pregame] wrote {OUT_PATH} rows={len(df)} season={season} at {datetime.now(timezone.utc).isoformat()}")

if __name__ == "__main__":
    main()
