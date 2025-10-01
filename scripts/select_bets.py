#!/usr/bin/env python3
"""
select_bets.py (pregame staking only) — v2

- Entrena el modelo (HGB + calibración + meta LR).
- Usa stats pregame históricas (4 temporadas previas) + temporada actual.
- Une con odds actuales (data/live/odds.csv) solo para temporada target.
- Calcula EV/edge, aplica estrategia con filtros y planifica stake pregame
  usando como bankroll base por semana el `bankroll_week_final` de la semana anterior.
- APPEND/UPSERT-ONLY: semanas finalizadas no se tocan; la semana actual sí puede reescribirse.
- Se añaden:
  * Métricas de entrenamiento/validación y test parcial si hay juegos finalizados.
  * Ajustes suaves con historial propio (calibración/beneficio por equipo y liga).
  * Mínimo 5 picks/semana con relajación progresiva.

IMPORTANTE:
- NO se escribe columna 'bankroll' en el CSV final (solo se usa para calcular stakes).
"""

import os, re, warnings, math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

warnings.filterwarnings("ignore")

# ----------------------------
# Debug helper
# ----------------------------
DEBUG = True
def dprint(*args):
    if DEBUG:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[DBG {ts}]", *args, flush=True)

# ----------------------------
# Constantes y paths
# ----------------------------
ET = ZoneInfo("America/New_York")
LIVE_ODDS_PATH = os.path.join("data", "live", "odds.csv")
LIVE_STATS_PATH = os.path.join("data", "live", "stats.csv")
ARCHIVE_DIR = os.path.join("data", "archive")
BETS_OUT_PATH = os.path.join("data", "live", "bets.csv")

TEAM_FIX = {
    "STL":"LA","LAR":"LA","SD":"LAC","SDG":"LAC","OAK":"LV","LVR":"LV","WSH":"WAS","JAC":"JAX",
    "GNB":"GB","KAN":"KC","NWE":"NE","NOR":"NO","SFO":"SF","TAM":"TB"
}

ORDER_LABELS = [f"Week {i}" for i in range(1,19)] + ["Wild Card","Divisional","Conference","Super Bowl"]
ORDER_INDEX  = {lab:i for i,lab in enumerate(ORDER_LABELS)}
MIN_PROB = 1e-6

# --------- Filtros base + políticas ---------
CFG = dict(
    INITIAL_BANKROLL = 1000.0,

    # Kelly & caps
    KELLY_FRACTION   = 0.25,
    KELLY_PCT_CAP    = 0.05,
    ABS_STAKE_CAP    = 300.0,
    WEEKLY_CAP_PCT   = 0.50,

    # Filtros globales (base)
    ODDS_MIN         = 1.20,
    ODDS_MAX         = 3.40,
    CONF_MIN         = 0.090,
    EDGE_TAU         = 0.060,
    EDGE_TAU_MIN     = 0.040,
    EV_BASE_MIN      = 0.012,
    EV_SLOPE         = 0.012,
    SKIP_W1_W2       = True,

    DEVIG_SINGLE_SIDE = "raw",

    KELLY_EDGE_WEIGHT = dict(EDGE_REF=0.12, MIN_SCALE=0.60, MAX_SCALE=1.00),

    # Límites de picks
    MIN_PICKS_PER_WEEK = 5,     # mínimo deseado
    MAX_BETS_PER_WEEK  = 8,     # techo suave (si hay valor, puede acercarse)
    MAX_BIG_DOGS_PER_WEEK = 2,

    # Bandas
    BANDS = dict(
        FAV = dict(odds_lt=1.60,                 tau=0.050, conf=0.075, ev_slope=0.010),
        MID = dict(odds_ge=1.60, odds_lt=2.40,   tau=0.055, conf=0.080, ev_slope=0.012),
        DOG = dict(odds_ge=2.40, odds_le=3.40,   tau=0.065, conf=0.095, ev_slope=0.017,
                   extra_if_ge3_10=dict(tau=0.070, conf=0.100))
    ),

    # Relajación progresiva para cumplir mínimo semanal
    RELAX_STEPS = [
        dict(EDGE_TAU=-0.010, CONF_MIN=-0.010, EV_DOWN=-0.004),
        dict(EDGE_TAU=-0.015, CONF_MIN=-0.015, EV_DOWN=-0.006),
        dict(EDGE_TAU=-0.020, CONF_MIN=-0.020, EV_DOWN=-0.008),
    ],

    # Peso máximo del “ajuste propio” (para no sobreajustar)
    SELF_LEARN_MAX_SHIFT = 0.03,    # +/- 3 puntos de prob como tope
)

# ----------------------------
# Helpers básicos
# ----------------------------
def getenv_int(name: str, default_val: int) -> int:
    v = os.environ.get(name, None)
    try:
        s = str(v).strip()
        return int(s) if s != "" else default_val
    except (TypeError, ValueError):
        return default_val

def getenv_int_or_none(name: str):
    v = os.environ.get(name, None)
    try:
        s = str(v).strip()
        return int(s) if s != "" else None
    except (TypeError, ValueError):
        return None

def detect_target_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 3 else now.year - 1

def norm_team(s: str) -> str:
    s = str(s).upper().strip()
    return TEAM_FIX.get(s, s)

def week_label_from_num(n:int) -> str:
    return f"Week {n}" if 1<=n<=18 else {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(n, f"Week {n}")

def add_week_order(df):
    out = df.copy()
    out["week_label"] = out["week_label"].astype(str)
    out["week_order"] = out["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    return out

def implied_from_decimal(d):
    d = float(d); return np.clip(1.0/max(d,1e-9), MIN_PROB, 1-MIN_PROB)

def decimal_to_american(d):
    if pd.isna(d): return np.nan
    d = float(d)
    return round((d - 1) * 100, 0) if d >= 2.0 else round(-100 / (d - 1), 0)

def make_game_id_row(week_label: str, team: str, opp: str) -> str:
    a, b = str(team), str(opp)
    pair = a + "_" + b if a < b else b + "_" + a
    return f"{week_label} | {pair}"

# ----------------------------
# Carga datasets
# ----------------------------
def load_current_odds() -> pd.DataFrame:
    dprint("LOAD odds live path:", LIVE_ODDS_PATH, "| exists:", os.path.exists(LIVE_ODDS_PATH))
    if not os.path.exists(LIVE_ODDS_PATH):
        raise FileNotFoundError(f"{LIVE_ODDS_PATH} not found.")
    df = pd.read_csv(LIVE_ODDS_PATH, low_memory=False)
    for c in ("season","week"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if "schedule_date" in df.columns:
        df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce", utc=True)
    for c in ("home_team","away_team"):
        if c in df.columns: df[c] = df[c].astype(str).map(norm_team)
    if "home_win" not in df.columns and {"score_home","score_away"}.issubset(df.columns):
        df["home_win"] = (pd.to_numeric(df["score_home"], errors="coerce")
                          > pd.to_numeric(df["score_away"], errors="coerce")).astype("Int64")
    return df

def load_pregame_stats_for_seasons(target_season: int) -> pd.DataFrame:
    frames = []
    for y in range(target_season-4, target_season+1):  # incluye actual si ya existe live
        p_live = LIVE_STATS_PATH if y == target_season else os.path.join(ARCHIVE_DIR, f"season={y}", "stats.csv")
        if os.path.exists(p_live):
            tmp = pd.read_csv(p_live, low_memory=False)
            tmp["season"] = pd.to_numeric(tmp["season"], errors="coerce").astype("Int64")
            tmp["week"]   = pd.to_numeric(tmp["week"], errors="coerce").astype("Int64")
            tmp["team"]   = tmp["team"].astype(str).map(norm_team)
            frames.append(tmp)
    if not frames:
        raise FileNotFoundError("No se encontraron stats pregame para entrenar/pred.")
    df = pd.concat(frames, ignore_index=True)
    df = (df.sort_values(["season","week","team"])
            .drop_duplicates(["season","week","team"], keep="last"))
    return df

# ----------------------------
# Merge odds + pregame a nivel juego
# ----------------------------
def ensure_home_line_from_odds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "home_line" not in out.columns:
        if "spread_home" in out.columns:
            out["home_line"] = pd.to_numeric(out["spread_home"], errors="coerce")
        else:
            dec_h = pd.to_numeric(out.get("decimal_home", np.nan), errors="coerce")
            dec_a = pd.to_numeric(out.get("decimal_away", np.nan), errors="coerce")
            out["home_line"] = np.where(dec_h.notna() & dec_a.notna(),
                                        np.where(dec_h < dec_a, -1.5, 1.5), np.nan)
    return out

def add_pref(df: pd.DataFrame, pref: str, key: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in ("season","week", key)]
    ren = {c:f"{pref}{c}" for c in cols}
    return df.rename(columns=ren)

def build_master(odds_df: pd.DataFrame, pre_df: pd.DataFrame) -> pd.DataFrame:
    out = odds_df.copy()
    out["home_team"] = out["home_team"].map(norm_team)
    out["away_team"] = out["away_team"].map(norm_team)
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["week"]   = pd.to_numeric(out["week"],   errors="coerce").astype("Int64")
    if "week_label" not in out.columns:
        out["week_label"] = out["week"].apply(week_label_from_num)
    out = ensure_home_line_from_odds(out)

    pre = pre_df.copy()
    pre["team"]   = pre["team"].map(norm_team)
    pre["season"] = pd.to_numeric(pre["season"], errors="coerce").astype("Int64")
    pre["week"]   = pd.to_numeric(pre["week"],   errors="coerce").astype("Int64")
    pre = (pre.sort_values(["season","week","team"])
             .drop_duplicates(["season","week","team"], keep="last"))

    home_pre = add_pref(pre.rename(columns={"team":"home_team"}), "home_", key="home_team")
    away_pre = add_pref(pre.rename(columns={"team":"away_team"}), "away_", key="away_team")

    m = (out.merge(home_pre, on=["season","week","home_team"], how="left")
            .merge(away_pre, on=["season","week","away_team"], how="left"))

    if "schedule_date" in m.columns:
        m["schedule_date"] = pd.to_datetime(m["schedule_date"], errors="coerce", utc=True)
    m = m.sort_values(["schedule_date","season","week","home_team"]).reset_index(drop=True)
    return m

# ----------------------------
# Historial propio (season actual, semanas finalizadas)
# ----------------------------
def build_self_learning_adjusters(target_season: int) -> tuple[pd.DataFrame, dict]:
    """
    Lee data/live/bets.csv y data/live/odds.csv para semanas ya FINALIZADAS
    y construye pequeños ajustes por equipo y liga (delta de calibración y pnl/unidad).
    Retorna:
      - df con columnas: team, team_calib_delta, team_unit_pnl
      - dict con 'league_calib_delta', 'league_unit_pnl'
    """
    if not (os.path.exists(BETS_OUT_PATH) and os.path.exists(LIVE_ODDS_PATH)):
        return pd.DataFrame(columns=["team","team_calib_delta","team_unit_pnl"]), {"league_calib_delta":0.0,"league_unit_pnl":0.0}

    try:
        bets = pd.read_csv(BETS_OUT_PATH, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=["team","team_calib_delta","team_unit_pnl"]), {"league_calib_delta":0.0,"league_unit_pnl":0.0}

    # Normaliza
    for c in ("season","week"):
        if c in bets.columns: bets[c] = pd.to_numeric(bets[c], errors="coerce").astype("Int64")
    if "week_label" not in bets.columns and "week" in bets.columns:
        bets["week_label"] = bets["week"].apply(week_label_from_num)
    if "schedule_date" in bets.columns:
        bets["schedule_date"] = pd.to_datetime(bets["schedule_date"], errors="coerce", utc=True)
    bets["team"] = bets["team"].astype(str).map(norm_team)
    bets["opponent"] = bets["opponent"].astype(str).map(norm_team)

    # Une con odds para saber resultado si no está
    odds = pd.read_csv(LIVE_ODDS_PATH, low_memory=False)
    for c in ("season","week"):
        if c in odds.columns: odds[c] = pd.to_numeric(odds[c], errors="coerce").astype("Int64")
    if "schedule_date" in odds.columns:
        odds["schedule_date"] = pd.to_datetime(odds["schedule_date"], errors="coerce", utc=True)
    for c in ("home_team","away_team"):
        if c in odds.columns: odds[c] = odds[c].astype(str).map(norm_team)
    if "home_win" not in odds.columns and {"score_home","score_away"}.issubset(odds.columns):
        odds["home_win"] = (pd.to_numeric(odds["score_home"], errors="coerce")
                          > pd.to_numeric(odds["score_away"], errors="coerce")).astype("Int64")

    # mapea resultado para cada pick
    odds_key = ["season","week","home_team","away_team"]
    pick_key = ["season","week","team","opponent","side"]
    # para saber si la pick ganó:
    # - side=home: ganó si home_win==1
    # - side=away: ganó si home_win==0
    dfm = odds.copy()
    dfm["week_label"] = dfm["week"].apply(week_label_from_num)
    if "home_win" not in dfm.columns:
        return pd.DataFrame(columns=["team","team_calib_delta","team_unit_pnl"]), {"league_calib_delta":0.0,"league_unit_pnl":0.0}

    # picks con resultado disponible (solo weeks finalizadas)
    if "week_is_final" in bets.columns:
        bets_fin = bets[bets["week_is_final"].fillna(False).astype(bool)].copy()
    else:
        # Fallback: semanas estrictamente anteriores a la actual
        now = datetime.now(timezone.utc)
        cur_week = int(pd.to_numeric(odds.loc[odds["season"].eq(target_season), "week"], errors="coerce").max() or 0)
        bets_fin = bets[pd.to_numeric(bets["week"], errors="coerce").fillna(0) < cur_week].copy()

    if bets_fin.empty:
        return pd.DataFrame(columns=["team","team_calib_delta","team_unit_pnl"]), {"league_calib_delta":0.0,"league_unit_pnl":0.0}

    # Realiza merge para obtener home_win y las decimales que usamos
    ocols = ["season","week","home_team","away_team","home_win","decimal_home","decimal_away"]
    dfm = dfm[ocols].copy()

    # Une por caso home/away
    bets_fin["home_team_m"] = np.where(bets_fin["side"].astype(str).str.lower().eq("home"), bets_fin["team"], bets_fin["opponent"])
    bets_fin["away_team_m"] = np.where(bets_fin["side"].astype(str).str.lower().eq("away"), bets_fin["team"], bets_fin["opponent"])
    bjoin = bets_fin.merge(dfm, left_on=["season","week","home_team_m","away_team_m"],
                           right_on=["season","week","home_team","away_team"], how="left")

    # Resultado real del pick
    bjoin["won"] = np.where(bjoin["side"].str.lower().eq("home"),
                            (bjoin["home_win"] == 1),
                            (bjoin["home_win"] == 0))
    # Prob de mercado de la pick (sin vigorish normalizado en live)
    bjoin["dec_used"] = np.where(bjoin["side"].str.lower().eq("home"), bjoin["decimal_home"], bjoin["decimal_away"])
    bjoin["p_mkt"] = 1.0 / pd.to_numeric(bjoin["dec_used"], errors="coerce").clip(lower=1e-9)
    # Prob modelo guardada (si existía)
    bjoin["p_model"] = pd.to_numeric(bjoin.get("model_prob", np.nan), errors="coerce")

    # Métricas por equipo (media de (won - p_model) y profit unitario)
    g = bjoin.dropna(subset=["team","won"])
    if g.empty:
        return pd.DataFrame(columns=["team","team_calib_delta","team_unit_pnl"]), {"league_calib_delta":0.0,"league_unit_pnl":0.0}

    g["unit_pnl"] = np.where(g["won"], (g["dec_used"] - 1.0), -1.0)
    # usa p_model si existe; si no, usa p_mkt para delta 0
    g["p_ref"] = g["p_model"].fillna(g["p_mkt"])
    team_stats = (g.groupby("team", as_index=False)
                    .agg(team_calib_delta=("won", lambda x: x.mean()) )
                 )
    # resta p_ref promedio por equipo (si había p_model): calib delta = win_rate - p_model_mean
    per_team_pref = g.groupby("team", as_index=False)["p_ref"].mean().rename(columns={"p_ref":"p_ref_mean"})
    team_stats = team_stats.merge(per_team_pref, on="team", how="left")
    team_stats["team_calib_delta"] = team_stats["team_calib_delta"] - team_stats["p_ref_mean"]
    team_pnl = g.groupby("team", as_index=False)["unit_pnl"].mean().rename(columns={"unit_pnl":"team_unit_pnl"})
    team_adj = team_stats.merge(team_pnl, on="team", how="left")
    team_adj = team_adj.fillna({"team_calib_delta":0.0,"team_unit_pnl":0.0})

    # Liga
    league_calib_delta = (g["won"].mean() - g["p_ref"].mean())
    league_unit_pnl    = g["unit_pnl"].mean()

    # Clip para no sobreajustar
    team_adj["team_calib_delta"] = team_adj["team_calib_delta"].clip(-CFG["SELF_LEARN_MAX_SHIFT"], CFG["SELF_LEARN_MAX_SHIFT"])
    league_calib_delta = float(np.clip(league_calib_delta, -CFG["SELF_LEARN_MAX_SHIFT"], CFG["SELF_LEARN_MAX_SHIFT"]))

    return team_adj[["team","team_calib_delta","team_unit_pnl"]], {
        "league_calib_delta": float(league_calib_delta),
        "league_unit_pnl": float(league_unit_pnl)
    }

def inject_self_features(master_all: pd.DataFrame, team_adj: pd.DataFrame, league_adj: dict) -> pd.DataFrame:
    """Inyecta features suaves de calibración/beneficio por equipo y liga."""
    df = master_all.copy()
    # liga
    df["league_calib_delta"] = float(league_adj.get("league_calib_delta", 0.0))
    df["league_unit_pnl"]    = float(league_adj.get("league_unit_pnl", 0.0))
    # por equipo (home/away)
    if not team_adj.empty:
        df = df.merge(team_adj.rename(columns={"team":"home_team", "team_calib_delta":"home_team_calib_delta",
                                               "team_unit_pnl":"home_team_unit_pnl"}),
                      on="home_team", how="left")
        df = df.merge(team_adj.rename(columns={"team":"away_team", "team_calib_delta":"away_team_calib_delta",
                                               "team_unit_pnl":"away_team_unit_pnl"}),
                      on="away_team", how="left")
    for c in ["home_team_calib_delta","away_team_calib_delta","home_team_unit_pnl","away_team_unit_pnl"]:
        if c not in df.columns: df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

# ----------------------------
# Modelo + métricas
# ----------------------------
def run_model(master_all: pd.DataFrame, target_season: int):
    df = master_all.copy()

    # Fallback home_line
    def compute_home_line(r):
        s  = r.get("spread_favorite", np.nan)
        tf = str(r.get("team_favorite_id", ""))
        if pd.isna(s) or tf == "" or tf == "nan":
            return r.get("home_line", np.nan)
        if "home_team" in r and tf == r["home_team"]:
            return float(s)
        if "away_team" in r and tf == r["away_team"]:
            return float(-s)
        return r.get("home_line", np.nan)

    if "home_line" not in df.columns or df["home_line"].isna().any():
        df["home_line"] = df.apply(compute_home_line, axis=1)

    df["abs_spread"] = pd.to_numeric(df["home_line"], errors="coerce").astype(float).abs()
    df["fav_home"]   = (pd.to_numeric(df["home_line"], errors="coerce").astype(float) < 0).astype(int)
    df["ou"]         = pd.to_numeric(df.get("over_under_line", np.nan), errors="coerce").astype(float)
    df["spread_x_ou"]   = df["abs_spread"] * df["ou"]
    df["fav_x_spread"]  = df["fav_home"]   * df["abs_spread"]
    df["home_line_sq"]  = df["home_line"]  * df["home_line"]

    # Features pregame + mercado
    pre_cols = [c for c in df.columns if re.search(r'(?:_pre(_ewm)?|_pre_ytd|_pre_l8)$', c)]
    mkt_cols = [c for c in ["home_line","abs_spread","fav_home","ou","spread_x_ou","fav_x_spread","home_line_sq"] if c in df.columns]

    # Self-learn (equipo/liga)
    self_cols = [c for c in ["league_calib_delta","league_unit_pnl",
                             "home_team_calib_delta","away_team_calib_delta",
                             "home_team_unit_pnl","away_team_unit_pnl"] if c in df.columns]

    feat_cols = [c for c in (pre_cols + mkt_cols + self_cols) if c in df.columns]

    hist_mask = df["season"] < target_season
    hist_years = sorted(df.loc[hist_mask, "season"].dropna().unique().tolist())
    if len(hist_years) < 2:
        raise RuntimeError("Se requieren al menos 2 temporadas históricas para entrenar/validar.")

    if len(hist_years) >= 4:
        train_years = hist_years[-4:-1]; val_year = hist_years[-1]
    else:
        train_years = hist_years[:-1];   val_year = hist_years[-1]
    test_year  = target_season

    base = df.dropna(subset=["home_line"]).copy()
    train_df = base[base["season"].isin(train_years)].copy()
    val_df   = base[base["season"].eq(val_year)].copy()
    test_df  = base[base["season"].eq(test_year)].copy()

    # Etiquetas
    if "home_win" not in base.columns:
        raise RuntimeError("Falta columna home_win en odds master para calcular métricas.")

    y_train = train_df["home_win"].dropna().astype(int).values
    y_val   = val_df["home_win"].dropna().astype(int).values

    X_train = train_df[feat_cols].copy()
    X_val   = val_df[feat_cols].copy()
    X_test  = test_df[feat_cols].copy()

    train_meds = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_meds); X_val = X_val.fillna(train_meds); X_test = X_test.fillna(train_meds)

    monotonic_cst = [(-1 if c == "home_line" else 0) for c in feat_cols]

    hgb = HistGradientBoostingClassifier(
        learning_rate=0.06, max_leaf_nodes=21, min_samples_leaf=60,
        l2_regularization=2.0, max_iter=320, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=40,
        monotonic_cst=monotonic_cst, random_state=42
    )
    hgb.fit(X_train, y_train)

    p_tr = hgb.predict_proba(X_train)[:,1]
    p_va = hgb.predict_proba(X_val)[:,1]
    p_te = hgb.predict_proba(X_test)[:,1] if len(X_test) else np.array([])

    # Calibración
    bin_edges = [-21, -10.5, -6.5, -3.5, -0.5, 0.5, 3.5, 6.5, 10.5, 21]
    cut_bins = lambda x: pd.cut(x, bins=bin_edges, include_lowest=True)
    val_bins  = cut_bins(val_df["home_line"])
    test_bins = cut_bins(test_df["home_line"]) if len(test_df) else pd.Series(dtype="category")

    iso_global = IsotonicRegression(out_of_bounds="clip")
    iso_global.fit(p_va, y_val)

    bin_min = 120
    iso_by_bin = {}
    for b in val_bins.unique():
        idx = (val_bins == b)
        if idx.sum() >= bin_min:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_va[idx], y_val[idx]); iso_by_bin[b] = iso

    def apply_bin_calibration(p_raw, bins_series):
        if len(p_raw) == 0: return p_raw
        if not iso_by_bin: return np.clip(iso_global.transform(p_raw), 1e-6, 1-1e-6)
        out = np.empty_like(p_raw, dtype=float)
        for b in bins_series.unique():
            m = (bins_series == b).values
            out[m] = (iso_by_bin[b].transform(p_raw[m]) if b in iso_by_bin else iso_global.transform(p_raw[m]))
        return np.clip(out, 1e-6, 1-1e-6)

    p_tr_cal = iso_global.transform(p_tr)
    p_va_cal = apply_bin_calibration(p_va, val_bins)
    p_te_cal = apply_bin_calibration(p_te, test_bins)

    # Priors simples por spread
    prior_lr = LogisticRegression(C=2.0, solver="liblinear", max_iter=200)
    prior_lr.fit(train_df[["home_line"]], y_train)
    p_tr_sp = prior_lr.predict_proba(train_df[["home_line"]])[:,1]
    p_va_sp = prior_lr.predict_proba(val_df[["home_line"]])[:,1]
    p_te_sp = prior_lr.predict_proba(test_df[["home_line"]])[:,1] if len(test_df) else np.array([])

    # Meta LR
    def meta_matrix(df_in, p_hgb_cal, p_spread):
        M = pd.DataFrame({
            "p_hgb":   p_hgb_cal,
            "p_sp":    p_spread,
            "home_line": df_in["home_line"].values,
            "abs_spread": df_in["abs_spread"].values,
            "ou": df_in["ou"].values
        })
        return M.fillna(M.median(numeric_only=True))

    meta_tr   = meta_matrix(train_df, p_tr_cal, p_tr_sp)
    meta_val  = meta_matrix(val_df,   p_va_cal, p_va_sp)
    meta_test = meta_matrix(test_df,  p_te_cal, p_te_sp) if len(p_te_cal) else pd.DataFrame()

    meta_lr = LogisticRegression(C=1.0, solver="liblinear", max_iter=300)
    meta_lr.fit(meta_val, y_val)

    p_tr_meta = np.clip(meta_lr.predict_proba(meta_tr)[:,1], 1e-6, 1-1e-6)
    p_va_meta = np.clip(meta_lr.predict_proba(meta_val)[:,1], 1e-6, 1-1e-6)
    p_te_meta = (np.clip(meta_lr.predict_proba(meta_test)[:,1], 1e-6, 1-1e-6)
                 if len(meta_test) else np.array([]))

    # ----- Métricas -----
    def safe_metrics(y_true, p):
        if len(y_true)==0 or len(p)==0: return dict(acc=np.nan, auc=np.nan, logloss=np.nan)
        try:
            return dict(
                acc = float(accuracy_score(y_true, (p>=0.5).astype(int))),
                auc = float(roc_auc_score(y_true, p)),
                logloss = float(log_loss(y_true, p, labels=[0,1]))
            )
        except Exception:
            return dict(acc=np.nan, auc=np.nan, logloss=np.nan)

    m_train = safe_metrics(y_train, p_tr_meta)
    m_val   = safe_metrics(y_val,   p_va_meta)

    # Test parcial (solo juegos finalizados del año actual)
    y_test = test_df["home_win"].dropna().astype(int).values
    if len(y_test) > 0 and len(p_te_meta) == len(test_df):
        # filtra los que ya tienen y_test (algunos test aún no han jugado)
        mask_fin = test_df["home_win"].notna().values
        m_test = safe_metrics(y_test, p_te_meta[mask_fin])
    else:
        m_test = dict(acc=np.nan, auc=np.nan, logloss=np.nan)

    print("Model Results:")
    print(f"train | ACC {m_train['acc']:.3f} | ROC_AUC {m_train['auc']:.3f} | LOGLOSS {m_train['logloss']:.3f}")
    print(f"val   | ACC {m_val['acc']:.3f} | ROC_AUC {m_val['auc']:.3f} | LOGLOSS {m_val['logloss']:.3f}")
    if not math.isnan(m_test['acc']):
        print(f"test* | ACC {m_test['acc']:.3f} | ROC_AUC {m_test['auc']:.3f} | LOGLOSS {m_test['logloss']:.3f}")
    else:
        print("test* | (sin suficientes juegos finalizados del año para medir)")

    # Preds para temporada target
    test_df = test_df.copy()
    test_df["week_label"] = test_df["week"].apply(week_label_from_num)
    test_preds = test_df[["season","week","week_label","home_team","away_team"]].copy()
    test_preds["p_home_win_meta"] = p_te_meta[:len(test_df)] if len(p_te_meta) else np.array([])
    return test_preds

# ----------------------------
# EV, selección
# ----------------------------
def sanitize_ev(df: pd.DataFrame, cfg) -> pd.DataFrame:
    need = ["season","week","week_label","schedule_date","side","team","opponent","decimal_odds","model_prob"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"EV table missing columns: {missing}")

    ev = df.copy()
    ev["schedule_date"] = pd.to_datetime(ev["schedule_date"], errors="coerce", utc=True)
    ev["model_prob"] = pd.to_numeric(ev["model_prob"], errors="coerce").astype(float).clip(MIN_PROB, 1-MIN_PROB)
    ev["decimal_odds"] = pd.to_numeric(ev["decimal_odds"], errors="coerce").astype(float)

    ev = add_week_order(ev)
    ev["game_id"] = ev.apply(lambda r: make_game_id_row(r["week_label"], r["team"], r["opponent"]), axis=1)

    ev = ev[ev["decimal_odds"].between(cfg["ODDS_MIN"], cfg["ODDS_MAX"])]
    ev = ev[np.abs(ev["model_prob"] - 0.5) >= cfg["CONF_MIN"]]
    if cfg["SKIP_W1_W2"]:
        ev = ev[~ev["week_label"].isin(["Week 1","Week 2"])]

    ev = ev.sort_values(["week_order","schedule_date","game_id"]).reset_index(drop=True)
    return ev

def devig_per_game(ev_df: pd.DataFrame, single_side_mode: str) -> pd.DataFrame:
    ev = ev_df.copy()
    if "market_prob_nv" not in ev.columns:
        ev["market_prob_nv"] = ev["decimal_odds"].apply(implied_from_decimal)

    tmp = ev.copy()
    tmp["p_raw"] = 1.0 / tmp["decimal_odds"].clip(1e-9)
    sides_per_game = tmp.groupby("game_id")["side"].transform("nunique")
    sum_p = tmp.groupby("game_id")["p_raw"].transform("sum")

    mask_pairs = sides_per_game >= 2
    ev.loc[mask_pairs, "market_prob_nv"] = (tmp.loc[mask_pairs, "p_raw"] / sum_p[mask_pairs]).clip(1e-6, 1-1e-6)

    ev["market_prob_nv"] = ev["market_prob_nv"].clip(1e-6, 1-1e-6)
    return ev

def ev_floor(decimal_odds, ev_base_min, ev_slope):
    d = float(decimal_odds)
    return ev_base_min + max(0.0, d - 2.0) * ev_slope

def ev_slope_for_row(row, cfg):
    d = float(row["decimal_odds"])
    b = cfg["BANDS"]
    if d < b["FAV"]["odds_lt"]: return b["FAV"]["ev_slope"]
    if d < b["MID"]["odds_lt"]: return b["MID"]["ev_slope"]
    return b["DOG"]["ev_slope"]

def apply_band_rules(df: pd.DataFrame, cfg) -> pd.DataFrame:
    b = cfg.get("BANDS", {})
    if not b: return df
    x = df.copy()
    d = x["decimal_odds"].astype(float)
    conf = np.abs(x["model_prob"] - 0.5)
    edge = x["edge"].astype(float)
    keep = pd.Series(True, index=x.index)

    if "FAV" in b:
        m = d < b["FAV"]["odds_lt"]
        keep &= (~m) | ((edge >= b["FAV"]["tau"]) & (conf >= b["FAV"]["conf"]))
    if "MID" in b:
        m = (d >= b["MID"]["odds_ge"]) & (d < b["MID"]["odds_lt"])
        keep &= (~m) | ((edge >= b["MID"]["tau"]) & (conf >= b["MID"]["conf"]))
    if "DOG" in b:
        m = (d >= b["DOG"]["odds_ge"]) & (d <= b["DOG"]["odds_le"])
        keep &= (~m) | ((edge >= b["DOG"]["tau"]) & (conf >= b["DOG"]["conf"]))
        if "extra_if_ge3_10" in b["DOG"]:
            m2 = d >= 3.10
            keep &= (~m2) | ((edge >= b["DOG"]["extra_if_ge3_10"]["tau"]) & (conf >= b["DOG"]["extra_if_ge3_10"]["conf"]))
    return x[keep].reset_index(drop=True)

def pick_per_game_best(ev: pd.DataFrame, cfg) -> pd.DataFrame:
    tau = max(cfg["EDGE_TAU"], cfg["EDGE_TAU_MIN"])
    def ok_ev(row):
        slope = ev_slope_for_row(row, cfg)
        return row["ev"] >= ev_floor(row["decimal_odds"], cfg["EV_BASE_MIN"], slope)
    c = ev[ev["edge"] >= tau].copy()
    if c.empty: return c
    c = c[c.apply(ok_ev, axis=1)]
    c = (c.sort_values(["week_order","schedule_date","game_id","edge"], ascending=[True,True,True,False])
           .drop_duplicates("game_id"))
    c = apply_band_rules(c, cfg)

    # Respeta techos
    if cfg.get("MAX_BIG_DOGS_PER_WEEK"):
        out = []
        for wk, g in c.groupby("week_label", sort=False):
            m_big = g["decimal_odds"] >= 3.10
            if m_big.sum() > cfg["MAX_BIG_DOGS_PER_WEEK"]:
                g_big = g[m_big].sort_values("edge", ascending=False).head(cfg["MAX_BIG_DOGS_PER_WEEK"])
                g_rest = g[~m_big]
                g = pd.concat([g_rest, g_big], ignore_index=True)
            out.append(g)
        c = pd.concat(out, ignore_index=True)

    if cfg.get("MAX_BETS_PER_WEEK"):
        out = []
        for wk, g in c.groupby("week_label", sort=False):
            g = g.sort_values(["ev","edge"], ascending=[False,False]).head(cfg["MAX_BETS_PER_WEEK"])
            out.append(g)
        c = pd.concat(out, ignore_index=True)
    return c.reset_index(drop=True)

def relax_until_minimum(c_base: pd.DataFrame, ev_base: pd.DataFrame, cfg) -> pd.DataFrame:
    """Si alguna semana tiene < MIN picks, relaja progresivamente edge/conf/ev floor."""
    if c_base.empty: return c_base
    def count_by_week(df): return df.groupby("week_label", sort=False).size().to_dict()

    picked = count_by_week(c_base)
    need_min = cfg["MIN_PICKS_PER_WEEK"]
    weeks = list(ev_base["week_label"].astype(str).unique())
    weeks_needing = [w for w in weeks if picked.get(w, 0) < need_min]

    if not weeks_needing:
        return c_base

    cur = c_base.copy()
    base_tau   = cfg["EDGE_TAU"]
    base_conf  = cfg["CONF_MIN"]
    base_evmin = cfg["EV_BASE_MIN"]

    dprint("Relaxation: weeks below min ->", weeks_needing)
    for step in cfg.get("RELAX_STEPS", []):
        cfg["EDGE_TAU"]   = max(cfg["EDGE_TAU_MIN"], base_tau   + step["EDGE_TAU"])
        cfg["CONF_MIN"]   = max(0.0,                base_conf  + step["CONF_MIN"])
        cfg["EV_BASE_MIN"]= max(0.0,                base_evmin + step["EV_DOWN"])
        dprint(f"Relax step -> EDGE_TAU={cfg['EDGE_TAU']:.3f} | CONF_MIN={cfg['CONF_MIN']:.3f} | EV_BASE_MIN={cfg['EV_BASE_MIN']:.3f}")

        cand = pick_per_game_best(ev_base, cfg)
        picked = count_by_week(cand)
        weeks_needing = [w for w in weeks if picked.get(w, 0) < need_min]
        cur = cand
        if not weeks_needing:
            break

    # Restaura límites base para no contaminar próximas semanas/ejecuciones
    cfg["EDGE_TAU"]   = base_tau
    cfg["CONF_MIN"]   = base_conf
    cfg["EV_BASE_MIN"]= base_evmin

    return cur

# -------- Kelly fraccional + caps ----------
def kelly_fraction_scaled(edge, cfg):
    wd = cfg.get("KELLY_EDGE_WEIGHT")
    if not wd: return cfg["KELLY_FRACTION"]
    scale = np.clip(edge / wd["EDGE_REF"], wd["MIN_SCALE"], wd["MAX_SCALE"])
    return cfg["KELLY_FRACTION"] * float(scale)

def kelly_stake(bk_week0, p, d, edge, cfg):
    b = max(d - 1.0, 1e-9)
    f_star = (b*p - (1-p)) / b
    f_base = max(0.0, f_star)
    f_adj  = kelly_fraction_scaled(edge, cfg)
    stake = bk_week0 * f_base * f_adj
    stake = min(stake, bk_week0*cfg["KELLY_PCT_CAP"], cfg["ABS_STAKE_CAP"])
    return float(np.floor(stake*100)/100.0)

def plan_pregame_stakes(bets_in: pd.DataFrame, cfg, bk_by_week: dict):
    """Asigna stake pregame por semana usando BK0 de bk_by_week. NO escribe columna bankroll."""
    if bets_in.empty:
        return bets_in.assign(stake=0.0, profit=np.nan), pd.DataFrame(), {
            "bets_planned": 0, "total_planned_stake": 0.0, "bankroll_assumed": None
        }

    data = add_week_order(bets_in).sort_values(["week_order","schedule_date","game_id"]).reset_index(drop=True)

    stakes = []
    for _, r in data.iterrows():
        BK0 = float(bk_by_week.get(r["week_label"], cfg["INITIAL_BANKROLL"]))
        st = kelly_stake(BK0, float(r["model_prob"]), float(r["decimal_odds"]), float(r["edge"]), cfg)
        stakes.append(st)

    out = data.copy()
    out["stake"] = np.round(stakes, 2)
    out["profit"] = np.nan

    wk_summary = (out.groupby("week_label", sort=False)
                    .agg(planned_stake=("stake","sum"))
                    .reset_index())

    summary = {
        "bets_planned": int(len(out)),
        "total_planned_stake": float(np.round(out["stake"].sum(), 2)),
        "bankroll_assumed": {wk: float(bk_by_week.get(wk, cfg["INITIAL_BANKROLL"])) for wk in out["week_label"].unique()}
    }
    return out, wk_summary, summary

# ----------------------------
# APPEND/UPSERT + bankroll_week_final
# ----------------------------
APPEND_KEYS = ["season","week","game_id","side"]

def build_bk_by_week(existing: pd.DataFrame, planned_weeks: list, cfg) -> dict:
    """Usa bankroll_week_final de semanas previas como BK0. Si no hay, usa INITIAL."""
    bk_by_week = {}
    if existing is None or existing.empty or "bankroll_week_final" not in existing.columns:
        for wk in planned_weeks: bk_by_week[wk] = cfg["INITIAL_BANKROLL"]
        return bk_by_week

    ex = existing.copy()
    if "week_label" not in ex.columns and "week" in ex.columns:
        ex["week_label"] = ex["week"].apply(week_label_from_num)
    if "week_order" not in ex.columns:
        ex["week_order"] = ex["week_label"].map(ORDER_INDEX).fillna(999).astype(int)

    wk2final = (ex.dropna(subset=["bankroll_week_final"])
                  .sort_values(["week_order"])
                  .groupby("week_label")["bankroll_week_final"]
                  .last()
                  .to_dict())

    for wk in planned_weeks:
        wk_ord = ORDER_INDEX.get(str(wk), 999)
        prev_vals = [(wo, val) for wl, val in wk2final.items()
                     if ORDER_INDEX.get(str(wl), 999) < wk_ord
                     for wo in [ORDER_INDEX.get(str(wl), 999)]]
        if prev_vals:
            prev_vals.sort(key=lambda x: x[0], reverse=True)
            bk_by_week[wk] = float(prev_vals[0][1])
        else:
            bk_by_week[wk] = cfg["INITIAL_BANKROLL"]
    return bk_by_week

def upsert_current_weeks(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Mantiene semanas finalizadas intactas.
    Sobrescribe/añade solo semanas NO finalizadas (p.ej., semana actual).
    """
    if existing is None or existing.empty:
        return new_rows.copy()

    ex = existing.copy()
    if "week_label" not in ex.columns and "week" in ex.columns:
        ex["week_label"] = ex["week"].apply(week_label_from_num)

    # Determina qué semanas están 'finalizadas'
    finished_weeks = set(ex.loc[ex.get("week_is_final", False)==True, "week_label"].astype(str).unique())
    modifiable_weeks = set(new_rows["week_label"].astype(str).unique()) - finished_weeks

    dprint("Weeks finalizadas (frozen):", sorted(finished_weeks))
    dprint("Weeks modificables (upsert):", sorted(modifiable_weeks))

    if not modifiable_weeks:
        dprint("No hay semanas modificables; se conserva el CSV tal cual.")
        return ex

    # Elimina filas existentes de semanas modificables que colisionen por clave
    mask_keep = ~ex["week_label"].isin(modifiable_weeks)
    kept = ex[mask_keep].copy()

    ex_mod = ex[ex["week_label"].isin(modifiable_weeks)]
    have = set(map(tuple, ex_mod[APPEND_KEYS].astype(object).to_numpy().tolist())) if not ex_mod.empty else set()
    new_key_list = list(map(tuple, new_rows[APPEND_KEYS].astype(object).to_numpy().tolist()))
    mask_new = ~pd.Series(new_key_list).isin(have)
    n_remove = len(ex_mod)
    dprint(f"Upsert: se eliminarán {n_remove} filas existentes de semanas modificables por clave match (upsert).")

    only_new = new_rows[mask_new.values].copy()
    combined = pd.concat([kept, only_new], ignore_index=True)
    return combined

# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(os.path.dirname(BETS_OUT_PATH), exist_ok=True)

    target_season = getenv_int("TARGET_SEASON", detect_target_season())
    target_week_override = getenv_int_or_none("TARGET_WEEK")
    dprint("Resolved -> target_season:", target_season, "| TARGET_WEEK override:", target_week_override)

    # Carga odds y stats
    odds_cur = load_current_odds()
    stats_all = load_pregame_stats_for_seasons(target_season)

    # Odds solo de temporada target (+ semana si override)
    odds_cur = odds_cur[odds_cur["season"].eq(target_season)].copy()
    if target_week_override is not None:
        odds_cur = odds_cur[odds_cur["week"].eq(target_week_override)].copy()

    # Master target + histórico
    # Cargar odds históricas (4 temporadas previas)
    hist_frames = []
    for y in range(target_season-4, target_season):
        p = os.path.join(ARCHIVE_DIR, f"season={y}", "odds.csv")
        if os.path.exists(p):
            tmp = pd.read_csv(p, low_memory=False)
            for c in ("season","week"):
                if c in tmp.columns: tmp[c] = pd.to_numeric(tmp[c], errors="coerce").astype("Int64")
            if "schedule_date" in tmp.columns:
                tmp["schedule_date"] = pd.to_datetime(tmp["schedule_date"], errors="coerce", utc=True)
            for c in ("home_team","away_team"):
                if c in tmp.columns: tmp[c] = tmp[c].astype(str).map(norm_team)
            if "home_win" not in tmp.columns and {"score_home","score_away"}.issubset(tmp.columns):
                tmp["home_win"] = (pd.to_numeric(tmp["score_home"], errors="coerce")
                                  > pd.to_numeric(tmp["score_away"], errors="coerce")).astype("Int64")
            hist_frames.append(tmp)

    team_adj, league_adj = build_self_learning_adjusters(target_season)
    if not team_adj.empty or league_adj.get("league_calib_delta", 0.0) != 0.0:
        dprint("Self-learn adjusters -> teams rows:", len(team_adj), "| league:", league_adj)

    master_target = build_master(odds_cur, stats_all)
    master_target = inject_self_features(master_target, team_adj, league_adj)

    master_hist = pd.DataFrame()
    if hist_frames:
        odds_hist = pd.concat(hist_frames, ignore_index=True)
        master_hist = build_master(odds_hist, stats_all)
        # en histórico no aplicamos self-learn de esta temporada (no existía)
        master_hist["league_calib_delta"]    = 0.0
        master_hist["league_unit_pnl"]       = 0.0
        master_hist["home_team_calib_delta"] = 0.0
        master_hist["away_team_calib_delta"] = 0.0
        master_hist["home_team_unit_pnl"]    = 0.0
        master_hist["away_team_unit_pnl"]    = 0.0

    master_all = pd.concat([master_hist, master_target], ignore_index=True) if not master_hist.empty else master_target.copy()

    dprint("Master target rows:", len(master_target))
    # Auditoría de presencia de columnas pregame
    pre_cols_sample = [c for c in master_target.columns if c.endswith(("_pre_ytd","_pre_ewm","_pre_l8"))]
    dprint("Audit: columnas pregame presentes:", len(pre_cols_sample), "OK" if len(pre_cols_sample)>0 else "MISSING")
    miss_home = master_target.filter(regex="^home_.*_pre_").isna().all(axis=1).sum()
    miss_away = master_target.filter(regex="^away_.*_pre_").isna().all(axis=1).sum()
    dprint(f"Audit: filas sin pregame (home)={miss_home} / (away)={miss_away} de {len(master_target)}")
    dprint("Audit: 'week' NA rows:", master_target["week"].isna().sum())

    # Modelo + métricas
    test_preds = run_model(master_all, target_season)

    # Unir probs con odds target
    cols_key = ["season","week","home_team","away_team"]
    need_cols = cols_key + ["week_label","schedule_date",
                            "decimal_home","decimal_away","ml_home","ml_away",
                            "market_prob_home_nv","market_prob_away_nv","home_line","spread_favorite","over_under_line"]
    merge_df = master_target[[c for c in need_cols if c in master_target.columns]].copy()
    if "week_label" not in merge_df.columns:
        merge_df["week_label"] = merge_df["week"].apply(week_label_from_num)

    pred = test_preds[["season","week","home_team","away_team","p_home_win_meta"]].copy()
    dfm  = merge_df.merge(pred, on=cols_key, how="inner")

    # EV home/away
    homes = pd.DataFrame({
        "season": dfm["season"], "week": dfm["week"], "week_label": dfm["week_label"], "schedule_date": dfm["schedule_date"],
        "side":"home", "team": dfm["home_team"], "opponent": dfm["away_team"],
        "decimal_odds": dfm.get("decimal_home", np.nan), "ml": dfm.get("ml_home", np.nan),
        "market_prob_nv": dfm.get("market_prob_home_nv", np.nan),
        "model_prob": dfm["p_home_win_meta"]
    })
    aways = pd.DataFrame({
        "season": dfm["season"], "week": dfm["week"], "week_label": dfm["week_label"], "schedule_date": dfm["schedule_date"],
        "side":"away", "team": dfm["away_team"], "opponent": dfm["home_team"],
        "decimal_odds": dfm.get("decimal_away", np.nan), "ml": dfm.get("ml_away", np.nan),
        "market_prob_nv": dfm.get("market_prob_away_nv", np.nan),
        "model_prob": 1.0 - dfm["p_home_win_meta"]
    })
    ev_predictions = pd.concat([homes, aways], ignore_index=True)

    # Completar ML si falta
    if "ml" in ev_predictions.columns and "decimal_odds" in ev_predictions.columns:
        m = ev_predictions["ml"].isna() & ev_predictions["decimal_odds"].notna()
        ev_predictions.loc[m, "ml"] = ev_predictions.loc[m, "decimal_odds"].apply(decimal_to_american)

    ev_predictions = ev_predictions[pd.to_numeric(ev_predictions["decimal_odds"], errors="coerce").notna()].copy()
    ev_predictions["decimal_odds"] = ev_predictions["decimal_odds"].astype(float)

    # EV / edge
    ev_predictions["ev"]   = ev_predictions["model_prob"]*(ev_predictions["decimal_odds"]-1) - (1-ev_predictions["model_prob"])
    ev_predictions = ev_predictions.sort_values(["schedule_date","week","team","side"]).reset_index(drop=True)

    # Limpieza/sanitizado y devig
    ev_base = sanitize_ev(ev_predictions, CFG)
    ev_base["game_id"] = ev_base.apply(lambda r: make_game_id_row(r["week_label"], r["team"], r["opponent"]), axis=1)
    ev_devig = devig_per_game(ev_base.copy(), single_side_mode=CFG["DEVIG_SINGLE_SIDE"])

    p = ev_devig["model_prob"].astype(float)
    d = ev_devig["decimal_odds"].astype(float)
    mkt = ev_devig["market_prob_nv"].astype(float)
    ev_devig["edge"] = p - mkt
    ev_devig["ev"]   = p*(d-1) - (1-p)

    # 1) Selección base
    cand_base = pick_per_game_best(ev_devig, CFG)

    # 2) Relajar si no se cumple mínimo
    cand = relax_until_minimum(cand_base, ev_devig, CFG)

    # --------- BANKROLL POR SEMANA (desde bets.csv existente) ----------
    if os.path.exists(BETS_OUT_PATH):
        try:
            existing = pd.read_csv(BETS_OUT_PATH, low_memory=False)
            if "schedule_date" in existing.columns:
                existing["schedule_date"] = pd.to_datetime(existing["schedule_date"], errors="coerce", utc=True)
            if "week_label" not in existing.columns and "week" in existing.columns:
                existing["week_label"] = existing["week"].apply(week_label_from_num)
        except Exception as e:
            dprint("WARN: no se pudo leer bets.csv existente; se asumirá INITIAL_BANKROLL. err:", e)
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    planned_weeks = sorted(cand["week_label"].astype(str).unique(), key=lambda wl: ORDER_INDEX.get(wl, 999))
    bk_by_week = build_bk_by_week(existing, planned_weeks, CFG)
    dprint("BK0 por semana (usando bankroll_week_final previo o fallback):", bk_by_week)

    # --------- PLAN PREGAME (usando BK0 por semana) ----------
    planned_bets_raw = cand.copy()
    planned_bets, weekly_plan, S = plan_pregame_stakes(planned_bets_raw, CFG, bk_by_week)

    # ----------------------------
    # UPSERT (no tocar semanas finalizadas) y ESCRITURA SIN 'bankroll'
    # ----------------------------
    # Asegura columnas mínimas
    for c in ["bankroll","bankroll_after","bankroll_week_final"]:
        if c in planned_bets.columns:
            planned_bets = planned_bets.drop(columns=[c])

    if existing is not None and not existing.empty:
        for c in ["bankroll","bankroll_after","bankroll_week_final"]:
            if c in existing.columns:
                existing = existing.drop(columns=[c])

    # upsert solo semanas no finalizadas
    combined = upsert_current_weeks(existing, planned_bets)

    combined.to_csv(BETS_OUT_PATH, index=False)
    print(f"[select_bets] upsert wrote {BETS_OUT_PATH} | prev_rows={len(existing)} | total={len(combined)} | weeks_upserted={sorted(planned_weeks)}")

if __name__ == "__main__":
    main()

