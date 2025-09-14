# nfl_dash/odds_scores.py
from __future__ import annotations
import pandas as pd
from pathlib import Path

from .paths import ARCHIVE_DIR, LIVE_DIR
from .utils import norm_abbr, week_label_from_num

def _read_odds_for_season(year: int) -> pd.DataFrame:
    """Carga odds para la temporada dada:
       - Si existe en archive/season=YYYY/odds.csv lo usa.
       - Si no, intenta data/live/odds.csv (para la temporada actual).
       Normaliza tipos y equipos."""
    # 1) archive primero
    arch = ARCHIVE_DIR / f"season={year}" / "odds.csv"
    if arch.exists():
        df = pd.read_csv(arch, low_memory=False)
    else:
        # 2) live como fallback
        live = LIVE_DIR / "odds.csv"
        if live.exists():
            df = pd.read_csv(live, low_memory=False)
        else:
            return pd.DataFrame()

    # Normalizaciones suaves
    for c in ("season", "week"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Asegura week_label
    if "week_label" not in df.columns and "week" in df.columns:
        df["week_label"] = df["week"].apply(week_label_from_num).astype(str)
    if "week_label" in df.columns:
        df["week_label"] = df["week_label"].astype(str)

    # Normaliza equipos si existen
    for c in ("home_team", "away_team"):
        if c in df.columns:
            df[c] = df[c].astype(str).map(norm_abbr)

    # Convierte moneylines/decimales a numéricos si existen
    num_cols = [
        "ml_home","ml_away","decimal_home","decimal_away",
        "ml_home_raw","ml_away_raw","decimal_home_raw","decimal_away_raw",
        "spread_home","spread_away","over_under_line",
        "score_home","score_away",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _canonicalize_bets_columns(bets: pd.DataFrame) -> pd.DataFrame:
    """Genera columnas estándar en el ledger para poder unir con odds."""
    view = bets.copy()

    # season
    if "season" in view.columns:
        view["season"] = pd.to_numeric(view["season"], errors="coerce")

    # week_label (si solo hay 'week')
    if "week_label" not in view.columns and "week" in view.columns:
        view["week_label"] = view["week"].apply(week_label_from_num).astype(str)
    if "week_label" in view.columns:
        view["week_label"] = view["week_label"].astype(str)

    # Equipos: si no trae home/away, intenta derivarlos de team/opponent
    if "home_team" not in view.columns or "away_team" not in view.columns:
        team     = view.get("team")
        opponent = view.get("opponent")
        side     = view.get("side")

        if team is not None and opponent is not None:
            # Si hay 'side', usa esa pista. Si no, asume team=home (mejor que nada).
            if side is not None:
                side_u = side.astype(str).str.lower()
                # cuando side indica 'home'/'away'
                is_home = side_u.isin(["home","h","local","house"])
                is_away = side_u.isin(["away","a","visitor","visitante"])

                view["home_team"] = view.get("home_team", pd.Series(index=view.index))
                view["away_team"] = view.get("away_team", pd.Series(index=view.index))

                # donde sea home -> team es home
                view.loc[is_home, "home_team"] = team[is_home]
                view.loc[is_home, "away_team"] = opponent[is_home]
                # donde sea away -> team es away
                view.loc[is_away, "home_team"] = opponent[is_away]
                view.loc[is_away, "away_team"] = team[is_away]

                # resto (sin pista): asume team=home
                mask_unknown = ~(is_home | is_away)
                view.loc[mask_unknown, "home_team"] = team[mask_unknown]
                view.loc[mask_unknown, "away_team"] = opponent[mask_unknown]
            else:
                view["home_team"] = team
                view["away_team"] = opponent

    # Normaliza equipos
    for c in ("home_team","away_team","team","opponent"):
        if c in view.columns:
            view[c] = view[c].astype(str).map(norm_abbr)

    return view


def _choose_merge_keys(bets: pd.DataFrame, odds: pd.DataFrame) -> list[str]:
    """Elige llaves de unión confiables según columnas disponibles."""
    if "event_id" in bets.columns and "event_id" in odds.columns:
        return ["event_id"]
    # Preferimos (season, week_label, home_team, away_team)
    candidates = ["season", "week_label", "home_team", "away_team"]
    keys = [c for c in candidates if c in bets.columns and c in odds.columns]
    if len(keys) >= 3:
        return keys
    # Alternativa con team/opponent
    candidates2 = ["season", "week_label", "team", "opponent"]
    keys2 = [c for c in candidates2 if c in bets.columns and c in odds.columns]
    if len(keys2) >= 3:
        return keys2
    # Último recurso: (season, schedule_date, home/away) si existe
    candidates3 = ["season", "schedule_date", "home_team", "away_team"]
    keys3 = [c for c in candidates3 if c in bets.columns and c in odds.columns]
    if len(keys3) >= 2:
        return keys3
    # Si no hay combinaciones suficientes, devolvemos lo que haya en común
    commons = [c for c in bets.columns if c in odds.columns]
    return commons[:1]  # evita crashear


def enrich_bets_with_scores(bets: pd.DataFrame, season: int) -> pd.DataFrame:
    """Une el ledger con odds/scores de esa temporada sin romper si faltan columnas."""
    if bets is None or bets.empty:
        return bets

    view = _canonicalize_bets_columns(bets)
    odds = _read_odds_for_season(season)

    if odds.empty:
        # No hay archivo de odds disponible; devuelve lo mismo
        return view

    keys = _choose_merge_keys(view, odds)

    # Columnas que sería ideal traer si existen
    want_cols = [
        "ml_home","ml_away","decimal_home","decimal_away",
        "ml_home_raw","ml_away_raw","decimal_home_raw","decimal_away_raw",
        "spread_home","spread_away","over_under_line",
        "score_home","score_away",
        "event_id","schedule_date","week","week_label",
        "home_team","away_team",
    ]
    keep_cols = [c for c in want_cols if c in odds.columns]

    # Evita KeyError seleccionando sólo lo disponible
    subset = list(dict.fromkeys(keys + keep_cols))
    left = view.copy()
    right = odds[subset].copy()

    merged = left.merge(right, on=keys, how="left", validate="m:1")

    # Si no hay scores en odds, deja columnas en NaN (bet_card mostrará "TBD")
    for sc in ("score_home","score_away"):
        if sc not in merged.columns:
            merged[sc] = pd.NA

    return merged
