import pandas as pd
from .data_io import load_odds_for_season
from .utils import norm_abbr

def _pair(a: str, b: str) -> str:
    a = str(a).upper().strip()
    b = str(b).upper().strip()
    return f"{a}_{b}" if a < b else f"{b}_{a}"

def enrich_bets_with_scores(bets: pd.DataFrame, season: int) -> pd.DataFrame:
    if bets is None or bets.empty:
        return bets
    o = load_odds_for_season(season)
    if o.empty:
        return bets

    # normaliza
    for c in ("home_team","away_team"):
        if c in o.columns:
            o[c] = o[c].astype(str).map(norm_abbr)
    odds = o.copy()
    odds["pair"] = [_pair(h, a) for h, a in zip(odds.get("home_team"), odds.get("away_team"))]
    if "week_label" not in odds.columns and "week" in odds.columns:
        def wl(n):
            try:
                n = int(n)
            except:
                return "Week 999"
            if 1 <= n <= 18:
                return f"Week {n}"
            return {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(n, f"Week {n}")
        odds["week_label"] = odds["week"].apply(wl)

    view = bets.copy()
    for c in ("team","opponent"):
        if c in view.columns:
            view[c] = view[c].astype(str).map(norm_abbr)
    view["pair"] = [_pair(t, o) for t, o in zip(view.get("team"), view.get("opponent"))]

    keys = ["pair", "week_label"]
    cols_keep = ["score_home","score_away","home_team","away_team","schedule_date"]
    m = view.merge(odds[keys + cols_keep], on=keys, how="left", validate="m:1")

    # construir marcador alineando lados
    def mk_score(row):
        sh, sa = row.get("score_home"), row.get("score_away")
        if pd.isna(sh) or pd.isna(sa):
            return None
        # quién es "team" respecto a home/away
        t, h, a = row.get("team"), row.get("home_team"), row.get("away_team")
        if str(t).upper().strip() == str(h).upper().strip():
            # team es local
            return f"{int(sh)} — {int(sa)}"
        elif str(t).upper().strip() == str(a).upper().strip():
            return f"{int(sa)} — {int(sh)}"
        else:
            # no pudimos alinear; mostrar como home-away
            return f"{int(sh)} — {int(sa)}"

    m["scoreline"] = m.apply(mk_score, axis=1)

    # si falta 'won' y tenemos score, computarlo
    if "won" not in m.columns and "score_home" in m.columns and "score_away" in m.columns:
        def calc_won(row):
            sh, sa = row.get("score_home"), row.get("score_away")
            if pd.isna(sh) or pd.isna(sa):
                return None
            t, h, a = row.get("team"), row.get("home_team"), row.get("away_team")
            home_won = int(sh > sa)
            if str(t).upper().strip() == str(h).upper().strip():
                return home_won
            elif str(t).upper().strip() == str(a).upper().strip():
                return 1 - home_won
            return None
        m["won"] = m.apply(calc_won, axis=1)

    return m
