import pandas as pd

ORDER_LABELS = [f"Week {i}" for i in range(1,19)] + ["Wild Card","Divisional","Conference","Super Bowl"]
ORDER_INDEX  = {lab:i for i,lab in enumerate(ORDER_LABELS)}

TEAM_FIX = {
    "STL":"LA","LAR":"LA","SD":"LAC","SDG":"LAC","OAK":"LV","LVR":"LV",
    "WSH":"WAS","JAC":"JAX","GNB":"GB","KAN":"KC","NWE":"NE","NOR":"NO","SFO":"SF","TAM":"TB",
    "LA RAMS":"LA","LOS ANGELES RAMS":"LA","SAN DIEGO CHARGERS":"LAC","OAKLAND RAIDERS":"LV",
}

def norm_abbr(s: str) -> str:
    s = str(s).upper().strip()
    return TEAM_FIX.get(s, s)

def american_to_decimal(m):
    if m is None:
        return None
    try:
        m = float(m)
    except:
        return None
    return 1 + (100/abs(m) if m < 0 else m/100)

def add_week_order(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "week_label" not in out.columns and "week" in out.columns:
        def wl(n):
            try:
                n = int(n)
            except:
                return "Week 999"
            if 1 <= n <= 18:
                return f"Week {n}"
            return {19:"Wild Card",20:"Divisional",21:"Conference",22:"Super Bowl"}.get(n, f"Week {n}")
        out["week_label"] = out["week"].apply(wl)
    out["week_label"] = out["week_label"].astype(str)
    out["week_order"] = out["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    return out

def kpis_from_pnl(pnl: pd.DataFrame):
    profits = pd.to_numeric(pnl.get("profit"), errors="coerce").fillna(0.0)
    stakes  = pd.to_numeric(pnl.get("stake"), errors="coerce").fillna(0.0)
    bank    = pd.to_numeric(pnl.get("bankroll"), errors="coerce")
    initial_bankroll = float(bank.dropna().iloc[0]) if bank.dropna().size else 0.0
    final_bankroll   = float(bank.dropna().iloc[-1]) if bank.dropna().size else initial_bankroll
    total_profit     = float(profits.sum())
    total_stake      = float(stakes.sum())
    yield_pct        = float(100.0 * total_profit / total_stake) if total_stake > 0 else 0.0
    return initial_bankroll, final_bankroll, total_profit, total_stake, yield_pct, profits, stakes

def season_stage(year: int, pnl_df: pd.DataFrame) -> str:
    # clasificación robusta por fecha actual
    now_ts = pd.Timestamp.now(tz="UTC")
    # Heurística de calendario NFL: inicio REG ~ segundo jueves de septiembre; fin ~ mediados de febrero
    reg_start = pd.Timestamp(year=year, month=9, day=1, tz="UTC")
    # desplazar al segundo jueves
    dow = reg_start.weekday()  # 0=Mon ... 3=Thu
    days_to_thu = (3 - dow) % 7
    first_thu = reg_start + pd.Timedelta(days=days_to_thu)
    second_thu = first_thu + pd.Timedelta(days=7)
    regular_start = second_thu
    season_end = pd.Timestamp(year=year+1, month=2, day=15, tz="UTC")

    if now_ts < regular_start:
        return "preseason"
    if now_ts > season_end:
        return "ended"
    # Si la serie PnL llega al Super Bowl, marcamos ended
    if "week_label" in pnl_df.columns and pnl_df["week_label"].astype(str).str.contains("Super Bowl").any():
        return "ended"
    return "in_season"
