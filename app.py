# app/streamlit_app.py
import os
import math
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# =========================================
# Configuración básica
# =========================================
st.set_page_config(page_title="NFL EV Dashboard", layout="wide")
st.title("NFL EV Betting — Overview & Bets")

# Rutas (ajústalas si las cambias en tu repo)
DATA_LATEST = os.getenv("DATA_URL", "data/processed/latest.csv")
BETS_DIR    = os.getenv("BETS_DIR", "data/bets")
INITIAL_BANKROLL = float(os.getenv("INITIAL_BANKROLL", "1000"))

# =========================================
# Utilidades
# =========================================
PLAYOFF_LABELS = {19:"Wild Card", 20:"Divisional", 21:"Conference", 22:"Super Bowl"}
ORDER_LABELS = [f"Week {i}" for i in range(1,19)] + list(PLAYOFF_LABELS.values())
ORDER_INDEX  = {lab:i for i,lab in enumerate(ORDER_LABELS)}

def week_label_from_num(n: int) -> str:
    try:
        n = int(n)
    except Exception:
        return str(n)
    if 1 <= n <= 18:
        return f"Week {n}"
    return PLAYOFF_LABELS.get(n, f"Week {n}")

def to_utc(ts) -> datetime:
    if pd.isna(ts):
        return None
    t = pd.to_datetime(ts, errors="coerce", utc=True)
    if t is pd.NaT:
        return None
    return t.to_pydatetime()

@st.cache_data(ttl=60*10)
def load_latest(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        # Normalización básica
        if "week_label" not in df.columns and "week" in df.columns:
            df["week_label"] = df["week"].apply(week_label_from_num)
        if "schedule_date" in df.columns:
            df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce", utc=True)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60*10)
def seasons_available(bets_dir: str) -> list[int]:
    out = []
    if not os.path.isdir(bets_dir):
        return out
    for name in sorted(os.listdir(bets_dir)):
        if name.startswith("season="):
            try:
                out.append(int(name.split("=")[1]))
            except Exception:
                pass
    return out

@st.cache_data(ttl=60*10)
def load_ledger(bets_dir: str, season: int) -> pd.DataFrame:
    path = os.path.join(bets_dir, f"season={season}", "ledger.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    # Campos recomendados (tolerante a faltantes)
    for c in ["season","week","week_label","kickoff_utc","placed_at_utc","status","result",
              "stake","decimal_odds","ml","profit","pnl_units","team","opponent",
              "home_team","away_team","score_home","score_away","market","side"]:
        if c not in df.columns:
            # crear columna vacía si no existe
            df[c] = np.nan
    # Tipos
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"]   = pd.to_numeric(df["week"],   errors="coerce").astype("Int64")
    df["week_label"] = df["week"].apply(week_label_from_num)
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True)
    df["placed_at_utc"] = pd.to_datetime(df["placed_at_utc"], errors="coerce", utc=True)
    if "profit" in df.columns:
        df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
    if "stake" in df.columns:
        df["stake"] = pd.to_numeric(df["stake"], errors="coerce")
    if "decimal_odds" in df.columns:
        df["decimal_odds"] = pd.to_numeric(df["decimal_odds"], errors="coerce")
    if "ml" in df.columns:
        df["ml"] = pd.to_numeric(df["ml"], errors="coerce")
    if "score_home" in df.columns:
        df["score_home"] = pd.to_numeric(df["score_home"], errors="coerce")
    if "score_away" in df.columns:
        df["score_away"] = pd.to_numeric(df["score_away"], errors="coerce")
    # Orden consistente
    df["week_order"] = df["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    df = df.sort_values(["week_order","kickoff_utc","team","opponent"], na_position="last").reset_index(drop=True)
    return df

def compute_nav_weekly(ledger: pd.DataFrame, initial_bankroll: float) -> pd.DataFrame:
    """
    Usa bets liquidadas para construir P&L semanal y NAV acumulado.
    Si no hay liquidadas, devuelve marco vacío con bankroll inicial.
    """
    if ledger.empty or "profit" not in ledger.columns:
        df = pd.DataFrame({"week_label": [], "profit": [], "stake": [], "bankroll": []})
        return df

    # Solo liquidadas
    settled = ledger.copy()
    if "status" in settled.columns:
        settled = settled[settled["status"].astype(str).str.lower().eq("settled")]

    if settled.empty:
        return pd.DataFrame({"week_label": [], "profit": [], "stake": [], "bankroll": []})

    agg = (settled.groupby("week_label", as_index=False, sort=False)
                  .agg(profit=("profit","sum"),
                       stake =("stake","sum")))

    # Orden por semana
    agg["week_order"] = agg["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    agg = agg.sort_values("week_order").reset_index(drop=True)

    # NAV acumulado
    bk = initial_bankroll
    bankroll = []
    for _, r in agg.iterrows():
        bk = float(np.round(bk + float(r["profit"]), 2))
        bankroll.append(bk)

    agg["bankroll"] = bankroll
    return agg[["week_label","profit","stake","bankroll"]]

def fmt_money(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "-"

def human_countdown(dt_utc: datetime) -> str:
    if dt_utc is None:
        return ""
    now = datetime.now(timezone.utc)
    delta = dt_utc - now
    secs = int(delta.total_seconds())
    if secs <= 0:
        return "Kickoff inminente"
    h = secs // 3600
    m = (secs % 3600) // 60
    return f"Kickoff en {h}h {m}m"

# =========================================
# CSS del “scoreboard”
# =========================================
st.markdown("""
<style>
.score-card {
  border-radius: 18px; padding: 14px 16px; margin-bottom: 14px;
  background: #0F172A; /* slate-900-ish */
  border: 1px solid rgba(148,163,184,0.18); /* slate-400 alpha */
  color: #E5E7EB;
  box-shadow: 0 4px 18px rgba(0,0,0,0.25);
}
.score-card .hdr {
  display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;
}
.badge {
  font-weight: 700; padding: 6px 10px; border-radius: 9999px; font-size: 13px; line-height: 1;
  text-transform: uppercase; letter-spacing: 0.03em;
}
.badge.win { background: #16a34a; color: white; }
.badge.loss { background: #dc2626; color: white; }
.badge.push { background: #64748b; color: white; }
.subtle { color: #94A3B8; font-size: 12px; }

.score-row {
  display: grid; grid-template-columns: 1fr 40px 1fr; gap: 8px; align-items: center;
  margin: 4px 0 8px 0;
}
.team {
  display: flex; align-items: center; gap: 10px; justify-content: flex-end;
}
.team.right { justify-content: flex-start; }
.logo {
  width: 48px; height: 48px; border-radius: 9999px; background: #111827; 
  display:flex; align-items:center; justify-content:center; 
  font-weight: 800; font-size: 18px; color: #e5e7eb; border: 1px solid rgba(148,163,184,0.25);
}
.score {
  font-size: 42px; font-weight: 800; line-height: 1; letter-spacing: 0.5px;
}
.sep {
  text-align: center; font-size: 24px; color: #94A3B8; font-weight: 700;
}
.meta {
  display:flex; align-items:center; justify-content: space-between; margin-top: 4px; color:#94A3B8; font-size:12px;
}
.pills {
  display:flex; gap:8px; flex-wrap:wrap; margin-top: 8px;
}
.pill {
  padding: 6px 10px; border-radius: 9999px; background:#0B1220; border:1px solid rgba(148,163,184,0.18);
  font-size: 12px; color:#E5E7EB;
}
.kicker {
  color:#CBD5E1; font-size:12px; margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# Componentes UI
# =========================================
def render_scoreboard_card(row: pd.Series):
    # Estado y badge
    status = str(row.get("status", "")).lower()
    result = str(row.get("result", "")).lower()
    is_win  = result == "win" or (("profit" in row) and pd.notna(row["profit"]) and float(row["profit"]) > 0)
    is_loss = result == "loss" or (("profit" in row) and pd.notna(row["profit"]) and float(row["profit"]) < 0)
    is_push = result == "push" or (("profit" in row) and abs(float(row["profit"] or 0.0)) < 1e-9 and status == "settled")

    if status == "pending":
        badge = '<span class="badge subtle">PENDING</span>'
    elif is_win:
        badge = '<span class="badge win">WIN</span>'
    elif is_loss:
        badge = '<span class="badge loss">LOSS</span>'
    elif is_push:
        badge = '<span class="badge push">PUSH</span>'
    else:
        badge = '<span class="badge subtle">—</span>'

    wk_label = row.get("week_label") or (week_label_from_num(row.get("week", "")) if pd.notna(row.get("week", np.nan)) else "")
    kickoff  = to_utc(row.get("kickoff_utc"))
    kickoff_txt = kickoff.strftime("%Y-%m-%d %H:%M UTC") if kickoff else ""
    market = str(row.get("market","")).title() if pd.notna(row.get("market", np.nan)) else "Moneyline"

    # Equipos y marcador
    # Si tienes home/away/score_* úsalo, si no, usa team/opponent
    h_team = row.get("home_team") if pd.notna(row.get("home_team", np.nan)) else row.get("team")
    a_team = row.get("away_team") if pd.notna(row.get("away_team", np.nan)) else row.get("opponent")
    s_h = row.get("score_home")
    s_a = row.get("score_away")
    # Cuando no hay marcador usa "-"
    sc_h = "-" if pd.isna(s_h) else int(s_h)
    sc_a = "-" if pd.isna(s_a) else int(s_a)

    # Logos: si no hay URL, usamos siglas en círculo
    def logo_span(name):
        abbr = str(name or "").upper()[:3] or "—"
        return f'<div class="logo">{abbr}</div>'

    # Pills (stake, odds, profit/EV)
    stake = fmt_money(row.get("stake", 0))
    dec   = row.get("decimal_odds", np.nan)
    ml    = row.get("ml", np.nan)
    odds_txt = f"{float(dec):.2f}" if pd.notna(dec) else (f"{int(ml):+d}" if pd.notna(ml) and not math.isnan(ml) else "—")

    profit = row.get("profit", np.nan)
    pnl_txt = fmt_money(profit) if pd.notna(profit) else "—"

    ev = row.get("ev", np.nan)
    edge = row.get("edge", np.nan)
    extra = []
    if pd.notna(ev):
        extra.append(f"EV {ev*100:+.2f}%")
    if pd.notna(edge):
        extra.append(f"Edge {edge*100:+.2f} pp")
    extra_txt = " · ".join(extra)

    # Subtítulo meta
    sub_left  = f"{wk_label} · {kickoff_txt} · {market}"
    if status == "pending":
        cd = human_countdown(kickoff)
        sub_right = cd
    else:
        sub_right = "Final" if (pd.notna(s_h) and pd.notna(s_a)) else ""

    html = f"""
    <div class="score-card">
      <div class="hdr">
        {badge}
        <div class="subtle">{sub_left}</div>
      </div>

      <div class="score-row">
        <div class="team">{logo_span(a_team)}<div class="score">{sc_a}</div></div>
        <div class="sep">—</div>
        <div class="team right"><div class="score">{sc_h}</div>{logo_span(h_team)}</div>
      </div>

      <div class="meta">
        <div class="subtle">{sub_right}</div>
        <div class="subtle">{str(row.get("side","")).upper()} · {str(row.get("team","")).upper()} vs {str(row.get("opponent","")).upper()}</div>
      </div>

      <div class="pills">
        <div class="pill">Stake: {stake}</div>
        <div class="pill">Odds: {odds_txt}</div>
        <div class="pill">Result: {pnl_txt}</div>
        {"<div class='pill'>"+extra_txt+"</div>" if extra_txt else ""}
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def nav_chart(df_nav: pd.DataFrame, height: int):
    if df_nav.empty:
        st.info("Sin datos de NAV todavía.")
        return
    c = (alt.Chart(df_nav)
          .mark_line(point=True)
          .encode(
              x=alt.X("week_label:N", sort=list(ORDER_INDEX.keys()), title="Semana"),
              y=alt.Y("bankroll:Q", title="NAV"),
              tooltip=["week_label","bankroll","profit","stake"]
          ).properties(height=height))
    st.altair_chart(c, use_container_width=True)

def pnl_chart(df_nav: pd.DataFrame, height: int):
    if df_nav.empty:
        st.info("Sin datos de P&L semanal todavía.")
        return
    c = (alt.Chart(df_nav)
          .mark_bar()
          .encode(
              x=alt.X("week_label:N", sort=list(ORDER_INDEX.keys()), title="Semana"),
              y=alt.Y("profit:Q", title="P&L semanal"),
              color=alt.condition(alt.datum.profit >= 0, alt.value("#16a34a"), alt.value("#dc2626")),
              tooltip=["week_label","profit","stake"]
          ).properties(height=height))
    st.altair_chart(c, use_container_width=True)

def render_bets_grid(df: pd.DataFrame, title: str, limit: int = 8):
    if df.empty:
        st.caption(f"{title}: sin registros.")
        return
    st.markdown(f"### {title}")
    # Grid de 2 columnas
    rows = []
    n = min(len(df), limit)
    for i in range(0, n, 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < n:
                with col:
                    render_scoreboard_card(df.iloc[i + j])

# =========================================
# Controles
# =========================================
left0, right0 = st.columns([1,1])
with left0:
    seasons = seasons_available(BETS_DIR)
    if not seasons:
        st.warning("No se encontraron carpetas en data/bets/season=YYYY. Muestra de datos limitada.")
        seasons = [datetime.now().year]  # fallback
    selected_season = st.selectbox("Temporada", options=seasons, index=len(seasons)-1)
with right0:
    st.caption("Consejo: cuando no haya partidos pendientes, los gráficos del overview ocuparán más alto para evitar espacios vacíos.")

# Carga de datos
latest = load_latest(DATA_LATEST)
ledger = load_ledger(BETS_DIR, selected_season)

# Estado: temporada activa o cerrada
pending = ledger[ledger["status"].astype(str).str.lower().eq("pending")] if not ledger.empty else pd.DataFrame()
has_pending = not pending.empty
season_closed = not has_pending  # Simple y claro: si no hay pendientes, consideramos “cerrada” para el layout

# Alturas de gráficos según estado
height_nav = 520 if season_closed else 360
height_pnl = 520 if season_closed else 320

# =========================================
# LAYOUT (dos columnas siempre)
# =========================================
colL, colR = st.columns([2, 1])

with colL:
    st.subheader("Portfolio NAV (por semana)")
    nav_df = compute_nav_weekly(ledger, INITIAL_BANKROLL)
    nav_chart(nav_df, height_nav)

    st.subheader("P&L semanal")
    pnl_chart(nav_df, height_pnl)

with colR:
    if has_pending:
        this_week = pending.copy()
        # Si tienes week_label, muestra primero las de la semana más próxima
        if "kickoff_utc" in this_week.columns:
            this_week = this_week.sort_values("kickoff_utc")
        render_bets_grid(this_week, "Bets pendientes de esta semana", limit=6)

        # Últimos 7 días liquidadas
        recent = ledger.copy()
        if "status" in recent.columns:
            recent = recent[recent["status"].astype(str).str.lower().eq("settled")]
        if "kickoff_utc" in recent.columns:
            recent = recent.sort_values("kickoff_utc", ascending=False)
        render_bets_grid(recent.head(6), "Liquidadas recientes", limit=6)
    else:
        # Temporada “cerrada” para el layout: deja la columna derecha con resumen compacto
        st.subheader("Resumen rápido")
        total_bets = len(ledger) if not ledger.empty else 0
        wins = int((ledger.get("profit", pd.Series(dtype=float)) > 0).sum()) if not ledger.empty else 0
        losses = int((ledger.get("profit", pd.Series(dtype=float)) < 0).sum()) if not ledger.empty else 0
        pnl_total = float(ledger.get("profit", pd.Series(dtype=float)).sum()) if not ledger.empty else 0.0
        st.metric("Bets totales", total_bets)
        st.metric("Win / Loss", f"{wins} / {losses}")
        st.metric("P&L total", fmt_money(pnl_total))

        st.caption("Cuando la temporada esté activa, aquí verás tarjetas tipo marcador con tus bets pendientes y recientes.")

# =========================================
# (Opcional) Pestaña “Bets” detallada
# =========================================
with st.expander("Ver tabla detallada de Bets"):
    if ledger.empty:
        st.info("Aún no hay registros en el ledger de esta temporada.")
    else:
        # Filtros simples
        c1, c2, c3 = st.columns(3)
        with c1:
            market_filter = st.multiselect("Mercado", sorted(ledger["market"].dropna().astype(str).unique()), default=None)
        with c2:
            status_filter = st.multiselect("Estado", sorted(ledger["status"].dropna().astype(str).unique()), default=None)
        with c3:
            week_filter = st.multiselect("Semana", sorted(ledger["week_label"].dropna().astype(str).unique()), default=None)

        dfv = ledger.copy()
        if market_filter:
            dfv = dfv[dfv["market"].astype(str).isin(market_filter)]
        if status_filter:
            dfv = dfv[dfv["status"].astype(str).isin(status_filter)]
        if week_filter:
            dfv = dfv[dfv["week_label"].astype(str).isin(week_filter)]

        show_cols = [c for c in [
            "kickoff_utc","week_label","status","result","team","opponent","side","market",
            "decimal_odds","ml","stake","profit","edge","ev"
        ] if c in dfv.columns]
        st.dataframe(dfv[show_cols].sort_values(["kickoff_utc","week_label"], na_position="last"), use_container_width=True)
