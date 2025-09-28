import streamlit as st
import pandas as pd
import re
from .logos import get_logo_url
from .utils import norm_abbr, decimal_to_american


def _s(val) -> str:
    """String 'seguro': '' si es None/NaN/pd.NA, str(val) en otro caso."""
    return "" if pd.isna(val) else str(val)


def _looks_live_from_short(short: str) -> bool:
    """
    Si no tenemos `state`, inferimos 'live' desde el shortDetail de ESPN.
    Consideramos LIVE si hay 'Q1/Q2/Q3/Q4', 'OT' o un timestamp tipo '12:34'.
    """
    s = (short or "").upper()
    if not s:
        return False
    if any(tag in s for tag in ["Q1", "Q2", "Q3", "Q4", "OT"]):
        return True
    # patrón mm:ss (evita 'ET' de timezone)
    return bool(re.search(r"\b\d{1,2}:\d{2}\b", s)) and "ET" not in s


def bet_card(row: pd.Series):
    # ===== Datos base seguros (evita TypeError con pd.NA) =====
    htm = norm_abbr(_s(row.get("home_team", ""))) or norm_abbr(_s(row.get("team", "")))
    atm = norm_abbr(_s(row.get("away_team", ""))) or norm_abbr(_s(row.get("opponent", "")))
    side = _s(row.get("side", "")).title()

    # Scores: aceptar ambos nombres posibles
    sh = row.get("score_home", None)
    sa = row.get("score_away", None)
    if pd.isna(sh) and "home_score" in row:
        sh = row.get("home_score", None)
    if pd.isna(sa) and "away_score" in row:
        sa = row.get("away_score", None)
    has_score = pd.notna(sh) and pd.notna(sa)

    # Estado / short detail
    state = _s(row.get("state", row.get("game_state", ""))).lower()
    status_short = _s(row.get("status_short", row.get("short", "")))

    # Apuesta
    wl_profit = pd.to_numeric(row.get("profit"), errors="coerce")
    stake = pd.to_numeric(row.get("stake"), errors="coerce")
    dec = pd.to_numeric(row.get("decimal_odds"), errors="coerce")
    ml  = pd.to_numeric(row.get("ml"), errors="coerce")
    if pd.isna(ml) and pd.notna(dec):
        ml = decimal_to_american(dec)

    # Header info
    wk = _s(row.get("week_label", row.get("week", "")))
    date_txt = _s(row.get("schedule_date", ""))[:10]

    # ===== LIVE / FINAL / OPEN =====
    is_final = (state == "post")
    is_live = (state == "in") or (not state and _looks_live_from_short(status_short))

    if is_live:
        dot_color = "#F59E0B"  # amarillo
        status_label = "LIVE"
    else:
        if pd.isna(wl_profit):
            dot_color = "#9CA3AF"  # gris
            status_label = "OPEN"
        elif wl_profit > 0:
            dot_color = "#10B981"  # verde
            status_label = "WIN"
        elif wl_profit < 0:
            dot_color = "#EF4444"  # rojo
            status_label = "LOSS"
        else:
            dot_color = "#8884D8"  # morado (push)
            status_label = "PUSH"

    status_color = dot_color

    # Logos
    def logo_tag(team_abbr: str, fallback_bg: str = "#222", size: int = 44):
        url = get_logo_url(team_abbr) if team_abbr else None
        if url:
            return f"<img src='{url}' width='{size}' height='{size}' style='object-fit:contain;'/>"
        t = (team_abbr or 'NA')[:3]
        return (
            f"<div style='width:{size}px;height:{size}px;border-radius:50%;"
            f"background:{fallback_bg};color:#fff;display:flex;align-items:center;"
            f"justify-content:center;font-weight:800;'>{t}</div>"
        )

    left_logo  = logo_tag(htm, "#1f2937", 44)
    right_logo = logo_tag(atm, "#374151", 44)

    # Marcador + sub-etiqueta de estado (tiempo) al centro
    score_main = (
        f"<div style='font-weight:900;font-size:26px;letter-spacing:.5px;'>{int(sh)} — {int(sa)}</div>"
        if has_score else "<div style='opacity:.55;font-weight:700;'>TBD</div>"
    )
    score_sub = ""
    if is_live and status_short:
        score_sub = f"<div style='font-size:12px;opacity:.8;font-weight:700;margin-top:2px;'>{status_short}</div>"
    elif is_final:
        score_sub = "<div style='font-size:12px;opacity:.7;font-weight:600;margin-top:2px;'>Final</div>"

    # Sublíneas bajo logos
    subline_left  = f"{htm}{(' • '+side) if side else ''}".strip()
    subline_right = f"{atm}".strip()

    # Textos de ML / stake / profit
    ml_txt    = f"{ml:+.0f}" if pd.notna(ml) else "—"
    stake_txt = f"${stake:,.2f}" if pd.notna(stake) else "—"
    prof_txt  = f"${wl_profit:,.2f}" if pd.notna(wl_profit) else "—"

    # Pill “Pick: TEAM”
    team_pick = norm_abbr(_s(row.get("team", "")))
    pick_badge_html = f"<span class='pill pill-pick'>Pick: {team_pick}</span>" if team_pick else ""

    # Estilos (pill)
    st.markdown("""
    <style>
    .pill{padding:.12rem .45rem;border:1px solid rgba(255,255,255,.15);
          border-radius:999px;font-size:.80rem;margin-right:.5rem;opacity:.95}
    .pill-pick{background:rgba(59,130,246,.15);border-color:rgba(59,130,246,.35)}
    </style>
    """, unsafe_allow_html=True)

    # Render
    st.markdown(f"""
    <div style="
        border:1px solid #e9e9e9;border-radius:12px;padding:12px; margin-bottom:10px;
        background:linear-gradient(180deg, rgba(0,0,0,0.02), rgba(0,0,0,0.00));
        font-size:12.5px;
    ">
      <!-- Header -->
      <div style="display:flex; align-items:center; justify-content:space-between;">
        <div style="display:flex; align-items:center; gap:8px;">
          <div style="width:8px;height:8px;border-radius:50%;background:{dot_color};"></div>
          <div style="font-weight:600;">{wk}</div>
          <div style="opacity:.7;">{date_txt}</div>
        </div>
        <div style="font-weight:800;color:{status_color}">{status_label}</div>
      </div>

      <!-- Logos + centro -->
      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:10px;">
        <div style="width:44px; display:flex; align-items:center; justify-content:center;">
          {left_logo}
        </div>
        <div style="flex:1; text-align:center;">
          {score_main}{score_sub}
        </div>
        <div style="width:44px; display:flex; align-items:center; justify-content:center;">
          {right_logo}
        </div>
      </div>

      <!-- Abrevs -->
      <div style="display:flex; align-items:center; justify-content:space-between; margin-top:6px;">
        <div style="font-size:12px; opacity:.7; font-weight:600;">{subline_left}</div>
        <div style="font-size:12px; opacity:.5; font-weight:700;">vs</div>
        <div style="font-size:12px; opacity:.7; font-weight:600; text-align:right;">{subline_right}</div>
      </div>

      <!-- Línea, stake, profit + pill de pick -->
      <div style="display:flex; gap:14px; margin-top:10px; flex-wrap:wrap; align-items:center;">
        {pick_badge_html}
        <div><span style="opacity:.6;">Moneyline:</span> <strong>{ml_txt}</strong></div>
        <div><span style="opacity:.6;">Stake:</span> <strong>{stake_txt}</strong></div>
        <div><span style="opacity:.6;">Profit:</span> <strong style="color:{status_color};">{prof_txt}</strong></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def game_card(row: pd.Series):
    home = _s(row.get("home_team", ""))
    away = _s(row.get("away_team", ""))
    hs   = row.get("home_score", None)
    as_  = row.get("away_score", None)
    state = _s(row.get("state","")).lower()
    short = _s(row.get("short",""))
    # LIVE / FINAL
    live = (state == "in") or _looks_live_from_short(short)
    final = (state == "post")
    if live and not final:
        color = "#E11D48"
        label = "LIVE"
    elif final:
        color = "#10B981"
        label = "FINAL"
    else:
        color = "#6B7280"
        label = short if short else "SCHEDULED"

    def logo_tag(name: str, fallback_bg="#222", size: int = 50):
        url = get_logo_url(name) if name else None
        if url:
            return f"<img src='{url}' width='{size}' height='{size}' style='object-fit:contain;'/>"
        t = (name or 'NA')[:3]
        return (
            f"<div style='width:{size}px;height:{size}px;border-radius:50%;"
            f"background:{fallback_bg};color:#fff;display:flex;align-items:center;"
            f"justify-content:center;font-weight:800;'>{t}</div>"
        )

    left_logo  = logo_tag(home, "#1f2937", 50)
    right_logo = logo_tag(away, "#374151", 50)

    has_score = pd.notna(hs) and pd.notna(as_)
    score_html = (
        f"<div style='font-weight:900;font-size:28px;letter-spacing:.5px;'>{int(hs)} — {int(as_)}</div>"
        if has_score else "<div style='opacity:.55;font-weight:700;'>—</div>"
    )

    st.markdown(f"""
    <div style="
        border:1px solid #ececec;border-radius:12px;padding:14px; margin-bottom:12px;
        background:linear-gradient(180deg, rgba(0,0,0,0.02), rgba(0,0,0,0));
        font-size:13px;
    ">
      <div style="display:flex; align-items:center; justify-content:space-between;">
        <div style="display:flex; align-items:center; gap:8px;">
          <div style="width:8px;height:8px;border-radius:50%;background:{color};"></div>
          <div style="font-weight:700;">{home} vs {away}</div>
        </div>
        <div style="font-weight:800;color:{color}">{label}</div>
      </div>

      <div style="display:flex; align-items:center; justify-content:space-between; gap:14px; margin-top:12px;">
        <div style="width:54px; display:flex; align-items:center; justify-content:center;">
          {left_logo}
        </div>
        <div style="flex:1; text-align:center;">
          {score_html}
        </div>
        <div style="width:54px; display:flex; align-items:center; justify-content:center;">
          {right_logo}
        </div>
      </div>

      <div style="display:flex; align-items:center; justify-content:center; margin-top:8px;">
        <div style="font-size:12px; opacity:.7; font-weight:600;">{short}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
