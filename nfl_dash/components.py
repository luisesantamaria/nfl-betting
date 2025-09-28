import streamlit as st
import pandas as pd
from .logos import get_logo_url
from .utils import norm_abbr, decimal_to_american

# ----------------------------
# Helpers seguros contra pd.NA
# ----------------------------
def _s(x) -> str:
    """Texto seguro (sin NA/None)."""
    try:
        return "" if pd.isna(x) else str(x)
    except Exception:
        return str(x or "")

def _f(x):
    """Número float seguro (None si no es convertible)."""
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None

def _is_pos(x) -> bool:
    v = _f(x)
    return (v is not None) and (v > 0)

def _is_neg(x) -> bool:
    v = _f(x)
    return (v is not None) and (v < 0)

# ----------------------------
# Bet Card
# ----------------------------
def bet_card(row: pd.Series):
    # Equipos (preferimos home/away si vienen de ESPN, si no, team/opponent)
    htm = norm_abbr(_s(row.get("home_team"))) or norm_abbr(_s(row.get("team")))
    atm = norm_abbr(_s(row.get("away_team"))) or norm_abbr(_s(row.get("opponent")))

    # Lado apostado -> etiqueta Pick
    side = _s(row.get("side")).lower()
    pick = htm if side == "home" else (atm if side == "away" else "")

    # ML / decimal / stake / profit
    dec = _f(row.get("decimal_odds"))
    ml  = _f(row.get("ml"))
    if ml is None and dec is not None:
        ml = decimal_to_american(dec)

    stake = _f(row.get("stake"))
    profit = _f(row.get("profit"))

    # Estado (de ESPN si lo hay)
    state = _s(row.get("state")).lower()   # pre | in | post | ""
    short = _s(row.get("status_short")) or _s(row.get("short"))

    live  = state == "in"
    final = state == "post"

    # Status color y label (si está LIVE, domina el color amarillo)
    if live:
        status_color, status_label = "#F59E0B", "LIVE"
    else:
        if profit is None:
            status_color, status_label = "#9CA3AF", "OPEN"
        elif profit > 0:
            status_color, status_label = "#10B981", "WIN"
        elif profit < 0:
            status_color, status_label = "#EF4444", "LOSS"
        else:
            status_color, status_label = "#8884D8", "PUSH"

    # Semana/fecha
    wk = _s(row.get("week_label") or row.get("week"))
    date_txt = _s(row.get("schedule_date"))[:10]

    # Score (si viene de ESPN)
    sh = row.get("home_score")
    sa = row.get("away_score")
    has_score = (not pd.isna(sh)) and (not pd.isna(sa))

    # Logos
    def logo_tag(team_abbr: str, fallback_bg: str = "#222", size: int = 44):
        team_abbr = _s(team_abbr)
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

    score_html = (
        f"<div style='font-weight:900;font-size:26px;letter-spacing:.5px;'>{int(sh)} — {int(sa)}</div>"
        if has_score else "<div style='opacity:.55;font-weight:700;'>TBD</div>"
    )
    # Línea debajo del score: short (minuto/quarter/OT) cuando aplica
    sub_status_html = ""
    if live and short:
        sub_status_html = f"<div style='font-size:12px; opacity:.8; margin-top:2px;'>{short}</div>"
    elif (not final) and short:
        sub_status_html = f"<div style='font-size:12px; opacity:.6; margin-top:2px;'>{short}</div>"

    # Subtítulos bajo logos
    subline_left  = f"{htm}{(' • '+side.title()) if side else ''}".strip()
    subline_right = f"{atm}".strip()

    # Textos
    ml_txt    = f"{ml:+.0f}" if ml is not None else "—"
    stake_txt = f"${stake:,.2f}" if stake is not None else "—"
    prof_txt  = f"${profit:,.2f}" if profit is not None else "—"
    pick_txt  = f"Pick: {pick}" if pick else ""

    st.markdown(f"""
    <div style="
        border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:12px;margin-bottom:10px;
        background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.00));
        font-size:12.5px;
    ">
      <div style="display:flex; align-items:center; justify-content:space-between;">
        <div style="display:flex; align-items:center; gap:8px;">
          <div style="width:8px;height:8px;border-radius:50%;background:{status_color};"></div>
          <div style="font-weight:600;">{wk}</div>
          <div style="opacity:.7;">{date_txt}</div>
        </div>
        <div style="font-weight:800;color:{status_color}">{status_label}</div>
      </div>

      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:10px;">
        <div style="width:44px; display:flex; align-items:center; justify-content:center;">
          {left_logo}
        </div>
        <div style="flex:1; text-align:center;">
          {score_html}
          {sub_status_html}
        </div>
        <div style="width:44px; display:flex; align-items:center; justify-content:center;">
          {right_logo}
        </div>
      </div>

      <div style="display:flex; align-items:center; justify-content:space-between; margin-top:6px;">
        <div style="font-size:12px; opacity:.7; font-weight:600;">{subline_left}</div>
        <div style="font-size:12px; opacity:.5; font-weight:700;">vs</div>
        <div style="font-size:12px; opacity:.7; font-weight:600; text-align:right;">{subline_right}</div>
      </div>

      <div style="display:flex; gap:14px; margin-top:10px; flex-wrap:wrap;">
        <div><span style="opacity:.6;">{pick_txt}</span></div>
        <div><span style="opacity:.6;">ML:</span> <strong>{ml_txt}</strong></div>
        <div><span style="opacity:.6;">Stake:</span> <strong>{stake_txt}</strong></div>
        <div><span style="opacity:.6;">Profit:</span> <strong style="color:{status_color};">{prof_txt}</strong></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Game Card (tab Live)
# ----------------------------
def game_card(row: pd.Series):
    home = _s(row.get("home_team"))
    away = _s(row.get("away_team"))
    hs   = row.get("home_score", None)
    as_  = row.get("away_score", None)
    state = _s(row.get("state")).lower()
    short = _s(row.get("short"))
    start = _s(row.get("start_time"))

    live  = (state == "in")
    final = (state == "post")
    if live:
        color = "#F59E0B"
        label = "LIVE"
    elif final:
        color = "#10B981"
        label = "FINAL"
    else:
        color = "#6B7280"
        label = short if short else "SCHEDULED"

    def logo_tag(name: str, fallback_bg="#222", size: int = 50):
        name = _s(name)
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

    has_score = (not pd.isna(hs)) and (not pd.isna(as_))
    score_html = (
        f"<div style='font-weight:900;font-size:28px;letter-spacing:.5px;'>{int(hs)} — {int(as_)}</div>"
        if has_score else "<div style='opacity:.55;font-weight:700;'>—</div>"
    )

    st.markdown(f"""
    <div style="
        border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:14px; margin-bottom:12px;
        background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.00));
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
