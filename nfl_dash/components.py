# nfl_dash/components.py
import streamlit as st
import pandas as pd
from .logos import get_logo_url
from .utils import norm_abbr, decimal_to_american

def _safe_abbr(x) -> str:
    try:
        return norm_abbr(x) if pd.notna(x) else ""
    except Exception:
        return ""

def _fmt_money(x) -> str:
    try:
        v = float(x)
        return f"${v:,.2f}"
    except Exception:
        return "—"

def _status_color_and_label(profit):
    if pd.isna(profit):
        return ("#facc15", "LIVE/OPEN")  # amarillo para abierto/en vivo
    profit = float(profit)
    if profit > 0:
        return ("#16a34a", "WIN")
    if profit < 0:
        return ("#ef4444", "LOSS")
    return ("#8b5cf6", "PUSH")

def _logo_tag(team_abbr: str, fallback_bg: str = "#222", size: int = 44):
    team_abbr = (team_abbr or "").strip()
    url = get_logo_url(team_abbr) if team_abbr else None
    if url:
        return f"<img src='{url}' alt='{team_abbr}' width='{size}' height='{size}' style='object-fit:contain;'/>"
    t = (team_abbr or 'NA')[:3]
    return (
        f"<div style='width:{size}px;height:{size}px;border-radius:50%;"
        f"background:{fallback_bg};color:#fff;display:flex;align-items:center;"
        f"justify-content:center;font-weight:800;'>{t}</div>"
    )

def bet_card(row: pd.Series):
    # Equipos (abreviados) y side/pick
    home = _safe_abbr(row.get("home_team", "")) or _safe_abbr(row.get("team", ""))
    away = _safe_abbr(row.get("away_team", "")) or _safe_abbr(row.get("opponent", ""))

    side = str(row.get("side", "")).lower() if pd.notna(row.get("side")) else ""
    # pick: si hay 'team' úsalo; si no, inferir por side (home/away)
    pick_abbr = _safe_abbr(row.get("team", ""))
    if not pick_abbr:
        if side == "home":
            pick_abbr = home
        elif side == "away":
            pick_abbr = away

    # Resultados / estado
    sh = pd.to_numeric(row.get("score_home"), errors="coerce")
    sa = pd.to_numeric(row.get("score_away"), errors="coerce")
    state = str(row.get("status", "")).upper()
    short = str(row.get("short", ""))

    # Dinero
    stake = pd.to_numeric(row.get("stake"), errors="coerce")
    dec = pd.to_numeric(row.get("decimal_odds"), errors="coerce")
    ml = pd.to_numeric(row.get("ml"), errors="coerce")
    if pd.isna(ml) and pd.notna(dec):
        ml = decimal_to_american(dec)

    profit = pd.to_numeric(row.get("profit"), errors="coerce")
    status_color, status_label = _status_color_and_label(profit)

    # Encabezado (Week + fecha + dot de estado)
    wk = str(row.get("week_label", row.get("week", "")))
    date_txt = str(row.get("schedule_date", ""))[:10] if pd.notna(row.get("schedule_date")) else ""
    live = (state == "IN")
    dot_color = "#facc15" if live or pd.isna(profit) else ("#16a34a" if float(profit or 0) > 0 else "#ef4444")

    # Logos
    left_logo = _logo_tag(home, "#1f2937", 54)
    right_logo = _logo_tag(away, "#374151", 54)

    # Marcador o "TBD" y línea de estado (short)
    has_score = pd.notna(sh) and pd.notna(sa)
    center_top = (
        f"<div style='font-weight:900;font-size:36px;letter-spacing:.5px;'>{int(sh)} — {int(sa)}</div>"
        if has_score else "<div style='opacity:.55;font-weight:800;font-size:18px;'>TBD</div>"
    )
    center_bottom = ""
    if short:
        # mostrarlos con opacidad media
        center_bottom = f"<div style='font-size:13px;opacity:.65;font-weight:700;margin-top:4px;'>{short}</div>"

    # Números compactos
    ml_txt = f"{ml:+.0f}" if pd.notna(ml) else "—"
    stake_txt = _fmt_money(stake)
    prof_txt = _fmt_money(profit)

    # ---- Render ----
    st.markdown(
        f"""
    <div style="
        border:1px solid rgba(255,255,255,.08);
        border-radius:14px;
        padding:16px 16px 14px 16px;
        margin-bottom:12px;
        background:rgba(255,255,255,.02);
    ">
      <!-- header -->
      <div style="display:flex; align-items:center; justify-content:space-between;">
        <div style="display:flex; align-items:center; gap:10px;">
          <div style="width:10px;height:10px;border-radius:50%;background:{dot_color};"></div>
          <div style="font-size:20px;font-weight:800;">{wk}</div>
          <div style="opacity:.7">{date_txt}</div>
        </div>
        <div style="font-weight:900;letter-spacing:.4px;color:{status_color};">{status_label}</div>
      </div>

      <!-- logos + marcador -->
      <div style="display:flex; align-items:center; justify-content:space-between; gap:16px; margin-top:12px;">
        <div style="width:56px; display:flex; align-items:center; justify-content:center;">
          {left_logo}
        </div>
        <div style="flex:1; text-align:center;">
          {center_top}
          {center_bottom}
        </div>
        <div style="width:56px; display:flex; align-items:center; justify-content:center;">
          {right_logo}
        </div>
      </div>

      <!-- subline equipos -->
      <div style="display:flex; align-items:center; justify-content:space-between; margin-top:8px;">
        <div style="font-size:13px; opacity:.8; font-weight:700;">{home} • Home</div>
        <div style="font-size:12px; opacity:.6; font-weight:800;">vs</div>
        <div style="font-size:13px; opacity:.8; font-weight:700; text-align:right;">{away} • Away</div>
      </div>

      <!-- meta row compacto: Pick · ML · Stake · Profit  -->
      <div style="display:flex; align-items:center; gap:12px; flex-wrap:nowrap; margin-top:12px;">
        <div style="display:flex; align-items:center; gap:6px; font-size:12px;">
          <span style="opacity:.7;">Pick:</span>
          <span style="
              display:inline-block; padding:.15rem .55rem;
              border-radius:999px; border:1px solid rgba(255,255,255,.18);
              font-weight:800; letter-spacing:.3px; font-size:11px; opacity:.95;
          ">{pick_abbr or '—'}</span>
        </div>
        <div style="font-size:12px;"><span style="opacity:.7;">ML:</span> <strong>{ml_txt}</strong></div>
        <div style="font-size:12px;"><span style="opacity:.7;">Stake:</span> <strong>{stake_txt}</strong></div>
        <div style="font-size:12px;">
          <span style="opacity:.7;">Profit:</span> <strong style="color:{status_color};">{prof_txt}</strong>
        </div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def game_card(row: pd.Series):
    home = row.get("home_team", "")
    away = row.get("away_team", "")
    hs = row.get("home_score", None)
    as_ = row.get("away_score", None)
    state = str(row.get("state", "")).lower()
    short = str(row.get("short", ""))
    live = (state == "in")
    final = (state == "post")

    if live:
        color = "#facc15"  # amarillo para live
        label = "LIVE"
    elif final:
        color = "#10B981"
        label = "FINAL"
    else:
        color = "#6B7280"
        label = short if short else "SCHEDULED"

    left_logo = _logo_tag(home, "#1f2937", 50)
    right_logo = _logo_tag(away, "#374151", 50)

    has_score = pd.notna(hs) and pd.notna(as_)
    score_html = (
        f"<div style='font-weight:900;font-size:28px;letter-spacing:.5px;'>{int(hs)} — {int(as_)}</div>"
        if has_score else "<div style='opacity:.55;font-weight:700;'>—</div>"
    )

    st.markdown(
        f"""
    <div style="
        border:1px solid rgba(255,255,255,.08);
        border-radius:12px;padding:14px; margin-bottom:12px;
        background:rgba(255,255,255,.02);
    ">
      <div style="display:flex; align-items:center; justify-content:space-between;">
        <div style="display:flex; align-items:center; gap:8px;">
          <div style="width:10px;height:10px;border-radius:50%;background:{color};"></div>
          <div style="font-weight:800;">{home} vs {away}</div>
        </div>
        <div style="font-weight:900;color:{color}">{label}</div>
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
        <div style="font-size:12px; opacity:.75; font-weight:700;">{short}</div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
