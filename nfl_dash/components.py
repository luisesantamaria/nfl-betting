# nfl_dash/components.py
from __future__ import annotations
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

from .logos import get_logo_url
from .utils import norm_abbr, decimal_to_american


def _safe_str(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def _logo_tag(team_abbr: str, fallback_bg: str = "#222", size: int = 44) -> str:
    t = _safe_str(team_abbr).strip()
    url = get_logo_url(t) if t else None
    if url:
        return f"<img src='{url}' alt='{t}' width='{size}' height='{size}' style='object-fit:contain;'/>"
    # fallback simple
    txt = (t or "NA")[:3]
    return (
        f"<div style='width:{size}px;height:{size}px;border-radius:50%;"
        f"background:{fallback_bg};color:#fff;display:flex;align-items:center;"
        f"justify-content:center;font-weight:800;'>{txt}</div>"
    )


def bet_card(row: pd.Series):
    """
    Card para apuestas (Overview y Bets). Renderizado en iframe para evitar que
    HTML suelto se 'escape' como texto (especialmente en seasons pasadas).
    """
    # --------- datos base / normalización ----------
    wk = _safe_str(row.get("week_label") or row.get("week") or "")
    date_txt = ""
    sd = row.get("schedule_date")
    if pd.notna(sd):
        # ya vendrá en tz-aware o no; sólo mostramos YYYY-MM-DD
        date_txt = _safe_str(sd)[:10]

    side = _safe_str(row.get("side")).lower()
    team = norm_abbr(_safe_str(row.get("team")))
    opp  = norm_abbr(_safe_str(row.get("opponent")))
    home_team = norm_abbr(_safe_str(row.get("home_team")))
    away_team = norm_abbr(_safe_str(row.get("away_team")))

    # quién es pick (chip)
    pick = ""
    if side in ("home", "away"):
        pick = home_team if side == "home" else away_team
    elif team:
        pick = team

    # odds / stake / profit
    stake = pd.to_numeric(row.get("stake"), errors="coerce")
    dec = pd.to_numeric(row.get("decimal_odds"), errors="coerce")
    ml = pd.to_numeric(row.get("ml"), errors="coerce")
    if pd.isna(ml) and pd.notna(dec):
        ml = decimal_to_american(dec)
    profit = pd.to_numeric(row.get("profit"), errors="coerce")

    # estado / scores (provenientes de odds/ESPN)
    state = _safe_str(row.get("state")).lower()
    short = _safe_str(row.get("short"))  # p.ej. "3:29 • 4th" si viene
    sh = row.get("score_home")
    sa = row.get("score_away")
    has_score = pd.notna(sh) and pd.notna(sa)

    # Determinar local/visit según lo que tengamos
    htm = home_team or (team if side == "home" else "")
    atm = away_team or (opp if side == "home" else "")
    if not htm or not atm:
        # fallback al par team/opponent normalizados
        htm = home_team or norm_abbr(_safe_str(row.get("home"))) or norm_abbr(_safe_str(row.get("team")))
        atm = away_team or norm_abbr(_safe_str(row.get("away"))) or norm_abbr(_safe_str(row.get("opponent")))

    # color / etiqueta estado (usa state si existe; si no, deriva de profit)
    if state == "in":
        dot_color, status_label = "#FACC15", "LIVE"
    elif state == "post":
        if pd.isna(profit):
            dot_color, status_label = "#6B7280", "FINAL"
        elif profit > 0:
            dot_color, status_label = "#10B981", "WIN"
        elif profit < 0:
            dot_color, status_label = "#EF4444", "LOSS"
        else:
            dot_color, status_label = "#A78BFA", "PUSH"
    elif state == "pre":
        dot_color, status_label = "#6B7280", "OPEN"
    else:
        # si no tenemos state, basamos en profit o abierto
        if pd.isna(profit):
            dot_color, status_label = "#6B7280", "OPEN"
        elif profit > 0:
            dot_color, status_label = "#10B981", "WIN"
        elif profit < 0:
            dot_color, status_label = "#EF4444", "LOSS"
        else:
            dot_color, status_label = "#A78BFA", "PUSH"

    # strings formateadas
    ml_txt    = f"{ml:+.0f}" if pd.notna(ml) else "—"
    stake_txt = f"${stake:,.2f}" if pd.notna(stake) else "—"
    prof_txt  = "—" if pd.isna(profit) else f"${profit:,.2f}"
    prof_col  = "#E5E7EB"  # default gris
    if not pd.isna(profit):
        prof_col = "#10B981" if profit > 0 else ("#EF4444" if profit < 0 else "#A78BFA")

    left_logo  = _logo_tag(htm, "#1f2937", 44)
    right_logo = _logo_tag(atm, "#374151", 44)

    score_html = (
        f"<div style='font-weight:900;font-size:26px;letter-spacing:.5px;'>{int(sh)} — {int(sa)}</div>"
        if has_score else "<div style='opacity:.55;font-weight:700;'>TBD</div>"
    )

    # línea de estado en el centro (sólo live/final si hay info)
    center_sub = ""
    if state == "in" and short:
        center_sub = f"<div style='font-size:12px;opacity:.7;font-weight:600;margin-top:4px;'>{short}</div>"

    # chip "Pick: XXX"
    pick_chip = ""
    if pick:
        pick_chip = (
            f"<span style='display:inline-block;padding:.15rem .5rem;border:1px solid rgba(255,255,255,.18);"
            f"border-radius:999px;font-weight:700;font-size:.75rem;background:rgba(255,255,255,.06);'>"
            f"Pick: {pick}</span>"
        )

    # ------------ HTML card (aislado en iframe) -------------
    card_html = f"""
    <div style="
        border:1px solid rgba(255,255,255,.10);border-radius:14px;padding:14px;
        background:rgba(255,255,255,.03);font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,'Helvetica Neue',Arial;
        color:#e5e7eb;
    ">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div style="display:flex;align-items:center;gap:8px;">
          <div style="width:8px;height:8px;border-radius:50%;background:{dot_color};"></div>
          <div style="font-weight:700;">{wk}</div>
          <div style="opacity:.7;">{date_txt}</div>
        </div>
        <div style="font-weight:800;color:{dot_color}">{status_label}</div>
      </div>

      <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:12px;">
        <div style="width:44px;display:flex;align-items:center;justify-content:center;">
          {left_logo}
        </div>
        <div style="flex:1;text-align:center;">
          {score_html}
          {center_sub}
        </div>
        <div style="width:44px;display:flex;align-items:center;justify-content:center;">
          {right_logo}
        </div>
      </div>

      <div style="display:flex;align-items:center;justify-content:space-between;margin-top:6px;">
        <div style="font-size:12px;opacity:.7;font-weight:600;">{htm} • Home</div>
        <div style="font-size:12px;opacity:.5;font-weight:700;">vs</div>
        <div style="font-size:12px;opacity:.7;font-weight:600;text-align:right;">{atm} • Away</div>
      </div>

      <div style="display:flex;gap:14px;margin-top:10px;flex-wrap:wrap;align-items:center;">
        {pick_chip}
        <div><span style="opacity:.6;">ML:</span> <strong>{ml_txt}</strong></div>
        <div><span style="opacity:.6;">Stake:</span> <strong>{stake_txt}</strong></div>
        <div><span style="opacity:.6;">Profit:</span> <strong style="color:{prof_col};">{prof_txt}</strong></div>
      </div>
    </div>
    """

    # altura aproximada (un poco más si hay sublínea live)
    height = 210 if (state == "in" and short) else 190
    st_html(card_html, height=height, scrolling=False)


def game_card(row: pd.Series):
    """
    Card para la pestaña Live (se deja con markdown tal como lo tenías).
    """
    home = _safe_str(row.get("home_team"))
    away = _safe_str(row.get("away_team"))
    hs   = row.get("home_score")
    as_  = row.get("away_score")
    state = _safe_str(row.get("state")).lower()
    short = _safe_str(row.get("short"))
    live = (state == "in")
    final = (state == "post")

    if live:
        color = "#E11D48"; label = "LIVE"
    elif final:
        color = "#10B981"; label = "FINAL"
    else:
        color = "#6B7280"; label = short if short else "SCHEDULED"

    left_logo  = _logo_tag(home, "#1f2937", 50)
    right_logo = _logo_tag(away, "#374151", 50)

    has_score = pd.notna(hs) and pd.notna(as_)
    score_html = (
        f"<div style='font-weight:900;font-size:28px;letter-spacing:.5px;'>{int(hs)} — {int(as_)}</div>"
        if has_score else "<div style='opacity:.55;font-weight:700;'>—</div>"
    )

    st.markdown(
        f"""
        <div style="
            border:1px solid #ececec;border-radius:12px;padding:14px;margin-bottom:12px;
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
        """,
        unsafe_allow_html=True,
    )
