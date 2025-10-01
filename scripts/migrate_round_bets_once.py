# scripts/migrate_round_bets_once.py
from pathlib import Path
import pandas as pd

BETS = Path("data/live/bets.csv")

NUM_COLS = ["stake", "profit", "bankroll_after", "bankroll_week_final"]

def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def main():
    if not BETS.exists():
        print("No existe data/live/bets.csv")
        return
    df = pd.read_csv(BETS, low_memory=False)

    # Asegura numéricos
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = _safe_num(df[c]).round(2)

    # (opcional) redondear también decimal_odds a 6 para consistencia visual
    if "decimal_odds" in df.columns:
        df["decimal_odds"] = _safe_num(df["decimal_odds"]).round(6)

    df.to_csv(BETS, index=False)
    print("✅ Migración hecha: redondeo aplicado a 2 decimales en columnas clave (una sola vez).")

if __name__ == "__main__":
    main()
