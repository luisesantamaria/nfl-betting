import altair as alt
import pandas as pd
from .utils import ORDER_INDEX


def _week_sort(domain=None):
    # dominio por defecto: secuencia de ORDER_INDEX
    if domain is None:
        domain = list(ORDER_INDEX.keys())
    return alt.Sort(domain=domain)


def chart_sparkline_cumprofit(cum_df: pd.DataFrame, height: int = 200) -> alt.Chart:
    """
    Línea/área de profit acumulado.
    Espera columnas: week_label, cum_profit
    """
    df = cum_df.copy()
    df["week_label"] = df["week_label"].astype(str)

    base = alt.Chart(df, height=height).encode(
        x=alt.X("week_label:N", sort=_week_sort(), title=None),
        y=alt.Y("cum_profit:Q", title="Cumulative Profit ($)"),
        tooltip=[
            "week_label:N",
            alt.Tooltip("cum_profit:Q", format="$.2f"),
        ],
    )

    area = base.mark_area(opacity=0.25)
    line = base.mark_line(point=True)
    return (area + line).properties(width="container")


def chart_last8_profit(pnl_df: pd.DataFrame, profits_series: pd.Series, last: int = 8, height: int = 200) -> alt.Chart:
    """
    Barras de los últimos N profits semanales. Respeta `height`.
    """
    df = pd.DataFrame(
        {"week_label": pnl_df["week_label"].astype(str), "profit": profits_series.values}
    ).copy()
    df["__order"] = df["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    df = df.sort_values(["__order"]).tail(last)

    return (
        alt.Chart(df, height=height)
        .mark_bar()
        .encode(
            x=alt.X("week_label:N", sort=_week_sort(), title=None),
            y=alt.Y("profit:Q", title="Last-Week Profit ($)"),
            tooltip=["week_label:N", alt.Tooltip("profit:Q", format="$.2f")],
        )
        .properties(width="container")
    )


def chart_bankroll(bank_df: pd.DataFrame, height: int = 220) -> alt.Chart:
    """
    Línea de bankroll por semana. Columnas: week_label, bankroll
    """
    df = bank_df.copy()
    df["week_label"] = df["week_label"].astype(str)

    return (
        alt.Chart(df, height=height)
        .mark_line(point=True)
        .encode(
            x=alt.X("week_label:N", sort=_week_sort(), title=None),
            y=alt.Y("bankroll:Q", title="Bankroll ($)"),
            tooltip=["week_label:N", alt.Tooltip("bankroll:Q", format="$.2f")],
        )
        .properties(width="container")
    )


def chart_profit_bars(prof_df: pd.DataFrame, height: int = 220) -> alt.Chart:
    """
    Barras de profit semanal. Columnas: week_label, profit
    """
    df = prof_df[["week_label", "profit"]].copy()
    df["week_label"] = df["week_label"].astype(str)

    return (
        alt.Chart(df, height=height)
        .mark_bar()
        .encode(
            x=alt.X("week_label:N", sort=_week_sort(), title=None),
            y=alt.Y("profit:Q", title="Weekly Profit ($)"),
            tooltip=["week_label:N", alt.Tooltip("profit:Q", format="$.2f")],
        )
        .properties(width="container")
    )


def chart_drawdown_area(bank_df: pd.DataFrame, height: int = 220) -> alt.Chart:
    """
    Área de drawdown desde el máximo histórico del bankroll.
    Columnas: week_label, bankroll
    """
    df = bank_df.copy()
    df["week_label"] = df["week_label"].astype(str)

    # drawdown = (peak - bankroll)
    peak = df["bankroll"].cummax()
    df["drawdown"] = (peak - df["bankroll"]).clip(lower=0)

    return (
        alt.Chart(df, height=height)
        .mark_area(opacity=0.25)
        .encode(
            x=alt.X("week_label:N", sort=_week_sort(), title=None),
            y=alt.Y("drawdown:Q", title="Drawdown ($)"),
            tooltip=[
                "week_label:N",
                alt.Tooltip("drawdown:Q", format="$.2f"),
            ],
        )
        .properties(width="container")
    )


def chart_stake_bars(stake_df: pd.DataFrame, height: int = 220) -> alt.Chart:
    """
    Barras de stake por semana. Columnas: week_label, stake
    """
    df = stake_df.copy()
    df["week_label"] = df["week_label"].astype(str)

    return (
        alt.Chart(df, height=height)
        .mark_bar()
        .encode(
            x=alt.X("week_label:N", sort=_week_sort(), title=None),
            y=alt.Y("stake:Q", title="Stake ($)"),
            tooltip=["week_label:N", alt.Tooltip("stake:Q", format="$.2f")],
        )
        .properties(width="container")
    )
