import altair as alt
import pandas as pd
from .utils import ORDER_INDEX

def chart_last8_profit(pnl_df: pd.DataFrame, profits_series: pd.Series, last: int = 8, height: int = 200) -> alt.Chart:
    """
    Barras de los últimos N profits semanales.
    Respeta el parámetro `height` para poder agrandarlo cuando no haya bets.
    """
    df = pd.DataFrame({
        "week_label": pnl_df["week_label"].astype(str),
        "profit": profits_series.values
    }).copy()

    # ordenar por la secuencia de semanas / playoffs
    df["__order"] = df["week_label"].map(ORDER_INDEX).fillna(999).astype(int)
    df = df.sort_values(["__order"]).tail(last)

    return (
        alt.Chart(df, height=height)
        .mark_bar()
        .encode(
            x=alt.X("week_label:N", sort=list(ORDER_INDEX.keys()), title=None),
            y=alt.Y("profit:Q", title="Last-Week Profit ($)"),
            tooltip=["week_label:N", alt.Tooltip("profit:Q", format="$.2f")],
        )
        .properties(width="container")
    )
