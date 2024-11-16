import polars as pl
from polars import selectors as cs
from typeguard import typechecked


@typechecked
def drop_invalid_values(data: pl.LazyFrame, threshold: float = 75.0) -> pl.LazyFrame:
    """Filter out rows where person_age exceeds the specified threshold.

    Parameters
    ----------
    data : pl.LazyFrame
        Input LazyFrame conßtaining person_age column.
    threshold : float, default=75.0
        Maximum allowed age value.

    Returns
    -------
    pl.LazyFrame
        Filtered LazyFrame containing only rows where person_age <= threshold.
    """
    data = data.filter(pl.col("person_age").le(threshold))
    return data


@typechecked
def clamp_numerical_values(
    data: pl.LazyFrame,
    column: str,
    lower_bound: float,
    upper_bound: float,
    lower_bound_replacement: float,
    upper_bound_replacement: float,
) -> pl.LazyFrame:
    """Replace values outside specified bounds with replacement values.

    Parameters
    ----------
    data : pl.LazyFrame
        Input LazyFrame containing the column to be clamped.
    column : str
        Name of the column to apply clamping.
    lower_bound : float
        Lower threshold value for clamping.
    upper_bound : float
        Upper threshold value for clamping.
    lower_bound_replacement : float
        Value to replace data points below lower_bound.
    upper_bound_replacement : float
        Value to replace data points above upper_bound.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with clamped values in the specified column.
    """
    data = data.with_columns(
        pl.when(pl.col(column).lt(lower_bound))
        .then(pl.lit(lower_bound_replacement))
        .otherwise(
            pl.when(pl.col(column).gt(upper_bound))
            .then(pl.lit(upper_bound_replacement))
            .otherwise(pl.col(column))
        )
        .alias(column)
    )
    return data


@typechecked
def get_unique_values(data: pl.LazyFrame) -> dict[str, list[str]]:
    """Get unique values for each string column in a LazyFrame.

    Parameters
    ----------
    data : pl.LazyFrame
        Input LazyFrame containing string columns.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping column names to lists of unique values.
    """
    result: dict[str, list[str]] = {}

    str_cols: list[str] = data.select(cs.string()).columns
    for col in str_cols:
        result[col] = data.select(col).unique().collect().to_numpy().flatten().tolist()

    return result


@typechecked
def probability_to_credit_score(probability: float) -> int:
    """Convert a probability value to a credit score.

    This function takes a probability value and converts it to a credit score
    in the range of 300-800. A small alpha value is added to the probability
    to ensure proper scaling.

    Parameters
    ----------
    probability : float
        Input probability value, typically between 0 and 1

    Returns
    -------
    int
        Calculated credit score between 300 and 800
    """
    alpha: float = 0.005  # This adds a small increment to the probability
    min_score: int = 300
    max_score: int = 800
    score_range: int = max_score - min_score

    # Calculate the credit score based on the probability
    credit_score: int = round(max_score - ((probability + alpha) * score_range))

    return credit_score
