import polars as pl
import polars.selectors as cs

from typeguard import typechecked


@typechecked
def convert_values_to_lowercase(data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Convert all string column values to lowercase.

    Parameters
    ----------
    data : pl.LazyFrame
        Input LazyFrame containing string columns.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with all string values converted to lowercase.
    """
    str_cols: list[str] = data.select(cs.string()).columns

    for col in str_cols:
        data = data.with_columns(pl.col(col).str.to_lowercase().alias(col))

    return data


@typechecked
def rename_loan_intent_values(data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Rename loan intent values using a predefined mapping.

    Parameters
    ----------
    data : pl.LazyFrame
        Input LazyFrame containing loan_intent column.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with renamed loan_intent values.
    """
    loan_intent_mapper: dict[str, str] = {
        "education": "education",
        "medical": "medical",
        "venture": "venture",
        "debtconsolidation": "debt consolidation",
        "personal": "personal",
        "homeimprovement": "home improvement",
    }
    data = data.with_columns(
        pl.col("loan_intent")
        .map_elements(lambda x: loan_intent_mapper.get(x))
        .alias("loan_intent")
    )
    return data
