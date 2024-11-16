import polars as pl
import polars.selectors as cs

from typeguard import typechecked


@typechecked
def convert_values_to_lowercase(data: pl.LazyFrame) -> pl.LazyFrame:
    str_cols: list[str] = data.select(cs.string()).columns

    for col in str_cols:
        data = data.with_columns(pl.col(col).str.to_lowercase().alias(col))

    return data


@typechecked
def rename_values(data: pl.LazyFrame) -> pl.LazyFrame:
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
