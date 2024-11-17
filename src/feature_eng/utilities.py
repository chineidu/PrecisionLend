from datetime import datetime as dt

import numpy as np
from omegaconf import DictConfig
import polars as pl
from polars import selectors as cs
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
from typeguard import typechecked

from src.data_eng.transformations import (
    convert_values_to_lowercase,
    rename_loan_intent_values,
)
from src.utilities import load_config


CONFIG: DictConfig = load_config()


@typechecked
def drop_invalid_values(
    data: pl.LazyFrame,
    column: str,
    lower_threshold: float = 18.0,
    upper_threshold: float = 75.0,
) -> pl.LazyFrame:
    """Filter out invalid values from a LazyFrame based on specified thresholds.

    Parameters
    ----------
    data : pl.LazyFrame
        Input LazyFrame containing the column to be filtered.
    column : str
        Name of the column to apply filtering.
    lower_threshold : float, default=18.0
        Lower threshold value for filtering.
    upper_threshold : float, default=75.0
        Upper threshold value for filtering.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with filtered values in the specified column.
    """
    data = data.filter(
        (pl.col(column).ge(lower_threshold) | pl.col(column).le(upper_threshold))
    )
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

    str_cols: list[str] = data.select(cs.string()).collect_schema().names()
    for col in str_cols:
        result[col] = data.select(col).unique().collect().to_numpy().flatten().tolist()

    return result


@typechecked
def clean_training_data(data: pl.LazyFrame) -> pl.LazyFrame:
    """Clean and preprocess training data by applying various transformations.

    Parameters
    ----------
    data : pl.LazyFrame
        Input LazyFrame containing the training data.

    Returns
    -------
    pl.LazyFrame
        Cleaned and preprocessed LazyFrame with transformed values.
    """
    lf_data: pl.LazyFrame = convert_values_to_lowercase(data=data)
    lf_data = rename_loan_intent_values(data=lf_data)
    lf_data = drop_invalid_values(
        data=lf_data,
        column=CONFIG.credit_score.steps_features.age_col,
        lower_threshold=CONFIG.credit_score.steps_features.lower_bound,
        upper_threshold=CONFIG.credit_score.steps_features.upper_bound,
    )
    return lf_data


@typechecked
def transform_array_to_lazyframe(
    array: np.ndarray, processor_pipe: Pipeline
) -> pl.LazyFrame:
    """Transform a NumPy array into a Polars LazyFrame using a scikit-learn Pipeline.

    Parameters
    ----------
    array : np.ndarray
        Input array of shape (n_samples, n_features) containing the processed features.
    processor_pipe : Pipeline
        Scikit-learn Pipeline object that was used to process the data and contains
        feature names information.

    Returns
    -------
    pl.LazyFrame
        A Polars LazyFrame containing the transformed data with feature names from
        the pipeline.
    """
    data: pl.LazyFrame = pl.from_numpy(
        array, schema=processor_pipe.get_feature_names_out().tolist()
    ).lazy()
    return data


class ScikitLearnPipeline(BaseModel):
    """A Pydantic model for scikit-learn Pipeline objects.

    This class wraps a scikit-learn Pipeline object and provides JSON serialization
    capabilities. It stores the pipeline, its parameters, and creation timestamp.

    Attributes
    ----------
    pipeline : Pipeline
        The scikit-learn Pipeline object containing the preprocessing/modeling steps.
    parameters : dict[str, str | int | float] | None
        Optional dictionary of pipeline parameters with their values.
    created_at : str | datetime
        Timestamp indicating when the pipeline was created.
    """

    pipeline: Pipeline
    parameters: dict[str, str | int | float] | None = None
    created_at: str | dt = Field(
        default_factory=lambda: dt.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            Pipeline: lambda pipe: {
                "steps": [(name, str(estimator)) for name, estimator in pipe.steps],
                "parameters": pipe.get_params(),
            }
        }


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
