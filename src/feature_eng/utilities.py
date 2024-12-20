from datetime import datetime as dt
import json
from typing import Any

import numpy as np
from omegaconf import DictConfig
import pandas as pd
import polars as pl
from polars import selectors as cs
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
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
        (pl.col(column).ge(lower_threshold) & pl.col(column).le(upper_threshold))
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
def transform_array_to_dataframe(
    array: np.ndarray, processor_pipe: Pipeline
) -> pl.DataFrame:
    """Transform a NumPy array into a Polars DataFrame using a scikit-learn Pipeline.

    Parameters
    ----------
    array : np.ndarray
        Input array of shape (n_samples, n_features) containing the processed features.
    processor_pipe : Pipeline
        Scikit-learn Pipeline object that was used to process the data and contains
        feature names information.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the transformed data with feature names from
        the pipeline.
    """
    data: pl.DataFrame = pl.from_numpy(
        array, schema=processor_pipe.get_feature_names_out().tolist()
    )
    if "remainder__loan_status" in data.columns:
        data = data.rename({"remainder__loan_status": "loan_status"})
        return data
    return data


@typechecked
def numpy_to_json(obj: Any) -> Any:
    """Convert numpy arrays and numpy generic types to JSON serializable objects.

    Parameters
    ----------
    obj : Any
        Input object to convert, can be numpy array or numpy generic type.

    Returns
    -------
    Any
        JSON serializable object:
        - For np.ndarray: returns a Python list
        - For np.generic: returns a Python scalar

    Raises
    ------
    TypeError
        If the input object is not a numpy array or numpy generic type.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Type {type(obj)} not serializable")


@typechecked
def polars_to_json(obj: Any) -> Any:
    """Convert Polars data types to JSON serializable format.

    Parameters
    ----------
    obj : Any
        Input object to be converted to JSON serializable format. Can be a Polars
        numeric type (Float64, Int64, UInt64), DataTypeClass, or Python primitive
        type (str, int, float, bool).

    Returns
    -------
    Any
        JSON serializable version of the input object. Numeric types are converted
        to their Python equivalents, DataTypeClass to string, and primitive types
        are returned as-is.

    Raises
    ------
    TypeError
        If the input object type is not supported for JSON serialization.
    """

    if isinstance(obj, (pl.Float64, pl.Int64, pl.UInt64)):
        return obj.value

    # Handle Polars' DataTypeClass
    elif isinstance(obj, pl.datatypes.DataTypeClass):
        return str(obj)

    elif isinstance(obj, (str, int, float, bool)):
        return obj

    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


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

    pipe: Pipeline
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
def _get_datatrame_metadata(
    data: pl.DataFrame,
) -> dict[str, dict[str, Any] | list[str]]:
    """Get metadata information from a Polars DataFrame."""

    result = {
        "shape": {
            "n_rows": data.shape[0],
            "n_columns": data.shape[1],
        },
        "schema": {col: str(dtype) for col, dtype in data.schema.items()},
        "missing_values": data.null_count().to_dicts()[0],
        "num_duplicate_rows": data.is_duplicated().sum(),
    }

    return json.loads(json.dumps(result, default=polars_to_json))


@typechecked
def _get_sklearn_pipeline_metadata(pipe: Pipeline) -> dict[str, Any]:
    """Get metadata information from a scikit-learn Pipeline.

    This function extracts metadata information from a scikit-learn Pipeline
    by converting it to a ScikitLearnPipeline object.

    Parameters
    ----------
    pipe : Pipeline
        Input scikit-learn Pipeline object containing preprocessing/modeling steps

    Returns
    -------
    dict[str, Any]
        Dictionary containing Pipeline metadata including steps, parameters,
        and creation timestamp
    """
    if isinstance(pipe, Pipeline):
        pipe_: ScikitLearnPipeline = ScikitLearnPipeline(pipe=pipe)
    return {"output": pipe_.model_dump()}


@typechecked
def get_metadata(input: pl.DataFrame | pd.DataFrame) -> dict[str, Any]:
    """Get metadata information from DataFrame or Pipeline objects.

    This function serves as a dispatcher to get metadata from either
    DataFrame objects or scikit-learn Pipeline objects.

    Parameters
    ----------
    input : pl.DataFrame | pd.DataFrame | Pipeline
        Input object to extract metadata from. Can be either:
        - DataFrame of shape (n_rows, n_columns)

    Returns
    -------
    dict[str, Any]
        Dictionary containing metadata specific to the input object type
    """
    if isinstance(input, pd.DataFrame | pl.DataFrame):
        return _get_datatrame_metadata(data=input)
    raise TypeError(
        f"Input must be a pandas or polars DataFrame, got {type(input).__name__} instead"
    )


@typechecked
def split_data_into_train_test(
    data: pl.DataFrame, target: str, test_size: float, random_state: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split data into training and test sets using stratified sampling.

    Parameters
    ----------
    data : pl.DataFrame
        Input dataframe to split, shape (n_samples, n_features)
    target : str
        Name of the target column for stratification
    test_size : float
        Proportion of the dataset to include in the test split, between 0.0 and 1.0
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        A tuple containing:
        - X_train: Training data, shape (n_samples * (1-test_size), n_features)
        - X_test: Test data, shape (n_samples * test_size, n_features)
    """
    X_train, X_test = train_test_split(
        data, stratify=data[target], test_size=test_size, random_state=random_state
    )

    return X_train, X_test


@typechecked
def get_inference_features(
    data: pl.DataFrame, pipe: Pipeline, target: str, has_target: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract features from data using a fitted pipeline for inference.

    Parameters
    ----------
    data : pl.DataFrame
        Input data to transform, shape (n_samples, n_features)
    pipe : Pipeline
        Fitted scikit-learn pipeline for feature transformation
    target : str
        Name of the target column
    has_target : bool, default=False
        Whether the input data contains the target column

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        A tuple containing:
        - X_test_arr: Transformed feature array, shape (n_samples, n_transformed_features)
        - y_test_arr: Target array if has_target=True, shape (n_samples,), else None
    """
    if not has_target:
        data = data.with_columns(pl.lit(99).alias(target))

    # Transform data using pipeline
    arr_matrix: np.ndarray = pipe.transform(data.to_pandas())
    features_df: pl.DataFrame = transform_array_to_dataframe(
        array=arr_matrix, processor_pipe=pipe
    )

    # Split features and target
    X_test_arr: np.ndarray = features_df.drop(target).to_numpy()
    y_test_arr: np.ndarray | None = (
        features_df.select(target).to_numpy().flatten() if has_target else None
    )

    return X_test_arr, y_test_arr


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
