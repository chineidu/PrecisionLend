from typing import Annotated

import numpy as np
from omegaconf import DictConfig
import pandas as pd
import polars as pl
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from zenml import step, log_artifact_metadata
from zenml.integrations.polars.materializers import PolarsMaterializer
from zenml.integrations.sklearn.materializers import SklearnMaterializer
import mlflow
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker

from src.data_eng.extraction import ingest_data
from src.utilities import logger, load_config
from src.feature_eng.pipelines import credit_loan_status_preprocessing_pipeline
from src.feature_eng.utilities import (
    clean_training_data,
    transform_array_to_dataframe,
)


experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for"
        " this to work."
    )
CONFIG: DictConfig = load_config()


@step(output_materializers=PolarsMaterializer)
def load_data(path: str) -> Annotated[pl.DataFrame, "data"]:
    """Load data from a file path into a Polars DataFrame.

    Parameters
    ----------
    path : str
        The file path to load data from.

    Returns
    -------
    pl.DataFrame
        The loaded data as a Polars DataFrame.
    """
    data: pl.DataFrame = ingest_data(path).collect()
    log_artifact_metadata(
        artifact_name="cleaned_data",
        metadata={
            "shape": {
                "n_rows": data.shape[0],
                "n_columns": data.shape[1],
            },
            "columns": data.columns,
        },
    )
    return data


@step(output_materializers=PolarsMaterializer)
def prepare_data(data: pl.DataFrame) -> Annotated[pl.DataFrame, "cleaned_data"]:
    """Preprocess the input data by cleaning and transforming.

    Parameters
    ----------
    data : pl.DataFrame
        Input DataFrame to be preprocessed.

    Returns
    -------
    pl.DataFrame
        Cleaned and preprocessed DataFrame.

    Raises
    ------
    Exception
        If any error occurs during preprocessing.
    """
    logger.info("Preprocessing data")
    try:
        cleaned_data: pl.DataFrame = clean_training_data(data=data.lazy()).collect()
        log_artifact_metadata(
            artifact_name="cleaned_data",
            metadata={
                "shape": {
                    "n_rows": cleaned_data.shape[0],
                    "n_columns": cleaned_data.shape[1],
                },
                "columns": cleaned_data.columns,
            },
        )
        return cleaned_data

    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise e


@step(output_materializers=SklearnMaterializer)
def load_training_processor() -> Annotated[Pipeline, "pipe"]:
    """Load and initialize the credit loan status preprocessing pipeline.

    Returns
    -------
    Pipeline
        Scikit-learn Pipeline object configured for credit loan status preprocessing.
    """
    logger.info("Loading training processor")
    pipe: Pipeline = credit_loan_status_preprocessing_pipeline()
    log_artifact_metadata(
        artifact_name="pipe", metadata={"parameters": str(pipe.get_params())}
    )
    return pipe


@step(
    output_materializers={
        "features_df": PolarsMaterializer,
        "pipe": SklearnMaterializer,
    }
)
def create_training_features(
    data: pl.DataFrame, pipe: Pipeline
) -> tuple[Annotated[pl.DataFrame, "features_df"], Annotated[Pipeline, "pipe"]]:
    """Create training features by transforming input data using preprocessing pipeline.

    Parameters
    ----------
    data : pl.DataFrame
        Input DataFrame to be transformed.
    pipe : Pipeline
        Scikit-learn preprocessing pipeline.

    Returns
    -------
    tuple[pl.DataFrame, Pipeline]
        A tuple containing:
        - features_df: Transformed data as a Polars DataFrame
        - pipe: Fitted preprocessing pipeline

    Notes
    -----
    The transformed array shape depends on the preprocessing steps in the pipeline.
    """
    try:
        logger.info("Creating training features")
        arr_matrix: np.ndarray = pipe.fit_transform(data.to_pandas())
        features_df: pl.DataFrame = transform_array_to_dataframe(
            array=arr_matrix, processor_pipe=pipe
        )
        log_artifact_metadata(
            artifact_name="cleaned_data",
            metadata={
                "shape": {
                    "n_rows": features_df.shape[0],
                    "n_columns": features_df.shape[1],
                },
                "columns": features_df.columns,
            },
        )
        log_artifact_metadata(
            artifact_name="pipe", metadata={"parameters": str(pipe.get_params())}
        )
        return (features_df, pipe)

    except Exception as e:
        logger.error(f"Error creating training features: {e}")
        raise e


@step(experiment_tracker=experiment_tracker.name)
def train_model(data: pl.DataFrame) -> ClassifierMixin:
    target: str = CONFIG.credit_score.features.target

    data: pd.DataFrame = data.to_pandas()  # type: ignore
    X_train: pd.DataFrame = data.drop(columns=[target])
    y_train: pd.Series = data[target]
    model: ClassifierMixin = LogisticRegression()
    mlflow.sklearn.autolog()

    logger.info("Training model")
    model.fit(X_train, y_train)
    return model
