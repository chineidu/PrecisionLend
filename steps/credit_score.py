from typing import Annotated, Any, Literal

import numpy as np
from omegaconf import DictConfig
import pandas as pd
import polars as pl
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from zenml import step, log_artifact_metadata
from zenml.integrations.polars.materializers import PolarsMaterializer
from zenml.integrations.numpy.materializers.numpy_materializer import NumpyMaterializer
from zenml.integrations.sklearn.materializers import SklearnMaterializer
import mlflow
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker

from src.data_eng.extraction import ingest_data
from src.training import evaluate, train_model_with_cross_validation
from src.utilities import logger, load_config
from src.feature_eng.pipelines import credit_loan_status_preprocessing_pipeline
from src.feature_eng.utilities import (
    clean_training_data,
    split_data_into_train_test,
    transform_array_to_dataframe,
    get_metadata,
)


CONFIG: DictConfig = load_config()
ESTIMATOR_NAME = Literal[
    "LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"
]
experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for"
        " this to work."
    )


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
        artifact_name="data",
        metadata=get_metadata(input=data),
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
            metadata=get_metadata(input=cleaned_data),
        )
        return cleaned_data

    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise e


@step(
    output_materializers={
        "train_data": PolarsMaterializer,
        "test_data": PolarsMaterializer,
    }
)
def split_data(
    data: pl.DataFrame, target: str, test_size: float, random_state: int
) -> tuple[Annotated[pl.DataFrame, "train_data"], Annotated[pl.DataFrame, "test_data"]]:
    """Split input data into training and test sets.

    Parameters
    ----------
    data : pl.DataFrame
        Input DataFrame to be split, shape (n_samples, n_features).
    target : str
        Name of the target column to stratify the split.
    test_size : float
        Proportion of the dataset to include in the test split, between 0.0 and 1.0.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        A tuple containing:
        - train_data: Training data as a Polars DataFrame,
        shape ((1-test_size) * n_samples, n_features)
        - test_data: Test data as a Polars DataFrame,
        shape (test_size * n_samples, n_features)
    """
    logger.info("Splitting data into train and test sets")

    train_data: pl.DataFrame
    test_data: pl.DataFrame
    train_data, test_data = split_data_into_train_test(
        data=data, target=target, test_size=test_size, random_state=random_state
    )
    log_artifact_metadata(
        artifact_name="train_data",
        metadata=get_metadata(input=train_data),
    )
    log_artifact_metadata(
        artifact_name="test_data",
        metadata=get_metadata(input=test_data),
    )
    return train_data, test_data


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
            artifact_name="features_df",
            metadata=get_metadata(input=features_df),
        )
        log_artifact_metadata(
            artifact_name="pipe", metadata={"parameters": str(pipe.get_params())}
        )
        return (features_df, pipe)

    except Exception as e:
        logger.error(f"Error creating training features: {e}")
        raise e


@step(
    enable_cache=False,
    output_materializers={
        "X_test_arr": NumpyMaterializer,
        "y_test_arr": NumpyMaterializer,
    },
)
def create_inference_features(
    data: pl.DataFrame, pipe: Pipeline, has_target: bool = False
) -> tuple[
    Annotated[np.ndarray, "X_test_arr"], Annotated[np.ndarray | None, "y_test_arr"]
]:
    target = CONFIG.credit_score.features.target

    try:
        logger.info("Creating inference features")

        # Ensure target column presence
        if not has_target:
            data = data.with_columns(pl.lit(99).alias(target))

        # Transform data using pipeline
        arr_matrix = pipe.transform(data.to_pandas())
        features_df = transform_array_to_dataframe(
            array=arr_matrix, processor_pipe=pipe
        )

        # Split features and target
        X_test_arr = features_df.drop(target).to_numpy()
        y_test_arr = (
            features_df.select(target).to_numpy().flatten() if has_target else None
        )
        # Log artifact metadata
        log_artifact_metadata(
            artifact_name="X_test_arr",
            metadata={
                "shape": {"rows": X_test_arr.shape[0], "features": X_test_arr.shape[1]},
                "dtype": str(X_test_arr.dtype),
            },
        )

        log_artifact_metadata(
            artifact_name="y_test_arr",
            metadata={
                "shape": {"rows": y_test_arr.shape[0] if y_test_arr is not None else 0},
                "dtype": str(y_test_arr.dtype if y_test_arr is not None else None),
            },
        )

        return X_test_arr, y_test_arr

    except Exception as e:
        logger.error(f"Error creating inference features: {e}")
        raise


@step(output_materializers=SklearnMaterializer)
def load_estimator_object(
    estimator_type: Literal["classifier", "regressor"], estimator_name: ESTIMATOR_NAME
) -> Annotated[ClassifierMixin | RegressorMixin, "estimator"]:
    logger.info("Loading estimator object")

    if estimator_type == "classifier" and estimator_name == "LogisticRegression":
        logger.info("Using LogisticRegression")
        estimator: Any = LogisticRegression(
            penalty=CONFIG.estimators.classifier.LogisticRegression.penalty,
            C=CONFIG.estimators.classifier.LogisticRegression.C,
            solver=CONFIG.estimators.classifier.LogisticRegression.solver,
            max_iter=CONFIG.estimators.classifier.LogisticRegression.max_iter,
            multi_class=CONFIG.estimators.classifier.LogisticRegression.multi_class,
            random_state=CONFIG.general.random_state,
        )

    elif estimator_type == "classifier" and estimator_name == "RandomForestClassifier":
        logger.info("Using RandomForestClassifier")
        estimator: Any = RandomForestClassifier(  # type: ignore
            n_estimators=CONFIG.estimators.classifier.RandomForestClassifier.n_estimators,
            criterion=CONFIG.estimators.classifier.RandomForestClassifier.criterion,
            max_depth=CONFIG.estimators.classifier.RandomForestClassifier.max_depth,
            min_samples_split=CONFIG.estimators.classifier.RandomForestClassifier.min_samples_split,
            min_samples_leaf=CONFIG.estimators.classifier.RandomForestClassifier.min_samples_leaf,
            max_features=CONFIG.estimators.classifier.RandomForestClassifier.max_features,
            max_leaf_nodes=CONFIG.estimators.classifier.RandomForestClassifier.max_leaf_nodes,
            random_state=CONFIG.general.random_state,
        )

    elif (
        estimator_type == "classifier"
        and estimator_name == "GradientBoostingClassifier"
    ):
        logger.info("Using GradientBoostingClassifier")
        estimator: Any = GradientBoostingClassifier(  # type: ignore
            loss=CONFIG.estimators.classifier.GradientBoostingClassifier.loss,
            learning_rate=CONFIG.estimators.classifier.GradientBoostingClassifier.learning_rate,
            n_estimators=CONFIG.estimators.classifier.GradientBoostingClassifier.n_estimators,
            criterion=CONFIG.estimators.classifier.GradientBoostingClassifier.criterion,
            min_samples_split=CONFIG.estimators.classifier.GradientBoostingClassifier.min_samples_split,
            min_samples_leaf=CONFIG.estimators.classifier.GradientBoostingClassifier.min_samples_leaf,
            max_depth=CONFIG.estimators.classifier.GradientBoostingClassifier.max_depth,
            max_features=CONFIG.estimators.classifier.GradientBoostingClassifier.max_features,
            max_leaf_nodes=CONFIG.estimators.classifier.GradientBoostingClassifier.max_leaf_nodes,
            random_state=CONFIG.general.random_state,
        )
    return estimator


# @step(experiment_tracker=experiment_tracker.name)
# def train_model(data: pl.DataFrame) -> Annotated[ClassifierMixin, "model"]:
#     target: str = CONFIG.credit_score.features.target

#     data: pd.DataFrame = data.to_pandas()  # type: ignore
#     X_train: pd.DataFrame = data.drop(columns=[target])
#     y_train: pd.Series = data[target]
#     model: ClassifierMixin = LogisticRegression()
#     mlflow.sklearn.autolog()

#     logger.info("Training model")
#     model.fit(X_train, y_train)
#     return model


@step(experiment_tracker=experiment_tracker.name)
def train_model(
    data: pl.DataFrame,
    estimator: ClassifierMixin | RegressorMixin,
    target: str,
    n_splits: int,
) -> Annotated[ClassifierMixin, "estimator"]:
    data: pd.DataFrame = data.to_pandas()  # type: ignore
    X_train: pd.DataFrame = data.drop(columns=[target]).to_numpy()
    y_train: pd.Series = data[target].to_numpy()

    mlflow.sklearn.autolog()
    logger.info("Training model")
    estimator, _, _, _ = train_model_with_cross_validation(
        X=X_train, y=y_train, estimator=estimator, n_splits=n_splits
    )

    return estimator


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    estimator: ClassifierMixin,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Annotated[dict[str, float], "results"]:
    logger.info("Evaluating model")
    results: dict[str, float] = evaluate(
        estimator=estimator, X_test=X_test, y_test=y_test
    )
    logger.info(f"Evaluation results: {results}")
    mlflow.log_metrics(results)
    log_artifact_metadata(artifact_name="results", metadata=results)
    return results
