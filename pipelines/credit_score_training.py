from typing import Annotated, Any, Literal

import numpy as np
from omegaconf import DictConfig
import polars as pl
from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW, SKLEARN

from steps.credit_score import (
    create_inference_features,
    create_training_features,
    evaluate_model,
    get_mlflow_experiment_status,
    load_data,
    load_estimator_object,
    load_training_processor,
    prepare_data,
    split_data,
    train_model,
)
from src.utilities import load_config, logger


CONFIG: DictConfig = load_config()
docker_settings: DockerSettings = DockerSettings(
    required_integrations=[MLFLOW, SKLEARN], requirements=["scikit-image"]
)
ESTIMATOR_NAME = Literal[
    "LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"
]


# @pipeline(enable_cache=False, settings={"docker": docker_settings})
@pipeline(enable_cache=False)
def credit_pipeline(
    estimator_type: Literal["classifier", "regressor"],
    estimator_name: ESTIMATOR_NAME,
) -> tuple[
    Annotated[Any, "trained_model"],
    Annotated[Any, "processor_pipe"],
]:
    target: str = CONFIG.credit_score.features.target
    n_splits: int = CONFIG.general.n_splits

    estimator: Any = load_estimator_object(estimator_type, estimator_name)

    try:
        logger.info("Running credit score pipeline")

        data: pl.LazyFrame = load_data(path=CONFIG.credit_score.data.path)
        cleaned_data: pl.LazyFrame = prepare_data(data=data)

        X_train, X_test = split_data(
            data=cleaned_data,
            target=target,
            test_size=CONFIG.general.test_size,
            random_state=CONFIG.general.random_state,
        )
        processor_pipe: Any = load_training_processor()

        # Training features
        X_train_feats_df, processor_pipe = create_training_features(
            data=X_train, pipe=processor_pipe
        )

        trained_model = train_model(
            data=X_train_feats_df, estimator=estimator, target=target, n_splits=n_splits
        )

        # Inference features
        X_test_arr: np.ndarray
        y_test_arr: np.ndarray
        X_test_arr, y_test_arr = create_inference_features(
            data=X_test, pipe=processor_pipe, has_target=True
        )
        # Model evaluation
        _: dict[str, float] = evaluate_model(
            estimator=trained_model, X_test=X_test_arr, y_test=y_test_arr
        )
        _, _ = get_mlflow_experiment_status()

        return trained_model, processor_pipe

    except Exception as e:
        logger.error(f"Error running credit score pipeline: {e}")
        raise


if __name__ == "__main__":
    estimator_type: str = CONFIG.general.model_selection.estimator_type
    estimator_name = CONFIG.general.model_selection.estimator_name
    run = credit_pipeline(estimator_type, estimator_name)
