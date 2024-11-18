from typing import Annotated, Any

from omegaconf import DictConfig
import polars as pl
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW, SKLEARN

from steps.credit_score import (
    load_training_processor,
    prepare_data,
    create_training_features,
    load_data,
    train_model,
)
from src.utilities import load_config, logger

docker_settings: DockerSettings = DockerSettings(
    required_integrations=[MLFLOW, SKLEARN], requirements=["scikit-image"]
)

CONFIG: DictConfig = load_config()


# @pipeline(enable_cache=False, settings={"docker": docker_settings})
@pipeline(enable_cache=False)
def credit_pipeline() -> (
    tuple[Annotated[ClassifierMixin, "trained_model"], Annotated[Any, "pipe"]]
):
    try:
        logger.info("Running credit score pipeline")
        data: pl.LazyFrame = load_data(path=CONFIG.credit_score.data.path)
        cleaned_data: pl.LazyFrame = prepare_data(data=data)
        pipe: Pipeline = load_training_processor()
        features_df, pipe = create_training_features(data=cleaned_data, pipe=pipe)
        trained_model = train_model(data=features_df)
        return trained_model, pipe

    except Exception as e:
        logger.error(f"Error running credit score pipeline: {e}")
        raise


if __name__ == "__main__":
    run = credit_pipeline()
