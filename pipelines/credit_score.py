from typing import Any

from omegaconf import DictConfig
import polars as pl
from sklearn.pipeline import Pipeline

from steps.credit_score import (
    load_training_processor,
    prepare_data,
    create_training_features,
    load_data,
)
from zenml import pipeline

from src.utilities import load_config, logger


CONFIG: DictConfig = load_config()


@pipeline(enable_cache=False)
def credit_pipeline() -> Any:
    try:
        logger.info("Running credit score pipeline")
        data: pl.LazyFrame = load_data(path=CONFIG.credit_score.data.path)
        lf_data: pl.LazyFrame = prepare_data(data=data)
        pipe: Pipeline = load_training_processor()
        lf_data, pipe = create_training_features(data=lf_data, pipe=pipe)
        return lf_data, pipe
    except Exception as e:
        logger.error(f"Error running credit score pipeline: {e}")
        raise


if __name__ == "__main__":
    run = credit_pipeline()
