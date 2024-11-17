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

from src.utilities import load_config


CONFIG: DictConfig = load_config()


@pipeline(enable_cache=False)
def main() -> Any:
    data: pl.LazyFrame = load_data(path=CONFIG.credit_score.data.path)
    lf_data: pl.LazyFrame = prepare_data(data=data)
    pipe: Pipeline = load_training_processor()
    lf_data, pipe = create_training_features(data=lf_data, pipe=pipe)
    # lf_data = create_training_features(data=lf_data)
    # return data
    return lf_data, pipe


if __name__ == "__main__":
    run = main()
