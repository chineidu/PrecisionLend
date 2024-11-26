import json
from typing import Annotated, Any

import numpy as np
from omegaconf import DictConfig
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from zenml import step
from zenml.steps.base_step import BaseStep
from zenml.integrations.pandas.materializers.pandas_materializer import (
    PandasMaterializer,
)
from zenml.integrations.evidently.metrics import EvidentlyMetricConfig
from zenml.config.retry_config import StepRetryConfig
from zenml.integrations.evidently.steps import (
    EvidentlyColumnMapping,
    evidently_report_step,
)


from src.data_eng.extraction import ingest_data
from src.utilities import load_config, logger


CONFIG: DictConfig = load_config()
STEP_RETRY_CONFIG = StepRetryConfig(max_retries=5, delay=5, backoff=2)


@step(
    retry=STEP_RETRY_CONFIG,
    output_materializers={
        "train_data": PandasMaterializer,
        "test_data": PandasMaterializer,
    },
)
def data_splitter(
    path: str,
) -> tuple[Annotated[pd.DataFrame, "train_data"], Annotated[pd.DataFrame, "test_data"]]:
    """Split data into reference and current datasets with added noise.

    Parameters
    ----------
    path : str
        Path to the input data file.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing:
        - train_data: Reference dataset with shape (n_samples, n_features)
        - test_data: Current dataset with added noise with shape (n_samples, n_features)
    """
    logger.info("Loading data")
    data: pl.DataFrame = ingest_data(path).collect()

    # Add id column and noise
    rng: np.random.Generator = np.random.default_rng(CONFIG.general.random_state)
    data = data.with_columns(
        id=pl.arange(0, data.shape[0]),
        target=pl.col(CONFIG.credit_score.features.target),
        prediction=(
            pl.col(CONFIG.credit_score.features.target) + rng.normal(-2.85, 1)
        ).clip(0, 1),
    ).drop([CONFIG.credit_score.features.target])

    logger.info("Splitting the data into reference and current datasets")
    # Split the data
    ref_data: pl.DataFrame
    cur_data: pl.DataFrame
    ref_data, cur_data = train_test_split(
        data,
        test_size=CONFIG.general.current_size,
        random_state=CONFIG.general.random_state,
    )
    cur_data = cur_data.with_columns(
        person_income=(pl.col("person_income") + rng.normal(12_000, 500)).round(1),
        person_emp_exp=pl.col("person_emp_exp") + rng.integers(0, 10),
        cb_person_cred_hist_length=pl.col("cb_person_cred_hist_length")
        + rng.integers(0, 4),
    )

    return ref_data.to_pandas(), cur_data.to_pandas()


@step(retry=STEP_RETRY_CONFIG)
def data_analyzer(report: str) -> Annotated[dict[str, Any], "result"]:
    """Analyze Evidently report data.

    Parameters
    ----------
    report : str
        JSON string containing Evidently metrics report data.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the results from the first metric in the report.
    """
    logger.info("Running Evidently analysis")
    result: dict[str, Any] = json.loads(report)["metrics"][0]["result"]

    return result


data_report: BaseStep = evidently_report_step.with_options(
    parameters=dict(
        column_mapping=EvidentlyColumnMapping(
            target="target",
            prediction="prediction",
            id="id",
            numerical_features=list(CONFIG.credit_score.features.num_cols),
            categorical_features=list(CONFIG.credit_score.features.cat_cols),
        ),
        metrics=[
            EvidentlyMetricConfig.metric("DataQualityPreset"),
            EvidentlyMetricConfig.metric("DataDriftPreset"),
            EvidentlyMetricConfig.metric("ClassificationPreset"),
        ],
    )
)
