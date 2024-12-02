from datetime import datetime as dt
from typing import Any

import click
from omegaconf import DictConfig

from pipelines import credit_pipeline, data_report_pipeline
from src.utilities import load_config, logger
from src.config import settings


CONFIG: DictConfig = load_config()


@click.command(help="Main entrypoint for the pipeline execution.")
@click.option(
    "--no-cache",
    "-cs",
    is_flag=True,
    help="Whether to enable caching for the pipeline.",
)
@click.option(
    "--run-credit-score-training",
    "-cs",
    is_flag=True,
    help="Whether to run the credit score training pipeline.",
)
@click.option(
    "--run-data-report",
    "-dr",
    is_flag=True,
    help="Whether to run the model and data drift report pipeline.",
)
@click.option(
    "--run-export-settings",
    "-xs",
    is_flag=True,
    help="Whether to export the settings to the ZenML secret store.",
)
def main(
    no_cache: bool = False,
    run_credit_score_training: bool = False,
    run_data_report: bool = False,
    run_export_settings: bool = False,
) -> None:
    """Execute the main pipeline processes based on provided flags.

    Parameters
    ----------
    no_cache : bool, default=False
        Flag to disable caching for the pipeline.
    run_credit_score_training : bool, default=False
        Flag to execute credit score training pipeline.
    run_data_report : bool, default=False
        Flag to execute data report pipeline.
    run_export_settings : bool, default=False
        Flag to export settings to ZenML secret store.

    Returns
    -------
    None
        This function doesn't return any value.

    Raises
    ------
    AssertionError
        If no process is selected to run.
    """
    assert (
        run_credit_score_training or run_data_report or run_export_settings
    ) is True, "Please select at least one process to run."

    if run_export_settings:
        logger.info("Exporting settings to the ZenML secret store.")
        settings.export()

    pipeline_args: dict[str, Any] = {
        "enable_cache": not no_cache,
    }

    if run_credit_score_training:
        run_args_cst_pipeline: dict[str, Any] = {}
        estimator_type: str = CONFIG.general.model_selection.estimator_type
        estimator_name: str = CONFIG.general.model_selection.estimator_name
        run_args_cst_pipeline["estimator_type"] = estimator_type
        run_args_cst_pipeline["estimator_name"] = estimator_name

        pipeline_args["run_name"] = (
            f"credit_score_training_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )
        credit_pipeline.with_options(**pipeline_args)(**run_args_cst_pipeline)

    if run_data_report:
        run_args_dr_pipeline: dict[str, Any] = {}
        pipeline_args["run_name"] = (
            f"data_report_pipeline_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )
        data_report_pipeline.with_options(**pipeline_args)(**run_args_dr_pipeline)


if __name__ == "__main__":
    main()
