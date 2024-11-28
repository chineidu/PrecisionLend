from typing import Any, Literal
import mlflow
from mlflow.tracking import MlflowClient
from typeguard import typechecked

from .config import settings


mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)


@typechecked
def list_experiments() -> list[mlflow.entities.Experiment] | None:
    """
    Lists all MLflow experiments in the tracking server.

    Returns
    -------
    list[mlflow.entities.Experiment] | None
        A list of MLflow Experiment objects containing experiment details,
        or None if no experiments are found.
    """
    client: MlflowClient = MlflowClient()

    # Retrieve all experiments
    all_experiments: list[mlflow.entities.Experiment] | None = (
        client.search_experiments()
    )

    return all_experiments


@typechecked
def get_experiment_id(experiment_name: str) -> str:
    """Get MLflow experiment ID for the credit pipeline.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment to retrieve.

    Returns
    -------
    str
        The experiment ID string from MLflow for the credit pipeline
        experiment.

    Notes
    -----
    This function retrieves the experiment ID from MLflow using the
    experiment name and returns it as a string.
    """
    current_experiment: dict[str, str] = dict(
        mlflow.get_experiment_by_name(experiment_name)
    )
    experiment_id: str = current_experiment["experiment_id"]
    return experiment_id


@typechecked
def get_experiment_details(experiment_id: str) -> MlflowClient.get_experiment:
    """
    Retrieve MLflow experiment details using the experiment ID.

    Parameters
    ----------
    experiment_id : str
        The unique identifier of the MLflow experiment.

    Returns
    -------
    MlflowClient.get_experiment
        The MLflow experiment object containing details like experiment name,
        artifact location, lifecycle stage, and other metadata.
    """
    client: MlflowClient = MlflowClient()
    experiment: MlflowClient.get_experiment = client.get_experiment(experiment_id)
    return experiment


@typechecked
def get_best_run_id(
    experiment_id: str, metric: Literal["auc_score", "rmse"] = "auc_score"
) -> str | None:
    """Get the run ID of the best performing model based on metric score.

    Parameters
    ----------
    experiment_id : str
        The MLflow experiment ID to search for runs.
    metric : Literal["auc_score", "rmse"], default="auc_score"
        The metric to use for sorting the runs. Can be either "auc_score" or "rmse".

    Returns
    -------
    str | None
        The run ID of the best performing model. Returns None if no runs are found.

    Notes
    -----
    The function sorts runs by the specified metric in descending order and returns
    the run ID of the top performing model.
    """
    client: MlflowClient = MlflowClient()
    runs: list[Any] = client.search_runs(experiment_ids=[experiment_id])
    best_runs: list[Any] = sorted(
        runs, key=lambda run: run.data.metrics[metric], reverse=True
    )
    top_n: int = 1  # Number of best models to fetch
    best_models: list[Any] = best_runs[:top_n]
    run_id: str = [run.info.run_id for run in best_models][0]

    return run_id


@typechecked
def load_best_model(experiment_name: str, artifact_path: str = "sklearn-model") -> Any:
    """Load the best performing model from MLflow tracking server.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment.
    artifact_path : str, default="sklearn-model"
        Path where the model is stored in MLflow.

    Returns
    -------
    Any
        The loaded model that can be either a classifier or regressor.
    """
    experiment_id: str = get_experiment_id(experiment_name)
    run_id: str = get_best_run_id(experiment_id)
    logged_model: str = f"runs:/{run_id}/{artifact_path}"
    loaded_model: Any = mlflow.pyfunc.load_model(logged_model)

    return loaded_model
