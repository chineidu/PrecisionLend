import mlflow
from mlflow.tracking import MlflowClient

from .config import settings


mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)


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
