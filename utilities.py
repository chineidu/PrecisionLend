from typing import Any
from typeguard import typechecked
from zenml.client import Client

from src.utilities import console


client = Client()


@typechecked
def get_mlflow_tracking_uri() -> None:
    tracking_uri: str = client.active_stack.experiment_tracker.get_tracking_uri()
    console.print(f"{tracking_uri=}")


@typechecked
def load_zenml_artifact(artifact_name: str) -> Any:
    """Loads an artifact from the ZenML store."""
    artifact = client.get_artifact_version(artifact_name)
    loaded_artifact = artifact.load()

    return loaded_artifact


if __name__ == "__main__":
    get_mlflow_tracking_uri()
