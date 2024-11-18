from zenml.client import Client

from src.utilities import console


client = Client()


def get_mlflow_tracking_uri() -> None:
    tracking_uri: str = client.active_stack.experiment_tracker.get_tracking_uri()
    console.print(f"{tracking_uri=}")


if __name__ == "__main__":
    get_mlflow_tracking_uri()
