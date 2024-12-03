from omegaconf import DictConfig
from pydantic_settings import BaseSettings
from zenml.client import Client
from zenml.exceptions import EntityExistsError

from .utilities import load_config, logger


CONFIG: DictConfig = load_config()
SECRET_NAME = CONFIG.general.secret_name


class Settings(BaseSettings):
    """Application settings class for managing MLflow configuration and credentials.

    Attributes
    ----------
    MLFLOW_HOST : str
        The host address for MLflow server, defaults to "0.0.0.0"
    MLFLOW_PORT : int
        The port number for MLflow server, defaults to 5500
    MLFLOW_BACKEND_STORE_URI : str
        The URI for MLflow backend storage, defaults to SQLite database
    MLFLOW_TRACKING_URI : str
        The complete tracking URI for MLflow server
    username : str | None
        Optional username for authentication
    password : str | None
        Optional password for authentication
    uri : str
        The tracking URI, defaults to MLFLOW_TRACKING_URI
    """

    MLFLOW_HOST: str = "http://localhost"
    MLFLOW_PORT: int = 5000
    MLFLOW_BACKEND_STORE_URI: str = "sqlite:///mlflow.db"
    MLFLOW_TRACKING_URI: str = f"{MLFLOW_HOST}:{MLFLOW_PORT}"

    username: str | None = None
    password: str | None = None
    uri: str = MLFLOW_TRACKING_URI

    @classmethod
    def load_settings(cls) -> "Settings":
        """Load settings from ZenML secret store or create default settings.

        Returns
        -------
        Settings
            An instance of Settings class with loaded or default values

        Raises
        ------
        RuntimeError
            If there's an error accessing the secret store
        ValueError
            If there's an error parsing the secret values
        """
        try:
            logger.info("Loading settings from the ZenML secret store.")

            settings_secrets = Client().get_secret(SECRET_NAME)
            settings = Settings(**settings_secrets.secret_values)

        except (RuntimeError, ValueError) as e:
            logger.warning(f"Error loading settings: {e}. Using default settings.")
            settings = Settings()

        return settings

    def export(self) -> None:
        """Export current settings to ZenML secret store.

        Stores all non-None values as strings in the secret store.

        Raises
        ------
        EntityExistsError
            If the secret with the same name already exists in the store
        """
        env_vars = settings.model_dump()

        for key, value in env_vars.items():
            if value is not None:
                env_vars[key] = str(value)

        client = Client()

        try:
            client.create_secret(name=SECRET_NAME, values=env_vars)

        except EntityExistsError:
            logger.warning(
                f"Secret scorep {SECRET_NAME!r} already exists. "
                f"Delete it manually by running `zenml secret delete {SECRET_NAME!r}` "
                "and try again."
            )

    class Config:
        """Configuration for environment variable loading"""

        env_file = ".env"
        env_file_encoding = "utf-8"


# Load environment variables
settings: Settings = Settings().load_settings()
