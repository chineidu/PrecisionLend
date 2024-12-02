from omegaconf import DictConfig
from pydantic_settings import BaseSettings
from zenml.client import Client
from zenml.exceptions import EntityExistsError

from .utilities import load_config, logger


CONFIG: DictConfig = load_config()
SECRET_NAME = CONFIG.general.secret_name


class Settings(BaseSettings):
    """Application settings"""

    MLFLOW_HOST: str = "0.0.0.0"
    MLFLOW_PORT: int = 5500
    MLFLOW_BACKEND_STORE_URI: str = "sqlite:///mlflow.db"
    MLFLOW_TRACKING_URI: str = f"http://0.0.0.0:{MLFLOW_PORT}"

    username: str | None = None
    password: str | None = None
    uri: str = MLFLOW_TRACKING_URI

    @classmethod
    def load_settings(cls) -> "Settings":
        try:
            logger.info("Loading settings from the ZenML secret store.")

            settings_secrets = Client().get_secret(SECRET_NAME)
            settings = Settings(**settings_secrets.secret_values)

        except (RuntimeError, ValueError) as e:
            logger.warning(f"Error loading settings: {e}. Using default settings.")
            settings = Settings()

        return settings

    def export(self) -> None:
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
