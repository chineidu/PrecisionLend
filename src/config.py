from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    MLFLOW_HOST: str | None = "0.0.0.0"
    MLFLOW_PORT: str | None = None
    MLFLOW_BACKEND_STORE_URI: str | None = None
    MLFLOW_TRACKING_URI: str | None = None

    class Config:
        """Configuration for environment variable loading"""

        env_file = ".env"
        env_file_encoding = "utf-8"


# Load environment variables
settings: Settings = Settings()
