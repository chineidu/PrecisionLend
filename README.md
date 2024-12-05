# PrecisionLend

Machine Learning-Powered Loan Processing and Credit Scoring

## Table of Contents

- [PrecisionLend](#precisionlend)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Key Features](#key-features)
  - [Technologies](#technologies)
  - [Development Roadmap](#development-roadmap)
  - [Installations](#installations)
  - [ZenML Integrations](#zenml-integrations)
    - [Useful Commands](#useful-commands)
    - [MLFlow Integration](#mlflow-integration)
      - [Get MLFlow Tracking URL](#get-mlflow-tracking-url)
      - [Configure MLFLOW Tracking In Steps And Pipelines](#configure-mlflow-tracking-in-steps-and-pipelines)
        - [A.) Steps](#a-steps)
        - [B.) Pipelines](#b-pipelines)
          - [1.) Run ZenML with MLFlow Docker Integration](#1-run-zenml-with-mlflow-docker-integration)
          - [2.) Run ZenML with A Connected MLFlow Server](#2-run-zenml-with-a-connected-mlflow-server)
    - [ZenML Evidently Integration](#zenml-evidently-integration)
    - [ZenML Stack CLI Commands](#zenml-stack-cli-commands)
  - [ML Services](#ml-services)
    - [1.) Credit Score Prediction Service](#1-credit-score-prediction-service)
  - [Docker](#docker)
    - [MLFlow Setup](#mlflow-setup)

## Overview

- PrecisionLend is an innovative lending solution leveraging machine learning for accurate credit scoring, optimized loan processing and personalized financial recommendations.

### Key Features

1. Creditworthiness scoring: Predictive modeling for informed lending decisions.
2. Loan application processing: Automated risk assessment and approval.
3. Loan amount estimation: Accurate loan amount estimation.
4. Loan term and rate estimation: Accurate loan term and rate estimation.
5. Customer segmentation: Data-driven targeting for enhanced customer experiences.

## Technologies

- Machine learning frameworks: Scikit-learn, XGBoost.
- Programming languages: Python.
- Cloud infrastructure: AWS.
- Data storage: PostgreSQL.
- Version control: Git.
- Pipelining, Experimentation and Model Serving: ZenML, MLFlow, Bentoml.
- Deployment: Docker and Kubernetes.
- Monitoring: Prometheus, Grafana.

## Development Roadmap

- Data collection and preprocessing
- Model training and evaluation
- Integration with core banking systems
- Continuous monitoring and refinement (retraining)
- Deployment and scaling
- User feedback and improvement

## Installations

- Install [UV](https://docs.astral.sh/uv/getting-started/installation/) by Astral.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- To create a virtual environment, run the following command:

```sh
# Create a new Python project (optional)
uv init

# Activate the virtual environment (Linux/macOS)
. .venv/bin/activate
```

- To sync the project's dependencies with the local environment, run the following command:

```sh
# Sync the project's dependencies with the local environment
uv sync

# Create a lockfile for the project's dependencies
uv lock
```

- Install `pre-commit` hooks:

```sh
uv add pre-commit
pre-commit install
```

## ZenML Integrations

### Useful Commands

```sh
# Start the ZenML dashboard locally.
zenml up

# Stop the ZenML dashboard
zenml down

# Downgrade zenml version in global config.
zenml downgrade

# Download and purge the local database and re-initialize the global configuration to bring it back to its default factory state
zenml downgrade && zenml clean

# show the ZenML dashboard.
zenml show
```

- It is assmumed that you have [ZenML](https://zenml.io/) installed and configured in your environment.

### [MLFlow Integration](https://docs.zenml.io/stack-components/experiment-trackers/mlflow)

- For remote tracking, check [here](https://mlflow.org/docs/latest/tracking/tutorials/remote-server.html).

```sh
# Add MLFlow to your stack
zenml integration install mlflow -y

# List all stacks
zenml stack list

# Register and set a stack with the new experiment tracker
# 1. Create a secret
SECRET_NAME=mlflow_secret
USERNAME=your_username
PASSWORD=your_password
URI=http://localhost:5000

zenml secret create ${SECRET_NAME} \
    --username=${USERNAME} \
    --password=${PASSWORD}

# 2. Create an experiment tracker component.
# Reference the username, password and uri in the experiment tracker component
zenml experiment-tracker register ${EXPERIMENT_TRACKER_NAME} \
    --flavor=mlflow \
    --tracking_username={{mlflow_secret.username}} \
    --tracking_password={{mlflow_secret.password}} \
    --tracking_uri={{mlflow_secret.uri}}

# 3. Register and set the new stack and its components
STACK_NAME=custom_stack
EXPERIMENT_TRACKER_NAME=mlflow
zenml stack register ${STACK_NAME} \
  -a default -o default \
  -e ${EXPERIMENT_TRACKER_NAME} --set


# 4. Start the MLFlow server locally
HOST=0.0.0.0
PORT=5000
BACKEND_STORE_URI=sqlite:///mlflow.db

mlflow server \
  --backend-store-uri ${BACKEND_STORE_URI} \
  --host ${HOST} \
  --port ${PORT}
```

#### Get MLFlow Tracking URL

```py
from zenml.client import Client

client = Client()
tracking_uri: str = client.active_stack.experiment_tracker.get_tracking_uri()

print(tracking_uri)
```

- To use the MLFlow tracking server, run the following command:

```sh
tracking_uri=<value_from_previous_step>
mlflow ui --backend-store-uri ${tracking_uri}
```

#### Configure MLFLOW Tracking In Steps And Pipelines

##### A.) Steps

```py
from typing import Annotated

from omegaconf import DictConfig
import polars as pl
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from zenml import step, log_artifact_metadata
from zenml.integrations.polars.materializers import PolarsMaterializer
from zenml.integrations.sklearn.materializers import SklearnMaterializer
import mlflow
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker

from src.utilities import logger, load_config
from src.feature_eng.utilities import  transform_array_to_dataframe



experiment_tracker = Client().active_stack.experiment_tracker
if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for"
        " this to work."
    )
CONFIG: DictConfig = load_config()



@step(
    output_materializers={
        "features_df": PolarsMaterializer,
        "pipe": SklearnMaterializer,
    }
)
def create_training_features(
    data: pl.DataFrame, pipe: Pipeline
) -> tuple[Annotated[pl.DataFrame, "features_df"], Annotated[Pipeline, "pipe"]]:
    """Create training features by transforming input data using preprocessing pipeline.

    Parameters
    ----------
    data : pl.DataFrame
        Input DataFrame to be transformed.
    pipe : Pipeline
        Scikit-learn preprocessing pipeline.

    Returns
    -------
    tuple[pl.DataFrame, Pipeline]
        A tuple containing:
        - features_df: Transformed data as a Polars DataFrame
        - pipe: Fitted preprocessing pipeline

    Notes
    -----
    The transformed array shape depends on the preprocessing steps in the pipeline.
    """
    try:
        logger.info("Creating training features")
        arr_matrix: np.ndarray = pipe.fit_transform(data.to_pandas())
        features_df: pl.DataFrame = transform_array_to_dataframe(
            array=arr_matrix, processor_pipe=pipe
        )
        log_artifact_metadata(
            artifact_name="cleaned_data",
            metadata={
                "shape": {
                    "n_rows": features_df.shape[0],
                    "n_columns": features_df.shape[1],
                },
                "columns": features_df.columns,
            },
        )
        log_artifact_metadata(
            artifact_name="pipe", metadata={"parameters": str(pipe.get_params())}
        )
        return (features_df, pipe)

    except Exception as e:
        logger.error(f"Error creating training features: {e}")
        raise e

# Add MLFlow tracking to the step
@step(experiment_tracker=experiment_tracker.name)
def train_model(data: pl.DataFrame) -> ClassifierMixin:
    target: str = CONFIG.credit_score.features.target

    data: pd.DataFrame = data.to_pandas()  # type: ignore
    X_train: pd.DataFrame = data.drop(columns=[target])
    y_train: pd.Series = data[target]
    model: ClassifierMixin = LogisticRegression()
    mlflow.sklearn.autolog()

    logger.info("Training model")
    model.fit(X_train, y_train)
    return model

```

##### B.) Pipelines

###### 1.) Run ZenML with MLFlow Docker Integration

```py
from typing import Annotated, Any

from omegaconf import DictConfig
import polars as pl
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW, SKLEARN

from steps.credit_score import (
    load_training_processor,
    prepare_data,
    create_training_features,
    load_data,
    train_model,
)
from src.utilities import load_config, logger

# Requirements installation order:
# Depending on the configuration of this object, requirements will be installed in the following order (each step optional):

# 1.) The packages installed in your local python environment
# 2.) The packages required by the stack unless this is disabled by setting install_stack_requirements=False.
# 3.) The packages specified via the required_integrations
# 4.) The packages specified via the requirements attribute
docker_settings: DockerSettings = DockerSettings(
    # List of ZenML integrations that should be installed inside the Docker image.
    required_integrations=[MLFLOW, SKLEARN],
    # Path to a requirements file or a list of required pip packages
    requirements=["scikit-image"]
)

CONFIG: DictConfig = load_config()

# Define the pipeline to run with Docker
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def credit_pipeline() -> (
    tuple[Annotated[ClassifierMixin, "trained_model"], Annotated[Any, "pipe"]]
):
    try:
        logger.info("Running credit score pipeline")
        data: pl.LazyFrame = load_data(path=CONFIG.credit_score.data.path)
        cleaned_data: pl.LazyFrame = prepare_data(data=data)
        pipe: Pipeline = load_training_processor()
        features_df, pipe = create_training_features(data=cleaned_data, pipe=pipe)
        trained_model = train_model(data=features_df)
        return trained_model, pipe

    except Exception as e:
        logger.error(f"Error running credit score pipeline: {e}")
        raise


if __name__ == "__main__":
    run = credit_pipeline()
```

###### 2.) Run ZenML with A Connected MLFlow Server

```py
from typing import Annotated, Any

from omegaconf import DictConfig
import polars as pl
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from zenml import pipeline

from steps.credit_score import (
    load_training_processor,
    prepare_data,
    create_training_features,
    load_data,
    train_model,
)
from src.utilities import load_config, logger


CONFIG: DictConfig = load_config()

# Define the pipeline to run WITHOUT Docker
@pipeline(enable_cache=False)
def credit_pipeline() -> (
    tuple[Annotated[ClassifierMixin, "trained_model"], Annotated[Any, "pipe"]]
):
    try:
        logger.info("Running credit score pipeline")
        data: pl.LazyFrame = load_data(path=CONFIG.credit_score.data.path)
        cleaned_data: pl.LazyFrame = prepare_data(data=data)
        pipe: Pipeline = load_training_processor()
        features_df, pipe = create_training_features(data=cleaned_data, pipe=pipe)
        trained_model = train_model(data=features_df)
        return trained_model, pipe

    except Exception as e:
        logger.error(f"Error running credit score pipeline: {e}")
        raise


if __name__ == "__main__":
    run = credit_pipeline()
```

### ZenML Evidently Integration

- To install the ZenML Evidently integration, run the following command:

```sh
# Install ZenML Evidently Integration With UV
zenml integration install evidently -y --uv
```

- Register a ZenML data validator:

```sh
VALIDATOR_NAME="evidently_data_validator"
zenml data-validator register ${VALIDATOR_NAME} \
  --flavor=evidently
```

- Update an existing ZenML stack to include the Evidently data validator:

```sh
zenml stack update -dv ${VALIDATOR_NAME}
```

### ZenML Stack CLI Commands

- To display the help message for the `zenml stack --help` command:

## ML Services

- [Best Practices](https://docs.ray.io/en/latest/serve/production-guide/best-practices.html)

### 1.) Credit Score Prediction Service

- Use highly configurable serve `config files` for production deployments.

```sh
# From the root of the project
export FILENAME="services.credit_score"
export DEPLOYMENT_NAME="credit_score_deployment"
export CONFIG_FILE_NAME="services/serve_config.yaml"

# Create a serve config file
serve build ${FILENAME}:${DEPLOYMENT_NAME} -o ${CONFIG_FILE_NAME}
```

- Update the serve config file with the correct host and port as per your requirements.

- Use [serve deploy](https://docs.ray.io/en/latest/serve/advanced-guides/deploy-vm.html#deploy-on-vm) (especially for deployments on VMs) for production deployments.

```sh
# Start Ray on the head node
ray start --head
...

# Deploy the Serve application on the head node using the config file
serve deploy ${CONFIG_FILE_NAME}
```

- You can also deploy to a remote VM by following the steps [here](https://docs.ray.io/en/latest/serve/advanced-guides/deploy-vm.html#using-a-remote-cluster)
- Add [autoscaling](https://docs.ray.io/en/latest/serve/autoscaling-guide.html) to your Serve deployment.

## Docker

### MLFlow Setup

```text
MLFLOW_HOST=0.0.0.0
MLFLOW_PORT=5000
# MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
MLFLOW_TRACKING_URI=http://$MLFLOW_HOST:$MLFLOW_PORT
# Use PostgreSQL as backend store (Docker)
MLFLOW_BACKEND_STORE_URI=postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB

export MLFLOW_HOST=mlflow-tracking-server
export MLFLOW_PORT=5000
export MLFLOW_TRACKING_URI=http://$MLFLOW_HOST:$MLFLOW_PORT
export MLFLOW_BACKEND_STORE_URI=postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB
export MLFLOW_ARTIFACT_STORE=/mlflow-artifact-store

export POSTGRES_HOST=mlflow-backend-store
export POSTGRES_PORT=5432
export POSTGRES_USER=mlflow
export POSTGRES_PASSWORD=mlflow
export POSTGRES_DB=mlflow_db
```
