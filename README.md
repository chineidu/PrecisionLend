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
    - [ZenML Stack CLI Commands](#zenml-stack-cli-commands)
    - [Create ZenML Secret](#create-zenml-secret)

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

# Register the MLFlow experiment tracker
zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow

# Register and set a stack with the new experiment tracker
STACK_NAME=custom_stack
EXPERIMENT_TRACKER_NAME=mlflow
zenml stack register $STACK_NAME -a default -o default \
  -e $EXPERIMENT_TRACKER_NAME --set

# Authentication Methods
# Create a secret called `mlflow_secret` with key-value pairs for the
# username and password to authenticate with the MLflow tracking server
zenml secret create <mlflow_secret_name> \
    --username=<USERNAME> \
    --password=<PASSWORD>

# Reference the username and password in our experiment tracker component
zenml experiment-tracker register ${EXPERIMENT_TRACKER_NAME} \
    --flavor=mlflow \
    --tracking_username={{mlflow_secret.username}} \
    --tracking_password={{mlflow_secret.password}} \
    ...


# Start the MLFlow server locally
HOST="127.0.0.1"
PORT="5000"

mlflow server --host ${HOST} --port ${PORT}
```

### Get MLFlow Tracking URL

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

### ZenML Stack CLI Commands

- To display the help message for the `zenml stack register --help` command:

```txt
Options:
  -a, --artifact-store TEXT       Name of the artifact store for this stack.
  -o, --orchestrator TEXT         Name of the orchestrator for this stack.
  -c, --container_registry TEXT   Name of the container registry for this
                                  stack.
  -r, --model_registry TEXT       Name of the model registry for this stack.
  -s, --step_operator TEXT        Name of the step operator for this stack.
  -f, --feature_store TEXT        Name of the feature store for this stack.
  -d, --model_deployer TEXT       Name of the model deployer for this stack.
  -e, --experiment_tracker TEXT   Name of the experiment tracker for this
                                  stack.
  -al, --alerter TEXT             Name of the alerter for this stack.
  -an, --annotator TEXT           Name of the annotator for this stack.
  -dv, --data_validator TEXT      Name of the data validator for this stack.
  -i, --image_builder TEXT        Name of the image builder for this stack.
  --set                           Immediately set this stack as active.
  -p, --provider [aws|azure|gcp]  Name of the cloud provider for this stack.
  -sc, --connector TEXT           Name of the service connector for this
```

- e.g.

```sh
# Register and set a stack with the new experiment tracker
STACK_NAME=custom_stack

zenml stack register ${STACK_NAME} \  # name of the stack
-a default \ # name of the artifact store
-o default \ # name of the orchestrator
-e ${STACK_NAME} --set  \ # set the stack using the STACK_NAME variable
```

### Create ZenML Secret

- Create a secret called `mlflow_secret` with key-value pairs for the username and password to authenticate with the MLflow tracking server.

```sh
zenml secret create <mlflow_secret_name> \
    --username=default \
    --password=password \
    --uri=<path_to_mlflow_uri>

# Reference the username and password in our experiment tracker component
zenml experiment-tracker register <mlflow_experiment_tracker_name> \
    --flavor=mlflow \
    --tracking_username={{mlflow_secret.username}} \
    --tracking_password={{mlflow_secret.password}} \
    --tracking_uri={{mlflow_secret.uri}}
```
