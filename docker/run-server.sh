#!/bin/bash
. .venv/bin/activate
which python

# Start MLflow server
mlflow server -h 0.0.0.0 \
    -p ${MLFLOW_PORT} \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --default-artifact-root ${MLFLOW_ARTIFACT_STORE}
