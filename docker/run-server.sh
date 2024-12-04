#!/bin/bash

# Activate the virtual environment
. /opt/venv/bin/activate
mlflow server -h 0.0.0.0 \
    -p ${MLFLOW_PORT} \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --default-artifact-root ${MLFLOW_ARTIFACT_STORE}
