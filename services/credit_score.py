from typing import Any
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
import polars as pl
from sklearn.base import ClassifierMixin

from src.feature_eng.utilities import (
    get_inference_features,
    probability_to_credit_score,
)
from src.mlflow_utils import (
    get_experiment_status,
    load_best_registered_model,
)

from utilities import load_zenml_artifact


class CreditRequestBody(BaseModel):
    person_age: int | float
    person_income: int | float
    person_emp_exp: int | float
    loan_amnt: int | float
    loan_int_rate: int | float
    loan_percent_income: int | float
    cb_person_cred_hist_length: int | float
    person_gender: str
    person_education: str
    person_home_ownership: str
    loan_intent: str
    previous_loan_defaults_on_file: str


app = FastAPI(title="Credit Score API", version="1.0.0")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "API is up and running!"}


@app.post("/score")
def get_credit_score(request_body: CreditRequestBody) -> Any:
    df: pl.DataFrame = pl.DataFrame(request_body.model_dump())
    pipe: Any = load_zenml_artifact(artifact_name="pipe")
    X_test_arr, y_test_arr = get_inference_features(
        data=df, pipe=pipe, target="loan_status"
    )
    experiment_id, run_id = get_experiment_status(
        experiment_name="credit_pipeline", metric="auc_score"
    )
    model: ClassifierMixin = load_best_registered_model(experiment_id, run_id)
    probability: np.ndarray = model.predict_proba(X_test_arr)[0][1]
    score: int = probability_to_credit_score(probability=probability)

    return {
        "probability_of_default": float(probability.round(2)),
        "credit_score": score,
    }
