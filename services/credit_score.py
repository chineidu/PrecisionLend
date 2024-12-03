import json
from typing import Any

from fastapi import FastAPI, HTTPException
import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator
from ray import serve
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline

from src.feature_eng.utilities import (
    get_inference_features,
    numpy_to_json,
    probability_to_credit_score,
)
from src.mlflow_utils import (
    get_experiment_status,
    load_best_registered_model,
)
from src.utilities import logger

from utilities import load_zenml_artifact


class CreditRequestBody(BaseModel):
    """Pydantic model for credit request validation and data parsing.

    This class defines the structure and validation rules for credit request data.
    All string fields are automatically converted to lowercase.
    """

    model_config = ConfigDict(from_attributes=True)

    person_age: int | float = Field(ge=17, le=85)
    person_income: float = Field(ge=1_000.0)
    person_emp_exp: int | float = Field(ge=0)
    loan_amnt: float = Field(ge=500.0, le=50_000.0)
    loan_int_rate: float = Field(ge=1.0, le=40.0)
    loan_percent_income: float = Field(ge=0.0, le=1.0)
    cb_person_cred_hist_length: int | float = Field(ge=0, le=20)
    person_gender: str
    person_education: str
    person_home_ownership: str
    loan_intent: str
    previous_loan_defaults_on_file: str

    @field_validator(
        "person_gender",
        "person_education",
        "person_home_ownership",
        "loan_intent",
        "previous_loan_defaults_on_file",
    )
    def convert_string_fields_to_lowercase(cls, v: str) -> str:
        """Convert string fields to lowercase.

        Parameters
        ----------
        v : str
            Input string value

        Returns
        -------
        str
            Lowercase version of input string
        """
        return v.lower().strip()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "person_age": 35,
                    "person_income": 72_000.00,
                    "person_emp_exp": 6,
                    "loan_amnt": 25_000.00,
                    "loan_int_rate": 14.2,
                    "loan_percent_income": 0.28,
                    "cb_person_cred_hist_length": 3,
                    "person_gender": "male",
                    "person_education": "bachelor",
                    "person_home_ownership": "rent",
                    "loan_intent": "education",
                    "previous_loan_defaults_on_file": "no",
                }
            ]
        }
    }


app = FastAPI(title="Credit Score API", version="1.0.0")


@serve.deployment(
    name="credit_score_deployment",
    num_replicas="auto",  # enable autoscaling
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
)
@serve.ingress(app)
class CreditScoreDeployment:
    def __init__(self) -> None:
        self.pipe: Pipeline = load_zenml_artifact(artifact_name="pipe")
        self.experiment_id, self.run_id = get_experiment_status(
            experiment_name="credit_pipeline", metric="auc_score"
        )
        self.model: ClassifierMixin = load_best_registered_model(
            self.experiment_id, self.run_id
        )

    @app.get("/")
    def root(self) -> dict[str, str]:
        return {"message": "API is up and running!"}

    @app.post("/score")
    def get_credit_score(self, request: CreditRequestBody) -> dict[str, Any]:
        try:
            data: pl.DataFrame = pl.DataFrame(request.model_dump())
            X_test_arr, _ = get_inference_features(
                data=data, pipe=self.pipe, target="loan_status"
            )
            probability: np.ndarray = self.model.predict_proba(X_test_arr)[0][1]
            score: int = probability_to_credit_score(probability=probability)
            result: dict[str, Any] = {
                "probability_of_default": probability.round(4),
                "credit_score": score,
            }
            logger.info(f"Result: {result}")
            return json.loads(json.dumps(result, default=numpy_to_json))

        except Exception as e:
            logger.error(f"Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


credit_score_deployment = CreditScoreDeployment.bind()  # type: ignore
# serve.run(credit_score_deployment, name="credit_score_deployment", route_prefix="/api/v1")
