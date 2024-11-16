import pandera.polars as pa
import polars as pl

from src.utilities import logger


class DataValidationSchema(pa.DataFrameModel):
    person_age: float = pa.Field(ge=18, le=100)
    person_gender: str = pa.Field(isin=["male", "female"])
    person_education: str = pa.Field(
        isin=["high school", "doctorate", "associate", "bachelor", "master"],
        raise_warning=False,
    )
    person_income: float = pa.Field(ge=5_000, le=10_000_000)
    person_emp_exp: int = pa.Field(ge=0, le=50)
    person_home_ownership: str = pa.Field(isin=["other", "mortgage", "rent", "own"])
    loan_amnt: float = pa.Field(ge=500, le=50_000)
    loan_intent: str = pa.Field(
        isin=[
            "education",
            "medical",
            "venture",
            "debt consolidation",
            "personal",
            "home improvement",
        ],
        raise_warning=False,
    )
    loan_int_rate: float = pa.Field(ge=1.0, le=30.0)
    loan_percent_income: float = pa.Field(ge=0.0, le=1.0)
    cb_person_cred_hist_length: float = pa.Field(ge=0.0, le=40.0)
    credit_score: int = pa.Field(ge=300, le=850)
    previous_loan_defaults_on_file: str = pa.Field(isin=["yes", "no"])
    loan_status: int = pa.Field(ge=0, le=1)


def validate_loan_data(data: pl.LazyFrame) -> pl.LazyFrame | None:
    try:
        data: pl.DataFrame = DataValidationSchema.validate(data.collect())  # type: ignore
        data = data.lazy()
        logger.info("Successfully validated the data.")
        return data
    except (pa.errors.SchemaErrors, pa.errors.SchemaError) as exc:
        logger.warning(f"Error validating the data: {exc}")
        return None
