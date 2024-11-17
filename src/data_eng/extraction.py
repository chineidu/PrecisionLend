import numpy as np
import polars as pl
from sklearn.preprocessing import KBinsDiscretizer
from typeguard import typechecked


from src.utilities import logger


@typechecked
def ingest_data(path: str) -> pl.LazyFrame:
    """Load data from CSV or Parquet file and remove credit score column.

    Parameters
    ----------
    path : str
        File path to load data from. Must be either .csv or .parquet format.

    Returns
    -------
    pl.LazyFrame
        LazyFrame containing loaded data with credit_score column removed.
        Shape: (n_samples, n_features - 1)

    Raises
    ------
    ValueError
        If file format is not supported or file cannot be loaded.
    """
    logger.info(f"Loading data from {path}")
    try:
        if path.endswith(".csv"):
            return pl.scan_csv(path).drop("credit_score")
        elif path.endswith(".parquet"):
            return pl.scan_parquet(path).drop("credit_score")
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise ValueError(f"Unsupported file format: {path}")


@typechecked
def load_tenure_data(data: pl.LazyFrame) -> pl.LazyFrame:
    """Process loan data to create tenure categories based on loan amount and interest ratio.

    Parameters
    ----------
    data : pl.LazyFrame
        Input LazyFrame containing loan data with columns:
        - loan_amnt: loan amount
        - loan_int_rate: interest rate
        - loan_status: loan status

    Returns
    -------
    pl.LazyFrame
        Processed LazyFrame with added tenure column and removed original loan columns.
        Shape: (n_samples, n_features - 3)
    """
    bin_discretizer: KBinsDiscretizer = KBinsDiscretizer(
        n_bins=3, encode="ordinal", strategy="kmeans", random_state=42
    )

    tenure_dict: dict[int, int] = {0: 4, 1: 8, 2: 12}

    data = data.with_columns(
        amount_interest_ratio=(pl.col("loan_amnt") / pl.col("loan_int_rate")).round(2)
    )
    t_array: np.ndarray = bin_discretizer.fit_transform(
        data.select(["amount_interest_ratio"]).collect().to_numpy().reshape(-1, 1)
    )
    temp: pl.DataFrame = pl.from_numpy(t_array, schema={"tenure": pl.Int64})
    tenure_data: pl.LazyFrame = (
        pl.concat([data, temp.lazy()], how="horizontal")
        .with_columns(
            tenure=pl.col("tenure").map_elements(lambda x: tenure_dict.get(x))
        )
        .drop(
            [
                "person_age",
                "person_gender",
                "person_education",
                "loan_int_rate",
                "amount_interest_ratio",
                "loan_percent_income",
                "previous_loan_defaults_on_file",
                "loan_status",
            ]
        )
    )

    return tenure_data
