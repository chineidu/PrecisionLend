from omegaconf import DictConfig
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from typeguard import typechecked
from src.utilities import load_config

# set_config(transform_output="polars")
CONFIG: DictConfig = load_config()


@typechecked
def credit_loan_status_preprocessing_pipeline() -> Pipeline:
    """Create a preprocessing pipeline for credit loan status prediction.

    The pipeline applies the following transformations:
    1. StandardScaler on numeric columns
    2. KBinsDiscretizer on numeric columns
    3. OneHotEncoder on categorical columns

    Returns
    -------
    Pipeline
        Scikit-learn pipeline with column transformations
    """
    # Convert OmegaConf lists to Python lists
    numeric_cols: list[str] = list(CONFIG.credit_score.features.num_cols)
    categorical_cols: list[str] = list(CONFIG.credit_score.features.cat_cols)

    column_transformer: ColumnTransformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "k_bins",
                KBinsDiscretizer(
                    n_bins=CONFIG.credit_score.discretizer.n_bins,
                    encode=CONFIG.credit_score.discretizer.encode,
                    strategy=CONFIG.credit_score.discretizer.strategy,
                    random_state=CONFIG.general.random_state,
                ),
                numeric_cols,
            ),
            ("cat", OneHotEncoder(), categorical_cols),
        ],
        remainder="passthrough",
    )
    pipe: Pipeline = Pipeline(
        steps=[
            ("column_transformer", column_transformer),
        ]
    )
    return pipe


@typechecked
def loan_tenure_preprocessing_pipeline() -> Pipeline:
    """Create a preprocessing pipeline for loan tenure prediction.

    The pipeline applies the following transformations:
    1. StandardScaler on numeric columns
    2. KBinsDiscretizer on numeric columns
    3. OneHotEncoder on categorical columns

    Returns
    -------
    Pipeline
        Scikit-learn pipeline with column transformations
    """
    # Convert OmegaConf lists to Python lists
    numeric_cols: list[str] = list(CONFIG.credit_score.features.num_cols)
    categorical_cols: list[str] = list(CONFIG.credit_score.features.cat_cols)

    column_transformer: ColumnTransformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "k_bins",
                KBinsDiscretizer(
                    n_bins=CONFIG.loan_tenure.discretizer.n_bins,
                    encode=CONFIG.loan_tenure.discretizer.encode,
                    strategy=CONFIG.loan_tenure.discretizer.strategy,
                    random_state=CONFIG.general.random_state,
                ),
                numeric_cols,
            ),
            ("cat", OneHotEncoder(), categorical_cols),
        ],
        remainder="passthrough",
    )
    pipe: Pipeline = Pipeline(
        steps=[
            ("column_transformer", column_transformer),
        ],
    )
    return pipe
