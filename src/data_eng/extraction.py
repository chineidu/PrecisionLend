import polars as pl
from typeguard import typechecked


from src.utilities import logger


@typechecked
def ingest_data(path: str) -> pl.LazyFrame:
    logger.info(f"Loading data from {path}")
    try:
        if path.endswith(".csv"):
            return pl.scan_csv(path)
        elif path.endswith(".parquet"):
            return pl.scan_parquet(path)
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise ValueError(f"Unsupported file format: {path}")
