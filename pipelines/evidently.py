from typing import Any

from omegaconf import DictConfig
from zenml import pipeline
from steps.evidently import data_analyzer, data_report, data_splitter

from src.utilities import load_config

CONFIG: DictConfig = load_config()


@pipeline(enable_cache=False)
def data_report_pipeline() -> Any:
    path: str = CONFIG.credit_score.data.path
    ref_data, cur_data = data_splitter(path=path)
    report, _ = data_report(reference_dataset=ref_data, comparison_dataset=cur_data)
    data_analyzer(report)


if __name__ == "__main__":
    data_report_pipeline()
