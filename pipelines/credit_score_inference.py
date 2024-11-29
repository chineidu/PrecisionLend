from typing import Any


from steps.credit_score import create_inference_features, load_trained_model_with_mlflow


def predict_credit_score() -> Any:
    # TODO: Update this function!!!
    estimator: Any = load_trained_model_with_mlflow()
    X_test_arr, _ = create_inference_features()
    return estimator, X_test_arr
