import time
from typing import Any
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)
from typeguard import typechecked


@typechecked
def train_model_with_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    estimator: ClassifierMixin | RegressorMixin,
    n_splits: int = 5,
) -> tuple[ClassifierMixin | RegressorMixin, list[float], float, float]:
    """
    Train a model using cross-validation and return performance metrics.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input features array.
    y : np.ndarray, shape (n_samples,)
        Target labels array.
    estimator : ClassifierMixin | RegressorMixin
        The machine learning model to be trained and evaluated.
    n_splits : int, optional
        Number of splits for cross-validation, by default 5.

    Returns
    -------
    tuple[ClassifierMixin | RegressorMixin, list[float], float, float]
        A tuple containing:
        - The trained estimator
        - List of accuracy scores for each fold
        - Mean accuracy across all folds
        - Standard deviation of accuracy across all folds
    """
    if isinstance(estimator, ClassifierMixin):
        y = y.astype(int)

    start_time: float = time.time()
    kfold: Any = StratifiedKFold(n_splits=n_splits).split(X, y)

    scores: list[float] = []

    for k, (train, test) in enumerate(kfold):
        estimator.fit(X[train], y[train])
        if isinstance(estimator, ClassifierMixin):
            score: float = estimator.score(X[test], y[test])
            scores.append(score)
            print(
                f"Fold: {k+1:2d} | Class dist.: {np.bincount(y[train])} | Acc: {score:.3f}"
            )  # noqa
        elif isinstance(estimator, RegressorMixin):
            # R^2  score
            score: float = estimator.score(X[test], y[test])  # type: ignore
            scores.append(score)
            print(f"Fold: {k+1:2d} | R^2: {score:.3f}")  # noqa

    mean_score: float = np.mean(scores)  # Accuracy or R^2
    std_accuracy: float = np.std(scores)
    stop_time: float = time.time()
    if isinstance(estimator, ClassifierMixin):
        print(f"\nCV accuracy: {mean_score:.3f} +/- {std_accuracy:.3f}")  # noqa
    elif isinstance(estimator, RegressorMixin):
        print(f"\nCV R^2: {mean_score:.3f} +/- {std_accuracy:.3f}")
    print(f"\nTime taken: {stop_time - start_time:.3f} seconds")  # noqa

    return estimator, scores, mean_score, std_accuracy


@typechecked
def evaluate(
    estimator: ClassifierMixin | RegressorMixin,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Evaluate the performance of a classifier or regressor model.

    Parameters
    ----------
    estimator : ClassifierMixin | RegressorMixin
        The trained model to evaluate.
    X_test : np.ndarray | pd.DataFrame, shape (n_samples, n_features)
        Test features data.
    y_test : np.ndarray | pd.Series, shape (n_samples,)
        Test target values.

    Returns
    -------
    dict[str, Any]
        Dictionary containing evaluation metrics:
        - For classifiers: accuracy_score, auc_score, f1_score
        - For regressors: r2_score, mse_score
    """

    if isinstance(estimator, ClassifierMixin):
        test_preds_proba: npt.NDArray[np.float64] = estimator.predict_proba(X_test)[
            :, 1
        ]  # shape (n_samples,)
        test_preds: npt.NDArray[np.float64] = estimator.predict(
            X_test
        )  # shape (n_samples,)
        # The metrics of the test data
        _acc_score: float = np.round(accuracy_score(y_test, test_preds), 4)
        _auc_score: float = np.round(roc_auc_score(y_test, test_preds_proba), 4)
        _f1_score: float = np.round(f1_score(y_test, test_preds), 4)
        results: dict[str, float] = {
            # Convert to regular float instead of np.float
            "accuracy_score": float(_acc_score),
            "auc_score": float(_auc_score),
            "f1_score": float(_f1_score),
        }

    if isinstance(estimator, RegressorMixin):
        test_preds: npt.NDArray[np.float64] = estimator.predict(  # type: ignore
            X_test
        )  # shape (n_samples,)
        # The metrics of the test data
        _r2_score: float = np.round(r2_score(y_test, test_preds), 4)
        _mse_score: float = np.round(mean_squared_error(y_test, test_preds), 4)
        results = {
            # Convert to regular float instead of np.float
            "r2_score": float(_r2_score),
            "mse_score": float(_mse_score),
        }

    return results
