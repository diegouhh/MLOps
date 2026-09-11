from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from app.ml.evaluators.classification import (
    aggregate_group_predictions,
    make_validation_split,
    summarize_fold_metrics,
)
from app.services.experiments import _predict_for_validation


class CountingClassifier(ClassifierMixin, BaseEstimator):
    fit_calls = 0

    def fit(self, X, y):
        type(self).fit_calls += 1
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.resize(self.classes_, len(X))

    def predict_proba(self, X):
        matrix = np.full((len(X), len(self.classes_)), 1 / len(self.classes_))
        return matrix


def test_group_aggregation_uses_mean_probability_when_available():
    grouped = aggregate_group_predictions(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1]),
        groups=np.array(["s1", "s1", "s2", "s2"]),
        probabilities=np.array(
            [
                [0.8, 0.2],
                [0.7, 0.3],
                [0.2, 0.8],
                [0.4, 0.6],
            ]
        ),
        probability_labels=np.array([0, 1]),
    )

    assert grouped["groups"].tolist() == ["s1", "s2"]
    assert grouped["y_true"].tolist() == [0, 1]
    assert grouped["y_pred"].tolist() == [0, 1]
    assert grouped["probabilities"].shape == (2, 2)


def test_fold_summary_reports_mean_and_population_std():
    summary = summarize_fold_metrics(
        [
            {"metrics": {"f1_macro": 0.6, "accuracy": 0.7}},
            {"metrics": {"f1_macro": 0.8, "accuracy": 0.9}},
        ]
    )
    assert summary["f1_macro"]["mean"] == 0.7
    assert np.isclose(summary["f1_macro"]["std"], 0.1)


def test_cross_validation_fits_once_per_fold_plus_final_fit():
    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = pd.Series([0, 1] * 6)
    strategy, splits = make_validation_split(
        "stratified_kfold",
        X,
        y,
        None,
        {"n_splits": 3},
        42,
    )
    estimator = Pipeline([("model", CountingClassifier())])
    CountingClassifier.fit_calls = 0

    result = _predict_for_validation(
        estimator,
        X,
        y,
        None,
        strategy,
        splits,
        supports_probability=True,
    )

    assert CountingClassifier.fit_calls == 4
    assert len(result["fold_metrics"]) == 3
    assert len(result["predictions"]) == len(y)
    assert result["probabilities"].shape == (len(y), 2)
