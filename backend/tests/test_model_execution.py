from __future__ import annotations

import joblib
import numpy as np
import pytest
from sklearn.datasets import load_iris

from app.ml.models.registry import model_registry


MODEL_IDS = [
    "logistic_regression",
    "random_forest",
    "svm",
    "knn",
    "gradient_boosting",
]


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_builtin_model_trains_predicts_probabilities_and_serializes(tmp_path, model_id):
    X, y = load_iris(return_X_y=True, as_frame=True)
    plugin = model_registry.get(model_id)
    model = plugin.build({}, random_state=42)

    model.fit(X, y)
    predictions = model.predict(X.iloc[:8])
    assert len(predictions) == 8

    if plugin.metadata.supports_probability:
        probabilities = model.predict_proba(X.iloc[:8])
        assert probabilities.shape == (8, len(np.unique(y)))
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    path = tmp_path / f"{model_id}.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    assert np.array_equal(restored.predict(X.iloc[:8]), predictions)
