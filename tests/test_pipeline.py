from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import (  # noqa: E402
    build_pipeline,
    load_dataset,
    split_data,
    train_and_evaluate,
    validate_dataset,
)


CONFIG = {"seed": 42, "data": {"test_size": 0.2}, "model": {"C": 1.0, "max_iter": 1000}}


def test_dataset_is_loaded_and_validated():
    X, y = load_dataset()
    validate_dataset(X, y)
    assert X.shape == (569, 30)
    assert sorted(y.unique()) == [0, 1]


def test_pipeline_trains_and_returns_metrics():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y, CONFIG)
    pipeline, metrics = train_and_evaluate(
        X_train, X_test, y_train, y_test, CONFIG
    )

    assert set(metrics) == {"accuracy", "f1", "roc_auc"}
    assert all(0 <= value <= 1 for value in metrics.values())
    assert len(pipeline.predict(X_test)) == len(y_test)


def test_validation_rejects_missing_values():
    X = pd.DataFrame({"feature": [1.0, None]})
    y = pd.Series([0, 1])

    with pytest.raises(ValueError, match="manquantes"):
        validate_dataset(X, y)
