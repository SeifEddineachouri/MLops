from pathlib import Path
import sys

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import (  # noqa: E402
    build_pipeline,
    load_config,
    load_dataset,
    run_training,
    split_data,
    train_and_evaluate,
    validate_dataset,
)


CONFIG = {"seed": 42, "data": {"test_size": 0.2}, "model": {"C": 1.0, "max_iter": 1000}}


def test_config_is_loaded_from_yaml():
    config = load_config("config/config.yaml")

    assert config["seed"] == 42
    assert config["data"]["test_size"] == 0.2
    assert config["model"]["C"] == 1.0
    assert config["mlflow"]["experiment_name"] == "tp-mlops-classification"


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


def test_training_logs_metrics_parameters_and_model(tmp_path):
    tracking_path = (tmp_path / "mlruns").resolve()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "data:",
                "  test_size: 0.2",
                "model:",
                "  C: 1.0",
                "  max_iter: 1000",
                "mlflow:",
                "  experiment_name: test-mlflow",
                f"  tracking_uri: file:{tracking_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    metrics = run_training(config_path)
    client = MlflowClient(tracking_uri=f"file:{tracking_path.as_posix()}")
    experiment = client.get_experiment_by_name("test-mlflow")
    runs = client.search_runs([experiment.experiment_id])

    assert len(runs) == 1
    assert runs[0].data.metrics["accuracy"] == metrics["accuracy"]
    assert runs[0].data.params["dataset"] == "Breast Cancer Wisconsin (scikit-learn)"
    logged_models = client.search_logged_models(
        [experiment.experiment_id],
        filter_string=f"source_run_id = '{runs[0].info.run_id}'"
    )
    assert len(logged_models) == 1
