"""Chargement, validation et entraînement du modèle de classification."""

from typing import Any
from pathlib import Path

import yaml

from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Charge la configuration YAML depuis la racine du dépôt."""
    with Path(path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("La configuration doit contenir un objet YAML.")
    return config


def load_dataset():
    """Charge le jeu Breast Cancer Wisconsin intégré à scikit-learn."""
    dataset = load_breast_cancer(as_frame=True)
    return dataset.data, dataset.target


def validate_dataset(X, y):
    """Vérifie la structure minimale attendue par le pipeline."""
    if X.empty:
        raise ValueError("Le jeu de données ne contient aucune observation.")

    if len(X) != len(y):
        raise ValueError("Le nombre d'observations et de cibles diffère.")

    if X.isnull().any().any():
        raise ValueError("Des valeurs manquantes ont été détectées.")

    if y.nunique() < 2:
        raise ValueError("La cible doit contenir au moins deux classes.")

    if not all(dtype.kind in "if" for dtype in X.dtypes):
        raise TypeError("Toutes les variables doivent être numériques.")


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    """Construit un pipeline sans fuite de données."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=config["model"]["C"],
                    max_iter=config["model"]["max_iter"],
                    random_state=config["seed"],
                ),
            ),
        ]
    )


def split_data(X, y, config: dict[str, Any]):
    """Sépare les données en conservant la proportion des classes."""
    return train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["seed"],
        stratify=y,
    )


def train_and_evaluate(X_train, X_test, y_train, y_test, config: dict[str, Any]):
    """Entraîne le pipeline puis calcule les métriques de classification."""
    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    return pipeline, metrics


if __name__ == "__main__":
    config = load_config()
    X, y = load_dataset()
    validate_dataset(X, y)
    print(f"Dimensions des variables : {X.shape}")
    print(f"Classes présentes : {sorted(y.unique())}")
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    _, metrics = train_and_evaluate(X_train, X_test, y_train, y_test, config)
    print(f"Métriques : {metrics}")
