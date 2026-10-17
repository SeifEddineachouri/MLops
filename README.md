# TP MLOps

Projet minimal de classification binaire avec le jeu Breast Cancer Wisconsin
intégré à scikit-learn.

## Structure

```text
mlops-tp/
├── config/config.yaml
├── src/train.py
├── tests/test_pipeline.py
├── artifacts/
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

## Entraînement et suivi MLflow

Depuis la racine du dépôt :

```powershell
$env:PYTHONPATH = "src"
python src/train.py
$env:MLFLOW_ALLOW_FILE_STORE = "true"
mlflow ui --backend-store-uri ./mlruns
```

Chaque exécution enregistre les paramètres, les métriques et le modèle
scikit-learn dans l'expérience `tp-mlops-classification`.

Avec les versions récentes de MLflow, le backend fichier est conservé pour
correspondre à la configuration du TP mais nécessite explicitement
`MLFLOW_ALLOW_FILE_STORE=true`. Pour une utilisation durable, SQLite peut être
utilisé à la place avec `mlflow server --backend-store-uri sqlite:///mlflow.db`.
