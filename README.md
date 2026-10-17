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
mlflow ui --backend-store-uri ./mlruns
```

Chaque exécution enregistre les paramètres, les métriques et le modèle
scikit-learn dans l'expérience `tp-mlops-classification`.
