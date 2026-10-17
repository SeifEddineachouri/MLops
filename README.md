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

## Installation et exécution

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python src/train.py
pytest
```

Le pipeline ajuste l'imputation et la normalisation uniquement pendant
l'entraînement. La séparation stratifiée conserve la proportion des deux classes
dans les ensembles d'entraînement et de test.

## Questions de réflexion

- La séparation doit précéder l'ajustement de l'imputation et de la
  normalisation pour empêcher les statistiques du jeu de test d'influencer le
  modèle.
- La stratification évite qu'une classe soit sous-représentée dans un des
  ensembles.
- L'accuracy peut masquer une classe minoritaire mal prédite. Le F1 mesure le
  compromis précision-rappel et le ROC-AUC évalue le classement des probabilités.
