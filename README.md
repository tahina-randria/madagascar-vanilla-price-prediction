# 🌿 Madagascar Vanilla Price Prediction

Prédiction du prix de la vanille malgache en utilisant des techniques de Machine Learning (Time Series Forecasting).

## 🎯 Objectif

Madagascar produit environ **80% de la vanille mondiale**. Ce projet vise à prédire les fluctuations de prix de la vanille pour aider :
- Les agriculteurs à planifier leurs ventes
- Les traders à anticiper le marché
- Les décideurs à comprendre les facteurs d'influence

## 📊 Sources de Données

| Source | Description | Lien |
|--------|-------------|------|
| **FAO** | Prix et production agricoles | [FAOSTAT](https://www.fao.org/faostat/) |
| **World Bank** | Commodity prices (Pink Sheet) | [Commodity Markets](https://www.worldbank.org/en/research/commodity-markets) |
| **UN Comtrade** | Données d'export/import | [Comtrade](https://comtradeplus.un.org/) |
| **INSTAT Madagascar** | Statistiques nationales | [INSTAT](https://www.instat.mg/) |

## 🔧 Structure du Projet

```
madagascar-vanilla-price-prediction/
├── data/
│   ├── raw/              # Données brutes téléchargées
│   └── processed/        # Données nettoyées et transformées
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── models/               # Modèles sauvegardés
├── outputs/
│   ├── figures/         # Visualisations
│   └── reports/         # Rapports d'analyse
├── requirements.txt
└── README.md
```

## 🧠 Modèles Utilisés

1. **Baseline**: ARIMA, SARIMA
2. **Machine Learning**: XGBoost, Random Forest
3. **Deep Learning**: LSTM, Prophet (Facebook)

## 📈 Features

- **Temporelles**: Saisonnalité, tendances, lag features
- **Économiques**: Taux de change USD/MGA, inflation
- **Climatiques**: Précipitations, cyclones (impact récolte)
- **Production**: Volume de production, surfaces cultivées

## 🚀 Installation

```bash
# Cloner le repo
git clone https://github.com/YOUR_USERNAME/madagascar-vanilla-price-prediction.git
cd madagascar-vanilla-price-prediction

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer dépendances
pip install -r requirements.txt
```

## 📦 Dépendances

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
prophet>=1.1.0
tensorflow>=2.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
requests>=2.31.0
jupyter>=1.0.0
```

## 📝 Utilisation

```python
# Exemple rapide
from src.models import VanillaPricePredictor

model = VanillaPricePredictor()
model.load_data('data/processed/vanilla_prices.csv')
model.train()
predictions = model.predict(horizon=12)  # 12 mois
```

## 📊 Résultats

*À compléter après l'analyse*

- **RMSE**: ...
- **MAE**: ...
- **MAPE**: ...

## 🌍 Contexte Madagascar

La vanille malgache (Vanilla planifolia) est cultivée principalement dans la région **SAVA** (nord-est). Les prix sont très volatils en raison de :
- Cyclones tropicaux
- Vols dans les plantations
- Spéculation internationale
- Récolte précoce (qualité variable)

## 👤 Auteur

**Tahina**

## 📄 Licence

MIT License

## 🙏 Remerciements

- FAO pour les données ouvertes
- World Bank pour les séries temporelles
- Communauté open source Python
