# Overdue-Installments-Prediction
# Overdue Installments Prediction

This repository contains a Jupyter notebook (`overdue.ipynb`) for predicting the number of overdue installments (`Nombre d’échéance en retard`) based on customer and loan attributes. The analysis includes data preprocessing, feature engineering, multiple regression models, evaluation metrics, and output submission generation.

## Table of Contents

* [Overview](#overview)
* [Project Structure](#project-structure)
* [Prerequisites](#prerequisites)
* [Data](#data)
* [Preprocessing](#preprocessing)
* [Feature Selection](#feature-selection)
* [Modeling](#modeling)
* [Evaluation](#evaluation)
* [Submission](#submission)
* [Usage](#usage)
* [License](#license)

## Overview

The goal of this project is to build predictive models for estimating the count of overdue loan installments. The workflow includes:

1. Reading training and test datasets.
2. Encoding categorical variables.
3. Handling missing values and capping outliers.
4. Selecting relevant features.
5. Training and comparing models:

   * Linear Regression
   * Random Forest Regressor
   * XGBoost Regressor
   * Support Vector Regressor (SVR)
   * Neural Network (TensorFlow/Keras)
6. Evaluating models using RMSE, MSE, and R².
7. Generating CSV submissions with predictions.

## Project Structure

```bash
├── data/
│   ├── train.csv         # Training data
│   └── test.csv          # Test data
├── overdue (1).ipynb     # Colab notebook with full pipeline
├── models/
│   ├── linear_model.py   # Linear Regression implementation
│   ├── rf_model.py       # Random Forest implementation
│   ├── xgb_model.py      # XGBoost implementation
│   ├── svr_model.py      # SVR implementation
│   └── nn_model.py       # Neural Network implementation
└── README.md             # This file
```

## Prerequisites

* Python 3.8+
* Required packages:

  * pandas
  * numpy
  * scikit-learn
  * xgboost
  * tensorflow
  * matplotlib
  * seaborn

Install dependencies with:

```bash
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn
```

## Data

* **train.csv**: Contains features and target `Nombre d’échéance en retard`.
* **test.csv**: Contains the same features without the target.

## Preprocessing

* **Categorical encoding**: LabelEncoder for columns like `Type de produit`, `Situation logement`, `Canal production`, etc.
* **Missing values**: Impute numeric columns (`Age`, `Ancienneté à lemploi`, etc.) with column means.
* **Outlier capping**: Cap `Ancienneté à lemploi` at 2400 days.

## Feature Selection

Dropped less important columns:

* `id`, `Genre`, `Revenu totaux`, `Montant échéance`.

Selected predictors include:

* `Taux dendettement`, `Salaire net mensuel.1`, `Age`, `Charges totales`, `Montant accepté`, `Montant demandé`, `Secteur d’activité`, `Durée du prêt`, etc.

## Modeling

Each model script follows this pattern:

1. Split data into train/test (30% test size).
2. Train the model on `X_train`, `Y_train`.
3. Predict on `X_test` and compute MSE, RMSE, R².
4. Predict on test set and round predictions to integers.
5. Save results to CSV (e.g., `uv.csv`, `2.csv`, `uv_11.csv`).

### Models Implemented

* **Linear Regression**
* **Random Forest Regressor**
* **XGBoost Regressor**
* **Support Vector Regressor (SVR)**
* **Neural Network** using TensorFlow/Keras

## Evaluation

Model performance metrics printed in the notebook:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R-squared (R²) Score

Feature importances are displayed for tree-based models.

## Submission

Predictions on the test set are exported as CSV files with two columns:

* `id`: from `test.csv`
* `Nombre d’échéance en retard`: predicted integer values

Example submission files: `uv.csv`, `2.csv`, `uv_11.csv`, `uv_10.csv`.

## Usage

1. Place `train.csv` and `test.csv` in the `data/` directory.

2. Open `overdue (1).ipynb` in Colab or Jupyter and run all cells.

3. Alternatively, run individual model scripts:

   ```bash
   python models/linear_model.py
   python models/rf_model.py
   # etc.
   ```

4. Check generated CSV in project root.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute.
