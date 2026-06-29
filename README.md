# Customer Churn Prediction

An end-to-end machine learning pipeline that predicts customer churn using the Telco Customer Churn dataset — covering data preprocessing, model comparison, threshold tuning, and deployment as a live web app.

🔗 **Live demo:** _add your Streamlit Cloud URL here_

---

## Overview

Telecom companies lose significant revenue to customer churn. This project builds a classifier that flags customers at risk of leaving, so retention efforts can be targeted before they go — rather than after.

**Key steps:**
- Data cleaning & feature encoding
- Dimensionality reduction (PCA) and scaling
- Model comparison across 4 algorithms
- Threshold tuning for business-relevant metrics
- Deployment as an interactive Streamlit app

---

## Dataset

[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 18 features (demographics, account info, services subscribed) plus a binary `Churn` target.

**Class balance:** ~73% No Churn / ~27% Churn.

---

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `matplotlib` / `seaborn` · `Streamlit` · `joblib`

---

## Pipeline

1. **Cleaning** — handled missing values and standardized categorical fields
2. **Encoding** — binary mapping for two-category fields, ordinal encoding for `Contract` (which has a natural order), one-hot encoding for the rest
3. **Train/test split** — performed before scaling/PCA to avoid data leakage
4. **Scaling & PCA** — standardized features, then reduced dimensionality
5. **Modeling** — compared Logistic Regression, Random Forest, XGBoost, and KNN via 5-fold cross-validation
6. **Threshold tuning** — adjusted the decision cutoff to prioritize recall, since missing a churner is costlier than a false alarm
7. **Deployment** — packaged the full pipeline (preprocessing + model + threshold) into a single object and served via Streamlit

---

## Model Comparison

5-fold cross-validated ROC-AUC on the training set:

| Model | Mean ROC-AUC |
|---|---|
| **Logistic Regression** | **0.845** |
| XGBoost | 0.839 |
| Random Forest | 0.826 |
| KNN (with PCA) | 0.786 |

Logistic Regression performed best, likely because churn drivers in this dataset (contract length, tenure, service type) interact in a fairly linear, additive way.

---

## Final Model Performance

Evaluated on a held-out test set, with the decision threshold tuned to prioritize recall on the churn class:

| Metric | Score |
|---|---|
| ROC-AUC | 0.84 |
| Accuracy | 0.75 |
| Precision (Churn) | 0.54 |
| Recall (Churn) | 0.77 |
| F1-score (Churn) | 0.63 |

A naive "predict everyone stays" baseline would score ~73% accuracy while catching zero churners. Since missing a real churner is more costly than a false retention flag, the threshold was tuned to favor recall while keeping precision reasonable.

---

## Project Structure

```
churn-prediction-app/
├── data/                       # raw dataset
├── models/
│   └── churn_model_final.pkl   # full pipeline: preprocessing + model + threshold
├── notebooks/
│   └── analysis.ipynb          # EDA, preprocessing, model comparison, tuning
├── app.py                      # Streamlit app
├── requirements.txt
└── README.md
```

---

## Running Locally

```bash
git clone https://github.com/Annjith/churn-prediction-app.git
cd churn-prediction-app
pip install -r requirements.txt
streamlit run app.py
```

---

## Future Improvements

- Use a separate validation set for threshold selection
- Additional feature engineering (service-bundle counts, tenure binning, charge ratios)
- Hyperparameter tuning via `GridSearchCV`
- Cost-based threshold optimization using real retention-offer vs. churn-loss estimates
