# ICR_ensemble
Kaggle [ICR](https://www.kaggle.com/competitions/icr-identify-age-related-conditions) - Identifying Age-Related Conditions

an implementation of a stacked ensemble model
1. cat_xgb.py contains an ensemble of XGboost, Lightgbm and Catboost by averaging
2. each model weights are optimized using [optuna](https://optuna.readthedocs.io/en/stable/)
3. stacking is TO BE done using Lightgbm.
