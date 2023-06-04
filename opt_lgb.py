import optuna
from functools import partial
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from loss.log_loss import balanced_log_loss, calc_log_loss_weight


# n_estimators = 1000
# random_state = 100
# lgb_params = {
#     'n_estimators': n_estimators,
#     'objective': 'binary',
#     'boosting_type': 'gbdt',
#     'learning_rate': 0.05,
#     'num_leaves': 5,
#     'colsample_bytree': 0.50,
#     'subsample': 0.80,
#     'reg_alpha': 2, 
#     'reg_lambda': 4,
#     'n_jobs': -1,
#     'is_unbalance':True,
#     # 'device': device,
#     'random_state': random_state,
#     'verbose': -1,
#     'early_stopping_round': early_stopping_rounds,
    
#     'class_weight': 'balanced'
# }

def objective(trial):
    train = pd.read_csv("ICR/ICR_data/processed_train.csv")
    
    X = train.drop(['Class'], axis=1)
    y = train.Class
        
    xtrain, xtest, ytrain, ytest = train_test_split(X, y)
    train_w0, train_w1 = calc_log_loss_weight(ytrain)

    # n_estimators= trial.suggest_int('n_estimators', 80, 120),
    # learning_rate=  trial.suggest_float('learning_rate', 0.001, 0.7),
    # booster= trial.suggest_categorical('booster', ['gbtree', 'dart']),
    # lambda_= trial.suggest_float('lambda', 0.00001, 0.00005),
    # alpha= trial.suggest_float('alpha', 0.0001, 0.0005),
    # subsample= trial.suggest_float('subsample', 0.1, 0.5),
    # verbosity= -1,
    # random_state= 100,
    # early_stopping_rounds= 1000,
    # scale_pos_weight= trial.suggest_float('scale_pos_weight', 1.0, 7.0),

    lgb_params = {
    # 'n_estimators':trial.suggest_int('n_estimators', 80, 120),
    'n_estimators': 90,
    'learning_rate':trial.suggest_float('learning_rate', 0.005, 5),
    # 'boosting_type':trial.suggest_categorical('booster', ['gbdt', 'dart']),
    # 'reg_lambda': trial.suggest_float('lambda', 1, 6),
    # 'reg_alpha':trial.suggest_float('alpha', 1, 6),
    'reg_alpha': 2.7, 
    'reg_lambda': 4.65,
    # 'subsample':trial.suggest_float('subsample', 0.1, 2),
    # 'verbose':-1,
    'random_state':100,
    # 'early_stopping_rounds':1000,
    # 'scale_pos_weight':trial.suggest_float('scale_pos_weight', 1.0, 7.0),
    # 'classweight': 'balanced',
    'objective': 'binary',
    'n_jobs': -1,
    'is_unbalance':True,
    'num_leaves': 5,
    'colsample_bytree': 0.50,
    }

    
    
    # clf = xgb.XGBClassifier(**{'n_estimators':n_estimators,
                            # 'learning_rate':learning_rate,
                            # 'booster':booster,
                            # 'reg_lambda': lambda_,
                            # 'alpha':alpha,
                            # 'subsample':subsample,
                            # 'verbosity':verbosity,
                            # 'random_state':random_state,
                            # 'early_stopping_rounds':early_stopping_rounds,
                            # 'scale_pos_weight':scale_pos_weight,})
    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(xtrain, ytrain, 
            sample_weight=ytrain.map({0: train_w0, 1: train_w1}),
            )
    test_pred = clf.predict(xtest)
    score = balanced_log_loss(ytest, test_pred)
    return score
    
# optuna.logging.set_verbosity(optuna.logging.ERROR)
# sampler = optuna.samplers.CmaEsSampler(seed=100)
sampler = optuna.samplers.RandomSampler(seed=100)
pruner = optuna.pruners.HyperbandPruner()
study = optuna.create_study(
    sampler=sampler, pruner=pruner,
    study_name='XGB_opt', direction='minimize')
study.optimize(objective
               , n_trials=1000
            )
print(study.best_params)

# print(f"score: {objective(xgb_params)}")