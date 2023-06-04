import pandas as pd
import numpy as np

from loss.log_loss import balanced_log_loss, calc_log_loss_weight
from opt.optweights import OptunaWeights
import optuna

from pprintpp import pprint
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import catboost as cat
import lightgbm as lgb



import resource
import sys


def memory_limit():
    """Limit max memory usage to half."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # Convert KiB to bytes, and divide in two to half
    resource.setrlimit(resource.RLIMIT_AS, (int(get_memory() * 1024 / 1.01), hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    # print(free_memory/1024)
    return free_memory  # KiB


n_estimators=100
# device="cpu"
# random_state=state
n_splits= 5
# random_state = 71
# n_estimators = 99999
verbose = False
early_stopping_rounds= 1000


def get_params(state):
    random_state = state
    # xgb_params = {
    #     'n_estimators': n_estimators,
    #     'learning_rate': 0.413327571405248,
    #     'booster': 'gbtree',
    #     'lambda': 0.0000263894617720096,
    #     'alpha': 0.000463768723479341,
    #     'subsample': 0.237467672874133,
    #     'colsample_bytree': 0.618829300507829,
    #     'max_depth': 5,
    #     'min_child_weight': 9,
    #     'eta': 2.09477807126539E-06,
    #     'gamma': 0.000847289463422307,
    #     'grow_policy': 'depthwise',
    #     'n_jobs': -1,
    #     'objective': 'binary:logistic',
    #     'eval_metric': 'logloss',
    #     'verbosity': 0,
    #     'random_state': random_state,
        
    #     'early_stopping_rounds': 1000,
    #     'scale_pos_weight': 4.71,
    # }
    """"optuna"""
    xgb_params={'n_estimators': 85, 
        'learning_rate': 0.5532875740096689,
        'booster': 'dart',
        'lambda': 2.210583397046728e-05,
        'alpha': 0.00011961099653447517,
        'subsample': 0.16149076133637538,
        'scale_pos_weight': 2.3999263887936273,
        'subsample': 0.237467672874133,
        'colsample_bytree': 0.618829300507829,
        'max_depth': 5,
        'min_child_weight': 9,
        'eta': 2.09477807126539E-06,
        'gamma': 0.000847289463422307,
        'grow_policy': 'depthwise',
        'n_jobs': -1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'verbosity': 0,
        'random_state': random_state,
        'early_stopping_rounds': 1000,

        }
            # if self.device == 'gpu':
            #     xgb_params['tree_method'] = 'gpu_hist'
            #     xgb_params['predictor'] = 'gpu_predictor'
            
    lgb_params = {
        'n_estimators': n_estimators,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 5,
        'colsample_bytree': 0.50,
        'subsample': 0.80,
        'reg_alpha': 2, 
        'reg_lambda': 4,
        'n_jobs': -1,
        'is_unbalance':True,
        # 'device': device,
        'random_state': random_state,
        'verbose': -1,
        'early_stopping_round': early_stopping_rounds,
        
        'class_weight': 'balanced'
    }
    """optuna"""
    # lgb_params = {
    #     'n_estimators': n_estimators,
    #     'objective': 'binary',
    #     'boosting_type': 'gbdt',
    #     'learning_rate': 0.,
    #     'num_leaves': 5,
    #     'colsample_bytree': 0.50,
    #     'subsample': 0.80,
    #     'reg_alpha': 2.7, 
    #     'reg_lambda': 4.65,
    #     'n_jobs': -1,
    #     'is_unbalance':True,
    #     # 'device': device,
    #     'random_state': random_state,
    #     'verbose': -1,
    #     'early_stopping_round': early_stopping_rounds,
        
    #     'class_weight': 'balanced'
    # }
    # lgb1_params = {
    #     'n_estimators': self.n_estimators,
    #     'learning_rate': 0.190197487721534,
    #     'reg_alpha': 0.00749112221417973,
    #     'reg_lambda': 0.000548118227209224,
    #     'num_leaves': 17,
    #     'colsample_bytree': 0.547257860506146,
    #     'subsample': 0.592628085686409,
    #     'subsample_freq': 2,
    #     'min_child_samples': 64,
    #     'objective': 'binary',
    #     #'metric': 'binary_error',
    #     'boosting_type': 'gbdt',
    #     'is_unbalance':True,
    #     'device': self.device,
    #     'random_state': self.random_state
    # } 
    # lgb2_params = {
    #     'n_estimators': self.n_estimators,
    #     'learning_rate': 0.181326407627473,
    #     'reg_alpha': 0.000030864084239014,
    #     'reg_lambda': 0.0000395714763869486,
    #     'num_leaves': 122,
    #     'colsample_bytree': 0.75076596295323,
    #     'subsample': 0.6303245788342,
    #     'subsample_freq': 3,
    #     'min_child_samples': 72,
    #     'objective': 'binary',
    #     #'metric': 'binary_error',
    #     'boosting_type': 'gbdt',
    #     'is_unbalance':True,
    #     'device': self.device,
    #     'random_state': self.random_state
    # } 
    cat_params = {
        'iterations': n_estimators,
        'colsample_bylevel': 0.0513276895988184,
        'depth': 2,
        'learning_rate': 0.0256579773375401,
        'l2_leaf_reg': 8.22319805476255,
        'random_strength': 0.11327724457066,
        'od_type': "Iter", 
        'od_wait': 72,
        'bootstrap_type': "Bayesian",
        'grow_policy': 'SymmetricTree',
        'bagging_temperature': 9.58737431845122,
        #'eval_metric': 'Logloss',
        #'loss_function': 'Logloss',
        'auto_class_weights': 'Balanced',
        # 'task_type': device.upper(),
        'random_state': random_state,
        
        # 'early_stopping_rounds': 1000,
    }
    # hist_params = {
    #     'l2_regularization': 0.01,
    #     'early_stopping': True,
    #     'learning_rate': 0.01,
    #     'max_iter': self.n_estimators,
    #     'max_depth': 4,
    #     'max_bins': 255,
    #     'min_samples_leaf': 10,
    #     'max_leaf_nodes':10,
    #     'class_weight':'balanced',
    #     'random_state': self.random_state
    # }
    return xgb_params, cat_params, lgb_params

def act_main(state):
    random_state = state
    
    xgb_params, cat_params, lgb_params = get_params(state)
    train = pd.read_csv("ICR/ICR_data/processed_train.csv")
    # test = pd.read_csv("ICR/ICR_data/processed_test.csv")

    X = train.drop('Class', axis=1)
    y = train['Class']
    
    trainx, testx, trainy, testy = train_test_split(X, y , test_size=0.1, random_state=random_state)

    
    clf1 = xgb.XGBClassifier(**xgb_params)
    clf2 = cat.CatBoostClassifier(**cat_params)
    clf3 = lgb.LGBMClassifier(**lgb_params)
    
    skf = StratifiedKFold(n_splits=n_splits)
    
    oof_predss = np.zeros((trainx.shape[0], 5))
    xgb_oof_pred = []
    cat_oof_pred = []
    lgb_oof_pred = []
    fold_target = []
    oof_ens_pred = []
    oof_loss = []
    weights = []
    ensemble_score = []
    for fold, (train_idx, valid_idx) in tqdm(enumerate(skf.split(trainx, trainy))):
        xtrain = trainx.iloc[train_idx]
        ytrain = trainy.iloc[train_idx]
        
        xvalid = trainx.iloc[valid_idx]
        yvalid = trainy.iloc[valid_idx]
        
        train_w0, train_w1 = calc_log_loss_weight(ytrain)
        valid_w0, valid_w1 = calc_log_loss_weight(yvalid)
        
        
        clf1.fit(
                xtrain, ytrain, sample_weight=ytrain.map({0: train_w0, 1: train_w1}), 
                eval_set=[(xvalid, yvalid)], sample_weight_eval_set=[yvalid.map({0: valid_w0, 1: valid_w1})],
                # early_stopping_rounds=early_stopping_rounds, 
                verbose=verbose)
        
        clf2.fit(cat.Pool(xtrain, ytrain, weight=ytrain.map({0: train_w0, 1: train_w1})), 
                eval_set=cat.Pool(xvalid, yvalid, weight=yvalid.map({0: valid_w0, 1: valid_w1})), 
                early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        
        clf3.fit(xtrain, ytrain, sample_weight=ytrain.map({0: train_w0, 1: train_w1}), 
                eval_set=[(xvalid, yvalid)], eval_sample_weight=[yvalid.map({0: valid_w0, 1: valid_w1})],
                callbacks=[lgb.log_evaluation(0)])
        
        xgb_pred = clf1.predict_proba(xvalid)[:, 1].reshape(-1)
        cat_pred = clf2.predict_proba(xvalid)[:, 1].reshape(-1)
        lgb_pred = clf3.predict_proba(xvalid)[:, 1].reshape(-1)
        ens_pred = (cat_pred + xgb_pred + lgb_pred)/3
        
        # cat_loss = balanced_log_loss(cat_pred, yvalid)
        # xgb_loss = balanced_log_loss(xgb_pred, yvalid)
        # lgb_loss = balanced_log_loss(lgb_pred, yvalid)
        ens_loss = balanced_log_loss(yvalid, ens_pred)
        oof_loss.append(ens_loss)
        oof_ens_pred.append(np.array(xgb_pred))
        oof_ens_pred.append(np.array(cat_pred))
        oof_ens_pred.append(np.array(lgb_pred))
        
        
        # cat_oof_pred.append(cat_pred)
        # xgb_oof_pred.append(xgb_pred)
        # lgb_oof_pred.append(lgb_loss)
        # fold_target.append(yvalid)lgb1_params
    
        # print(f"fold: {fold}, catboost loss: {cat_loss}")
        # print(f"fold: {fold}, xgbboost loss: {xgb_loss}")
        # print(f"fold: {fold}, ens loss: {ens_loss}")
        
        # oof_ens_pred.append(ens_pred)
        
        # print(np.array(oof_ens_pred).T)
        
        
        """OPTUNA"""
        # data = pd.read_csv("ICR/ICR_data/pred_for_lgb1.csv")
        # oof_preds = data.drop('Class', axis=1)
        # y_val = data.Class
        optweights = OptunaWeights(random_state=random_state)
        y_val_pred = optweights.fit_predict(yvalid.values, oof_ens_pred)
        
        
        score = balanced_log_loss(yvalid, y_val_pred)
        # score_ = roc_auc_score(y_val, y_val_pred)
        print(f'--> Ensemble [FOLD-{fold}] BalancedLogLoss score {score:.5f}')
        ensemble_score.append(score)
        # ensemble_score_ = [score_]
    weights.append(optweights.weights)
        
        
    print("*"*45)
    # print(f"mean_oof_loss: {np.mean(oof_loss)}")
    print(f"mean_cv_loss: {np.mean(ensemble_score)}")
    
    
    """"test score"""
    # ens_test_pred = []
    xgb_test_pred = clf1.predict_proba(testx)[:, 1].reshape(-1)
    cat_test_pred = clf2.predict_proba(testx)[:, 1].reshape(-1)
    lgb_test_pred = clf3.predict_proba(testx)[:, 1].reshape(-1)
    ens_test_pred = (cat_test_pred + xgb_test_pred + lgb_test_pred)/3
    # ens_test_pred.append(xgb_test_pred)
    # ens_test_pred.append(cat_test_pred)
    # ens_test_pred.append(lgb_test_pred)
    # weighted_ens_test_pred = np.average(np.array(ens_test_pred).T, axis=1, weights=np.mean(np.array(weights).T))
    print(f"ens_test_pred_loss: {balanced_log_loss(testy, ens_test_pred)}")
    # pred = cat_oof_pred + xgb_oof_pred + fold_target
        
    # pd.DataFrame(pred).T.to_csv("ICR/ICR_data/oof_pred.csv", index=False)
    return ens_test_pred
    


if __name__ == '__main__':
    memory_limit()
    try:
        score = []
        for state in range(100, 1000, 100):
            score.append(act_main(state))
        
        print(np.mean(score))
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)