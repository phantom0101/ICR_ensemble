import pandas as pd 
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from loss.log_loss import balanced_log_loss, calc_log_loss_weight

n_estimators = 10
random_state = 71
n_splits = 5

lgb1_params = {
    'n_estimators': n_estimators,
    # 'learning_rate': 0.190197487721534,
    'learning_rate': 0.7,
    'reg_alpha': 0.00749112221417973,
    'reg_lambda': 0.000548118227209224,
    'num_leaves': 17,
    'colsample_bytree': 0.547257860506146,
    'subsample': 0.592628085686409,
    'subsample_freq': 2,
    'min_child_samples': 64,
    'objective': 'binary',
    #'metric': 'binary_error',
    'boosting_type': 'gbdt',
    'is_unbalance': True,
    # 'device': device,
    # 'random_state': random_state
    # 'class_weight': 'balanced'
} 



if __name__ == "__main__":
    
    train= (pd.read_csv("ICR/ICR_data/pred_for_lgb1.csv"))
    test= (pd.read_csv("ICR/ICR_data/test_for_lgb1.csv"))
    
    trainx = train.drop('Class', axis=1)
    trainy = train.Class
    testx = test.drop('Class', axis=1)
    testy = test.Class
    
    
    clf = lgb.LGBMClassifier(**lgb1_params)
    # clf = LogisticRegression()
    
    skf = StratifiedKFold(n_splits=n_splits)
    lgb1_oof_loss = []
    
    for fold, (train_idx, valid_idx) in tqdm(enumerate(skf.split(trainx, trainy))):
        xtrain = trainx.iloc[train_idx]
        ytrain = trainy.iloc[train_idx]
        
        xvalid = trainx.iloc[valid_idx]
        yvalid = trainy.iloc[valid_idx]
        
        train_w0, train_w1 = calc_log_loss_weight(ytrain)
        valid_w0, valid_w1 = calc_log_loss_weight(yvalid)
        
        clf.fit(xtrain, ytrain, sample_weight=ytrain.map({0: train_w0, 1: train_w1}),
                eval_set=[(xvalid, yvalid)], eval_sample_weight=[yvalid.map({0: valid_w0, 1: valid_w1})],
                callbacks=[lgb.log_evaluation(0)])
        # clf.fit(xtrain, ytrain, sample_weight=ytrain.map({0: train_w0, 1: train_w1}))
        
        lgb1_pred = clf.predict(xvalid)
        loss = balanced_log_loss(lgb1_pred, yvalid)
        lgb1_oof_loss.append(loss)
        print(f"oof_loss: {loss}")
        
    print(f"lgb1_loss: {np.mean(lgb1_oof_loss)}")
    
    test_pred = clf.predict(testx)
    print(f"test_pred_loss: {balanced_log_loss(test_pred ,testy)}")
            