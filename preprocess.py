import pandas as pd
import numpy as np
from pprintpp import pprint
import matplotlib.pyplot as plt
import seaborn as sns
# pd.set_option("display.max_columns", 200)

def preprocessing(train):
    # fillna with mean values
    na_keys = train.isna().sum()[train.isna().sum() != 0].keys()
    na_keys_mean = train[na_keys].mean().to_dict()
    train.fillna(na_keys_mean, inplace=True)
    
    # train = train[[
    #     # 'Id', 
    #     'AB', 'AF', 'AH', 'AM', 'AR', 'AX', 'AY', 'AZ',
    #     # 'BC', 
    #     'BD ', 'BN',
    #     'BP', 'BQ', 'BR', 'BZ', 'CB', 'CC', 'CD ', 'CF', 'CH', 'CL', 'CR', 'CS',
    #     'CU', 'CW ', 'DA', 'DE', 'DF', 'DH', 'DI', 'DL', 'DN', 'DU', 'DV', 'DY',
    #     'EB', 'EE', 'EG', 
    #     # 'EH', 
    #     'EJ', 'EL', 'EP', 'EU', 'FC', 'FD ', 'FE', 'FI',
    #     'FL', 'FR', 'FS', 'GB', 'GE', 'GF', 'GH', 'GI', 'GL', 'Class']]
    
    train = train.drop(['Id', 'BC', 'EH'], axis=1)
    train.EJ = train.EJ.map({'A':1, 'B':0})
    
    return train

if __name__ == "__main__":
    train = pd.read_csv("/home/yash/Desktop/Code/py/ml/ICR/ICR_data/train.csv")
    test = pd.read_csv("/home/yash/Desktop/Code/py/ml/ICR/ICR_data/test.csv")
    
    train = preprocessing(train)
    test = preprocessing(test)
    
    train.to_csv("ICR/ICR_data/processed_train.csv", index=False)
    test.to_csv("ICR/ICR_data/processed_test.csv", index=False)
    # correlation
    # print(train.corr().query("@train.corr()<1 and @train.corr()>0.9").dropna(thresh=1).dropna(thresh=1, axis=1)) 
    
    