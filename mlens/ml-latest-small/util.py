import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from copy import deepcopy
from sklearn.metrics import mean_squared_error

def split_train_test(ratings):
    train_idxs=[]
    test_idxs=[]
    for u in ratings.userId.unique():
        r_u=ratings[ratings.userId==u].sort_values('timestamp',ascending=False)
        test_idxs += list(r_u.iloc[:int(len(r_u)*0.1)].index)
        train_idxs += list(r_u.iloc[int(len(r_u)*0.1):].index)

    X_train=ratings.loc[train_idxs]
    y_train=X_train.rating
    
    X_test=ratings.loc[test_idxs]
    y_test=X_test.rating
    
    return X_train,y_train,X_test,y_test
