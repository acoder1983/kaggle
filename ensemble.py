
# coding: utf-8

# In[ ]:

from brew.base import Ensemble
from brew.stacking import EnsembleStackClassifier,EnsembleStack
import sklearn

class EnsembleStackClassifierEx(EnsembleStackClassifier):
    def __init__(self, stack, combiner=None):
        EnsembleStackClassifier.__init__(self, stack,combiner)
        
    def score(self, X, y, sample_weight=None):
        return (self.predict(X)==y).astype(float).sum()/len(X)
    
    def get_params(self, deep=True):
        return {'stack':self.stack}
    


# [stacked regressor](https://www.otexts.org/1536)

# In[4]:

from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split
import pandas as pd

class StackRegressor(RegressorMixin):
    def __init__(self,base_regs,second_reg,train_size):
        self.base_regs=base_regs
        self.second_reg=second_reg
        self.train_size=train_size
        
    def fit(self,X,y):
        X_train, X_test, y_train, y_test=train_test_split(X,y,train_size=self.train_size)
        df=pd.DataFrame()
        for r in self.base_regs:
            r.fit(X_train,y_train)
            df[r.__class__.__name__]=r.predict(X_test)
        self.second_reg.fit(df,y_test)
    
    def predict(self,X):
        df=pd.DataFrame()
        for r in self.base_regs:
            df[r.__class__.__name__]=r.predict(X)
        return self.second_reg.predict(df)
        
    def get_params(self,deep=True):
        return {'base_regs':self.base_regs,'second_reg':self.second_reg,'train_size':self.train_size}

