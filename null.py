from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class NotNull(BaseEstimator,TransformerMixin):
    '''
    if has value
    '''
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
            
    
    def transform(self, X, y=None):
        return 1-pd.isnull(X).astype(int)
        
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)