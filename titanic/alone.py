
# coding: utf-8

# In[6]:

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class IsAlone(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
            
    
    def transform(self, X, y=None):
        return (X == 0).astype(int)
        
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
import unittest as ut

from sklearn.pipeline import Pipeline

class Test(ut.TestCase):
    def test(self):
        a=IsAlone()
        
        df=pd.DataFrame({'size':[0,1]})
        data=a.transform(df[['size']].values)  
        
        self.assertTrue(np.array_equal(np.array([[1],[0]]), data))
        
                
if __name__ == '__main__':
    ut.main(argv=['ignored', '-v'], exit=False)

