
# coding: utf-8

# In[37]:

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class HasCabin(BaseEstimator,TransformerMixin):
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
    
import unittest as ut

from sklearn.pipeline import Pipeline

class Test(ut.TestCase):
    def testRange(self):
        h=HasCabin()
        
        df=pd.DataFrame({'cabin':['a','b',np.nan]})
        data=h.transform(df[['cabin']].values)  
        
        self.assertTrue(np.array_equal(np.array([[1],[1],[0]]), data))
        
                
if __name__ == '__main__':
    ut.main(argv=['ignored', '-v'], exit=False)

