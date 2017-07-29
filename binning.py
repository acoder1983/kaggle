
# coding: utf-8

# In[15]:

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Binner(BaseEstimator,TransformerMixin):
    '''
    binner for continuous num array
    '''
    def __init__(self,strategy):
        self.strategy=strategy
        
    def fit(self, X, y=None):
        return self
            
    
    def transform(self, X, y=None):
        if isinstance(self.strategy,list):
            arr=np.zeros((len(X),1), dtype=np.int)
            for i in range(len(X)):
                arr[i][0]=len(self.strategy)
                for j in range(len(self.strategy)):
                    if X[i][0] < self.strategy[j]:
                        arr[i][0]=j
                        break
                
            return arr
        
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
import unittest as ut

from sklearn.pipeline import Pipeline

class Test(ut.TestCase):
    def testRange(self):
        b=Binner(strategy=list(np.linspace(10,100,10)))
        
        df=pd.DataFrame({'age':[3.,17.,66.]})
        data=b.transform(df[['age']].values)  
        
        self.assertTrue(np.array_equal(np.array([[0],[1],[6]]), data))
        
                
if __name__ == '__main__':
    ut.main(argv=['ignored', '-v'], exit=False)

