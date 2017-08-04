
# coding: utf-8

# In[16]:

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AddColumns(BaseEstimator,TransformerMixin):
    '''
    add multiple columns to one
    '''
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
            
    
    def transform(self, X, y=None):
        return np.array([X.sum(axis=1)]).transpose()
        
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
import unittest as ut

from sklearn.pipeline import Pipeline

class Test(ut.TestCase):
    def testRange(self):
        a=AddColumns()
        
        df=pd.DataFrame({'age':[1,2,3],'fare':[3,2,1]})
        data=a.transform(df[['age','fare']].values)  
        
        self.assertTrue(np.array_equal(np.array([[4],[4],[4]]), data))
        
                
if __name__ == '__main__':
    ut.main(argv=['ignored', '-v'], exit=False)


# In[14]:

import numpy as np
a=np.array([[3,8],[1,2]])
np.array([a.sum(axis=1)]).transpose()
# np.array([[1,2]]).transpose()

