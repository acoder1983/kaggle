
# coding: utf-8

# In[30]:

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Ordinar(BaseEstimator,TransformerMixin):
    '''
    transform category value to ordinal
    '''
    def __init__(self,cat_list):
        '''
        list of category values, index means ordinal
        '''
        self.cat_list=cat_list
        
    def fit(self, X, y=None):
        return self
            
    
    def transform(self, X, y=None):
        df=pd.DataFrame(X,columns=['x'])
        not_missing_index=df[np.logical_not(df['x'].isnull())].index
        
        df[df['x'].isnull()]=-1
        for i in not_missing_index:
            for j in range(len(self.cat_list)):
                if self.cat_list[j].lower() == X[i][0].lower() :
                    df.loc[i,'x'] = j
                    break
                
        return df.astype(float).values
                    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
import unittest as ut

from sklearn.pipeline import Pipeline

class Test(ut.TestCase):
    def test(self):
        df=pd.DataFrame({'level':['ex','gd','ex',np.nan]})
        
        o=Ordinar(['GD','ex'])
        data=o.fit_transform(df[['level']].values)
        
        self.assertTrue(np.array_equal(np.array([[1],[0],[1],[-1]]), data))
        
if __name__ == '__main__':
    ut.main(argv=['ignored', '-v'], exit=False)

