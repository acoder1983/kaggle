
# coding: utf-8

# In[4]:

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LabelBinarizerEx(BaseEstimator,TransformerMixin):
    '''
    extend for sklearn label binarizer
    take account for missing values
    '''
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        '''
        @X dataFrame with one column
        '''
        df=pd.get_dummies(X)
        self.classes=[c[c.rindex('_')+1:] for c in df.columns]
        self.columns=list(df.columns)
        return self
    
    def transform(self,X,y=None):
        all_zero=[0 for c in self.classes]
        values=np.array([all_zero for i in range(len(X))])
        for i in range(len(X)):
            j=np.searchsorted(self.classes, X.iloc[i][0])
            if j<len(self.classes):
                values[i][j]=1
        return values
    
    def fit_transform(self,X,y=None):
        self.fit(X)
        return self.transform(X)
    
import unittest as ut

from sklearn.pipeline import Pipeline

class Test(ut.TestCase):
    def testLabelize(self):
        df=pd.DataFrame({'sex':['m','f',np.nan]})
        
        l=LabelBinarizerEx()
        l.fit(df)
        self.assertEqual(['f','m'],l.classes)
        self.assertEqual(['sex_f','sex_m'],l.columns)
        
        data=l.transform(pd.DataFrame({'sex':['m','f',np.nan]}))  
        self.assertTrue(np.array_equal(np.array([[0,1],[1,0],[0,0]]), data))
                
if __name__ == '__main__':
    ut.main(argv=['ignored', '-v'], exit=False)

