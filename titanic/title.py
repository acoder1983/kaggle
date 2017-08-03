
# coding: utf-8

# In[9]:

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import re

class TitleExtractor(BaseEstimator,TransformerMixin):
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        titles=np.array([['RareOrNone'] for i in range(len(X))])
        for i in range(len(X)):
            
            m = re.search(' \w+\\.',X[i][0])
            if m:
                t=m.group()[1:-1]
                if t in {'Mr','Miss','Mrs','Master'}:
                    titles[i][0] = t
        return titles
    
    def fit_transform(self,X,y=None):
        self.fit(X)
        return self.transform(X)
    
import unittest as ut

from sklearn.pipeline import Pipeline

class Test(ut.TestCase):
    def testRange(self):
        t=TitleExtractor()
        
        df=pd.DataFrame({'name':['Braund, Mr. Owen Harris','Cumings, Don. John Bradley (Florence Briggs Thayer)']})
        data=t.transform(df[['name']].values)  
        
        self.assertTrue(np.array_equal(np.array([['Mr'],['Rare']]), data))
        
                
if __name__ == '__main__':
    ut.main(argv=['ignored', '-v'], exit=False)

