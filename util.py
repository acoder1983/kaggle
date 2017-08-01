
# coding: utf-8

# In[37]:

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelecter(BaseEstimator,TransformerMixin):
    def __init__(self,cols):
        self.cols=cols
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        return X[self.cols].values
    
class DataFrameSelecter2(BaseEstimator,TransformerMixin):
    def __init__(self,cols):
        self.cols=cols
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        return X[self.cols]

class DataFrameSelecter3(BaseEstimator,TransformerMixin):
    def __init__(self,cols):
        self.cols=cols
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        return list(X[self.cols])
        
# In[ ]:

class DataFrameDropper(BaseEstimator,TransformerMixin):
    def __init__(self,cols):
        self.cols=cols
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        return X.drop(self.cols,axis=1)


# In[ ]:

class NanDropper(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        return X.dropna()


# In[43]:

import unittest as ut
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline

class Test(ut.TestCase):
    def testDataFrameSelecter(self):
        df=pd.DataFrame({'a':[1,2],'b':['1','2']})
        Selecter=DataFrameSelecter('a')
        self.assertTrue(np.array_equal(np.array([1,2]),Selecter.transform(df)))
        
        pipe=Pipeline([('Selecter',DataFrameSelecter('b'))])
        data=pipe.fit_transform(df)
        self.assertTrue(np.array_equal(np.array(['1','2']),data))
        
if __name__ == '__main__':
    ut.main(argv=['ignored', '-v'], exit=False)

