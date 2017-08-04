
# coding: utf-8

# In[59]:

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class GroupImputer(BaseEstimator,TransformerMixin):
    '''
    impute according to group cols
    default use median
    '''
    def __init__(self,cols):
        '''
        group cols + target col
        '''
        self.cols=cols
        self.group_cols=cols[:-1]
        self.impute_col=cols[-1]
        
    def fit(self, X, y=None):
        df = pd.DataFrame(X,columns=self.cols)
        self.medians=df.groupby(self.group_cols).median()[self.impute_col]
        return self
            
    
    def transform(self, X, y=None):
        df = pd.DataFrame(X,columns=self.cols)
        
        missing_index=df[df[self.impute_col].isnull()].index
        
        for i in missing_index:
            idx_values=[df.loc[i,c] for c in self.group_cols]
            if len(self.group_cols) > 1:
                df.loc[i,self.impute_col] = self.medians[tuple(idx_values)]
            else:
                df.loc[i,self.impute_col] = self.medians[idx_values[0]]
        return df[[self.impute_col]].values
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
import unittest as ut

from sklearn.pipeline import Pipeline

class Test(ut.TestCase):
    def test(self):
        df=pd.DataFrame({'sex':[0,0,0,1,1,1,0,1],'age':[1,2,3,4,5,6,np.nan,np.nan]})
        
        g=GroupImputer(['sex','age'])
        data=g.fit_transform(df[['sex','age']].values)
        self.assertEqual(['sex'],g.group_cols)
        self.assertEqual('age',g.impute_col)
        
        df_ret=pd.DataFrame(data,columns=['age'])
        
        self.assertTrue(np.array_equal(np.array([1,2,3,4,5,6,2,5]), df_ret['age'].values))
        
        df=pd.DataFrame({'sex':[0,1,0],'class':[0,1,0],'age':[1,2,np.nan]})
        
        g=GroupImputer(['sex','class','age'])
        data=g.fit_transform(df[['sex','class','age']].values)
        
        df_ret=pd.DataFrame(data,columns=['age'])
        
        self.assertTrue(np.array_equal(np.array([1,2,1]), df_ret['age'].values))
                
if __name__ == '__main__':
    ut.main(argv=['ignored', '-v'], exit=False)

