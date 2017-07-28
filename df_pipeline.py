
# coding: utf-8

# In[3]:

import pandas as pd
from sklearn.preprocessing import Imputer,StandardScaler,LabelBinarizer,OneHotEncoder,LabelEncoder
from sklearn.pipeline import Pipeline,FeatureUnion
from label_binary import LabelBinarizerEx

class DfPipeline:
    '''
    pipeline for dataframe
    auto generate transformed column names
    '''
    def __init__(self,pipelines):
        '''
        @pipelines,(col_name,Pipeline)
        '''
        self.pipelines=pipelines
        self.full_pipeline=FeatureUnion(transformer_list=pipelines)
    
    def fit_transform(self,df):
        data=self.full_pipeline.fit_transform(df)
        
        self.columns=[]
        for p in self.pipelines:
            last_step=p[1].steps[-1][1]
            if isinstance(last_step, LabelBinarizerEx):
                self.columns += last_step.columns
            else:
                self.columns.append(p[0])
        
        return pd.DataFrame(data,columns=self.columns)
    
    def transform(self,df):
        data=self.full_pipeline.transform(df)
        return pd.DataFrame(data,columns=self.columns)
    
import unittest as ut
import numpy as np
from util import *

class Test(ut.TestCase):        
    def testImputer(self):
        df=pd.DataFrame({'id':[1,2,1,np.nan]})
        
        dp=DfPipeline([('id',Pipeline([('select',DataFrameSelecter(cols=['id'])),
                                       ('impute',Imputer(strategy='most_frequent'))]))
                      ])
        t_df=dp.fit_transform(df)
        
        self.assertEqual([1.,2.,1.,1.],list(t_df['id']))
        
    def testScaler(self):
        df=pd.DataFrame({'id':[1.,1.]})
        
        dp=DfPipeline([('id',Pipeline([('select',DataFrameSelecter(cols=['id'])),
                                       ('scale',StandardScaler())]))
                      ])
        t_df=dp.fit_transform(df)
        
        self.assertEqual([0.,0.],list(t_df['id']))
        
        
    def testMutipleCols(self):
        df=pd.DataFrame({'id':[1.,1.],'name':[2.,np.nan]})
        
        dp=DfPipeline([('id',Pipeline([('select',DataFrameSelecter(cols=['id'])),
                                       ('scale',StandardScaler())])),
                       ('name',Pipeline([('select',DataFrameSelecter(cols=['name'])),
                                       ('impute',Imputer(strategy='most_frequent'))])),
                      ])
        t_df=dp.fit_transform(df)
        
        self.assertTrue(np.array_equal(np.array([[0.,2.],
                                  [0.,2.]]),
                         t_df.values))
        self.assertEqual(['id','name'],list(t_df.columns))
        
    def testLabelBinary2(self):
        df=pd.DataFrame({'sex':['male','female',np.nan]})
        
        dp=DfPipeline([('sex',Pipeline([('select',DataFrameSelecter(cols=['sex'])),
                                       ('onehot',LabelBinarizerEx())]))
                      ])
        t_df=dp.fit_transform(df)
        
        self.assertEqual([0,1,0],list(t_df['sex_female']))
        self.assertEqual([1,0,0],list(t_df['sex_male']))
        
    def testDf(self):
        df=pd.DataFrame({'id':[1,2,np.nan],'sex':['male','female',np.nan]})
        
        dp=DfPipeline([('id',Pipeline([('select',DataFrameSelecter(cols=['id'])),
                                       ('impute',Imputer(strategy='most_frequent'))])),
                       ('sex',Pipeline([('select',DataFrameSelecter(cols=['sex'])),
                                       ('onehot',LabelBinarizerEx())]))
                      ])
        t_df=dp.fit_transform(df)
        
        self.assertEqual(['id','sex_female','sex_male'], list(t_df.columns))
        self.assertEqual([0,1,0],list(t_df['sex_female']))
        self.assertEqual([1,0,0],list(t_df['sex_male']))
        
if __name__ == '__main__':
    ut.main(argv=['ignored', '-v'], exit=False)

