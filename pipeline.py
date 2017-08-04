
# coding: utf-8

# In[35]:

import pandas as pd
from sklearn.preprocessing import Imputer,StandardScaler,LabelBinarizer,OneHotEncoder,LabelEncoder
from sklearn.pipeline import Pipeline,FeatureUnion
from onehot import LabelBinarizerEx
from addcols import AddColumns

class FeaturePipeline:
    def __init__(self,input_col,output_col,pipeline):
        if isinstance(input_col,list):
            self.input_cols=input_col
        else:
            self.input_cols=[input_col]
        self.output_col=output_col
        self.pipeline=pipeline

class DataFramePipeline:
    '''
    pipeline for dataframe
    '''
    def __init__(self,pipelines):
        '''
        @pipelines for dataframe
        element is FeaturePipeline
        
        '''
        self.pipelines=pipelines
    
    def fit_transform(self,df):
        '''
        @return transformed dataframe
        '''
        
        df_ret=df.copy()
        
        for p in self.pipelines:
            data=p.pipeline.fit_transform(df_ret[p.input_cols].values)
            last_step=p.pipeline.steps[-1][1]
            
            if isinstance(last_step,LabelBinarizerEx):
                df_tmp=pd.DataFrame(p.pipeline.fit_transform(df_ret[p.input_cols].values),columns=last_step.columns)
                df_ret=pd.concat([df_ret,df_tmp],axis=1)
            else:
                df_ret[p.output_col]=p.pipeline.fit_transform(df_ret[p.input_cols].values)
        
        return df_ret
    
    def transform(self,df):
        data=self.full_pipeline.transform(df)
        return pd.DataFrame(data,columns=self.columns)
    
import unittest as ut
import numpy as np
from util import *

class Test(ut.TestCase):        
    def testImputer(self):
        df=pd.DataFrame({'id':[1,2,1,np.nan]})
        
        dp=DataFramePipeline([FeaturePipeline('id','id_imputed', 
                                              Pipeline([('impute',Imputer(strategy='most_frequent'))]))
                      ])
        t_df=dp.fit_transform(df)
        
        self.assertEqual([1.,2.,1.,1.],list(t_df['id_imputed']))
        
    def testScaler(self):
        df=pd.DataFrame({'id':[1.,1.]})
        
        dp=DataFramePipeline([FeaturePipeline('id','id_imputed',
                               Pipeline([('scale',StandardScaler())]))
                      ])
        t_df=dp.fit_transform(df)
        
        self.assertEqual([0.,0.],list(t_df['id_imputed']))
        
        
    def testLabelBinary2(self):
        df=pd.DataFrame({'sex':['male','female',np.nan]})
        
        dp=DataFramePipeline([FeaturePipeline('sex','',Pipeline([('onehot',LabelBinarizerEx(['sex']))]))
                      ])
        t_df=dp.fit_transform(df)
        
        self.assertEqual([0,1,0],list(t_df['sex_female']))
        self.assertEqual([1,0,0],list(t_df['sex_male']))
        
    def testMutipleCols(self):
        df=pd.DataFrame({'id':[1.,1.],'name':[2.,np.nan],'fare':[2.,3.]})
        
        dp=DataFramePipeline([FeaturePipeline('id','id_scaled',Pipeline([('scale',StandardScaler())])),
                              FeaturePipeline('name','name_imputed',Pipeline([('impute',Imputer(strategy='most_frequent'))])),
                              FeaturePipeline(['id','fare'],'id_fare',Pipeline([('add',AddColumns())])),
                      ])
        t_df=dp.fit_transform(df)
        
        self.assertTrue(np.array_equal(np.array([
            [0.,2.,3.],
            [0.,2.,4.],
        ]),t_df[['id_scaled','name_imputed','id_fare']].values))
        
if __name__ == '__main__':
    ut.main(argv=['ignored', '-v'], exit=False)


