# coding: utf-8
from __future__ import division
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import Counter
import time
import numpy as np

BASE_CLASSIFY_MODELS=[
    LogisticRegression(random_state=42,n_jobs=1),
    # RandomForestClassifier(random_state=42),
    # SVC(),
]

BASE_MODEL_PARAMS=[
    {'C':[0.001,0.01,0.1,1,10,100,1000],'penalty':['l1','l2'],'class_weight':['balanced',None]},
    # {'penalty':['l1','l2']}
]

def buildBaseModels(models,params):
    baseModels=[]
    for m,p in zip(models,params):
        keys = list(p.keys())
        k_cursors=[[len(p[k]),0] for k in keys]
        while k_cursors[0][1]<k_cursors[0][0]:
            mo = sklearn.clone(m)
            kv=mo.get_params()
            for i,kc in enumerate(k_cursors):
                kv[keys[i]]=p[keys[i]][kc[1]]
            mo.set_params(**kv)
            baseModels.append(mo)
            for i in range(len(keys)-1,-1,-1):
                if k_cursors[i][1]<k_cursors[i][0]-1:
                    k_cursors[i][1]+=1
                    break
                else:
                    if i == 0:
                        k_cursors[i][1]+=1
                    else:
                        k_cursors[i][1]=0
                
    return baseModels

class BinaryClassifier(BaseEstimator):
    
    def __init__(self):
        self.base_models=buildBaseModels(BASE_CLASSIFY_MODELS, BASE_MODEL_PARAMS)
        
    def fit(self,X,y):
        skf=StratifiedKFold(random_state=42)
        for train_idx,test_idx in skf.split(X,y):
            X_train=X[train_idx]
            y_train=y[train_idx]
            X_test =X[test_idx]
            y_test =y[test_idx]
            break
        model_probs=[self._calc_prob(m,X_train,y_train,X_test) for m in self.base_models]
        ensembles=[Counter({i:1}) for i in range(len(self.base_models))]
        en_scores=[self._calc_en_score(en,model_probs,y_test) for en in ensembles]
        MIN_EPS=1e-3
        for i, en in enumerate(ensembles):
            while True:
                find_m=False
                for k in range(len(self.base_models)):
                    en_copy=ensembles[i].copy()
                    en_copy.update({k:1})
                    en_copy_score=self._calc_en_score(en_copy,model_probs,y_test)
                    if en_copy_score - en_scores[i] > MIN_EPS:
                        ensembles[i]=en_copy
                        en_scores[i]=en_copy_score
                        find_m=True
                        break
                if not find_m:
                    break

        print(sorted(zip(en_scores,ensembles),key=lambda x:x[0],reverse=True)[:5])
        return self
    
    def _calc_prob(self,model, X_train,y_train,X_test):
        model.fit(X_train,y_train)
        prob=model.predict_proba(X_test)
        return prob


    def _calc_en_score(self,ensemble,model_probs,y_test):
        prob=0.
        cnt=0
        for m_i,m_c in ensemble.items():
            prob += model_probs[m_i]*m_c
            cnt+=m_c
        avg_prob=prob/cnt
        cls_idx=(avg_prob[:,0]<avg_prob[:,1]).astype('int')
        y_pred=np.array([self.base_models[0].classes_[i] for i in cls_idx])
        return accuracy_score(y_test,y_pred)

    def predict(self,X):
        return self.best_model.predict(X)
    
    def fit_score(self):
        return self.best_score

    
    
import unittest as ut

class TestAutoClassifier(ut.TestCase):
    def testBuildBaseModels(self):
        models=buildBaseModels([LogisticRegression()],
        [{'C':[0.001,0.01,0.1,1,10,100,1000],'penalty':['l1','l2'],'class_weight':['balanced',None]}])
        # self.assertEqual(2,len(models))
        # self.assertEqual(10, models[1].C)
        self.assertEqual(28,len(models))

    def testFit(self):
        pass

if __name__ == '__main__':
    ut.main()
