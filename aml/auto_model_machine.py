# coding: utf-8

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from collections import Counter

BASE_CLASSIFY_MODELS=[
    LogisticRegression(random_state=0),
    RandomForestClassifier(random_state=0),
    SVC(),
]

class BinaryClassifier(BaseEstimator):
    
    def __init__(self):
        self.base_models=_buildBaseModels()
        
    def fit(self,X,y):
        base_scores=_calc_base_model_scores(self.base_models)
        models_and_scores=enumerate(base_scores)
        last_ens_score=0.
        ens_score=base_scores[0]
        MIN_EPS=1e-3
        emsemble=Counter()
        while True:
            for m in self.base_models:
                ens_copy=ensemble.copy()
                ens_copy.append(m)
                ens_copy_score=_calc_score(ens_copy)
                if ens_copy_score > ens_score:
                    ensemble.append(m)
                    last_ens_score=ens_score
                    ens_score=ens_copy_score
                    if ens_score-last_ens_score<=MIN_EPS:
                        break

        for m_i in ensemble:
            
        scores=[cross_val_score(m,X,y,scoring='accuracy',cv=5).mean() 
            for m in self.base_models]
        self.best_model, self.best_score=sorted(zip(self.base_models,scores),key=lambda x:x[1],reverse=True)[0]
        # print(self.best_model,self.best_score)
        return self
    
    def predict(self,X):
        return self.best_model.predict(X)
    
    def fit_score(self):
        return self.best_score

    
    
import unittest as ut

class TestAutoClassifier(ut.TestCase):
    def test(self):
        pass

if __name__ == '__main__':
    ut.main()
