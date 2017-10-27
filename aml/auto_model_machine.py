# coding: utf-8

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

BASE_CLASSIFY_MODELS=[LogisticRegression(random_state=0),RandomForestClassifier(random_state=0),SVC()]

class AutoClassifier(BaseEstimator):
    
    def __init__(self,base_models=BASE_CLASSIFY_MODELS):
        self.base_models=[sklearn.clone(m) for m in base_models]
        
    def fit(self,X,y):
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
