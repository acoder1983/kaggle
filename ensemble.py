
# coding: utf-8

# In[37]:

from brew.base import Ensemble
from brew.stacking import EnsembleStackClassifier,EnsembleStack
import sklearn

class EnsembleStackClassifierEx(EnsembleStackClassifier):
    def __init__(self, stack, combiner=None):
        EnsembleStackClassifier.__init__(self, stack,combiner)
        
    def score(self, X, y, sample_weight=None):
        return (self.predict(X)-y).astype(float).sum()/len(X)
    
    def get_params(self, deep=True):
        return {'stack':self.stack}
    

