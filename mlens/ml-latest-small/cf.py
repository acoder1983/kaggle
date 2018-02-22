import numpy as np
import pandas as pd
from copy import deepcopy


class IIcf:
    def __init__(self,k=3):
        self.k=k
        
    
    def get_params(self,deep=True):
        return {'k':self.k}
    
    def set_params(self,**params):
        self.k=params['k']
        
    def fit(self,X,y):
        self.ratings_matrix = X[['userId','movieId','rating']].pivot_table(index='userId',columns='movieId')
        self.ratings_matrix.columns = self.ratings_matrix.columns.levels[1]
        
        self.item_sims = self.ratings_matrix.corr(min_periods=5)
        ratings_summ = self.ratings_matrix.describe().T
        self.r_items_mean = ratings_summ['mean']
        self.r_items_std = ratings_summ['std']
        
        self.users_train=set(self.ratings_matrix.index)
        self.items_train=set(self.ratings_matrix.columns)
        
        return self
    
    def predict(self,X):
        
        y_pred=[]

        for i in X.index:
            item=X.loc[i,'movieId']
            user=X.loc[i,'userId']
            
            pred = 0
            if item in self.items_train and user in self.users_train:
                pred=self.r_items_mean[item]
                sim_items=self.item_sims[item].sort_values(ascending=False)[1:self.k+1]

                r_sum=0
                r_w=0
                for j in sim_items.index:
                    w=sim_items[j]
                    if not np.isnan(w) and not np.isnan(self.ratings_matrix.loc[user,j]) and self.r_items_std[j] != 0.:
                        r_j_norm=(self.ratings_matrix.loc[user,j]-self.r_items_mean[j])/self.r_items_std[j]
                        r_sum+=r_j_norm*w
                        r_w+=np.abs(w)

                if r_w != 0:
                    pred+= r_sum*self.r_items_std[item]/r_w
                
            y_pred.append(pred)

        return y_pred