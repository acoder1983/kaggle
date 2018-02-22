import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.base import BaseEstimator

class BasePredictor(BaseEstimator):
    
    def __init__(self,epsilon=100,alpha=0.01,learning_rate=0.01,max_iter=100):
        self.epsilon=epsilon
        self.alpha=alpha
        self.learning_rate=learning_rate
        self.max_iter=max_iter
        
        self.init_fit=False

    def init_fitting(self,X,y):
        print(str(self.get_params()))
        
        if not self.init_fit:
            users_train=set(X.userId.unique())
            items_train=set(X.movieId.unique())
            
            self.b_users=pd.Series(np.random.randn(len(users_train)),index=users_train)
            self.b_items=pd.Series(np.random.randn(len(items_train)),index=items_train)

            self.g_mean=np.mean(y)

            self.init_fit = True

              
    def fit(self, X, y):
        self.init_fitting(X,y)

        last_cost = np.inf    
        for it in range(self.max_iter):
            cost=self.alpha*(np.sum(self.b_users**2)+np.sum(self.b_items**2))
                    
            for idx in X.index:
                u=X.loc[idx,'userId']
                i=X.loc[idx,'movieId']
                e_ui=y[idx]-self.g_mean-self.b_users[u]-self.b_items[i]
                cost+=(e_ui)**2
                
                self.b_users[u] += self.learning_rate*(e_ui - self.alpha*self.b_users[u])
                self.b_items[i] += self.learning_rate*(e_ui - self.alpha*self.b_items[i])
            
            print('iter %d, cost %.2f'%(it+1,cost))

            if np.isnan(cost) or (last_cost > cost and last_cost-cost < self.epsilon) or last_cost<cost:
                break

            last_cost = cost
                
            self.learning_rate*=0.9
              
        return self

    def predict(self,X):
        y_pred=[self.g_mean for i in range(len(X))]

        for i in range(len(X)):
            u=X.iloc[i]['userId']
            m=X.iloc[i]['movieId']
            if u in self.b_users.index and m in self.b_items.index:
                y_pred[i] += self.b_users[u] + self.b_items[m]
                    
        return y_pred
                
    def get_params(self,deep=True):
        return {'epsilon':self.epsilon,
                'alpha':self.alpha,'learning_rate':self.learning_rate,'max_iter':self.max_iter}
    
    def set_params(self,**params):
        self.epsilon=params['epsilon']
        self.alpha=params['alpha']
        self.learning_rate=params['learning_rate']
        self.max_iter=params['max_iter']