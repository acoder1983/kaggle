import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.base import BaseEstimator

class Svd(BaseEstimator):
    
    def __init__(self,factors=3,epsilon=100,alpha=0.01,learning_rate=0.01,max_iter=100):
        self.factors=factors
        self.epsilon=epsilon
        self.alpha=alpha
        self.learning_rate=learning_rate
        self.max_iter=max_iter
        
        self.init_fit=False

    def init_fitting(self,X,y):
        print(str(self.get_params()))
        
        if not self.init_fit:
            users_train=set(X.userId.unique())
            self.f_users={}
            for u in users_train:
                self.f_users[u]=np.ones(self.factors)/10

            items_train=set(X.movieId.unique())
            self.f_items={}
            for i in items_train:
                self.f_items[i]=np.ones(self.factors)/10

            self.b_users=pd.Series(np.random.randn(len(users_train)),index=users_train)
            self.b_items=pd.Series(np.random.randn(len(items_train)),index=items_train)

            self.g_mean=np.mean(y)

            self.init_fit = True

    def fit_factors_onebyone(self,X,y):
        self.init_fitting(X,y)

        early_stop=False
        for f in range(self.factors):
            print('train factors %d'%(f+1))
            
            learning_rate = self.learning_rate
            last_cost=np.inf
            for it in range(self.max_iter):
                cost=self.alpha*(np.sum(self.b_users**2)+np.sum(self.b_items**2))
                for fm in self.f_users,self.f_items:
                    for _,f_i in fm.items():
                        cost+=self.alpha*np.sum(f_i**2)

                for idx in X.index:
                    u=X.loc[idx,'userId']
                    i=X.loc[idx,'movieId']
                    r_pred=np.dot(self.f_users[u],self.f_items[i])
                    e_ui=y[idx] - self.g_mean - self.b_users[u] - self.b_items[i] - r_pred
                    cost+=(e_ui)**2

                    f_uf = self.f_users[u][f]
                    self.f_users[u][f] += self.learning_rate*(e_ui*self.f_items[i][f] - self.alpha*self.f_users[u][f])
                    self.f_items[i][f] += self.learning_rate*(e_ui*f_uf - self.alpha*self.f_items[i][f])

                    self.b_users[u] += self.learning_rate*(e_ui - self.alpha*self.b_users[u])
                    self.b_items[i] += self.learning_rate*(e_ui - self.alpha*self.b_items[i])

                print('iter %d, cost %.2f'%(it+1,cost))

                if np.isnan(cost) or (last_cost > cost and last_cost-cost < self.epsilon) or last_cost<cost:
                    early_stop = it < 2 and last_cost < cost
                    break

                last_cost = cost

                learning_rate*=0.9
                
            if early_stop:
                break
            

        return self
              
    def fit_factors_all(self, X, y):
        self.init_fitting(X,y)

        last_cost = np.inf    
        learning_rate = self.learning_rate
        for it in range(self.max_iter):
            cost=self.alpha*(np.sum(self.b_users**2)+np.sum(self.b_items**2))
            for f in self.f_users,self.f_items:
                for _,f_i in f.items():
                    cost+=self.alpha*np.sum(f_i**2)
                    
            for idx in X.index:
                u=X.loc[idx,'userId']
                i=X.loc[idx,'movieId']
                r_pred=np.dot(self.f_users[u],self.f_items[i])
                e_ui=y[idx]-self.g_mean-self.b_users[u]-self.b_items[i]-r_pred
                cost+=(e_ui)**2
                
                f_u = deepcopy(self.f_users[u])
                self.f_users[u] += learning_rate*(e_ui*self.f_items[i] - self.alpha*self.f_users[u])
                self.f_items[i] += learning_rate*(e_ui*f_u - self.alpha*self.f_items[i])

                self.b_users[u] += learning_rate*(e_ui - self.alpha*self.b_users[u])
                self.b_items[i] += learning_rate*(e_ui - self.alpha*self.b_items[i])
            
            print('iter %d, cost %.2f'%(it+1,cost))

            if np.isnan(cost) or (last_cost > cost and last_cost-cost < self.epsilon) or last_cost<cost:
                break

            last_cost = cost
                
            learning_rate*=0.9
              
        return self

    def predict(self,X):
        y_pred=[self.g_mean for i in range(len(X))]

        for i in range(len(X)):
            u=X.iloc[i]['userId']
            m=X.iloc[i]['movieId']
            if u in self.f_users and m in self.f_items:
                y_pred[i] += self.b_users[u] + self.b_items[m] + np.dot(self.f_users[u],self.f_items[m])
                    
        return y_pred
                
    def get_params(self,deep=True):
        return {'factors':self.factors,'epsilon':self.epsilon,
                'alpha':self.alpha,'learning_rate':self.learning_rate,'max_iter':self.max_iter}
    
    def set_params(self,**params):
        self.factors=params['factors']
        self.epsilon=params['epsilon']
        self.alpha=params['alpha']
        self.learning_rate=params['learning_rate']
        self.max_iter=params['max_iter']