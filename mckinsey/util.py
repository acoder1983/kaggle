from sklearn.metrics import roc_auc_score
def score_auc(c,x,y):
    p=c.predict_proba(x)[:,1]
    return roc_auc_score(y,p)

def score_auc2(c,x,y):
    p=c.predict_proba(x)[:,1]
    p[p<0.8]=0
    return roc_auc_score(y,p)