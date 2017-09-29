import multiprocessing
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import time
import traceback
import random
import arima

def predict_sales(train_data,test_data,depts,idx):
    results=[]
    for dept in depts:
        try:
            dept_train=train_data[np.logical_and(train_data.Store==dept[0],train_data.Dept == dept[1])]
            dept_test=test_data[np.logical_and(test_data.Store==dept[0],test_data.Dept == dept[1])]
            dept_train=pd.DataFrame(dept_train.Weekly_Sales.values,index=pd.DatetimeIndex(dept_train.Date),columns=['Weekly_Sales'])
            dept_resample=dept_train.resample('1W').sum()
            dept_resample.fillna(0)
            dept_resample.index = dept_resample.index-(dept_resample.index[0]-dept_train.index[0])
            m ,o = arima.sarima_model(dept_resample.Weekly_Sales,3)
            predicts = m.forecast(len(dept_test))
            print('######################store %d dept %d, order %s' % (dept[0], dept[1], o))

            df = pd.DataFrame()
            df['Id']=np.core.defchararray.add("%d_%d_" % (dept[0],dept[1]),predicts.index.strftime("%Y-%m-%d"))
            df['Weekly_Sales']=predicts.values.astype(int)
            results.append(df)
            time.sleep(1)
        except Exception as e:
            print('********************** %s'%str(dept))
            print(e)
            traceback.print_exc()

    pd.concat(results).to_csv('result_%d.csv'%idx, index=False,header=False)


    # df = pd.DataFrame()
    # df['Id']=np.core.defchararray.add("%d_%d_" % (dept[0],dept[1]),predicts.index.strftime("%Y-%m-%d"))
    # df['Weekly_Sales']=predicts.values.astype(int)
    # result=pd.concat([result,df])

if __name__ == '__main__':
    train_data = pd.read_csv('raw_data/train.csv')
    test_data = pd.read_csv('raw_data/test.csv')
    t = time.time()
    times=[]
    result=pd.DataFrame()
    tasks=2
    pool = multiprocessing.Pool(processes = 4)

    dept_ids=test_data.groupby(['Store','Dept']).count().index
    depts=[]
    for i in range(4*tasks):
        idx=random.randrange(1,len(dept_ids))
        depts.append(dept_ids[idx])
    for i in range(0,4*tasks,tasks):
        pool.apply_async(predict_sales, (train_data,test_data,depts[i:i+tasks], i/tasks))
        # pool.apply_async(predict_sales,(train_data[np.logical_and(train_data.Store==dept[0],train_data.Dept == dept[1])], test_data[np.logical_and(test_data.Store==dept[0],test_data.Dept == dept[1])], dept,))
        # times.append(int(time.time()-t))
        # t = time.time()

    pool.close()
    pool.join()

    print('total time %ds' % int(time.time()-t))