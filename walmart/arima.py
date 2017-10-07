import statsmodels.api as sm
import traceback

def run_model(sales, orders, season, speedup):
    best_r = None
    best_o = None
    min_aic=9999999999999
    min_bic=9999999999999
    
    for o in orders:
        try:
            if speedup:
                if season is None:
                    m=sm.tsa.statespace.SARIMAX(sales,order=o,
                         simple_differencing=True, enforce_stationarity=False, enforce_invertibility=False)
                else:
                    m=sm.tsa.statespace.SARIMAX(sales,order=o,seasonal_order=season,
                         simple_differencing=True, enforce_stationarity=False, enforce_invertibility=False)
                    
            else:
                if season is None:
                    m=sm.tsa.statespace.SARIMAX(sales,order=o)
                else:
                    m=sm.tsa.statespace.SARIMAX(sales,order=o,seasonal_order=season)
                
            r=m.fit(disp=False)
            if r.aic < min_aic and r.bic < min_bic:
                best_r = r
                best_o = o
                min_aic=r.aic
                min_bic=r.bic
                print(o,r.aic,r.bic)

        except Exception as e:
            print(e)
            # traceback.print_exc()
    return best_r,best_o

def sarima_model(sales,range_num,speedup=True):
    # orders=[(1,2,1),(2,2,1),(0,1,1)]
    # best_r,best_o=run_model(sales,orders,(0,2,0,52),speedup)
    # if best_r is not None:
    #     return best_r,best_o

#     orders=make_orders(range_num,3)
#     best_r,best_o=run_model(sales,orders,(0,2,0,52),speedup)
#     if best_r is not None:
#         return best_r,best_o

#     best_r,best_o=run_model(sales,orders,None,speedup)

    best_r = None
    best_o = None
    min_aic=9999999999999
    min_bic=9999999999999
    
    # orders=make_orders(range_num,6)
    orders=[(0, 1, 1, 0, 2, 0)]
    for o in orders:
        try:
            m=sm.tsa.statespace.SARIMAX(sales,order=(o[0],o[1],o[2]),seasonal_order=(o[3],o[4],o[5],52))
                
            r=m.fit(disp=False)
            if r.aic < min_aic and r.bic < min_bic:
                best_r = r
                best_o = o
                min_aic=r.aic
                min_bic=r.bic
                print(o,r.aic,r.bic)

        except Exception as e:
            pass
    return best_r,best_o

def make_orders(range_num, seq_num):
    if seq_num == 0:
        return [[]]
    else:
        orders=[]
        sub_orders=make_orders(range_num,seq_num-1)
        for o in sub_orders:
            for i in range(range_num):
                s=o.copy()
                s.append(i)
                orders.append(s)
        return orders