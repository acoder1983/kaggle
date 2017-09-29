def sarima_model(sales,range_num):
    best_r = None
    best_o = None
    min_aic=9999999999999
    min_bic=9999999999999

    orders=[(1,2,1),(2,2,1),(0,1,1)]
    for o in orders:
        try:
            m=sm.tsa.statespace.SARIMAX(sales,order=(o[0],o[1],o[2]),seasonal_order=(0,2,0,52))
            r=m.fit(disp=False)
            if r.aic < min_aic and r.bic < min_bic:
                best_r = r
                best_o = o
                min_aic=r.aic
                min_bic=r.bic

        except Exception as e:
            pass
    if best_r is not None:
        return best_r,best_o

    orders=make_orders(range_num,3)
    for o in orders:
        try:
            m=sm.tsa.statespace.SARIMAX(sales,order=(o[0],o[1],o[2]),seasonal_order=(0,2,0,52))
            r=m.fit(disp=False)
            if r.aic < min_aic and r.bic < min_bic:
                best_r = r
                best_o = o
                min_aic=r.aic
                min_bic=r.bic

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