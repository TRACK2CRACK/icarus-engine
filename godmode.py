import numpy as np
import pandas as pd
import math
# from functools import partial

def numpy_ewma(data, window):
    alpha = 2 /(window + 1.0)
    scale = 1/(1-alpha)
    n = data.shape[0]
    scale_arr = (1-alpha)**(-1*np.arange(n))
    weights = (1-alpha)**np.arange(n)
    pw0 = (1-alpha)**(n-1)
    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = cumsums*scale_arr[::-1] / weights.cumsum()

    return out

def numpy_ewma_vectorized_v2(data, window):

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out
def EMA(df, n):  
    EMA = pd.Series(df.ewm(span = n, min_periods = n - 1), name = 'EMA_' + str(n))  
    df = df.join(EMA)  
    return df
######################
# window = 10
# data = pd.read_excel('cs-movavg.xls')
# df = pd.DataFrame(data, columns=['Date','Price','10-day EMA'],index=['Date'])

# print(df)
# print (EMA(df['Price'],window))


#### intc_data.csv ####
# df = pd.read_csv('intc_data.csv', parse_dates=['Date'], index_col=['Date'])
# df['backward_ewm'] = df['Close'].ewm(span=20,min_periods=0,adjust=False,ignore_na=False).mean()
# df = df.sort_index()
# df['ewm'] = df['Close'].ewm(span=20,min_periods=0,adjust=False,ignore_na=False).mean()
# print(df[['ewm', 'backward_ewm']].tail())

df = pd.read_csv('cs-movavg.csv', parse_dates=['Date'], index_col=['Date'])
df['ewm'] = df['Price'].ewm(span=10,min_periods=10,adjust=False,ignore_na=False).mean()
print(df[['10-day EMA','ewm']].head(30))

