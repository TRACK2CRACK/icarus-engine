import numpy as np
import pandas as pd
import math
# from functools import partial

#https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm/42905202

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

def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out
######################

# def ema():
#     multiplier = 2 / (length + 1)
#     ema = (target[source] * multiplier) + (previous['ema'] * (1 - multiplier))

# # https://www.learndatasci.com/tutorials/python-finance-part-3-moving-average-trading-strategy/
# # https://gist.github.com/jtwyles/45b364220f7f4c5a7cf030540a675459#file-dema-py-L45

# # Calculates the SMA of an array of candles using the `source` price.
# def calculate_sma(candles, source):
#     length = len(candles)
#     sum = reduce((lambda last, x: { source: last[source] + x[source] }), candles)
#     sma = sum[source] / length
#     return sma
# # Calculates the EMA of an array of candles using the `source` price.
# def calculate_ema(candles, source):
#     length = len(candles)
#     target = candles[0]
#     previous = candles[1]

#     # if there is no previous EMA calculated, then EMA=SMA
#     if 'ema' not in previous or previous['ema'] == None:
#         return calculate_sma(candles, source)

#     else:
#         # multiplier: (2 / (length + 1))
#         # EMA: (close * multiplier) + ((1 - multiplier) * EMA(previous))
#         multiplier = 2 / (length + 1)
#         ema = (target[source] * multiplier) + (previous['ema'] * (1 - multiplier))

#         return ema

######################

#### intc_data.csv ####
# df = pd.read_csv('intc_data.csv', parse_dates=['Date'], index_col=['Date'])
# df['backward_ewm'] = df['Close'].ewm(span=20,min_periods=0,adjust=False,ignore_na=False).mean()
# df = df.sort_index()
# df['ewm'] = df['Close'].ewm(span=20,min_periods=0,adjust=False,ignore_na=False).mean()
# print(df[['ewm', 'backward_ewm']].tail())

window = 10
df = pd.read_csv('cs-movavg.csv', parse_dates=['Date'], index_col=['Date'])

df['ewm'] = df['Price'].ewm(span=window,min_periods=window-1,adjust=False,ignore_na=False).mean()
df['native-ewm'] = pd.Series.ewm(df['Price'], span=window).mean()
df['vectorV2-EMA'] = numpy_ewma_vectorized_v2(df['Price'], window)
df['np-EMA'] = numpy_ewma(df['Price'], window)
df['c-EMA'] = calculate_ema(df['Price'], window)

print(df[['Price','10-day EMA','ewm','vectorV2-EMA','np-EMA','native-ewm']].head(30))

