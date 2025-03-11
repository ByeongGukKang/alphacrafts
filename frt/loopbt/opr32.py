import numpy as np

from alphacrafts.frt.loopbt.opr_src import _1d_flt32, _2d_flt32, _ts_flt32

# Wrapper for numba compiled functions
# Avoid using decorator for better performance (less overheads)
# Decorator is called every time the function is called
# TODO """ docstringed functions are currently slower than the pure numpy functions

def __opr_src_wrapper(func_name):
    def wrapped_func(array: np.ndarray, axis: int, **kargs):
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array, dtype=np.float32)
        if array.ndim == 1:
            return getattr(_1d_flt32, func_name)(array, **kargs)
        elif array.ndim == 2:
            return getattr(_2d_flt32, func_name)(array, axis, **kargs)
        else:
            ValueError(f'Only 1D and 2D arrays are supported, but got {array.ndim}D')
    return wrapped_func

def __opr_src_wrapper_1donly(func_name):
    def wrapped_func(array: np.ndarray, **kargs):
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array, dtype=np.float32)
        if array.ndim == 1:
            return getattr(_1d_flt32, func_name)(array, **kargs)
        else:
            ValueError(f'Only 1D and 2D arrays are supported, but got {array.ndim}D')
    return wrapped_func

def __oprts_src_wrapper(func_name):
    def wrapped_func(array: np.ndarray, **kargs):
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array, dtype=np.float32)
        if array.ndim == 2:
            return getattr(_ts_flt32, func_name)(array, **kargs)
        else:
            ValueError(f'Only 2D array is supported, but got {array.ndim}D')
    return wrapped_func


##### Compiled Numpy Functions #####
__opr_mean = __opr_src_wrapper('mean')
def opr_mean(array: np.ndarray, axis: int = 0):
    """mean function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
         in 1D array, axis is ignored (only for 2D array).
    
    """
    return __opr_mean(array, axis)

__opr_nanmean = __opr_src_wrapper('nanmean')
def opr_nanmean(array: np.ndarray, axis: int = 0):
    """nanmean function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
         in 1D array, axis is ignored (only for 2D array).
    
    """
    return __opr_nanmean(array, axis)

__opr_std = __opr_src_wrapper('std')
def opr_std(array: np.ndarray, axis: int = 0):
    """std function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
         in 1D array, axis is ignored (only for 2D array).
    
    """
    return __opr_std(array, axis)

__opr_nanstd = __opr_src_wrapper('nanstd')
def opr_nanstd(array: np.ndarray, axis: int = 0):
    """nanstd function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
         In 1D array, axis is ignored (only for 2D array).
    
    """
    return __opr_nanstd(array, axis)

__opr_min = __opr_src_wrapper('min')
def opr_min(array: np.ndarray, axis: int = 0):
    """min function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
         in 1D array, axis is ignored (only for 2D array)
    
    """
    return __opr_min(array, axis)

__opr_nanmin = __opr_src_wrapper('nanmin')
def opr_nanmin(array: np.ndarray, axis: int = 0):
    """Nanmin function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
         in 1D array, axis is ignored (only for 2D array).
    
    """
    return __opr_nanmin(array, axis)

__opr_max = __opr_src_wrapper('max')
def opr_max(array: np.ndarray, axis: int = 0):
    """max function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): False for row, True for column. Default is False
         in 1D array, axis is ignored (only for 2D array)
    
    """
    return __opr_max(array, axis)

__opr_nanmax = __opr_src_wrapper('nanmax')
def opr_nanmax(array: np.ndarray, axis: int = 0):
    """nanmax function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): False for row, True for column. Default is False
            in 1D array, axis is ignored (only for 2D array)
    
    """
    return __opr_nanmax(array, axis)

__opr_percentile = __opr_src_wrapper('percentile')
def opr_percentile(array: np.ndarray, axis: int = 0, hurdle: float = 50.0):
    """percentile function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): False for row, True for column. Default is False
         in 1D array, axis is ignored (only for 2D array)
        hurdle (float 0 ~ 100): percentile. Default is 50.0
    
    """
    return __opr_percentile(array, axis, hurdle=hurdle)

__opr_nanpercentile = __opr_src_wrapper('nanpercentile')
def opr_nanpercentile(array: np.ndarray, axis: int = 0, hurdle: float = 50.0):
    """nanpercentile function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
         in 1D array, axis is ignored (only for 2D array).
        hurdle (float 0 ~ 100): percentile. Default is 50.0
    
    """
    return __opr_nanpercentile(array, axis, hurdle=hurdle)

__opr_sum = __opr_src_wrapper('sum')
def opr_sum(array: np.ndarray, axis: int = 0):
    """sum function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
         in 1D array, axis is ignored (only for 2D array).
    
    """
    return __opr_sum(array, axis)

__opr_nansum = __opr_src_wrapper('nansum')
def opr_nansum(array: np.ndarray, axis: int = 0):
    """nansum function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
         in 1D array, axis is ignored (only for 2D array).
    
    """
    return __opr_nansum(array, axis)

# TODO Handle these small functions in a better way
"""
def opr_isnan(array: np.ndarray):
    isnan function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
    
    
    return np.isnan(array)
"""

"""
def opr_abs(array: np.ndarray):
    abs function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
    
    
    return np.abs(array)
"""

##### Operators #####
# isall
def opr_isall(array: np.ndarray, value: float):
    """Check if all the elements are equal to the value

    Args:
        array (np.ndarray): 1D array
        value (float): value to check
    
    """
    if array.dtype != np.float32:
        array = array.astype(np.float32)

    if array.ndim != 1:
        ValueError(f'Only 1D array is supported, but got {array.ndim}D')

    return _1d_flt32.isall(array.ravel('C'), value)

# isany
def opr_isany(array: np.ndarray, value: float):
    """Check if any element is equal to the value

    Args:
        array (np.ndarray): 1D array
        value (float): value to check
    
    """
    if array.dtype != np.float32:
        array = array.astype(np.float32)
        
    if array.ndim != 1:
        ValueError(f'Only 1D array is supported, but got {array.ndim}D')

    return _1d_flt32.isany(array.ravel('C'), value)

# Z-score
__opr_zscore = __opr_src_wrapper('zscore')
def opr_zscore(array: np.ndarray, axis: int = 0):
    """Z-score Normalization

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
         in 1D array, axis is ignored (only for 2D array).
    
    """
    return __opr_zscore(array, axis)

__oprts_zscore = __oprts_src_wrapper('zscore')
def oprts_zscore(array: np.ndarray):
    """[TS] Z-score Normalization 

    - only 2D array is supported

    Args:
        array (np.ndarray): 2D array
    
    """
    return __oprts_zscore(array)

# MinMax
__opr_minmax = __opr_src_wrapper('minmax')
def opr_minmax(array: np.ndarray, axis: int = 0):
    """Min-Max Normalization

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
         in 1D array, axis is ignored (only for 2D array).
    
    """
    return __opr_minmax(array, axis)        

__oprts_minmax = __oprts_src_wrapper('minmax')
def oprts_minmax(array):
    """[TS] Min-Max Normalization

    - only 2D array is supported

    Args:
        array (np.ndarray): 2D array
    
    """
    return __oprts_minmax(array)

# Rank
__opr_rank = __opr_src_wrapper('rank')
def opr_rank(array: np.ndarray, axis: int = 0, method: str = 'dense'):
    """Rank Normalization

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
         in 1D array, axis is ignored (only for 2D array).
        method (str): 'dense', 'average', 'min', 'max', 'ordinal'. Default is 'dense'
        
    """
    return __opr_rank(array, axis, method=method)        

__oprts_rank = __oprts_src_wrapper('rank')
def oprts_rank(array: np.ndarray, method: str = 'dense'):
    """[TS] Rank Normalization

    - only 2D array is supported

    Args:
        array (np.ndarray): 2D array
        method (str): 'dense', 'average', 'min', 'max', 'ordinal'. Default is 'dense'
        
    """
    return __oprts_rank(array, method=method)

# Percentile Rank
__opr_rank_pct = __opr_src_wrapper('rank_pct')
def opr_rank_pct(array: np.ndarray, axis: int = 0, method: str ='dense'):
    """Percentile Rank Normalization

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
         in 1D array, axis is ignored (only for 2D array).
        method (str): 'dense', 'average', 'min', 'max', 'ordinal'. Default is 'dense'
    
    """
    return __opr_rank_pct(array, axis, method=method)

__oprts_rank_pct = __oprts_src_wrapper('rank_pct')
def oprts_rank_pct(array: np.ndarray, method: str ='dense'):
    """[TS] Percentile Rank Normalization

    - only 2D array is supported

    Args:
        array (np.ndarray): 2D array
        method (str): 'dense', 'average', 'min', 'max', 'ordinal'. Default is 'dense'
    
    """
    return __oprts_rank_pct(array, method=method)

# Winsorization by Z-score
__opr_winsor_by_zscore = __opr_src_wrapper('winsor_by_zscore')
def opr_winsor_by_zscore(array: np.ndarray, axis: int = 0, hurdle: float=2.58):
    """Winsorization by Z-score

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
         in 1D array, axis is ignored (only for 2D array).
        hurdle (float): Z-score threshold. Default is 2.58
    
    """
    return __opr_winsor_by_zscore(array, axis, hurdle=hurdle)        

__oprts_winsor_by_zscore = __oprts_src_wrapper('winsor_by_zscore')
def oprts_winsor_by_zscore(array: np.ndarray, hurdle: float = 2.58):
    """[TS] Winsorization by Z-score

    - only 2D array is supported

    Args:
        array (np.ndarray): 2D array
        hurdle (float): Z-score threshold. Default is 2.58
    
    """
    return __oprts_winsor_by_zscore(array, hurdle=hurdle)

# Winsorization by Percentile
__opr_winsor_by_percentile = __opr_src_wrapper('winsor_by_percentile')
def opr_winsor_by_percentile(array: np.ndarray, axis: int = 0, hurdle: float = 1.0):
    """Winsorization by Percentile

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
         in 1D array, axis is ignored (only for 2D array).
        hurdle (float 0 ~ 1): top/bottom n percentile to winsorize the data. Default is 1.0
         hurdle=1.0 means 1% from top and 1% from bottom
    
    """
    return __opr_winsor_by_percentile(array, axis, hurdle=hurdle)        

__oprts_winsor_by_percentile = __oprts_src_wrapper('winsor_by_percentile')
def oprts_winsor_by_percentile(array: np.ndarray, hurdle: float = 1.0):
    """[TS] Winsorization by Percentile

    - only 2D array is supported

    Args:
        array (np.ndarray): 2D array
        hurdle (float 0 ~ 1): top/bottom n percentile to winsorize the data. Default is 1.0.
         hurdle=1.0 means 1% from top and 1% from bottom
    
    """
    return __oprts_winsor_by_percentile(array, hurdle=hurdle)

# Softmax
__opr_softmax = __opr_src_wrapper('softmax')
def opr_softmax(array: np.ndarray, axis: int = 0):
    """Softmax Normalization

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
         in 1D array, axis is ignored (only for 2D array).
    
    """
    return __opr_softmax(array, axis)

__oprts_softmax = __oprts_src_wrapper('softmax')
def oprts_softmax(array: np.ndarray):
    """[TS] Softmax Normalization

    - only 2D array is supported

    Args:
        array (np.ndarray): 2D array
    
    """
    return __oprts_softmax(array)

# Top
__opr_top = __opr_src_wrapper('top')
def opr_top(array: np.ndarray, axis: int = 0, n: int = 5):
    """Top n

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
         in 1D array, axis is ignored (only for 2D array).
        n (int): top n values. Default is 5.

    Returns:
        np.ndarray: array of T/F mask for top n values
    
    """
    return __opr_top(array, axis, n=n)

# Rolling
__opr_rolling = __opr_src_wrapper_1donly('rolling')
def opr_rolling(array: np.ndarray, window: int):
    """Rolling Window

    - only 1D array is supported

    Args:
        array (np.ndarray): 1D or 2D array
        window (int): window size
    
    """
    return __opr_rolling(array, window=window)

# Rolling Apply
def opr_rolling_apply(array: np.ndarray, window: int, opr, kargs = {'axis':1}):
    """Apply function along the axis

    - only 2D array is supported
    - operation kargs 'axis':0 may occur error, use 'axis':1
    
    Args:
        array (np.ndarray): 2D array.
        window (int): window size.
        opr (Callable): operation that supports 2D array.
        kargs (dict): keyword arguments for the operation. Default is {'axis':1}.
        
    """
    if window < 2:
        raise ValueError('Window size should be larger than 2')
    if array.ndim != 2:
        ValueError(f'Only 2D array is supported, but got {array.ndim}D')
   
    res = np.empty(array.size, dtype=np.float32)
    res[window-1:] = opr(opr_rolling(array.flatten(), window), **kargs)
    res = res.reshape((array.shape[1], array.shape[0])).T[window-1:,:]
    return res