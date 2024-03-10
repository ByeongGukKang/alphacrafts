
import numpy as np

from alphacrafts.frt.loopbt.opr_src import _1d_flt64, _2d_flt64, _ts_flt64


# Wrapper for numba compiled functions
# Avoid using decorator for better performance (less overheads)
# Decorator is called every time the function is called
def __opr_src_wrapper(func_name):

    def wrapped_func(array, axis, **kargs):
        # TODO : Add support for float32
        if array.dtype != np.float64:
            array = array.astype(np.float64)

        if array.ndim == 1:
            return getattr(_1d_flt64, func_name)(array.ravel('C'), **kargs)
        elif array.ndim == 2:
            return getattr(_2d_flt64, func_name)(array, axis, **kargs)
        else:
            ValueError(f'Only 1D and 2D arrays are supported, but got {array.ndim}D')
    
    return wrapped_func

##### Compiled Numpy Functions #####
__opr_nanmean = __opr_src_wrapper('nanmean')
def opr_nanmean(array, axis):
    """nanmean function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
        - in 1D array, axis is ignored (only for 2D array)
    
    """
    return __opr_nanmean(array, axis)

__opr_nanstd = __opr_src_wrapper('nanstd')
def opr_nanstd(array, axis):
    """nanstd function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
        - in 1D array, axis is ignored (only for 2D array)
    
    """
    return __opr_nanstd(array, axis)

__opr_nanmin = __opr_src_wrapper('nanmin')
def opr_nanmin(array, axis):
    """Nanmin function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
        - in 1D array, axis is ignored (only for 2D array)
    
    """
    return __opr_nanmin(array, axis)

__opr_nanmax = __opr_src_wrapper('nanmax')
def opr_nanmax(array, axis):
    """nanmax function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
        - in 1D array, axis is ignored (only for 2D array)
    
    """
    return __opr_nanmax(array, axis)

__opr_nanpercentile = __opr_src_wrapper('nanpercentile')
def opr_nanpercentile(array, axis, hurdle):
    """nanpercentile function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column
        - in 1D array, axis is ignored (only for 2D array)
        hurdle (float): percentile
    
    """
    return __opr_nanpercentile(array, axis, hurdle=hurdle)

__opr_nansum = __opr_src_wrapper('nansum')
def opr_nansum(array, axis):
    """nansum function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
        - in 1D array, axis is ignored (only for 2D array)
    
    """
    return __opr_nansum(array, axis)

__opr_isnan = __opr_src_wrapper('isnan')
def opr_isnan(array):
    """isnan function for 1D and 2D arrays

    Args:
        array (np.ndarray): 1D or 2D array
    
    """
    return __opr_isnan(array, 0) # here axis=0 is dummy 

##### Operations #####

# Z-score
__opr_zscore = __opr_src_wrapper('zscore')
def opr_zscore(array, axis):
    """Z-score Normalization

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
        - in 1D array, axis is ignored (only for 2D array)
    
    """
    return __opr_zscore(array, axis)

def oprts_zscore(array):
    """[TS] Z-score Normalization 

    Args:
        array (np.ndarray): 2D array
    
    """
    if (array.ndim==2) & (array.dtype == np.float64):
         return _ts_flt64.zscore(array)
    else:
        ValueError(f'Only 2D array, float64 are supported, but got ndim:{array.ndim}, dtype:{array.dtype}')

# MinMax
__opr_minmax = __opr_src_wrapper('minmax')
def opr_minmax(array, axis):
    """Min-Max Normalization

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column. Default is 0
        - only for 2D array
    
    """
    return __opr_minmax(array, axis)        

def oprts_minmax(array):
    """[TS] Min-Max Normalization

    Args:
        array (np.ndarray): 2D array
    
    """
    if (array.ndim==2) & (array.dtype == np.float64):
         return _ts_flt64.minmax(array)
    else:
        ValueError(f'Only 2D array, float64 are supported, but got ndim:{array.ndim}, dtype:{array.dtype}')

# Rank
__opr_rank = __opr_src_wrapper('rank')
def opr_rank(array, axis, method='average'):
    """Rank Normalization

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column
        - in 1D array, axis is ignored (only for 2D array)
        method (str): 'average', 'min', 'max', 'dense', 'ordinal'
        
    """
    return __opr_rank(array, axis, method=method)        

def oprts_rank(array, method='average'):
    """[TS] Rank Normalization

    Args:
        array (np.ndarray): 2D array
        method (str): 'average', 'min', 'max', 'dense', 'ordinal'
        
    """
    if (array.ndim==2) & (array.dtype == np.float64):
         return _ts_flt64.rank(array)
    else:
        ValueError(f'Only 2D array, float64 are supported, but got ndim:{array.ndim}, dtype:{array.dtype}')

# Percentile Rank
__opr_rank_pct = __opr_src_wrapper('rank_pct')
def opr_rank_pct(array, axis, method='average'):
    """Percentile Rank Normalization

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column
        - in 1D array, axis is ignored (only for 2D array)
        method (str): 'average', 'min', 'max', 'dense', 'ordinal'
    
    """
    return __opr_rank_pct(array, axis, method=method)

def oprts_rank_pct(array, method='average'):
    """[TS] Percentile Rank Normalization

    Args:
        array (np.ndarray): 2D array
        method (str): 'average', 'min', 'max', 'dense', 'ordinal'
    
    """
    if (array.ndim==2) & (array.dtype == np.float64):
         return _ts_flt64.rank(array)
    else:
        ValueError(f'Only 2D array, float64 are supported, but got ndim:{array.ndim}, dtype:{array.dtype}')

# Winsorization by Z-score
__opr_winsor_by_zscore = __opr_src_wrapper('winsor_by_zscore')
def opr_winsor_by_zscore(array, axis, hurdle=2.58):
    """Winsorization by Z-score

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column
        - in 1D array, axis is ignored (only for 2D array)
        hurdle (float): Z-score threshold. Default is 2.58
    
    """
    return __opr_winsor_by_zscore(array, axis, hurdle=hurdle)        

def oprts_winsor_by_zscore(array, hurdle=2.58):
    """[TS] Winsorization by Z-score

    Args:
        array (np.ndarray): 2D array
        hurdle (float): Z-score threshold. Default is 2.58
    
    """
    if (array.ndim==2) & (array.dtype == np.float64):
         return _ts_flt64.winsor_by_zscore(array, hurdle)
    else:
        ValueError(f'Only 2D array, float64 are supported, but got ndim:{array.ndim}, dtype:{array.dtype}')

# Winsorization by Percentile
__opr_winsor_by_percentile = __opr_src_wrapper('winsor_by_percentile')
def opr_winsor_by_percentile(array, axis, hurdle=1.0):
    """Winsorization by Percentile

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column
        - in 1D array, axis is ignored (only for 2D array)
        hurdle (float): top/bottom n percentile to winsorize the data
        - hurdle=1.0 means 1% from top and 1% from bottom
    
    """
    return __opr_winsor_by_percentile(array, axis, hurdle=hurdle)        

def oprts_winsor_by_percentile(array, hurdle=1.0):
    """[TS] Winsorization by Percentile

    Args:
        array (np.ndarray): 2D array
        hurdle (float): top/bottom n percentile to winsorize the data
        - hurdle=1.0 means 1% from top and 1% from bottom
    
    """
    if (array.ndim==2) & (array.dtype == np.float64):
         return _ts_flt64.winsor_by_percentile(array, hurdle)
    else:
        ValueError(f'Only 2D array, float64 are supported, but got ndim:{array.ndim}, dtype:{array.dtype}')

# Softmax
__opr_softmax = __opr_src_wrapper('softmax')
def opr_softmax(array, axis):
    """Softmax Normalization

    Args:
        array (np.ndarray): 1D or 2D array
        axis (int): 0 for row, 1 for column
        - in 1D array, axis is ignored (only for 2D array)
    
    """
    return __opr_softmax(array, axis)

def oprts_softmax(array):
    """[TS] Softmax Normalization

    Args:
        array (np.ndarray): 2D array
    
    """
    if (array.ndim==2) & (array.dtype == np.float64):
         return _ts_flt64.softmax(array)
    else:
        ValueError(f'Only 2D array, float64 are supported, but got ndim:{array.ndim}, dtype:{array.dtype}')