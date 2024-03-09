
import numpy as np

from alphacrafts.frt.loopbt.opr_src import _1d_flt64, _2d_flt64, _ts_flt64

##### Compiled Numpy Functions #####
def opr2d_nanmean(mat, axis):
    if mat.dtype == np.float64:
        return _2d_flt64.nanmean(mat, axis)
    elif mat.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {mat.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {mat.dtype}')

def opr2d_nanstd(mat, axis):
    if mat.dtype == np.float64:
        return _2d_flt64.nanstd(mat, axis)
    elif mat.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {mat.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {mat.dtype}')

def opr2d_nanmin(mat, axis):
    if mat.dtype == np.float64:
        return _2d_flt64.nanmin(mat, axis)
    elif mat.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {mat.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {mat.dtype}')

def opr2d_nanmax(mat, axis):
    if mat.dtype == np.float64:
        return _2d_flt64.nanmax(mat, axis)
    elif mat.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {mat.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {mat.dtype}')

##### Operations #####

# Z-score
def opr1d_zscore(arr):
    if arr.dtype == np.float64:
         return _1d_flt64.zscore(arr)
    elif arr.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {arr.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {arr.dtype}')

def oprts_zscore(arr):
    if arr.dtype == np.float64:
         return _ts_flt64.zscore(arr)
    elif arr.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {arr.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {arr.dtype}')

def opr2d_zscore(mat, axis):
    if mat.dtype == np.float64:
         return _2d_flt64.zscore(mat, axis)
    elif mat.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {mat.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {mat.dtype}')

# MinMax
def opr1d_minmax(arr):
    if arr.dtype == np.float64:
         return _1d_flt64.minmax(arr)
    elif arr.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {arr.dtype}')
    else:
        Exception(f'Only float64 and float32 are supported, but got {arr.dtype}')

def oprts_minmax(arr):
    if arr.dtype == np.float64:
         return _ts_flt64.minmax(arr)
    elif arr.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {arr.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {arr.dtype}')

def opr2d_minmax(mat, axis):
    if mat.dtype == np.float64:
         return _2d_flt64.minmax(mat, axis)
    elif mat.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {mat.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {mat.dtype}')

# Rank
def opr1d_rank(arr, method='average'):
    if arr.dtype == np.float64:
         return _1d_flt64.rank(arr, method)
    elif arr.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {arr.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {arr.dtype}')

def oprts_rank(arr, method='average'):
    if arr.dtype == np.float64:
         return _ts_flt64.rank(arr, method)
    elif arr.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {arr.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {arr.dtype}')

def opr2d_rank(mat, axis, method='average'):
    if mat.dtype == np.float64:
         return _2d_flt64.rank(mat, axis, method)
    elif mat.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {mat.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {mat.dtype}')

# Percentile Rank
def opr1d_rank_pct(arr, method='average'):
    if arr.dtype == np.float64:
         return _1d_flt64.rank_pct(arr, method)
    elif arr.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {arr.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {arr.dtype}')

def oprts_rank_pct(arr, method='average'):
    if arr.dtype == np.float64:
         return _ts_flt64.rank_pct(arr, method)
    elif arr.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {arr.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {arr.dtype}')

def opr2d_rank_pct(mat, axis, method='average'):
    if mat.dtype == np.float64:
         return _2d_flt64.rank_pct(mat, axis, method)
    elif mat.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {mat.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {mat.dtype}')

# Winsorization by Z-score
def opr1d_winsor_by_zscore(arr, hurdle=2.58):
    if arr.dtype == np.float64:
         return _1d_flt64.winsor_by_zscore(arr, hurdle)
    elif arr.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {arr.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {arr.dtype}')

def oprts_winsor_by_zscore(arr, hurdle=2.58):
    if arr.dtype == np.float64:
         return _ts_flt64.winsor_by_zscore(arr, hurdle)
    elif arr.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {arr.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {arr.dtype}')

# Winsorization by Percentile
def opr1d_winsor_by_percentile(arr, hurdle=1.0):
    if arr.dtype == np.float64:
         return _1d_flt64.winsor_by_percentile(arr, hurdle)
    elif arr.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {arr.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {arr.dtype}')

def oprts_winsor_by_percentile(arr, hurdle=1.0):
    if arr.dtype == np.float64:
         return _ts_flt64.winsor_by_percentile(arr, hurdle)
    elif arr.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {arr.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {arr.dtype}')

# Softmax
def opr1d_softmax(arr):
    if arr.dtype == np.float64:
         return _1d_flt64.softmax(arr)
    elif arr.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {arr.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {arr.dtype}')

def oprts_softmax(arr):
    if arr.dtype == np.float64:
         return _ts_flt64.softmax(arr)
    elif arr.dtype == np.float32:
        raise NotImplementedError(f'Not implemented for {arr.dtype}')
    else:
        ValueError(f'Only float64 and float32 are supported, but got {arr.dtype}')