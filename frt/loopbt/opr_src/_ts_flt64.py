import numba as nb
import numpy as np

from alphacrafts.frt.loopbt.opr_src import _1d_flt64
from alphacrafts.frt.loopbt.opr_src import _2d_flt64
from alphacrafts.frt.loopbt.opr_src._2d_flt64 import (
    nanmean, nanstd, nanmin, nanmax, nanpercentile,
    nansum
)

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], ),
    locals = {},
    boundscheck = False,
    cache = True,
    nopython = True
)
def zscore(mat) -> nb.float64[::1]:
    """Z-score Normalization

    Args:
        mat (nb.float64[:,::1])

    """
    return (mat[mat.shape[0]-1,:] - nanmean(mat,0)) / nanstd(mat,0)

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], ),
    locals = {},
    boundscheck = False,
    cache = True,
    nopython = True
)
def minmax(mat) -> nb.float64[::1]:
    """Min-Max Normalization

    Args:
        mat (np.float64[:,::1])

    """
    return (mat[mat.shape[0]-1,:] - nanmin(mat,0)) / (nanmax(mat,0) - nanmin(mat,0))

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.types.unicode_type),
    locals = {
        'res':nb.float64[:,::1],
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def rank(mat, method) -> nb.float64[::1]:
    """Rank Normalization

    Args:
        mat (np.float64[:,::1])
        method (str): 'average', 'min', 'max', 'dense', 'ordinal'.

    """
    res = _2d_flt64.rank(mat, 0, method)
    return res[res.shape[0]-1,:]

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.types.unicode_type),
    locals = {
        'res':nb.float64[:,::1],
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def rank_pct(mat, method) -> nb.float64[::1]:
    """Percentile Rank Normalization

    Args:
        mat (np.float64[:,::1])
        method (str): 'average', 'min', 'max', 'dense', 'ordinal'.

    """
    res = _2d_flt64.rank_pct(mat, 0, method)
    return res[res.shape[0]-1,:]

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.float64),
    locals = {
        'res':nb.float64[::1], 'cap':nb.float64[::1], 'floor':nb.float64[::1]
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def winsor_by_zscore(mat, hurdle) -> nb.float64[::1]:
    """Winsorization by Z-score

    Args:
        mat (nb.float64[:,::1])
        zscore_hurdle (nb.float64): The z-score value to winsorize the data. e.g.) 1.96, 2.58, ...

    """
    res = mat[mat.shape[0]-1,:].copy()
    cap = nanmean(mat, 0) + (hurdle * nanstd(mat, 0))
    floor = nanmean(mat, 0) - (hurdle * nanstd(mat, 0))

    res[res > cap] = cap
    res[res < floor] = floor
    return res

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.float64),
    locals = {
        'res':nb.float64[::1], 'cap':nb.float64[::1], 'floor':nb.float64[::1]
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def winsor_by_percentile(mat, hurdle) -> nb.float64[::1]:
    """Winsorization by percentile

    Args:
        mat (nb.float64[:,::1])
        percentile_hurdle (nb.float64): top/bottom n percentile to winsorize the data. e.g.) 0.1, 0.001, ...

    """
    res = mat[mat.shape[0]-1,:].copy()
    cap = nanpercentile(mat, 0, 100 - hurdle)
    floor = nanpercentile(mat, 0, hurdle)
    res[res > cap] = cap
    res[res < floor] = floor
    return res

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], ),
    locals = {
        'exp_arr':nb.float64[:,::1]
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def softmax(mat) -> nb.float64[::1]:
    """Softmax Normalization

    Args:
        mat (nb.float64[::1])

    """
    exp_mat = np.exp(mat)
    return exp_mat[mat.shape[0]-1,:] / nansum(exp_mat, 0)

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.float64),
    boundscheck = False,
    cache = True,
    nopython = True
)
def decay(mat, rate):
    """Decay operation

    Args:
        mat (nb.float64[:,::1])
        rate (nb.float64): decay rate
    
    """
    mat_maxidx = mat.shape[0]-1
    result = mat[mat_maxidx,:].copy()
    for i in range(1, mat_maxidx+1):
        result += mat[mat_maxidx-i,:] * np.power(rate, i)
    return result

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.float64[::1]),
    boundscheck = False,
    cache = True,
    nopython = True
)
def decay_with(mat, rates):
    """Decay operation

    Args:
        mat (nb.float64[:,::1])
        rates (nb.float64): decay rate array
    
    """
    mat_maxidx = mat.shape[0]-1
    result = np.zeros(mat.shape[1], dtype=np.float64)
    for i in range(min(mat_maxidx, rates.size-1)+1):
        result += mat[mat_maxidx-i,:] * rates[i]
    return result
