import numba as nb
import numpy as np

from alphacrafts.frt.loopbt.opr_src import _1d_flt64
from alphacrafts.frt.loopbt.opr_src import _2d_flt64
from alphacrafts.frt.loopbt.opr_src._2d_flt64 import nanmean, nanstd, nanmin, nanmax

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
        'res':nb.float64[::1], 'cap':nb.float64, 'floor':nb.float64
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
    res = mat[mat.shape[0]-1,:]
    cap = nanmean(mat, 0) + (hurdle * nanstd(mat, 0))
    floor = nanmean(mat, 0) - (hurdle * nanstd(mat, 0))
    
    res[res > cap] = cap
    res[res < floor] = floor
    return res
