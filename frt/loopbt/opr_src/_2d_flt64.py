import numba as nb
import numpy as np

from alphacrafts.frt.loopbt.opr_src import _1d_flt64

### Complied Numpy Functions ###

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.bool_),
    locals = {
        'res':nb.float64[::1],'i':nb.int64
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def nanmean(mat, axis) -> nb.float64[::1]:
    """Compiled nanmean function for 2D array

    Args:
        mat (nb.float64[:,::1])
        axis (nb.bool_): 0 for row, 1 for column

    """
    if axis:
        res = np.empty(mat.shape[0])
        for i in range(mat.shape[0]):
            res[i] = np.nanmean(mat[i,:])
    else:
        res = np.empty(mat.shape[1])
        for i in range(mat.shape[1]):
            res[i] = np.nanmean(mat[:,i])
    return res

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.bool_),
    locals = {
        'res':nb.float64[::1],'i':nb.int64
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def nanstd(mat, axis) -> nb.float64[::1]:
    """Compiled nanstd function for 2D array

    Args:
        mat (nb.float64[:,::1])
        axis (nb.bool_): 0 for row, 1 for column

    """
    if axis:
        res = np.empty(mat.shape[0])
        for i in range(mat.shape[0]):
            res[i] = np.nanstd(mat[i,:])
    else:
        res = np.empty(mat.shape[1])
        for i in range(mat.shape[1]):
            res[i] = np.nanstd(mat[:,i])
    return res

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.bool_),
    locals = {'res':nb.float64[::1],'i':nb.int64},
    boundscheck = False,
    cache = True,
    nopython = True
)
def nanmin(mat, axis) -> nb.float64[::1]:
    """Compiled nanmin function for 2D array

    Args:
        mat (nb.float64[:,::1])
        axis (nb.bool_): 0 for row, 1 for column

    """
    if axis:
        res = np.empty(mat.shape[0])
        for i in range(mat.shape[0]):
            res[i] = np.nanmin(mat[i,:])
    else:
        res = np.empty(mat.shape[1])
        for i in range(mat.shape[1]):
            res[i] = np.nanmin(mat[:,i])
    return res

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.bool_),
    locals = {'res':nb.float64[::1],'i':nb.int64},
    boundscheck = False,
    cache = True,
    nopython = True
)
def nanmax(mat, axis) -> nb.float64[::1]:
    """Compiled nanmax function for 2D array

    Args:
        mat (nb.float64[:,::1])
        axis (nb.bool_): 0 for row, 1 for column

    """
    if axis:
        res = np.empty(mat.shape[0])
        for i in range(mat.shape[0]):
            res[i] = np.nanmax(mat[i,:])
    else:
        res = np.empty(mat.shape[1])
        for i in range(mat.shape[1]):
            res[i] = np.nanmax(mat[:,i])
    return res

###### Complied Operations ######
@nb.jit(
    nb.float64[:,::1](nb.float64[:,::1], nb.bool_),
    locals = {'res':nb.float64[:,::1], 'i':nb.int64},
    boundscheck = False,
    cache = True,
    nopython = True
)
def zscore(mat, axis) -> nb.float64[:,::1]:
    """Z-score Normalization

    Args:
        mat (nb.float64[:,::1])

    """
    res = np.empty(mat.shape, dtype=np.float64)
    if axis:
        for i in range(mat.shape[0]):
            res[i,:] = _1d_flt64.zscore(mat[i,:])
    else:
        for i in range(mat.shape[1]):
            res[:,i] = _1d_flt64.zscore(mat[:,i].flatten())
    return res

@nb.jit(
    nb.float64[:,::1](nb.float64[:,::1], nb.bool_),
    locals = {'res':nb.float64[:,::1], 'i':nb.int64},
    boundscheck = False,
    cache = True,
    nopython = True
)
def minmax(mat, axis) -> nb.float64[:,::1]:
    """Min-Max Normalization

    Args:
        mat (np.float64[:,::1])

    """
    res = np.empty(mat.shape, dtype=np.float64)
    if axis:
        for i in range(mat.shape[0]):
            res[i,:] = _1d_flt64.minmax(mat[i,:])
    else:
        for i in range(mat.shape[1]):
            res[:,i] = _1d_flt64.minmax(mat[:,i].flatten())
    return res

@nb.jit(
    nb.float64[:,::1](nb.float64[:,::1], nb.bool_, nb.types.unicode_type),
    locals = {'res':nb.float64[:,::1], 'i':nb.int64},
    boundscheck = False,
    cache = True,
    nopython = True
)
def rank(mat, axis, method) -> nb.float64[:,::1]:
    """Rank Normalization

    Args:
        mat (np.float64[:,::1])
        axis (bool): 0 for row, 1 for column
        method (str): 'average', 'min', 'max', 'dense', 'ordinal'.

    """
    res = np.empty(mat.shape, dtype=np.float64)
    if axis:
        for i in range(mat.shape[0]):
            res[i,:] = _1d_flt64.rank(mat[i,:], method)
    else:
        for i in range(mat.shape[1]):
            res[:,i] = _1d_flt64.rank(mat[:,i].flatten(), method)
    return res

@nb.jit(
    nb.float64[:,::1](nb.float64[:,::1], nb.bool_, nb.types.unicode_type),
    locals = {'res':nb.float64[:,::1], 'i':nb.int64},
    boundscheck = False,
    cache = True,
    nopython = True
)
def rank_pct(mat, axis, method) -> nb.float64[:,::1]:
    """Percentile Rank Normalization

    Args:
        mat (np.float64[:,::1])
        axis (bool): 0 for row, 1 for column
        method (str): 'average', 'min', 'max', 'dense', 'ordinal'.

    """
    res = np.empty(mat.shape, dtype=np.float64)
    if axis:
        for i in range(mat.shape[0]):
            res[i,:] = _1d_flt64.rank_pct(mat[i,:], method)
    else:
        for i in range(mat.shape[1]):
            res[:,i] = _1d_flt64.rank_pct(mat[:,i].flatten(), method)
    return res