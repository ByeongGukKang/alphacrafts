import numba as nb
import numpy as np

from alphacrafts.frt.loopbt.opr_src import _1d_flt64


### Complied Numpy Functions ###

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.int64),
    locals = {
        'res':nb.float64[::1],'i':nb.int64
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def mean(mat, axis) -> nb.float64[::1]:
    """Compiled mean function for 2D array

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column

    """
    if axis:
        res = np.empty(mat.shape[0], dtype=np.float64)
        for i in range(mat.shape[0]):
            res[i] = np.mean(mat[i,:])
    else:
        res = np.empty(mat.shape[1], dtype=np.float64)
        for i in range(mat.shape[1]):
            res[i] = np.mean(mat[:,i])
    return res

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.int64),
    locals = {
        'res':nb.float64[::1],'i':nb.int64
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def std(mat, axis) -> nb.float64[::1]:
    """Compiled std function for 2D array

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column

    """
    if axis:
        res = np.empty(mat.shape[0], dtype=np.float64)
        for i in range(mat.shape[0]):
            res[i] = np.std(mat[i,:])
    else:
        res = np.empty(mat.shape[1], dtype=np.float64)
        for i in range(mat.shape[1]):
            res[i] = np.std(mat[:,i])
    return res

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.int64),
    locals = {
        'res':nb.float64[::1],'i':nb.int64
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def min(mat, axis) -> nb.float64[::1]:
    """Compiled min function for 2D array

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column

    """
    if axis:
        res = np.empty(mat.shape[0], dtype=np.float64)
        for i in range(mat.shape[0]):
            res[i] = np.min(mat[i,:])
    else:
        res = np.empty(mat.shape[1], dtype=np.float64)
        for i in range(mat.shape[1]):
            res[i] = np.min(mat[:,i])
    return res

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.int64),
    locals = {
        'res':nb.float64[::1],'i':nb.int64
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def max(mat, axis) -> nb.float64[::1]:
    """Compiled max function for 2D array

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column

    """
    if axis:
        res = np.empty(mat.shape[0], dtype=np.float64)
        for i in range(mat.shape[0]):
            res[i] = np.max(mat[i,:])
    else:
        res = np.empty(mat.shape[1], dtype=np.float64)
        for i in range(mat.shape[1]):
            res[i] = np.max(mat[:,i])
    return res

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.int64, nb.float64),
    locals = {
        'res':nb.float64[::1],'i':nb.int64
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def percentile(mat, axis, hurdle) -> nb.float64[::1]:
    """Compiled percentile function for 2D array

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column
        hurdle (nb.float64): percentile value

    """
    if axis:
        res = np.empty(mat.shape[0], dtype=np.float64)
        for i in range(mat.shape[0]):
            res[i] = np.percentile(mat[i,:], hurdle)
    else:
        res = np.empty(mat.shape[1], dtype=np.float64)
        for i in range(mat.shape[1]):
            res[i] = np.percentile(mat[:,i], hurdle)
    return res

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.int64),
    locals = {
        'res':nb.float64[::1],'i':nb.int64
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def sum(mat, axis) -> nb.float64[::1]:
    """Compiled sum function for 2D array

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column

    """
    if axis:
        res = np.empty(mat.shape[0], dtype=np.float64)
        for i in range(mat.shape[0]):
            res[i] = np.sum(mat[i,:])
    else:
        res = np.empty(mat.shape[1], dtype=np.float64)
        for i in range(mat.shape[1]):
            res[i] = np.sum(mat[:,i])
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
def nanmean(mat, axis) -> nb.float64[::1]:
    """Compiled nanmean function for 2D array

    Args:
        mat (nb.float64[:,::1])
        axis (nb.bool_): 0 for row, 1 for column

    """
    if axis:
        res = np.empty(mat.shape[0], dtype=np.float64)
        for i in range(mat.shape[0]):
            res[i] = np.nanmean(mat[i,:])
    else:
        res = np.empty(mat.shape[1], dtype=np.float64)
        for i in range(mat.shape[1]):
            res[i] = np.nanmean(mat[:,i])
    return res

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.int64),
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
        axis (nb.int64): 0 for row, 1 for column

    """
    if axis:
        res = np.empty(mat.shape[0], dtype=np.float64)
        for i in range(mat.shape[0]):
            res[i] = np.nanstd(mat[i,:])
    else:
        res = np.empty(mat.shape[1], dtype=np.float64)
        for i in range(mat.shape[1]):
            res[i] = np.nanstd(mat[:,i])
    return res

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.int64),
    locals = {
        'res':nb.float64[::1],'i':nb.int64
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def nanmin(mat, axis) -> nb.float64[::1]:
    """Compiled nanmin function for 2D array

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column

    """
    if axis:
        res = np.empty(mat.shape[0], dtype=np.float64)
        for i in range(mat.shape[0]):
            res[i] = np.nanmin(mat[i,:])
    else:
        res = np.empty(mat.shape[1], dtype=np.float64)
        for i in range(mat.shape[1]):
            res[i] = np.nanmin(mat[:,i])
    return res

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.int64),
    locals = {
        'res':nb.float64[::1],'i':nb.int64
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def nanmax(mat, axis) -> nb.float64[::1]:
    """Compiled nanmax function for 2D array

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column

    """
    if axis:
        res = np.empty(mat.shape[0], dtype=np.float64)
        for i in range(mat.shape[0]):
            res[i] = np.nanmax(mat[i,:])
    else:
        res = np.empty(mat.shape[1], dtype=np.float64)
        for i in range(mat.shape[1]):
            res[i] = np.nanmax(mat[:,i])
    return res

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.int64, nb.float64),
    locals = {
        'res':nb.float64[::1],'i':nb.int64
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def nanpercentile(mat, axis, hurdle) -> nb.float64[::1]:
    """Compiled nanpercentile function for 2D array

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column
        hurdle (nb.float64): percentile value

    """
    if axis:
        res = np.empty(mat.shape[0], dtype=np.float64)
        for i in range(mat.shape[0]):
            res[i] = np.nanpercentile(mat[i,:], hurdle)
    else:
        res = np.empty(mat.shape[1], dtype=np.float64)
        for i in range(mat.shape[1]):
            res[i] = np.nanpercentile(mat[:,i], hurdle)
    return res

@nb.jit(
    nb.float64[::1](nb.float64[:,::1], nb.int64),
    locals = {
        'res':nb.float64[::1],'i':nb.int64
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def nansum(mat, axis) -> nb.float64[::1]:
    """Compiled nansum function for 2D array

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column

    """
    if axis:
        res = np.empty(mat.shape[0], dtype=np.float64)
        for i in range(mat.shape[0]):
            res[i] = np.nansum(mat[i,:])
    else:
        res = np.empty(mat.shape[1], dtype=np.float64)
        for i in range(mat.shape[1]):
            res[i] = np.nansum(mat[:,i])
    return res

# @nb.jit(
#     nb.bool_[:,::1](nb.float64[:,::1], nb.bool_),
#     boundscheck = False,
#     cache = True,
#     nopython = True
# )
# def isnan(mat, axis) -> nb.bool_[:,::1]: # here axis is dummy (not used)
#     """Compiled isnan function for 2D array

#     Args:
#         mat (nb.float64[:,::1])

#     """
#     return np.isnan(mat)

# @nb.jit(
#     nb.float64[:,::1](nb.float64[:,::1],),
#     boundscheck = False,
#     cache = True,
#     nopython = True
# )
# def abs(mat) -> nb.float64[:,::1]: # here axis is dummy (not used)
#     """Compiled abs function for 2D array

#     Args:
#         mat (nb.float64[:,::1])

#     """
#     return np.abs(mat)

###### Complied Operations ######
@nb.jit(
    nb.float64[:,::1](nb.float64[:,::1], nb.int64),
    locals = {'res':nb.float64[:,::1], 'i':nb.int64},
    boundscheck = False,
    cache = True,
    nopython = True
)
def zscore(mat, axis) -> nb.float64[:,::1]:
    """Z-score Normalization

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column

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
    nb.float64[:,::1](nb.float64[:,::1], nb.int64),
    locals = {'res':nb.float64[:,::1], 'i':nb.int64},
    boundscheck = False,
    cache = True,
    nopython = True
)
def minmax(mat, axis) -> nb.float64[:,::1]:
    """Min-Max Normalization

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column

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
    nb.float64[:,::1](nb.float64[:,::1], nb.int64, nb.types.unicode_type),
    locals = {'res':nb.float64[:,::1], 'i':nb.int64},
    boundscheck = False,
    cache = True,
    nopython = True
)
def rank(mat, axis, method) -> nb.float64[:,::1]:
    """Rank Normalization

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column
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
    nb.float64[:,::1](nb.float64[:,::1], nb.int64, nb.types.unicode_type),
    locals = {'res':nb.float64[:,::1], 'i':nb.int64},
    boundscheck = False,
    cache = True,
    nopython = True
)
def rank_pct(mat, axis, method) -> nb.float64[:,::1]:
    """Percentile Rank Normalization

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column
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

@nb.jit(
    nb.float64[:,::1](nb.float64[:,::1], nb.int64, nb.float64),
    locals = {'res':nb.float64[:,::1], 'i':nb.int64},
    boundscheck = False,
    cache = True,
    nopython = True
)
def winsor_by_zscore(mat, axis, hurdle) -> nb.float64[:,::1]:
    """Winsorization by Z-score

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column
        hurdle (float): Z-score value

    """
    res = np.empty(mat.shape, dtype=np.float64)
    if axis:
        for i in range(mat.shape[0]):
            res[i,:] = _1d_flt64.winsor_by_zscore(mat[i,:], hurdle)
    else:
        for i in range(mat.shape[1]):
            res[:,i] = _1d_flt64.winsor_by_zscore(mat[:,i].flatten(), hurdle)
    return res

@nb.jit(
    nb.float64[:,::1](nb.float64[:,::1], nb.int64, nb.float64),
    locals = {'res':nb.float64[:,::1], 'i':nb.int64},
    boundscheck = False,
    cache = True,
    nopython = True
)
def winsor_by_percentile(mat, axis, hurdle) -> nb.float64[:,::1]:
    """Winsorization by Percentile

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column
        hurdle (float): percentile value

    """
    res = np.empty(mat.shape, dtype=np.float64)
    if axis:
        for i in range(mat.shape[0]):
            res[i,:] = _1d_flt64.winsor_by_percentile(mat[i,:], hurdle)
    else:
        for i in range(mat.shape[1]):
            res[:,i] = _1d_flt64.winsor_by_percentile(mat[:,i].flatten(), hurdle)
    return res

@nb.jit(
    nb.float64[:,::1](nb.float64[:,::1], nb.int64),
    locals = {'res':nb.float64[:,::1], 'i':nb.int64},
    boundscheck = False,
    cache = True,
    nopython = True
)
def softmax(mat, axis) -> nb.float64[:,::1]:
    """Softmax Normalization

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column

    """
    res = np.empty(mat.shape, dtype=np.float64)
    if axis:
        for i in range(mat.shape[0]):
            res[i,:] = _1d_flt64.softmax(mat[i,:])
    else:
        for i in range(mat.shape[1]):
            res[:,i] = _1d_flt64.softmax(mat[:,i].flatten())
    return res

@nb.jit(
    nb.bool_[:,::1](nb.float64[:,::1], nb.int64, nb.int64),   
    locals = {'res':nb.bool_[:,::1], 'i':nb.int64}, 
    boundscheck = False,
    cache = True,
    nopython = True
)
def top(mat, axis, n) -> nb.bool_[:,::1]:
    """Return the mask of top n values of the array

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column

    """
    res = np.empty(mat.shape, dtype=np.bool_)
    if axis:
        for i in range(mat.shape[0]):
            res[i,:] = _1d_flt64.top(mat[i,:], n)
    else:
        for i in range(mat.shape[1]):
            res[:,i] = _1d_flt64.top(mat[:,i].flatten(), n)
    return res

@nb.jit(
    nb.float64[:,::1](nb.float64[:,::1], nb.int64),   
    boundscheck = False,
    cache = True,
    nopython = True
)
def weight(mat, axis) -> nb.float64[:,::1]:
    """Return the weight array of the array

    Args:
        mat (nb.float64[:,::1])
        axis (nb.int64): 0 for row, 1 for column

    """
    res = np.empty(mat.shape, dtype=np.float64)
    if axis:
        for i in range(mat.shape[0]):
            res[i,:] = _1d_flt64.weight(mat[i,:])
    else:
        for i in range(mat.shape[1]):
            res[:,i] = _1d_flt64.weight(mat[:,i].flatten())
    return res