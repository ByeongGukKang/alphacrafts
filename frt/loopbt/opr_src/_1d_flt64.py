import numba as nb
import numpy as np


### Complied Numpy Functions ###
@nb.jit(
    nb.float64(nb.float64[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def nanmean(arr) -> nb.float64:
    """Compiled nanmean function for 1D array

    Args:
        arr (nb.float64[::1])

    """
    return np.nanmean(arr)

@nb.jit(
    nb.float64(nb.float64[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def nanstd(arr) -> nb.float64:
    """Compiled nanstd function for 1D array

    Args:
        arr (nb.float64[:,::1])

    """
    return np.nanstd(arr)

@nb.jit(
    nb.float64(nb.float64[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def nanmin(arr) -> nb.float64:
    """Compiled nanmin function for 1D array

    Args:
        arr (nb.float64[::1])

    """
    return np.nanmin(arr)

@nb.jit(
    nb.float64(nb.float64[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def nanmax(arr) -> nb.float64:
    """Compiled nanmax function for 1D array

    Args:
        arr (nb.float64[:,::1])

    """
    return np.nanmax(arr)

@nb.jit(
    nb.float64(nb.float64[::1], nb.float64),
    boundscheck = False,
    cache = True,
    nopython = True
)
def nanpercentile(arr, hurdle) -> nb.float64[::1]:
    """Compiled nanpercentile function for 1D array

    Args:
        arr (nb.float64[::1])
        hurdle (nb.float64): percentile value

    """
    return np.nanpercentile(arr, hurdle)

@nb.jit(
    nb.float64(nb.float64[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def nansum(arr) -> nb.float64:
    """Compiled nansum function for 1D array

    Args:
        arr (nb.float64[::1])

    """
    return np.nansum(arr)

@nb.jit(
    nb.bool_[::1](nb.float64[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def isnan(arr) -> nb.float64:
    """Compiled isnan function for 1D array

    Args:
        arr (nb.float64[::1])

    """
    return np.isnan(arr)



###### Complied Operations ######
@nb.jit(
    nb.float64[::1](nb.float64[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def zscore(arr):
    """Z-score Normalization

    Args:
        arr (nb.float64[::1])

    """
    return (arr - np.nanmean(arr)) / np.nanstd(arr)

@nb.jit(
    nb.float64[::1](nb.float64[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def minmax(arr: nb.float64[::1]) -> nb.float64[::1]:
    """Min-Max Normalization

    Args:
        arr (np.float64[::1])

    """
    return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))

@nb.jit(
    nb.float64[::1](nb.float64[::1], nb.types.unicode_type),
    locals = {
        'res':nb.float64[::1], 'nan_mask':nb.boolean[::1], 'non_nan_arr':nb.float64[::1],
        'sorter':nb.int64[::1], 'obs':nb.boolean[::1], 'dense':nb.int64[::1], 
        'count':nb.int64[::1]
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def rank(arr: nb.float64[::1], method: nb.types.unicode_type) -> nb.float64[::1]:
    """Rank Normalization

    Args:
        arr (nb.float64[::1])
        method (str): 'average', 'min', 'max', 'dense', 'ordinal'.
            'average': The average of the ranks that would have been assigned to all the tied values is assigned to each value.
            'min': The minimum of the ranks that would have been assigned to all the tied values is assigned to each value. (This is also referred to as 'competition' ranking.)
            'max': The maximum of the ranks that would have been assigned to all the tied values is assigned to each value.
            'dense': Like 'min', but the rank of the next highest element is assigned the rank immediately after those assigned to the tied elements.
            'ordinal': All values are given a distinct rank, corresponding to the order that the values occur in a.

    Source:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html

    """
    res = np.empty_like(arr, dtype=np.float64)
    nan_mask = np.isnan(arr)
    non_nan_arr = arr[~nan_mask]
    sorter = np.argsort(non_nan_arr, kind='mergesort')

    if method == 'ordinal':
        res[~nan_mask] = np.argsort(sorter, kind='mergesort')
    else:
        non_nan_arr = non_nan_arr[sorter]
        # obs = np.r_[True, non_nan_arr[1:] != non_nan_arr[:-1]]
        obs = np.concatenate((np.array([True]), non_nan_arr[1:] != non_nan_arr[:non_nan_arr.shape[0] -1]))
        dense = obs.cumsum()[sorter]

        if method == 'dense':
            res[~nan_mask] = dense
        else:
            # cumulative counts of each unique value
            # count = np.r_[np.nonzero(obs)[0], len(obs)]
            count = np.concatenate((np.nonzero(obs)[0], np.array([len(obs)])))

            if method == 'max':
                res[~nan_mask] = count[dense]
            elif method == 'min':
                res[~nan_mask] = count[dense - 1] + 1
            elif method == 'average':
                res[~nan_mask] = 0.5 * (count[dense] + count[dense - 1] + 1)
            else:
                raise Exception("ValueError: Unknown argument for parameter 'method'")

    res[nan_mask] = np.nan
    return res

@nb.jit(
    nb.float64[::1](nb.float64[::1], nb.types.unicode_type),
    locals = {
        'rankval':nb.float64[::1], 'pctrank':nb.float64[::1]
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def rank_pct(arr: nb.float64[::1], method: nb.types.unicode_type) -> nb.float64[::1]:
    """Percentile Rank Normalization

    Args:
        arr (nb.float64[::1])
        method (str): 'average', 'min', 'max', 'dense', 'ordinal'.
            'average': The average of the ranks that would have been assigned to all the tied values is assigned to each value.
            'min': The minimum of the ranks that would have been assigned to all the tied values is assigned to each value. (This is also referred to as 'competition' ranking.)
            'max': The maximum of the ranks that would have been assigned to all the tied values is assigned to each value.
            'dense': Like 'min', but the rank of the next highest element is assigned the rank immediately after those assigned to the tied elements.
            'ordinal': All values are given a distinct rank, corresponding to the order that the values occur in a.
    
    Description:
        Returns percentile rank of data. It is simply rank(arr, method) / np.nanmax(rank(arr, method)).

    """

    rankval = rank(arr, method)
    pctrank = rankval / np.nanmax(rankval)

    return pctrank

@nb.jit(
    nb.float64[::1](nb.float64[::1], nb.float64),
    locals = {
        'cap':nb.float64, 'floor':nb.float64
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def winsor_by_zscore(arr: nb.float64[::1], hurdle: nb.float64) -> nb.float64[::1]:
    """Winsorization by Z-score

    Args:
        arr (nb.float64[::1])
        zscore_hurdle (nb.float64): The z-score value to winsorize the data. e.g.) 1.96, 2.58, ...

    """
    cap = np.nanmean(arr) + (hurdle * np.nanstd(arr))
    floor = np.nanmean(arr) - (hurdle * np.nanstd(arr))
    arr[arr > cap] = cap
    arr[arr < floor] = floor
    return arr

@nb.jit(
    nb.float64[::1](nb.float64[::1], nb.float64),
    locals = {
        'cap':nb.float64, 'floor':nb.float64
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def winsor_by_percentile(arr, hurdle) -> nb.float64[::1]:
    """Winsorization by percentile

    Args:
        arr (nb.float64[::1])
        percentile_hurdle (nb.float64): top/bottom n percentile to winsorize the data. e.g.) 0.1, 0.001, ...

    """
    cap = np.nanpercentile(arr, 100 - hurdle)
    floor = np.nanpercentile(arr, hurdle)
    arr[arr > cap] = cap
    arr[arr < floor] = floor
    return arr

@nb.jit(
    nb.float64[::1](nb.float64[::1], ),
    locals = {
        'exp_arr':nb.float64[::1]
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def softmax(arr) -> nb.float64[::1]:
    """Softmax Normalization

    Args:
        arr (nb.float64[::1])

    """
    exp_arr = np.exp(arr)
    return exp_arr / np.nansum(exp_arr) 