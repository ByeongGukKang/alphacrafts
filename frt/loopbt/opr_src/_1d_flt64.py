import numba as nb
import numpy as np


### Complied Numpy Functions ###

@nb.jit(
    nb.float64(nb.float64[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def mean(arr) -> nb.float64:
    """Compiled nanmean function for 1D array

    Args:
        arr (nb.float64[::1])

    """
    return np.mean(arr)

@nb.jit(
    nb.float64(nb.float64[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def std(arr) -> nb.float64:
    """Compiled nanstd function for 1D array

    Args:
        arr (nb.float64[::1])

    """
    return np.std(arr)

@nb.jit(
    nb.float64(nb.float64[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def min(arr) -> nb.float64:
    """Compiled min function for 1D array

    Args:
        arr (nb.float64[::1])

    """
    return np.min(arr)

@nb.jit(
    nb.float64(nb.float64[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def max(arr) -> nb.float64:
    """Compiled max function for 1D array

    Args:
        arr (nb.float64[::1])

    """
    return np.max(arr)

@nb.jit(
    nb.float64(nb.float64[::1], nb.float64),
    boundscheck = False,
    cache = True,
    nopython = True
)
def percentile(arr, hurdle) -> nb.float64[::1]:
    """Compiled percentile function for 1D array

    Args:
        arr (nb.float64[::1])
        hurdle (nb.float64): percentile value

    """
    return np.percentile(arr, hurdle)

@nb.jit(
    nb.float64(nb.float64[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def sum(arr) -> nb.float64:
    """Compiled sum function for 1D array

    Args:
        arr (nb.float64[::1])

    """
    return np.sum(arr)

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
        arr (nb.float64[::1])

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
        arr (nb.float64[::1])

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

# @nb.jit(
#     nb.bool_[::1](nb.float64[::1], ),
#     boundscheck = False,
#     cache = True,
#     nopython = True
# )
# def isnan(arr) -> nb.float64:
#     """Compiled isnan function for 1D array

#     Args:
#         arr (nb.float64[::1])

#     """
#     return np.isnan(arr)

# @nb.jit(
#     nb.float64[::1](nb.float64[::1], ),
#     boundscheck = False,
#     cache = True,
#     nopython = True
# )
# def abs(arr) -> nb.float64:
#     """Compiled abs function for 1D array

#     Args:
#         arr (nb.float64[::1])

#     """
#     return np.abs(arr)

###### Complied Operations ######

@nb.jit(
    nb.bool_(nb.float64[::1], nb.float64),
    boundscheck = False,
    cache = True,
    nopython = True
)
def isall(arr, value):
    """Check if all elements are equal to the value

    Args:
        arr (nb.float64[::1])
        value (nb.float64)

    Returns:
        nb.bool_: True if all elements in the array are equal to the value, False otherwise
    """
    for val in arr:
        if val != value:
            return False
    return True

@nb.jit(
    nb.bool_(nb.float64[::1], nb.float64),
    boundscheck = False,
    cache = True,
    nopython = True
)
def isany(arr, value):
    """Check if any element is equal to the value

    Args:
        arr (nb.float64[::1])
        value (nb.float64)

    Returns:
        nb.bool_: True if any elements in the array are different to the value, False otherwise
    """
    for val in arr:
        if val == value:
            return True
    return False

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
        'len_non_nan':nb.int64, 'sorter':nb.int64[::1], 'ordinal_ranks':nb.int64[::1], 'ordered_ranks':nb.float64[::1],
        'y':nb.float64[::1], 'i':nb.boolean[::1], 'indices':nb.int64[::1], 'counts':nb.int64[::1], 'ranks':nb.float64[::1],
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
    res = np.full_like(arr, np.nan, dtype=np.float64)

    nan_mask = np.isnan(arr)
    non_nan_arr = arr[~nan_mask]
    len_non_nan = len(non_nan_arr)

    sorter = np.argsort(non_nan_arr, kind='mergesort')
    ordinal_ranks = np.arange(1, len_non_nan+1, dtype=np.int64)
    ordered_ranks = np.empty(len_non_nan, dtype=np.float64)

    if method == 'ordinal':
        ordered_ranks[sorter] = ordinal_ranks
    else:
        # Sort array
        y = non_nan_arr[sorter]
        # Logical indices of unique elements
        i = np.concatenate((np.ones(1, dtype=np.bool_), y[:-1] != y[1:]))

        # Integer indices of unique elements
        indices = np.arange(len_non_nan, dtype=np.int64)[i]
        indices = np.append(indices, len_non_nan)
        # Counts of unique elements
        counts = np.diff(indices)

        # Compute `'min'`, `'max'`, and `'mid'` ranks of unique elements
        if method == 'dense':
            ranks = (np.cumsum(i)[i]).astype(np.float64)
        elif method == 'average':
            ranks = (ordinal_ranks[i] + (counts - 1)/2).astype(np.float64)
        elif method == 'min':
            ranks = (ordinal_ranks[i]).astype(np.float64)
        elif method == 'max':
            ranks = (ordinal_ranks[i] + counts - 1).astype(np.float64)
        else:
            raise ValueError("Unknown argument for parameter 'method'")
        
        ranks = np.repeat(ranks, counts)
        ordered_ranks[sorter] = ranks

    res[~nan_mask] = ordered_ranks
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

@nb.jit(
    nb.bool_[::1](nb.float64[::1], nb.int64),
    locals = {
        'rank_arr':nb.float64[::1]
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def top(arr, n) -> nb.bool_[::1]:
    """Return the mask array of the top n elements in the array

    Args:
        arr (nb.float64[::1])

    """
    rank_arr = rank(arr, 'dense')
    return rank_arr > (nanmax(rank_arr) - n)

@nb.jit(
    nb.float64[:,::1](nb.float64[::1], nb.int64),
    boundscheck = False,
    cache = True,
    nopython = True
)
def rolling(arr, window) -> nb.float64[:,::1]:
    """Return the rolling window of the array

    Args:
        arr (nb.float64[::1])

    """
    return np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(arr, window))

@nb.jit(
    nb.float64[::1](nb.float64[::1], nb.float64[::1], nb.float64),
    boundscheck = False,
    cache = True,
    nopython = True
)
def hump(arr, compare, threshold):
    """Return the hump array of the array

    Args:
        arr (nb.float64[::1])
        compare (nb.float64[::1])
        threshold (nb.float64)
    
    """
    return np.where(np.abs(arr-compare) > threshold, arr, compare)

@nb.jit(
    nb.float64[::1](nb.float64[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def weight(arr) -> nb.float64[::1]:
    """Return the weight array of the array

    Args:
        arr (nb.float64[::1])

    """
    arr_sum = np.nansum(np.abs(arr))
    if arr_sum == 0:
        raise ValueError("sum of abs(array) is zero")
    arr[np.isnan(arr)] = 0
    return arr / arr_sum