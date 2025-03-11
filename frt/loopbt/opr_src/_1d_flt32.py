import numba as nb
import numpy as np

# TODO Notice: commented functions are slower than the pure numpy functions

### Complied Numpy Functions ###
# @nb.jit(
#     nb.float32[::1](nb.float32[:],),
#     boundscheck = False,
#     cache = True,
#     nopython = True
# )
# def corder(array) -> nb.float32[::1]:
#     """Convert array to 'C' ordered array

#     Args:
#         array (nb.float32[1::])

#     """
#     return np.ascontiguousarray(array)

@nb.jit(
    nb.float32(nb.float32[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def mean(arr) -> nb.float32:
    """Compiled nanmean function for 1D array

    Args:
        arr (nb.float32[::1])

    """
    return np.mean(arr)

@nb.jit(
    nb.float32(nb.float32[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def std(arr) -> nb.float32:
    """Compiled nanstd function for 1D array

    Args:
        arr (nb.float32[::1])

    """
    return np.std(arr)

@nb.jit(
    nb.float32(nb.float32[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def min(arr) -> nb.float32:
    """Compiled min function for 1D array

    Args:
        arr (nb.float32[::1])

    """
    return np.min(arr)

@nb.jit(
    nb.float32(nb.float32[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def max(arr) -> nb.float32:
    """Compiled max function for 1D array

    Args:
        arr (nb.float32[::1])

    """
    return np.max(arr)

@nb.jit(
    nb.float32(nb.float32[::1], nb.float32),
    boundscheck = False,
    cache = True,
    nopython = True
)
def percentile(arr, hurdle) -> nb.float32[::1]:
    """Compiled percentile function for 1D array

    Args:
        arr (nb.float32[::1])
        hurdle (nb.float32): percentile value

    """
    return np.percentile(arr, hurdle)

@nb.jit(
    nb.float32(nb.float32[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def sum(arr) -> nb.float32:
    """Compiled sum function for 1D array

    Args:
        arr (nb.float32[::1])

    """
    return np.sum(arr)

@nb.jit(
    nb.float32(nb.float32[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def nanmean(arr) -> nb.float32:
    """Compiled nanmean function for 1D array

    Args:
        arr (nb.float32[::1])

    """
    return np.nanmean(arr)

@nb.jit(
    nb.float32(nb.float32[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def nanstd(arr) -> nb.float32:
    """Compiled nanstd function for 1D array

    Args:
        arr (nb.float32[::1])

    """
    return np.nanstd(arr)

@nb.jit(
    nb.float32(nb.float32[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def nanmin(arr) -> nb.float32:
    """Compiled nanmin function for 1D array

    Args:
        arr (nb.float32[::1])

    """
    return np.nanmin(arr)

@nb.jit(
    nb.float32(nb.float32[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def nanmax(arr) -> nb.float32:
    """Compiled nanmax function for 1D array

    Args:
        arr (nb.float32[::1])

    """
    return np.nanmax(arr)

@nb.jit(
    nb.float32(nb.float32[::1], nb.float32),
    boundscheck = False,
    cache = True,
    nopython = True
)
def nanpercentile(arr, hurdle) -> nb.float32[::1]:
    """Compiled nanpercentile function for 1D array

    Args:
        arr (nb.float32[::1])
        hurdle (nb.float32): percentile value

    """
    return np.nanpercentile(arr, hurdle)

@nb.jit(
    nb.float32(nb.float32[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def nansum(arr) -> nb.float32:
    """Compiled nansum function for 1D array

    Args:
        arr (nb.float32[::1])

    """
    return np.nansum(arr)

# @nb.jit(
#     nb.bool_[::1](nb.float32[::1], ),
#     boundscheck = False,
#     cache = True,
#     nopython = True
# )
# def isnan(arr) -> nb.float32:
#     """Compiled isnan function for 1D array

#     Args:
#         arr (nb.float32[::1])

#     """
#     return np.isnan(arr)

# @nb.jit(
#     nb.float32[::1](nb.float32[::1], ),
#     boundscheck = False,
#     cache = True,
#     nopython = True
# )
# def abs(arr) -> nb.float32:
#     """Compiled abs function for 1D array

#     Args:
#         arr (nb.float32[::1])

#     """
#     return np.abs(arr)

###### Complied Operations ######
@nb.jit(
    nb.bool_(nb.float32[::1], nb.float32),
    boundscheck = False,
    cache = True,
    nopython = True
)
def isall(arr, value):
    """Check if all elements are equal to the value

    Args:
        arr (nb.float32[::1])
        value (nb.float32)

    Returns:
        nb.bool_: True if all elements in the array are equal to the value, False otherwise
    """
    for val in arr:
        if val != value:
            return False
    return True

@nb.jit(
    nb.bool_(nb.float32[::1], nb.float32),
    boundscheck = False,
    cache = True,
    nopython = True
)
def isany(arr, value):
    """Check if any element is equal to the value

    Args:
        arr (nb.float32[::1])
        value (nb.float32)

    Returns:
        nb.bool_: True if any elements in the array are different to the value, False otherwise
    """
    for val in arr:
        if val == value:
            return True
    return False

@nb.jit(
    nb.float32[::1](nb.float32[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def zscore(arr):
    """Z-score Normalization

    Args:
        arr (nb.float32[::1])

    """
    return ((arr - np.nanmean(arr)) / np.nanstd(arr)).astype(np.float32)

@nb.jit(
    nb.float32[::1](nb.float32[::1], ),
    boundscheck = False,
    cache = True,
    nopython = True
)
def minmax(arr: nb.float32[::1]) -> nb.float32[::1]:
    """Min-Max Normalization

    Args:
        arr (np.float32[::1])

    """
    return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))

@nb.jit(
    nb.float32[::1](nb.float32[::1], nb.types.unicode_type),
    locals = {
        'res':nb.float32[::1], 'nan_mask':nb.boolean[::1], 'non_nan_arr':nb.float32[::1],
        'len_non_nan':nb.int32, 'sorter':nb.int32[::1], 'ordinal_ranks':nb.int32[::1], 'ordered_ranks':nb.float32[::1],
        'y':nb.float32[::1], 'i':nb.boolean[::1], 'indices':nb.int32[::1], 'counts':nb.int32[::1], 'ranks':nb.float32[::1],
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def rank(arr: nb.float32[::1], method: nb.types.unicode_type) -> nb.float32[::1]:
    """Rank Normalization

    Args:
        arr (nb.float32[::1])
        method (str): 'average', 'min', 'max', 'dense', 'ordinal'.
            'average': The average of the ranks that would have been assigned to all the tied values is assigned to each value.
            'min': The minimum of the ranks that would have been assigned to all the tied values is assigned to each value. (This is also referred to as 'competition' ranking.)
            'max': The maximum of the ranks that would have been assigned to all the tied values is assigned to each value.
            'dense': Like 'min', but the rank of the next highest element is assigned the rank immediately after those assigned to the tied elements.
            'ordinal': All values are given a distinct rank, corresponding to the order that the values occur in a.

    Source:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html

    """
    res = np.full_like(arr, np.nan, dtype=np.float32)

    nan_mask = np.isnan(arr)
    non_nan_arr = arr[~nan_mask]
    len_non_nan = len(non_nan_arr)

    sorter = np.argsort(non_nan_arr, kind='mergesort').astype(np.int32)
    ordinal_ranks = np.arange(1, len_non_nan+1, dtype=np.int32)
    ordered_ranks = np.empty(len_non_nan, dtype=np.float32)

    if method == 'ordinal':
        ordered_ranks[sorter] = ordinal_ranks
    else:
        # Sort array
        y = non_nan_arr[sorter]
        # Logical indices of unique elements
        i = np.concatenate((np.ones(1, dtype=np.bool_), y[:-1] != y[1:]))

        # Integer indices of unique elements
        indices = np.arange(len_non_nan, dtype=np.int32)[i]
        indices = np.append(indices, len_non_nan)
        # Counts of unique elements
        counts = np.diff(indices)

        # Compute `'min'`, `'max'`, and `'mid'` ranks of unique elements
        if method == 'dense':
            ranks = (np.cumsum(i)[i]).astype(np.float32)
        elif method == 'average':
            ranks = (ordinal_ranks[i] + (counts - 1)/2).astype(np.float32)
        elif method == 'min':
            ranks = (ordinal_ranks[i]).astype(np.float32)
        elif method == 'max':
            ranks = (ordinal_ranks[i] + counts - 1).astype(np.float32)
        else:
            raise Exception("ValueError: Unknown argument for parameter 'method'")
        
        ranks = np.repeat(ranks, counts)
        ordered_ranks[sorter] = ranks

    res[~nan_mask] = ordered_ranks
    return res

@nb.jit(
    nb.float32[::1](nb.float32[::1], nb.types.unicode_type),
    locals = {
        'rankval':nb.float32[::1], 'pctrank':nb.float32[::1]
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def rank_pct(arr: nb.float32[::1], method: nb.types.unicode_type) -> nb.float32[::1]:
    """Percentile Rank Normalization

    Args:
        arr (nb.float32[::1])
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
    nb.float32[::1](nb.float32[::1], nb.float32),
    locals = {
        'cap':nb.float32, 'floor':nb.float32
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def winsor_by_zscore(arr: nb.float32[::1], hurdle: nb.float32) -> nb.float32[::1]:
    """Winsorization by Z-score

    Args:
        arr (nb.float32[::1])
        zscore_hurdle (nb.float32): The z-score value to winsorize the data. e.g.) 1.96, 2.58, ...

    """
    cap = np.nanmean(arr) + (hurdle * np.nanstd(arr))
    floor = np.nanmean(arr) - (hurdle * np.nanstd(arr))
    arr[arr > cap] = cap
    arr[arr < floor] = floor
    return arr

@nb.jit(
    nb.float32[::1](nb.float32[::1], nb.float32),
    locals = {
        'cap':nb.float32, 'floor':nb.float32
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def winsor_by_percentile(arr, hurdle) -> nb.float32[::1]:
    """Winsorization by percentile

    Args:
        arr (nb.float32[::1])
        percentile_hurdle (nb.float32): top/bottom n percentile to winsorize the data. e.g.) 0.1, 0.001, ...

    """
    cap = np.nanpercentile(arr, 100 - hurdle)
    floor = np.nanpercentile(arr, hurdle)
    arr[arr > cap] = cap
    arr[arr < floor] = floor
    return arr

@nb.jit(
    nb.float32[::1](nb.float32[::1], ),
    locals = {
        'exp_arr':nb.float32[::1]
    },
    boundscheck = False,
    cache = True,
    nopython = True
)
def softmax(arr) -> nb.float32[::1]:
    """Softmax Normalization

    Args:
        arr (nb.float32[::1])

    """
    exp_arr = np.exp(arr)
    return exp_arr / np.nansum(exp_arr) 

@nb.jit(
    nb.bool_[::1](nb.float32[::1], nb.int32),
    locals = {
        'rank_arr':nb.float32[::1]
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
    nb.float32[:,::1](nb.float32[::1], nb.int32),
    boundscheck = False,
    cache = True,
    nopython = True
)
def rolling(arr, window) -> nb.float32[:,::1]:
    """Return the rolling window of the array

    Args:
        arr (nb.float32[::1])

    """
    return np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(arr, window))