
import numba as nb
import numpy as np


@nb.jit(
    (nb.float64[::1], nb.float64, nb.float64[::1], nb.float64, nb.float64[::1], nb.float64),
    # (nb.types.Array(nb.float64, 1, 'C'), nb.float64, nb.types.Array(nb.float64, 1, 'C'), nb.float64, nb.types.Array(nb.float64, 1, 'C'), nb.float64),
    locals = {
        'curr_wgt':nb.float64[::1], 'target_weight':nb.float64[::1], 'avail_pf_value': nb.float64,
        'dollar_weight': nb.float64[::1], 'orders': nb.float64[::1],
    },
    nopython = True,
    boundscheck = False,
    cache = True
)
def ordF_flt(current_price, fee, current_position, pf_value, weight_signal, leverage_factor):
    curr_wgt = (current_position * current_price) / pf_value
    target_weight = weight_signal / np.nansum(np.abs(weight_signal)) * leverage_factor

    avail_pf_value = pf_value * fee
    dollar_weight = avail_pf_value * (target_weight - curr_wgt)
    orders = (dollar_weight / current_price)
    return orders

@nb.jit(
    (nb.types.Array(nb.float64, 1, 'C'), nb.float64, nb.types.Array(nb.float64, 1, 'C'), nb.float64, nb.types.Array(nb.float64, 1, 'C'), nb.float64),
    locals = {
        'curr_wgt':nb.float64[::1], 'target_weight':nb.float64[::1], 'avail_pf_value': nb.float64,
        'dollar_weight': nb.float64[::1], 'orders': nb.int64[::1],
    },
    nopython = True,
    boundscheck = False,
    cache = True
)
def ordF_int(current_price, fee, current_position, pf_value, weight_signal, leverage_factor):
    curr_wgt = (current_position * current_price) / pf_value
    target_weight = (weight_signal / np.nansum(np.abs(weight_signal)) * leverage_factor)

    avail_pf_value = pf_value * fee
    dollar_weight = avail_pf_value * (target_weight - curr_wgt)
    orders = (dollar_weight / current_price).astype(np.int64)
    return orders

