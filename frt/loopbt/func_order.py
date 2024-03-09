
import numba as nb
import numpy as np


@nb.jit(
    nb.float64[::1](nb.float64[::1], nb.float64, nb.float64[::1], nb.float64, nb.float64[:], nb.float64),
    locals = {
        'curr_wgt':nb.float64[::1], 'target_wgt':nb.float64[::1], 'avail_pf_value': nb.float64,
        'dollar_wgt': nb.float64[::1], 'orders': nb.float64[::1],
    },
    nopython = True,
    boundscheck = False,
    cache = True
)
def ordF_flt(curr_prc, fee, curr_posit, pf_value, wgt_signal, leverage_factor):
    wgt_signal.ravel()

    curr_wgt = (curr_posit * curr_prc) / pf_value
    target_wgt = wgt_signal / np.nansum(np.abs(wgt_signal)) * leverage_factor

    avail_pf_value = pf_value * fee
    dollar_wgt = avail_pf_value * (target_wgt - curr_wgt)
    orders = (dollar_wgt / curr_prc)
    return orders

@nb.jit(
    nb.int64[::1](nb.float64[::1], nb.float64, nb.float64[::1], nb.float64, nb.float64[:], nb.float64),
    locals = {
        'curr_wgt':nb.float64[::1], 'target_wgt':nb.float64[::1], 'avail_pf_value': nb.float64,
        'dollar_wgt': nb.float64[::1], 'orders': nb.int64[::1],
    },
    nopython = True,
    boundscheck = False,
    cache = True
)
def ordF_int(curr_prc, fee, curr_posit, pf_value, wgt_signal, leverage_factor):
    wgt_signal.ravel()

    curr_wgt = (curr_posit * curr_prc) / pf_value
    target_wgt = wgt_signal / np.nansum(np.abs(wgt_signal)) * leverage_factor

    avail_pf_value = pf_value * fee
    dollar_wgt = avail_pf_value * (target_wgt - curr_wgt)
    orders = (dollar_wgt / curr_prc).astype(np.int64)
    return orders
