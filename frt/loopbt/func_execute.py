
from numba import njit
import numba as nb
import numpy as np


@njit(
    (nb.float64[::1], nb.float64[::1], nb.float64, nb.float64),
    locals = {'delta_position_cash':nb.float64[::1], 'buy_order': nb.int64[::1], 'sell_order': nb.int64[::1], 'delta_cash_sum': nb.float64},
    nopython = True,    
    boundscheck = False,
    cache = True
)
def execF_basic(current_price, order, buy_fee, sell_fee):
    delta_position_cash = np.zeros_like(order, dtype=np.float64)
    
    buy_order = np.where(order > 0)[0]
    sell_order = np.where(order < 0)[0]

    delta_position_cash[buy_order] = -1 * current_price[buy_order] * order[buy_order] * buy_fee
    delta_position_cash[sell_order] = -1 * current_price[sell_order] * order[sell_order] * sell_fee
    delta_cash_sum = np.nansum(delta_position_cash)

    return delta_position_cash, delta_cash_sum, order