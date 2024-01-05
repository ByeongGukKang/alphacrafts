
from numba import njit
import numpy as np

@njit
def basic_buy_execution(current_price, order, trader_cash, trader_buy_fee, orderbook_state):
    delta_cashflow = -1* current_price * order * trader_buy_fee
    delta_cash = np.nansum(delta_cashflow)
    delta_positions = order
    return delta_cashflow, delta_cash, delta_positions

@njit
def basic_sell_execution(current_price, order, trader_positions, trader_sell_fee, orderbook_state):
    delta_cashflow = -1 * current_price * order * trader_sell_fee
    delta_cash = np.nansum(delta_cashflow)
    delta_positions = order
    return delta_cashflow, delta_cash, delta_positions