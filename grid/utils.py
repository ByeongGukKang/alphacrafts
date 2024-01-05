from contextlib import contextmanager
from threading import Thread
from threading import Lock as ThreadLock

import numpy as np


class Ledger:

    def __init__(self, cash, codes):
        self._lock = ThreadLock()

        self._cash = cash
        
        self._book = np.zeros(
            shape = len(codes),
            # dtype = [("code", "U7"), ("order", "f8"), ("order_price", "f8"), ("position", "f8"), ("position_price", "f8")],
            dtype = [("code", "U7"), ("position", "f8"), ("position_price", "f8")],
            order = 'C' # C: row-major, F: column-major
        )
        self._book["code"] = codes

    ### Locking ###

    @property
    def islocked(self):
        return self._lock.locked()

    def lock(self):
        self._lock.acquire(blocking=False, timeout=-1)

    def unlock(self):
        self._lock.release()

    # Used as context manager to lock the ledger
    # with ledger.withlock():
    #    . . .
    @contextmanager
    def withlock(self):
        try:
            self.lock(self)
            yield
        finally:
            self.unlock(self)
    
    # Used as decorator to check if the lock is held by this instance
    def _chklock(func):
        def wrapper(*args, **kwargs):
            assert args[0].islocked, "Error: ledger access without lock"
            return func(*args, **kwargs)
        return wrapper

    ### Properties ###

    @property
    @_chklock
    def book(self):
        return self._book
    
    @property
    @_chklock
    def cash(self):
        return self._cash
    
    @_chklock
    def add_cash(self, delta_cash):
        self._cash += delta_cash

    @_chklock
    def update_cash(self, update_cash):
        self._cash = update_cash

    @property
    @_chklock
    def code(self):
        return self._book["code"]
    
    @_chklock
    def add_code(self, code):
        new_code_array = np.array(
            # (code, 0, 0, 0, 0),
            (code, 0.0, 0.0),
            # dtype = [("code", "U7"), ("order", "f8"), ("order_price", "f8"), ("position", "f8"), ("position_price", "f8")],
            dtype = [("code", "U7"), ("position", "f8"), ("position_price", "f8")],
            order = 'C' # C: row-major, F: column-major
        )
        self._book = np.append(self._book, new_code_array)

    # @property
    # @_chklock
    # def order(self):
    #     return self._book[["code", "order"]]
    
    # @_chklock
    # def add_order(self, code, delta_order):
    #     """
    #     code (1d array[U7]): code of the stock
    #     delta_order (1d array[float]): change in order
    #     new_price (1d array)[float]: new order price
    #     """
    #     idxCode = np.isin(self._book["code"], code)
    #     # add new code if not exist
    #     if not np.any(idxCode):
    #         self.add_code(code)
    #         idxCode = np.isin(self._book["code"], code)
    #     current_size = self._book["order"][idxCode]
    #     self._book["order"][idxCode] += delta_order
    #     self._book["order_price"][idxCode] = ((current_size * self._book["order_price"][idxCode]) + (delta_order * new_price)) / self._book["order"][idxCode]


    # @_chklock
    # def update_order(self, code, update_order, new_price):
    #     """
    #     code (1d array[U7]): code of the stock
    #     delta_order (1d array[float]): change in order
    #     new_price (1d array)[float]: new order price
    #     """
    #     idxCode = np.isin(self._book["code"], code)
    #     # add new code if not exist
    #     if not np.any(idxCode):
    #         self.add_code(code)
    #         idxCode = np.isin(self._book["code"], code)
    #     self._book["order"][idxCode] = update_order
    #     self._book["order_price"][idxCode] = new_price
    
    # @property
    # @_chklock
    # def order_price(self):
    #     return self._book[["code", "order_price"]]
    
    @property
    @_chklock
    def position(self):
        return self._book[["code", "position"]]
    
    @_chklock
    def add_position(self, code, delta_position, new_price):
        """
        code (1d array[U7]): code of the stock
        delta_position (1d array[float]): change in position
        new_price (1d array)[float]: new position price
        """
        idxCode = np.isin(self._book["code"], code)
        # add new code if not exist
        if not np.any(idxCode):
            self.add_code(code)
            idxCode = np.isin(self._book["code"], code)
        current_size = self._book["position"][idxCode]
        self._book["position"][idxCode] += delta_position
        self._book["position_price"][idxCode] = ((current_size * self._book["position_price"][idxCode]) + (delta_position * new_price)) / self._book["position"][idxCode]
    
    @_chklock
    def update_position(self, code, update_position, new_price):
        """
        code (1d array[U7]): code of the stock
        delta_position (1d array[float]): change in position
        new_price (1d array)[float]: new position price
        """
        idxCode = np.isin(self._book["code"], code)
        # add new code if not exist
        if not np.any(idxCode):
            self.add_code(code)
            idxCode = np.isin(self._book["code"], code)
        self._book["position"][idxCode] = update_position
        self._book["position_price"][idxCode] = new_price

    @property
    @_chklock
    def position_price(self):
        return self._book[["code", "position_price"]]