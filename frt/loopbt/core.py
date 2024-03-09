
import inspect
import types

import pandas as pd
import numpy as np
from tqdm import tqdm

from alphacrafts.frt.loopbt.func_execute import execF_basic


class PacketPacker: 

    def __init__(self, price_df):
        """
        price_df (pd.DataFrame): price data to be used for backtesting
        """
        self._frame_data = {}
        self._frame_data_index = {}
        self._frame_data_dtype = {}
        self._iscompiled = False

        # TODO price_df의 index 및 column의 dtype 확인
        # Pandas DataFrame check
        if not isinstance(price_df, pd.DataFrame):
            raise TypeError("Data must be pandas DataFrame!")
        # Index data type check
        if not isinstance(price_df.index, pd.DatetimeIndex):
            raise TypeError("Data index must be DatetimeIndex!")
        self._frame_prc = price_df.values
        self._frame_prc_index = price_df.index
        self._frame_prc_column = price_df.columns
        
    def add_data(self, key, data):
        """
        key : key of data to be added
        data (pd.DataFrame): data to be added
        """
        # Key check
        if key in self._frame_data.keys():
            raise KeyError(f"Key {key} already exists!")
        # Pandas DataFrame check
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be pandas DataFrame!")
        # Index data type check
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("Data index must be DatetimeIndex!")
        # Column data type check
        if len(set(data.dtypes)) != 1:
            raise TypeError("dtypes of Data columns are not consistent!")
        # Column data type check
        if data.dtypes.values[0] not in [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64, np.bool_]:
            raise TypeError("Data columns must be numeric!")
        
        self._frame_data[key] = data.values
        self._frame_data_index[key] = data.index
        self._frame_data_dtype[key] = data.dtypes.values[0]

    def get_data(self, key):
        """
        key : key of data to be retrieved
        """
        if key not in self._frame_data.keys():
            raise KeyError(f"Key {key} does not exist!")
        return pd.DataFrame(self._frame_data[key], index=self._frame_data_index[key], columns=self._frame_prc_column)

    def del_data(self, key):
        """
        key : key of data to be deleted
        """
        if key not in self._frame_data.keys():
            raise KeyError(f"Key {key} does not exist!")
        del self._frame_data[key]
        del self._frame_data_index[key]

    def compile(
            self,
            window_size: int|dict,
            drop_initials: bool = True
        ):
        """Compile data into packets

        Args:
            window_size (int, dict, np.inf): number of obersevations of data for each frame_data to be included in each packet
            - dictionary argument should be with key: frame_data key, value: window_size (int)
            - np.inf will work as an expanding window, it ignores drop_initials
            drop_initials (bool): drop initial packets that does not have enough data. Defaults to True

        """
        if isinstance(window_size, int):
            window_size = {key: window_size for key in self._frame_data.keys()}
        elif isinstance(window_size, dict):
            for key, value in window_size.items():
                if key not in self._frame_data.keys():
                    raise Exception(f"Key {key} does not exist in frame_data!")
                if not isinstance(value, int):
                    raise Exception(f"window_size must be integer!")
        elif window_size == np.inf:
            window_size = {key: np.inf for key in self._frame_data.keys()}
            drop_initials = False
        else:
            raise Exception("window_size must be integer or dict!")

        # Sender index
        self._sender_index = []
        # Memory to save complied data
        self._complied_data = {}

        # Loop for each date
        # cnt (int), end_date (datetime)
        for cnt, prc_end_date in enumerate(tqdm(self._frame_prc_index, desc="Compiling data packets...")):

            isInitial = False
            packet_idx = {} # packet that contains index of each frame_data, key:tuple 로 이루어짐 tuple(start_index, end_index)
            for key, data_index in self._frame_data_index.items():
                avail_data_idx = np.where(data_index < prc_end_date)[0] # array of available data index, before prc_end_date(CURRENT TIME)
                if drop_initials:
                    if len(avail_data_idx) < window_size[key]:
                        isInitial = True
                        break
                    str_idx = avail_data_idx[-1] - window_size[key] + 1
                else:
                    if len(avail_data_idx) == 0:
                        isInitial = True
                        break
                    str_idx = max(avail_data_idx[-1] - window_size[key] + 1, 0)

                end_idx = avail_data_idx[-1]
                packet_idx[key] = [str_idx, end_idx]

            if isInitial:
                continue
            else:
                # save packet, end_date equals to current time
                self._complied_data[prc_end_date] = (cnt, packet_idx) 
                # append index
                self._sender_index.append(prc_end_date)
        
        # iscomplied flag
        self._iscompiled = True

    # Return 
    @property
    def data_index(self):
        # check if data is complied
        if not self._iscompiled:
            raise Exception("Data is not complied yet!")
        return self._sender_index
    
    @property
    def data_column(self):
        # check if data is complied
        if not self._iscompiled:
            raise Exception("Data is not complied yet!")
        return self._frame_prc_column

    # TODO ?
    # Maybe we can move to normal function that does not use yield
    # Also applying numba
    def datafeed(self):
        """ Returns datafeed generator

        """
        # check if data is complied
        if not self._iscompiled:
            raise Exception("Data is not complied yet!")

        # generator
        def packet_generator():
            # packing data
            for time, packet_tuple in self._complied_data.items(): # key:packet_idx
                packet_val = {} # real packet with data value

                current_price = self._frame_prc[packet_tuple[0],:].ravel('C') # current price array
                # TODO Is it really helpful to give current price to a researcher?
                packet_val['prc'] = current_price 

                packet_idx = packet_tuple[1]
                for key, tuple_idx in packet_idx.items():
                    tmp_data = self._frame_data[key][tuple_idx[0]:tuple_idx[1]+1,:]
                    if isinstance(tmp_data, np.ndarray):
                        packet_val[key] = self._frame_data[key][tuple_idx[0]:tuple_idx[1]+1,:]
                    else:
                        packet_val[key] = np.array(tmp_data, dtype=self._frame_data_dtype[key])

                yield time, packet_val, current_price
        
        return packet_generator()
    

class LedgerBackTest:

    def __init__(self, init_cash, sender_index, sender_column):

        self._base_index = sender_index
        self._base_columns = sender_column

        self._cash = np.zeros(len(sender_index), dtype=np.float64)
        self._cash[0] = init_cash
        self._cash[-1] = init_cash
        self._pf_value = np.zeros(len(sender_index), dtype=np.float64)
        self._pf_value[0] = init_cash
        self._pf_value[-1] = init_cash
        
        self._book = np.zeros(
            shape = (len(sender_index), len(sender_column)),
            # dtype = [("order", "f8"), ("order_price", "f8"), ("position", "f8"), ("position_price", "f8")],
            dtype = [("posit", "f8"), ("posit_prc", "f8")],
            order = 'C' # C: row-major, F: column-major
        )

        # row index (acts like a pointer)
        self._curr_loc = -1 

    ### Backtest ###

    # Move to next time 
    def _bt_next(self, current_price):
        # move current balance to next time
        self._cash[self._curr_loc + 1] = self._cash[self._curr_loc]
        self._book[self._curr_loc + 1] = self._book[self._curr_loc]
        # update portfolio value
        self._pf_value[self._curr_loc + 1] = self._cash[self._curr_loc] + np.nansum(self._book["posit"][self._curr_loc] * current_price)
        # increase location pointer
        self._curr_loc += 1

    # Book for signal_func & order_func

    @property
    def _bt_book(self):
        return self._book[self._curr_loc]
    
    @property
    def _bt_cash(self):
        return self._cash[self._curr_loc]
    
    @property
    def _bt_pf_value(self):
        return self._pf_value[self._curr_loc]

    def _bt_add_cash(self, delta_cash):
        self._cash[self._curr_loc] += delta_cash

    def _bt_add_order(self, delta_order, new_price):
        """
        delta_order (1d array[float]): change in order
        new_price (1d array)[float]: new order price
        """
        current_size = self._book["order"][self._curr_loc]
        self._book["order"][self._curr_loc] += delta_order
        self._book["order_price"][self._curr_loc] = ((current_size * self._book["order_price"][self._curr_loc]) + (delta_order * new_price)) / self._book["order"][self._curr_loc]

    def _bt_add_position(self, delta_position, new_price):
        """
        delta_position (1d array[float]): change in position
        new_price (1d array)[float]: new position price
        """
        current_size = self._book["posit"][self._curr_loc]
        self._book["posit"][self._curr_loc] += delta_position
        self._book["posit_prc"][self._curr_loc] = ((current_size * self._book["posit_prc"][self._curr_loc]) + (delta_position * new_price)) / self._book["posit"][self._curr_loc]


    # TODO Remove and create performance
    ### Properties ###

    @property
    def cash(self):
        return pd.DataFrame(self._cash, index=self._base_index, columns=["cash"])
    
    @property
    def pf_value(self):
        return pd.DataFrame(self._pf_value, index=self._base_index, columns=["pf_value"])
    
    @property
    def order(self):
        return pd.DataFrame(self._book["order"], index=self._base_index, columns=self._base_columns)
    
    @property
    def order_price(self):
        return pd.DataFrame(self._book["order_price"], index=self._base_index, columns=self._base_columns)
    
    @property
    def position(self):
        return pd.DataFrame(self._book["posit"], index=self._base_index, columns=self._base_columns)
    
    @property
    def position_price(self):
        return pd.DataFrame(self._book["posit_prc"], index=self._base_index, columns=self._base_columns)
    

# Historical 용도 FINAL
class Trader:

    def __init__(
        self,
        name: str,
        init_cash: float,
        signal_func: types.FunctionType,
        order_func: types.FunctionType,
        transaction_cost: tuple = (0,0),
        leverage_factor: float = 1.0,
    ):
        """Trader class for backtesting

        Args:
            name (str): name of trader
            init_cash (float): initial cash
            signal_func (types.FunctionType): signal function, user defined
            order_func (types.FunctionType): order function, take one from alphacrafts.frt.loopbt.func_order
            transaction_cost (tuple, optional): transaction cost. (buy_fee, sell_fee). Defaults to (0,0).
            leverage_factor (float, optional): leverage factor. Defaults to 1.0.

        """

        # TODO inspect function parameters
        if list(inspect.signature(signal_func).parameters.keys()) != ["time","data","cash","pf_value","book","var_signal"]:
            raise Exception("Signal function must have ordered parameters (time, data, cash, pf_value, book, var_signal)")

        self.name = name
        self.init_cash = init_cash

        self.signal_func = signal_func
        self.order_func = order_func

        self.buy_fee = np.float64(1 + transaction_cost[0])
        self.sell_fee = np.float64(1 - transaction_cost[1])

        self.leverage_factor = leverage_factor

        self._var_signal = {}
        # self._var_order = {"buy_fee": self.buy_fee, "sell_fee": self.sell_fee}

        self._performance = None
    
    @property
    def var_signal(self):
        return self._var_signal
    
    @var_signal.setter
    def var_signal(self, key, value):
        self._var_signal[key] = value

    # @property
    # def var_order(self):
    #     return self._var_order
    
    # @var_order.setter
    # def var_order(self, key, value):
    #     self._var_order[key] = value

    @property
    def performance(self):
        if self._performance is None:
            raise Exception("Backtest not done yet!")
        return self._performance
    
    def run(self, packet_packer: PacketPacker):
        data_index = packet_packer.data_index
        data_column = packet_packer.data_column
        packet_generator = packet_packer.datafeed()

        self.ledger = LedgerBackTest(self.init_cash, data_index, data_column)
        larger_fee = min(self.buy_fee, self.sell_fee)
        var_signal = self.var_signal.copy()

        for _ in tqdm(range(len(data_index)-1), desc=f'Backtesting [{self.name}]...'):

            ### Get data packet
            time, trader_data, current_price = next(packet_generator)

            ### Move to next time
            self.ledger._bt_next(current_price)

            ### Generate Signals & Orders

            # TODO 분할매수/매도 가능하도록 수정
            weight_signal, var_signal = self.signal_func(
                time,                                
                trader_data,
                self.ledger._bt_cash,
                self.ledger._bt_pf_value,
                self.ledger._bt_book,
                var_signal
            )
            
            if weight_signal is not None:
                orders = self.order_func(
                    current_price,  
                    larger_fee,
                    self.ledger._bt_book['posit'].ravel('C'),
                    self.ledger._bt_pf_value,
                    weight_signal.astype(np.float64),
                    self.leverage_factor,
                )

                ### Execute Orders
                delta_position_cash, delta_cash_sum, delta_position = execF_basic(current_price, orders, self.buy_fee, self.sell_fee)

                # Update Ledger
                self.ledger._bt_add_cash(delta_cash_sum)
                self.ledger._bt_add_position(delta_position, current_price)
        
        # Backtest Done
        time, trader_data, current_price = next(packet_generator)
        self.ledger._bt_next(current_price)

    def run_speedcheck(self, packet_packer: PacketPacker):
        from datetime import datetime 
        data_index = packet_packer.data_index
        data_column = packet_packer.data_column
        packet_generator = packet_packer.datafeed()

        self.ledger = LedgerBackTest(self.init_cash, data_index, data_column)
        larger_fee = min(self.buy_fee, self.sell_fee)
        var_signal = self.var_signal.copy()

        start_time = datetime.now()
        self.generator_time = start_time
        self.signal_time = start_time
        self.order_time = start_time
        self.execution_time = start_time

        for _ in tqdm(range(len(data_index)-1), desc=f'Backtesting [{self.name}]...'):

            ### Get data packet
            tmp_start_time = datetime.now()
            time, trader_data, current_price = next(packet_generator)
            self.generator_time += datetime.now() - tmp_start_time

            ### Move to next time
            self.ledger._bt_next(current_price)

            ### Generate Signals & Orders

            # TODO 분할매수/매도 가능하도록 수정
            tmp_start_time = datetime.now()
            weight_signal, var_signal = self.signal_func(
                time,                                
                trader_data,
                self.ledger._bt_cash,
                self.ledger._bt_pf_value,
                self.ledger._bt_book,
                var_signal
            )
            self.signal_time += datetime.now() - tmp_start_time
            
            if weight_signal is not None:
                tmp_start_time = datetime.now()
                orders = self.order_func(
                    current_price,  
                    larger_fee,
                    self.ledger._bt_book['posit'].ravel('C'),
                    self.ledger._bt_pf_value,
                    weight_signal.astype(np.float64),
                    self.leverage_factor,
                )
                self.order_time += datetime.now() - tmp_start_time

                ### Execute Orders
                tmp_start_time = datetime.now()
                delta_position_cash, delta_cash_sum, delta_position = execF_basic(current_price, orders, self.buy_fee, self.sell_fee)
                self.execution_time += datetime.now() - tmp_start_time

                # Update Ledger
                self.ledger._bt_add_cash(delta_cash_sum)
                self.ledger._bt_add_position(delta_position, current_price)
        
        # Backtest Done
        time, trader_data, current_price = next(packet_generator)
        self.ledger._bt_next(current_price)
        