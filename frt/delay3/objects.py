
import inspect
import types

import pandas as pd
import numpy as np
from tqdm import tqdm

from alphacrafts.frt.tools import to_nb_datetime
from alphacrafts.frt.types import OrderResult, Performance
from alphacrafts.frt.delay3.execution import basic_buy_execution, basic_sell_execution


class ExchangeSender: 

    def __init__(self, price_df):
        """
        price_df (pd.DataFrame): price data to be used for backtesting
        """
        self._frame_data = {}
        self._iscompiled = False

        # TODO price_df의 index 및 column의 dtype 확인
        # self._frame_data["date"] = pd.DataFrame(price_df.index.to_numpy(), index=price_df.index, columns=["date"])
        self._frame_data["price"] = price_df
        
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
        if data.dtypes.values[0] not in [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64, np.bool8]:
            raise TypeError("Data columns must be numeric!")
        
        self._frame_data[key] = data

    def del_data(self, key):
        """
        key : key of data to be deleted
        """
        if key not in self._frame_data.keys():
            raise KeyError(f"Key {key} does not exist!")
        del self._frame_data[key]

    # TODO window_size for each frame_data
    def compile(
            self,
            window_size: int,
            drop_initials: bool = True
        ):
        """
        window_size (int): number of price obersevations to be included in each packet
        drop_initials (bool): drop initial packets that does not have enough data 
        """
        # Sender index
        self._sender_index = []

        # Match with Python index starts from 0
        window_size = window_size - 1 

        # Memory to save complied data
        self._complied_data = {}

        # Date index for loop
        date_index = self._frame_data["price"].index

        # Loop for each date
        for cnt, end_date in enumerate(tqdm(date_index, desc="Compiling data packets...")):

            # drop initials
            if cnt < window_size:
                if drop_initials:
                    continue

            # set start date
            str_date = date_index[max(cnt - window_size, 0)]

            # generate packet
            packet_idx = {} # packet that contains index of each frame_data, key:tuple 로 이루어짐 tuple(start_index,end_index)
            for key, value in self._frame_data.items():
                #print(key, value.index, str_date)
                idx_list = np.where((value.index >= str_date) & (value.index <= end_date))[0]
                if len(idx_list) == 0:
                    packet_idx[key] = None
                else:
                    str_idx = np.where((value.index >= str_date) & (value.index <= end_date))[0][0] 
                    end_idx = np.where((value.index >= str_date) & (value.index <= end_date))[0][-1] # TODO avoid using -1 index
                    packet_idx[key] = [str_idx, end_idx]

            # save packet, end_date equals to current time
            self._complied_data[end_date] = packet_idx
            # append index
            self._sender_index.append(end_date)
        
        # iscomplied flag
        self._iscompiled = True

    # Return 
    def data_index(self):
        # check if data is complied
        if not self._iscompiled:
            raise Exception("Data is not complied yet!")
        return self._sender_index
    
    def data_column(self):
        # check if data is complied
        if not self._iscompiled:
            raise Exception("Data is not complied yet!")
        return self._frame_data["price"].columns

    def datafeed(self):
        """
        Returns datafeed generator
        """
        # check if data is complied
        if not self._iscompiled:
            raise Exception("Data is not complied yet!")

        # TODO Numba 적용
        def packet_generator():
            # generate datafeed
            for time, packet_idx in self._complied_data.items(): # key:packet_idx
                packet_val = {} # real packet with data value

                for key, tuple_idx in packet_idx.items():
                    if tuple_idx == None:
                        packet_val[key] = np.nan
                    else:
                        packet_val[key] = self._frame_data[key].values[tuple_idx[0]:tuple_idx[1]+1,:]
                    # packet_val[key] = self._frame_data[key].values[tuple_idx[0]:tuple_idx[1]+1,:]

                yield time, packet_val 
        
        return packet_generator()
    

class ExchangeReceiver:

    def __init__(self, execute_buy, execute_sell):
        self.execute_buy = execute_buy
        self.execute_sell = execute_sell

    def buy_execute(self, current_price, order, trader_cash, trader_buy_fee, orderbook_state): # 들어온 주문을 시장에서 집행
        return self.execute_buy(current_price, order, trader_cash, trader_buy_fee, orderbook_state)

    def sell_execute(self, current_price, order, trader_positions, trader_sell_fee, orderbook_state): # 들어온 주문을 시장에서 집행
        return self.execute_sell(current_price, order, trader_positions, trader_sell_fee, orderbook_state)


class LedgerBackTest:

    def __init__(self, init_cash, sender_index, sender_column):

        self._base_index = sender_index
        self._base_columns = sender_column

        self._cash = np.empty(len(sender_index), dtype=np.float64)
        self._cash[0] = init_cash
        self._pf_value = np.empty(len(sender_index), dtype=np.float64)
        self._pf_value[0] = init_cash
        
        self._book = np.empty(
            shape = (len(sender_index), len(sender_column)),
            dtype = [("order", "f8"), ("order_price", "f8"), ("position", "f8"), ("position_price", "f8")],
            order = 'C' # C: row-major, F: column-major
        )

        # row index (acts like pointer)
        self._curr_loc = 0 

    ### Backtest ###

    # Move to next time 
    def _bt_next(self, current_price):
        # move current balance to next time
        self._cash[self._curr_loc + 1] = self._cash[self._curr_loc]
        self._book[self._curr_loc + 1] = self._book[self._curr_loc]
        # update portfolio value
        self._pf_value[self._curr_loc + 1] = self._cash[self._curr_loc] + np.nansum(self._book["position"][self._curr_loc] * current_price)
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
        current_size = self._book["position"][self._curr_loc]
        self._book["position"][self._curr_loc] += delta_position
        self._book["position_price"][self._curr_loc] = ((current_size * self._book["position_price"][self._curr_loc]) + (delta_position * new_price)) / self._book["position"][self._curr_loc]


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
        return pd.DataFrame(self._book["position"], index=self._base_index, columns=self._base_columns)
    
    @property
    def position_price(self):
        return pd.DataFrame(self._book["position_price"], index=self._base_index, columns=self._base_columns)
    

# Historical 용도 FINAL
class Trader:

    def __init__(
            self,
            name: str,
            init_cash: float,
            signal_func: types.FunctionType,
            order_func: types.FunctionType,
            transaction_cost: tuple = (0,0),
            var_signal: dict = {},
            var_order: dict = {}
        ):

        # TODO inspect function parameters
        if list(inspect.signature(signal_func).parameters.keys()) != ["time","data","cash","book","var_signal"]:
            raise Exception("Signal function must have ordered parameters (time, data, cash, book, var_signal)")
        if list(inspect.signature(order_func).parameters.keys()) != ["time","data","cash","book","signal","var_order"]:
            raise Exception("Order function must have ordered parameters (time, data, cash, book, signal, var_order)")

        self.name = name
        self.init_cash = init_cash

        self.signal = signal_func
        self.order = order_func

        self.buy_fee = 1 + transaction_cost[0]
        self.sell_fee = 1 - transaction_cost[1]

        self.var_signal = var_signal
        self.var_order = var_order

        # TODO remove and create performance
        self.ledger = None

    def initiate(self, sender_index, sender_column):
        return LedgerBackTest(self.init_cash, sender_index, sender_column)
    

class Exchange:

    def __init__(self, sender: ExchangeSender, receiver: ExchangeReceiver, trader: Trader):
        self.sender = sender
        self.receiver = receiver
        self.trader = trader

    def run(self): # 여기서 for iter 돌면서 테스트 이루어짐
        ### Initiate Sender (Datafeed)
        sender_index = self.sender.data_index()
        trader_packet_generator = self.sender.datafeed()
        # TODO order book, execution modeling
        # receiver_packet_generator = self.RECEIVER.datafeed()

        ### Initiate Trader (Ledger)
        trader_ledger = self.trader.initiate(
            sender_index = self.sender.data_index(),
            sender_column = self.sender.data_column()
        )
        var_signal = self.trader.var_signal
        var_order = self.trader.var_order

        for _ in tqdm(range(len(sender_index)-1), desc=f'Backtesting [{self.trader.name}]...'):

            ### Get data packet
            time, trader_data = next(trader_packet_generator)
            current_price = trader_data["price"][-1]

            ### Generate Signals & Orders
            trader_signal, var_signal = self.trader.signal(
                                            time,                                
                                            trader_data,
                                            trader_ledger._bt_cash,
                                            trader_ledger._bt_book,
                                            var_signal
                                        )

            if trader_signal is not None:
                trader_orders, var_order = self.trader.order(
                                                time,
                                                trader_data,
                                                trader_ledger._bt_cash,
                                                trader_ledger._bt_book,
                                                trader_signal,
                                                var_order
                                            )

                if trader_orders is not None:
                    ### Execute Orders
                    # Sell orders first
                    sell_orders = np.where(trader_orders<0, trader_orders, 0)
                    delta_cashflow, delta_cash, delta_positions = self.receiver.sell_execute(
                                                                        current_price,
                                                                        sell_orders,
                                                                        trader_ledger._bt_book["position"],
                                                                        self.trader.sell_fee,
                                                                        orderbook_state=0
                                                                    )

                    # Update Ledger
                    trader_ledger._bt_add_cash(delta_cash)
                    trader_ledger._bt_add_position(delta_positions, current_price)
                    
                    # Buy orders
                    buy_orders = np.where(trader_orders>0, trader_orders, 0)
                    delta_cashflow, delta_cash, delta_positions = self.receiver.buy_execute(
                                                                        current_price,
                                                                        buy_orders,
                                                                        trader_ledger._bt_cash,
                                                                        self.trader.buy_fee,
                                                                        orderbook_state=0
                                                                    )
                    # Update Ledger
                    trader_ledger._bt_add_cash(delta_cash)
                    trader_ledger._bt_add_position(delta_positions, current_price)

            ### Move to next time
            trader_ledger._bt_next(current_price)
        
        # TODO Return Performance itself
        self.trader.ledger = trader_ledger