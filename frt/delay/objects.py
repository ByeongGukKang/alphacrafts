
from abc import ABCMeta, abstractmethod

import numba as nb
import numpy as np
from tqdm import tqdm

from alphacraft.frt.tools import to_nb_datetime
from alphacraft.frt.types import OrderResult, Performance


# Historical 용도 FINAL
# 작업 중
class ExchangeSender: # (metaclass=ABCMeta)

    def __init__(self, price_df):
        self.price_df = price_df

        self.frame_data = {}
        # TODO self.orderbook_data = {}
        # TODO self.nonframe_data = {}

    def auto_complie(
            self,
            window_size,
            window_adjust,
            target_datetime=to_nb_datetime,
            low_memory=True,
            drop_initials=True,
            check_columns=True
        ):
        self.low_memory = low_memory

        self.price_index = []
        self.symbols = self.price_df.columns
        self.number_of_symbols = len(self.symbols)

        # frame_data 무결성 체크
        if check_columns:
            for key, value in self.frame_data.items():
                # if not np.array_equal(tmp_value.index, value.index):
                #     raise Exception(f'Complie Error! Please check the index of data {key}')
                if not np.array_equal(self.price_df.columns, value.columns):
                    raise Exception(f'Complie Error! Please check the columns of data {key}')

        self.complied_data = {}

        if low_memory:
            for i, end_index in enumerate(tqdm(self.price_df.index, desc='Complie Data Packets')):
                ### Find the Start Index / If Length of Data is Too Short, Continue
                tmp_idx = i-(window_size*window_adjust) + window_adjust
                if tmp_idx <= 0:
                    if drop_initials:
                        continue
                start_index = self.price_df.index[max(0, tmp_idx)]

                # print(start_index)

                price_idx = target_datetime(end_index)
                self.price_index.append(price_idx)

                packet = {} # key:tuple 로 이루어짐 tuple(start_index,end_index)

                for key, value in self.frame_data.items():
                    
                    # sliced_index = value.loc[start_index:end_index,:].index
                    # (list(value.index).index(sliced_index[0]), list(value.index).index(sliced_index[-1])+1)
                    
                    # print((list(value.index).index(sliced_index[0]), list(value.index).index(sliced_index[-1])+1))

                    idx_start = np.where(value.index <= start_index)[0][-1]
                    idx_end = np.where(value.index <= end_index)[0][-1]
                    # print(value.index[idx_start], value.index[(list(value.index).index(sliced_index[0]))], start_index)
                    # print((idx_start, idx_end), (list(value.index).index(sliced_index[0]), list(value.index).index(sliced_index[-1])+1))
                    # print(start_index, end_index)
                    packet[key] = (idx_start, idx_end)
    
                ### Current Price to Calculate MtM Portfolio Value
                current_price = self.price_df.values[i]

                ### 
                self.complied_data[price_idx] = [self.symbols, current_price, packet]

        else:
            for i, end_index in enumerate(tqdm(self.price_df.index, desc='Complie Data Packets')):
                ### Find the Start Index / If Length of Data is Too Short, Continue
                tmp_idx = i-(window_size*window_adjust) + window_adjust
                if tmp_idx <= 0:
                    if drop_initials:
                        continue
                start_index = self.price_df.index[max(0, tmp_idx)]

                price_idx = target_datetime(end_index)
                self.price_index.append(price_idx)

                ### Create Empty Packet
                packet = nb.typed.Dict.empty(
                    key_type=nb.types.unicode_type,
                    value_type=nb.types.float64[:,:],
                )

                ### Fill the Packet with Data
                for key, value in self.frame_data.items():
                    packet[key] = value.loc[start_index:end_index,:].values.reshape(-1, value.shape[1])

                ### Current Price to Calculate MtM Portfolio Value
                current_price = self.price_df.values[i]

                ### 
                self.complied_data[price_idx] = [self.symbols, current_price, packet]

        
        self.number_of_index = len(self.price_index)

    def complie(self, packing_func):
        compile_buffer = {}
        pass

    # receiver = datafeed() 로 제너레이터 정의하고 next() 매번 호출해서 사용
    def datafeed(self):
        if self.low_memory:
            for time, symbol_packet in self.complied_data.items():
                packet = nb.typed.Dict.empty(
                    key_type=nb.types.unicode_type,
                    value_type=nb.types.float64[:,:],
                )

                for key, index_info in symbol_packet[2].items():
                    packet[key] = self.frame_data[key].values[index_info[0]:index_info[1]+1,:]
                yield time, symbol_packet[0], symbol_packet[1], packet # time, symbols, current_price, packet
                
        else:
            for time, symbol_packet in self.complied_data.items():
                yield time, symbol_packet[0], symbol_packet[1], symbol_packet[2] # time, symbols, current_price, packet


# Historical 용도 FINAL
class Trader(metaclass=ABCMeta):

    def __init__(self, name, init_cash, fee=(0,0)):
        self.name = name
        self.init_cash = init_cash

        self.buy_fee = 1+fee[0]
        self.sell_fee = 1-fee[1]

        self.price_index = None
        self.symbols = None
        self.__performance = None

    def initiate(self, cash_balance, pf_value, positions, cashflow, signals, orders):
        self.cash_balance = cash_balance
        self.pf_value = pf_value
        self.positions = positions
        self.cashflow = cashflow
        self.signals = signals
        self.orders = orders

        self.cash_balance[0] = self.init_cash
        self.pf_value[0] = self.init_cash

    def pnl_analysis(self, vol_days=252):
        if not self.price_index: # Exchange에서 run 하면 자동으로 index 생성됨
            raise NotImplementedError('Backtesting is not yet implemented')
        self.__performance = Performance(trader=self, vol_days=vol_days)

    @property
    def performance(self):
        if not self.__performance:
            raise NotImplementedError('pnl_analysis is not yet implemented')
        return self.__performance

    @staticmethod
    @abstractmethod
    def signal(time, packet_data): # packet_pile에 데이터가 들어옴, Signal만 Return
        
        return

    @staticmethod
    @abstractmethod
    def order(time, signals, current_price, cash_balance, pf_value, positions): # 위에서 생성한 signal들을 주문으로 바꿈

        return


# Histroical 용도 FINAL
class ExchangeReceiver:

    def __init__(self, execute_buy, execute_sell):
        self.execute_buy = execute_buy
        self.execute_sell = execute_sell

    def buy_execute(self, current_price, order, trader_cash, trader_buy_fee, orderbook_state): # 들어온 주문을 시장에서 집행
        return self.execute_buy(current_price, order, trader_cash, trader_buy_fee, orderbook_state)

    def sell_execute(self, current_price, order, trader_positions, trader_sell_fee, orderbook_state): # 들어온 주문을 시장에서 집행
        return self.execute_sell(current_price, order, trader_positions, trader_sell_fee, orderbook_state)


# Historical 용도 FINAL
class Exchange:

    def __init__(self, sender:ExchangeSender, receiver:ExchangeReceiver, trader:Trader):
        self.SENDER = sender
        self.RECEIVER = receiver

        self.TRADER = trader

    def run(self): # 여기서 for iter 돌면서 테스트 이루어짐
        SENDER_FEED_DATA = self.SENDER.datafeed()
        # ORDER_BOOK_DATA = self.RECEIVER.datafeed()

        # CASH_BALANCE = np.zeros((self.SENDER.number_of_index, 1))
        # PF_VALUE = np.zeros((self.SENDER.number_of_index, 1))
        # POSITIONS = np.zeros((self.SENDER.number_of_index, self.SENDER.number_of_symbols))

        self.TRADER.initiate(
            np.zeros((self.SENDER.number_of_index, 1)), # Cash Balance
            np.zeros((self.SENDER.number_of_index, 1)), # Portfolio Value
            np.zeros((self.SENDER.number_of_index, self.SENDER.number_of_symbols)), # Positions
            np.zeros((self.SENDER.number_of_index, self.SENDER.number_of_symbols)), # Cashflow
            np.zeros((self.SENDER.number_of_index, self.SENDER.number_of_symbols)), # Signals
            np.zeros((self.SENDER.number_of_index, self.SENDER.number_of_symbols)) # Orders
        )

        trader_orders = np.zeros(len(self.SENDER.symbols))
        for iter, iter_time in enumerate(tqdm(self.SENDER.price_index, desc=f'Backtesting [{self.TRADER.name}]')):
            ### Get Yesterday's Balance
            self.TRADER.cash_balance[iter] = self.TRADER.cash_balance[max(0, iter-1)]
            self.TRADER.positions[iter] = self.TRADER.positions[max(0, iter-1)]

            ### Get Data Packet
            packet_time, symbols, current_price, packet_data = next(SENDER_FEED_DATA)
            # orderbook_state = next(ORDER_BOOK_DATA) # 5 4 3 2 1 0 1 2 3 4 5

            ### Calcualte Portfolio Value
            self.TRADER.pf_value[iter] = self.TRADER.cash_balance[iter]  + np.nansum(self.TRADER.positions[iter] * current_price)

            ### Generate Signals & Orders
            trader_signal = self.TRADER.signal(packet_time, packet_data)
            trader_orders = self.TRADER.order(
                                packet_time,
                                trader_signal,
                                current_price,
                                self.TRADER.cash_balance[:iter+1],
                                self.TRADER.pf_value[:iter+1],
                                self.TRADER.positions[:iter+1,:]
                            )

            #print()
            #print(self.TRADER.positions[iter])
            ### Execute Orders
            sell_orders = np.where(trader_orders<0, trader_orders, 0)
            #print(iter_time.hour, sell_orders)
            delta_cashflow, delta_cash, delta_positions = self.RECEIVER.sell_execute(
                                                                current_price,
                                                                sell_orders,
                                                                self.TRADER.positions[iter],
                                                                self.TRADER.sell_fee,
                                                                orderbook_state=0
                                                            )
            # print(delta_cashflow)
            # print(delta_cash)
            # print(delta_positions)
            self.TRADER.cash_balance[iter] = self.TRADER.cash_balance[max(0, iter-1)] + delta_cash
            self.TRADER.cashflow[iter] = self.TRADER.cashflow[max(0, iter-1)] + delta_cashflow
            self.TRADER.positions[iter] = self.TRADER.positions[max(0, iter-1)] + delta_positions
            
            buy_orders = np.where(trader_orders>0, trader_orders, 0)
            #print(iter_time.hour, buy_orders)
            delta_cashflow, delta_cash, delta_positions = self.RECEIVER.buy_execute(
                                                                current_price,
                                                                buy_orders,
                                                                self.TRADER.cash_balance[iter],
                                                                self.TRADER.buy_fee,
                                                                orderbook_state=0
                                                            )
            self.TRADER.cash_balance[iter] = self.TRADER.cash_balance[iter] + delta_cash
            self.TRADER.cashflow[iter] = self.TRADER.cashflow[iter] + delta_cashflow
            self.TRADER.positions[iter] = self.TRADER.positions[iter] + delta_positions

            self.TRADER.signals[iter] = trader_signal
            self.TRADER.orders[iter] = trader_orders

        # Performance 보여줄 index와 symbol column
        self.TRADER.price_index = self.SENDER.price_index
        self.TRADER.symbols = self.SENDER.symbols

    def run_lagged(self): # 여기서 for iter 돌면서 테스트 이루어짐
        SENDER_FEED_DATA = self.SENDER.datafeed()
        # ORDER_BOOK_DATA = self.RECEIVER.datafeed()

        # CASH_BALANCE = np.zeros((self.SENDER.number_of_index, 1))
        # PF_VALUE = np.zeros((self.SENDER.number_of_index, 1))
        # POSITIONS = np.zeros((self.SENDER.number_of_index, self.SENDER.number_of_symbols))

        self.TRADER.initiate(
            np.zeros((self.SENDER.number_of_index, 1)), # Cash Balance
            np.zeros((self.SENDER.number_of_index, 1)), # Portfolio Value
            np.zeros((self.SENDER.number_of_index, self.SENDER.number_of_symbols)), # Positions
            np.zeros((self.SENDER.number_of_index, self.SENDER.number_of_symbols)), # Cashflow
            np.zeros((self.SENDER.number_of_index, self.SENDER.number_of_symbols)), # Signals
            np.zeros((self.SENDER.number_of_index, self.SENDER.number_of_symbols)) # Orders
        )

        trader_orders = np.zeros(len(self.SENDER.symbols))
        for iter, iter_time in enumerate(tqdm(self.SENDER.price_index, desc=f'Backtesting [{self.TRADER.name}]')):
            ### Get Yesterday's Balance
            self.TRADER.cash_balance[iter] = self.TRADER.cash_balance[max(0, iter-1)]
            self.TRADER.positions[iter] = self.TRADER.positions[max(0, iter-1)]

            ### Get Data Packet
            packet_time, symbols, current_price, packet_data = next(SENDER_FEED_DATA)
            # orderbook_state = next(ORDER_BOOK_DATA) # 5 4 3 2 1 0 1 2 3 4 5

            #print()
            #print(self.TRADER.positions[iter])
            ### Execute Orders
            sell_orders = np.where(trader_orders<0, trader_orders, 0)
            #print(iter_time.hour, sell_orders)
            delta_cashflow, delta_cash, delta_positions = self.RECEIVER.sell_execute(
                                                                current_price,
                                                                sell_orders,
                                                                self.TRADER.positions[iter],
                                                                self.TRADER.sell_fee,
                                                                orderbook_state=0
                                                            )
            # print(delta_cashflow)
            # print(delta_cash)
            # print(delta_positions)
            self.TRADER.cash_balance[iter] = self.TRADER.cash_balance[max(0, iter-1)] + delta_cash
            self.TRADER.cashflow[iter] = delta_cashflow
            self.TRADER.positions[iter] = self.TRADER.positions[max(0, iter-1)] + delta_positions
            
            buy_orders = np.where(trader_orders>0, trader_orders, 0)
            #print(iter_time.hour, buy_orders)
            delta_cashflow, delta_cash, delta_positions = self.RECEIVER.buy_execute(
                                                                current_price,
                                                                buy_orders,
                                                                self.TRADER.cash_balance[iter],
                                                                self.TRADER.buy_fee,
                                                                orderbook_state=0
                                                            )
            self.TRADER.cash_balance[iter] = self.TRADER.cash_balance[iter] + delta_cash
            self.TRADER.cashflow[iter] = delta_cashflow
            self.TRADER.positions[iter] = self.TRADER.positions[iter] + delta_positions

            ### Calcualte Portfolio Value
            self.TRADER.pf_value[iter] = self.TRADER.cash_balance[iter]  + np.nansum(self.TRADER.positions[iter] * current_price)

            ### Generate Signals & Orders
            trader_signal = self.TRADER.signal(packet_time, packet_data)
            trader_orders = self.TRADER.order(
                                packet_time,
                                trader_signal,
                                current_price,
                                self.TRADER.cash_balance[:iter+1],
                                self.TRADER.pf_value[:iter+1],
                                self.TRADER.positions[:iter+1,:]
                            )

            self.TRADER.signals[iter] = trader_signal
            self.TRADER.orders[iter] = trader_orders

        # Performance 보여줄 index와 symbol column
        self.TRADER.price_index = self.SENDER.price_index
        self.TRADER.symbols = self.SENDER.symbols