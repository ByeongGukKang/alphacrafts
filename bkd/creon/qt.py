
from collections import deque
from datetime import datetime, timedelta

import numpy as np
from PyQt5.QtCore import (pyqtSignal, pyqtSlot, QObject, QThread)
from PyQt5.QtTest import QTest

from alphacrafts.bkd.share.qt import ThreadData
from alphacrafts.bkd.creon.wrapper.account import ObjCpCybos
from alphacrafts.bkd.creon.wrapper.trade import (ObjCpTdUtil, ObjCpTd0311)


# Balance Manager
class QtBalanceManager(QObject):

    def __init__(self, sorted_code_list):
        super().__init__()

        self._cash = 0
        self._pf_value = 0
        self._trade_condition = {}

        dt = np.dtype([('code', 'U20'), ('position', int)])
        self._position = np.array([(code, 0) for code in sorted_code_list], dtype=dt)
        
    @property
    def cash(self):
        return self._cash
    
    def add_cash(self, cashflow):
        """
        cashflow is added to current cash balance 
        """
        self._cash += cashflow

    def update_cash(self, cash):
        """
        cashflow is updated
        """
        self._cash = cash

    @property
    def position(self):
        return self._position

    def add_position(self, code, position_change):
        """
        position_change is added to current position
        if delte_zero==True, code with position of 0 is removed
        """
        self._position['position'][np.argwhere(self._position['code'] == code)] += position_change

    def update_position(self, code, new_position):
        """
        position is updated 
        """
        self._position['position'][np.argwhere(self._position['code'] == code)] = new_position

    # TODO Update 필요 Structed Array 맞춰서
    def get_position(self, code_array):
        raise NotImplementedError('get_position is not implemented yet')
        position_array = np.zeros_like(code_array)

        for idx, value in enumerate(np.isin(code_array, np.array(list(self.position.values())))):
            if value:
                position_array[idx] = self.position[code_array[idx]]

        return position_array
    
    @property
    def pf_value(self):
        return self._pf_value

    def update_pf_value(self, pf_value):
        """
        pf_value is updated 
        """
        self._pf_value = pf_value
    
    def calculate_pf_value(self, code_array, price_array):
        """
        pf_value = cash balance + sum(position*price)
        """
        position_array = self.get_position(code_array)

        self._pf_value = self.cash + np.nansum(position_array*price_array)

    @pyqtSlot(ThreadData)
    def update_order_result(self, evt_order_result):
        pass

    @property
    def trade_condition(self):
        return self._trade_condition
    
    def set_trade_condition(self, key, condition):
        self._trade_condition[key] = condition


# Market Observation
class QtObserverDiscrete(QObject):
    
    evt_market_data = pyqtSignal(ThreadData)

    def __init__(self, obj_observer_discrete):
        super().__init__()
        # Log Processing Time
        self._log_time = lambda: None

        # Data Request Frequency
        self.set_freq()
        self._last_observation_time = datetime.now() - self._freq
        # Header Setting
        self._get_header = False
        # Creon Object
        self.creon_obj = obj_observer_discrete

    # Log Processing Time
    def set_log_time(self, log_time):
        if log_time:
            self._log_time = datetime.now
        else:
            self._log_time = lambda: None

    # Data Request Freqency
    def set_freq(self, deltatime:timedelta=timedelta(seconds=1)):
        self._freq = deltatime
        self._timeout = deltatime.total_seconds() * 1000 # Convert second to millisecond

    # Creon Object 
    def set_input(self, **kargs):
        self._creon_obj_input = kargs

    def set_header_output(self, get_header=True, **kargs):
        self._get_header = get_header
        self._creon_obj_header_output = kargs

    def set_data_output(self, **kargs):
        self._creon_obj_data_output = kargs

    # Data request is triggered by Local Clock
    @pyqtSlot(datetime)
    def get_data(self, evt_local_time):
        if (evt_local_time - self._last_observation_time) >= self._freq:
            # Update Time
            self._last_observation_time = evt_local_time

            # Send Request to Creon
            if ObjCpCybos.get_remain_count(1) != 0:
                pass
            else:
                QTest.qWait(ObjCpCybos.get_refresh_time(1))
            
            self.creon_obj.set_input(**self._creon_obj_input)
            self.creon_obj.request(self._timeout) # Timeout as request frequency

            # Get Header & Data From Creon Object
            res_header = None
            if self._get_header:
                res_header = self.creon_obj.get_header(**self._creon_obj_header_output)
            res_data = self.creon_obj.get_data(**self._creon_obj_data_output)

            # Emit Data to Other Threads 
            self.evt_market_data.emit(ThreadData(res_header, res_data, evt_local_time, self._log_time()))


# Order Optimizer
class QtOrderOptimizer(QThread):

    evt_order_optimize_data = pyqtSignal(ThreadData)

    def __init__(self, order_optimization_function):
        super().__init__()
        # Log Processing Time
        self._log_time = lambda: None

        # Order Optimization Funcion
        self._optimization_function = order_optimization_function
        # Additional Data
        self._additional_data = None
        # Market(Orderbook) Data
        self._orderbook_data = None

        # Order Queue
        self._order_queue = deque()
        # Order Result Wait
        self._order_result_waitline = {}

        # Order Type Mapping
        self._order_type_mapping = {1:1, 2:-1}

    # Log Processing Time
    def set_log_time(self, log_time):
        if log_time:
            self._log_time = datetime.now
        else:
            self._log_time = lambda: None

    # Additional Data
    def set_additional_data(self, additional_data):
        self._additional_data = additional_data

    # Send Order
    def run(self):
        order_submit_queue = deque()
        while True:
            if len(self._order_queue) > 0:
                # final_order = {
                #     0: 1-매도, 2-매수
                #     1: 계좌번호
                #     2: 상품관리구분코드
                #     3: 종목코드
                #     4: 주문수량
                #     5: 주문단가
                #     7: 주문조건구분코드
                #     8: 주문호가구분코드
                # }
                start_time = self._log_time()
                is_realistic_order, final_order = self._optimization_function(self._order_queue.popleft(), self._orderbook_data)
                
                order_submit_queue.append(ThreadData(is_realistic_order, final_order, start_time, self._log_time()))

                # self.evt_order_optimize_data.emit(ThreadData(is_realistic_order, final_order, start_time, self._log_time()))
            else:
                self.evt_order_optimize_data.emit(ThreadData(False, order_submit_queue, None, None))
                self.quit()
                break

    # Market(OrderBook) Data
    @pyqtSlot(ThreadData)
    def update_orderbook(self, evt_orderbook_data):
        self._orderbook_data = evt_orderbook_data

    # Update Order Request
    @pyqtSlot(ThreadData)
    def optimize_order(self, evt_order_raw_data):
        # Split Array Order Info
        order_loc = np.where(evt_order_raw_data.data['order'] != 0)[0]
        for idx in order_loc:
            self._order_queue.append((
                evt_order_raw_data.header[idx],
                evt_order_raw_data.data['order'][idx],
                evt_order_raw_data.data['target_price'][idx]
            ))
        self.start()

    @pyqtSlot(ThreadData)
    def manage_order_result(self, evt_order_result):
        if evt_order_result.data['conclusion_type'] == '4':
            self._order_result_waitline[evt_order_result.data['order_number']] = (
                evt_order_result.data['code'], # Stock Code
                evt_order_result.data['size'], # Order Size
                evt_order_result.data['price'], # Order Price
                evt_order_result.data['order_type'] # Order Type
            )

        elif evt_order_result.data['conclusion_type'] == '1':
            print('order_result_waitline', self._order_result_waitline) 
            if (self._order_result_waitline[evt_order_result.data['order_number']][1] - evt_order_result.data['size']) == 0:
                del self._order_result_waitline[evt_order_result.data['order_number']]

            elif evt_order_result.data['order_condition_type'] != '0':
                self._order_queue.append((
                    evt_order_result.data['code'],
                    self._order_type_mapping[evt_order_result.data['order_type']] * (self._order_result_waitline[evt_order_result.data['order_number']] - evt_order_result.data['size']),
                    evt_order_result.data['price']
                ))

                del self._order_result_waitline[evt_order_result.data['order_number']]

        else:
            print('manage_order_result','Order Error')


class QtOrderExecutor(QThread):

    evt_time_stop = pyqtSignal(bool)

    def __init__(self, asset_list_fiter_num=1):
        super().__init__()

        ObjCpTdUtil.tradeinit()
        self._account_number = ObjCpTdUtil.get_account_number()
        self._asset_number = ObjCpTdUtil.get_asset_list(asset_list_fiter_num)

        self._order_queue = deque()
        self._freezer = False

    @pyqtSlot(ThreadData)
    def execute_order(self, evt_order_optimize_data):
        self.evt_time_stop.emit(True)
        for order in evt_order_optimize_data.data:
            if order.header == True:

                input_dict = order.data
                input_dict[1] = self._account_number[0]
                input_dict[2] = self._asset_number[0]

                if ObjCpCybos.get_remain_count(0) != 0:
                    pass
                else:
                    print('wait...')
                    QTest.qWait(ObjCpCybos.get_refresh_time(1))

                ObjCpTdUtil.tradeinit()
                ObjCpTd0311.set_input(input_dict)
                ObjCpTd0311.blockrequest()

            else:
                continue

        self.evt_time_stop.emit(False)

    # @pyqtSlot(ThreadData)
    # def execute_order(self, evt_order_optimize_data):
    #     if evt_order_optimize_data.header: # Send Realistic Order Only
    #         input_dict = evt_order_optimize_data.data
    #         input_dict[1] = self._account_number[0]
    #         input_dict[2] = self._asset_number[0]

    #         self._order_queue.append(input_dict)
        
    # def run(self):
    #     for order in self._order_queue:
    #         if ObjCpCybos.get_remain_count(0) != 0:
    #             pass
    #         else:
    #             QTest.qWait(ObjCpCybos.get_refresh_time(1))

    #         ObjCpTdUtil.tradeinit()
    #         ObjCpTd0311.set_input(self._order_queue.popleft())

    #         self.evt_time_stop.emit(True)
    #         ObjCpTd0311.blockrequest()

    #     while True:
    #         if not self._freezer:
    #             if len(self._order_queue) != 0:
    #                 print(self._order_queue)
                
    #                 self._freezer = True
                    
    #                 if ObjCpCybos.get_remain_count(0) != 0:
    #                     pass
    #                 else:
    #                     QTest.qWait(ObjCpCybos.get_refresh_time(1))

                    
    #                 order = self._order_queue.popleft()
    #                 ObjCpTdUtil.tradeinit()
    #                 ObjCpTd0311.set_input(order)

    #                 self.evt_time_stop.emit(True)
    #                 ObjCpTd0311.blockrequest()
    #                 print('success')
                    
    #                 self._freezer = False
    #         else:
    #             self.evt_time_stop.emit(False)
    #             QTest.qWait(500)