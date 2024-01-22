
from datetime import datetime
from typing import Callable

import numpy as np

from PySide2.QtCore import QObject, QThread, QEventLoop, QTimer
from PySide2.QtCore import Signal as QSignal
from PySide2.QtCore import Slot as QSlot


# ThreadData to communicate between QObjects
class ThreadData:

    def __init__(self, header, data, start_time=None, end_time=None):
        self._header = header
        self._data = data
        self._start_time = start_time
        self._end_time = end_time

    def __str__(self):
        return str({
            'header': str(self.header),
            'data': str(self.data),
            'start_time': str(self.start_time),
            'end_time': str(self.end_time)
        })

    @property
    def header(self):
        return self._header

    @property
    def data(self):
        return self._data
    
    @property
    def start_time(self):
        return self._start_time
    
    @property
    def end_time(self):
        return self._end_time


# Local Clock
class QtLocalClock(QThread):

    evt_local_time = QSignal(datetime)

    def __init__(self):
        """
        Default clock frequency is 500ms
        """
        super().__init__()
        self.freq = 500
        self.go_flag = True

    def set_freq(self, msec:int=500):
        """
        msec (int): clock frequency in millisecond
        """
        self.freq = msec

    def run(self):
        while self.go_flag:
            self.evt_local_time.emit(datetime.now())
            # Replacement for PyQt5.QTest.qWait(self.freq)
            loop = QEventLoop()
            QTimer.singleShot(self.freq, loop.quit)
            loop.exec_()

    @QSlot(bool)
    def manage_time_stop(self, evt_time_stop):
        self.time_stop = evt_time_stop

# QtStarter
class QtStarter(QObject):

    def __init__(self, start_time: datetime, start_jobs: list):
        super().__init__()
        self._start_time = start_time
        self._start_jobs = start_jobs

    @QSlot(datetime)
    def check_time(self, evt_local_time):
        if evt_local_time > self._start_time:
            for job in self._start_jobs:
                job()

# QtEnder
class QtEnder(QObject):

    def __init__(self, end_time: datetime, end_jobs: list):
        super().__init__()
        self._end_time = end_time
        self._end_jobs = end_jobs

    @QSlot(datetime)
    def check_time(self, evt_local_time):
        if evt_local_time > self._end_time:
            for job in self._end_jobs:
                job()

# Messanger
class QtMessanger(QObject):

    def __init__(self):
        super().__init__()
        self.freq = 5000

    @QSlot(ThreadData)
    def message_data(self, evt_market_data):
        pass

    @QSlot(ThreadData)
    def message_preprocess(self, evt_preprocessed_data):
        pass

    @QSlot(ThreadData)
    def message_signal(self, evt_signal_data):
        pass

    @QSlot(ThreadData)
    def message_order_raw(self, evt_order_raw_data):
        pass

    @QSlot(ThreadData)
    def message_order_optimize(self, evt_order_optimize_data):
        pass

    @QSlot(ThreadData)
    def message_order_conclusion(self, evt_order_conclusion_data):
        pass

    @QSlot(ThreadData)
    def message_userdef(self, evt_userdef_data):
        pass


# Message Command
class QtMessageCommand(QThread):

    def __init__(self):
        super().__init__()
        self.freq = 5000

    def set_freq(self, ms):
        self.freq = ms

    # Continouly Read Slack Messages
    def run(self):
        while True:

            # TODO Write Codes for Slack Bot

            QTest.qWait(self.freq)
            pass


# Log Manager
# TODO Save Log On PostgreSQL DB or Local File.
class QtLogger(QObject):

    def __init__(self):
        super().__init__()
    
    @QSlot(ThreadData)
    def log_data(self, evt_market_data):
        pass

    @QSlot(ThreadData)
    def log_preprocess(self, evt_preprocessed_data):
        pass

    @QSlot(ThreadData)
    def log_signal(self, evt_signal_data):
        pass

    @QSlot(ThreadData)
    def log_order_raw(self, evt_order_raw_data):
        pass

    @QSlot(ThreadData)
    def log_order_optimize(self, evt_order_optimize_data):
        pass

    @QSlot(ThreadData)
    def log_order_conclusion(self, evt_order_conclusion_data):
        pass

    @QSlot(ThreadData)
    def log_userdef(self, evt_userdef_data):
        pass


# Data Converter
class QtDataConverter(QObject):

    """
    Don't FORGET TO ALLOCATE self._memory & update_data Slot!!! \n 

    Define \n
    @QSlot(ThreadData) \n
    def update_memory[CHANGE HERE MAKE (MAKE IT SAME AS MEMORY DICT KEY)](self, evt_market_data): \n
        self._memory[CHANGE HERE] = self._update_function(evt_market_data)
        self.check_memory()
    
    """

    evt_preprocessed_data = QSignal(ThreadData)

    def __init__(self, number_of_memory_slot: int, preprocess_func: Callable, memory_condition_func: Callable, memory_update_func: Callable=lambda x:x):
        super().__init__()
        # Log Processing Time
        self._log_time = lambda: None

        # Data Preprocessig Funcion
        self._preprocess_function = preprocess_func
        # Memory For Stacking Data
        self._clear_memory = True # Clear memory at every emission
        self._memory_slot_count = number_of_memory_slot
        self._memory_slot = np.zeros(self._memory_slot_count)
        self._memory = {}
        self._memory_condition_funcion = memory_condition_func # Check memory data emit condition
        # Memory Update Function 
        self._memory_update_function = memory_update_func
        # Additional Data
        self._additional_data = None

    # Log Processing Time
    def set_log_time(self, log_time: bool):
        if log_time:
            self._log_time = datetime.now
        else:
            self._log_time = lambda: None

    # Additional Data
    def set_additional_data(self, additional_data):
        self._additional_data = additional_data

    # Clear Memory
    def set_clear_memory(self, clear_memory: bool=True):
        self._clear_memory = clear_memory

    # # Check Memory Condition
    # def run(self):
    #     while self._memory_condition_funcion(self._memory):
    #         # Start Time
    #         start_time = self._log_time
    #         # Preprocess Data
    #         res_preprocess = self._preprocess_function(self._memory, self._additional_data)
    #         # Emit Preprocessed Data to Other Threads
    #         self.evt_preprocessed_data.emit(ThreadData(None, res_preprocess, start_time, self._log_time))
    #         # Clear Memory
    #         if self._clear_memory:
    #             self._memory = self._memory_initial

    # Check Memory Condition
    def check_memory(self):
        if self._memory_condition_funcion(self._memory_slot, self._memory):
            # Start Time
            start_time = self._log_time()
            # Preprocess Data
            res_preprocess = self._preprocess_function(self._memory, self._additional_data)
            # Clear Memory
            # if self._clear_memory:
            #     self._memory = {}
            self._memory_slot = np.zeros(self._memory_slot_count)

            # Emit Preprocessed Data to Other Threads
            self.evt_preprocessed_data.emit(ThreadData(None, res_preprocess, start_time, self._log_time()))
        else:
            pass

    # Update Data
    @QSlot(ThreadData)
    def update_memory(self, evt_market_data):
        self._memory = self._memory_update_function(evt_market_data)
        self.check_memory()

# Signal Generator
class QtSignalGenerator(QObject):

    evt_signal_data = QSignal(ThreadData)

    def __init__(self, signal_func: Callable):
        super().__init__()
        # Log Processing Time
        self._log_time = lambda: None

        # Signal Generating Funcion
        self._signal_function = signal_func

    # Log Processing Time
    def set_log_time(self, log_time: bool):
        if log_time:
            self._log_time = datetime.now
        else:
            self._log_time = lambda: None
    
    # Generate Signal
    @QSlot(ThreadData)
    def generate_signal(self, evt_preprocessed_data):
        # Start Time
        start_time = self._log_time()
        # Generate Signal
        res_signal = self._signal_function(evt_preprocessed_data.data)
        # Emit Signal to Other Threads
        self.evt_signal_data.emit(ThreadData(evt_preprocessed_data.data['code'], res_signal, start_time, datetime.now()))


# Raw Order Generator 
class QtOrderGenerator(QObject):

    """
    Returns two arrays
        1d array or stock code
        1d array of orders
    """

    evt_order_raw_data = QSignal(ThreadData)

    def __init__(self, qt_balance_manager_instance, signal_to_order_func: Callable):
        super().__init__()
        # Log Processing Time
        self._log_time = lambda: None

        # Balance/Position Managing Object
        self._balance_manager = qt_balance_manager_instance
        # Singal to Order Funcion
        self._order_function = signal_to_order_func
        # Additional Data
        self._additional_data = None

    # Log Processing Time
    def set_log_time(self, log_time: bool):
        if log_time:
            self._log_time = datetime.now
        else:
            self._log_time = lambda: None

    # Additional Data
    def set_additional_data(self, additional_data):
        self._additional_data = additional_data

    # Returns three array, one for stock_code, one for order, and one for target price.
    # res_order_raw = dict{'code':[], 'order':[], 'target_price':[]}
    @QSlot(ThreadData)
    def generate_order(self, evt_signal_data):
        # Start Time
        start_time = self._log_time()
        # Generate Order
        res_order_raw = self._order_function(self._balance_manager, evt_signal_data.data, self._additional_data)
        # Emit Order to Other Threads
        self.evt_order_raw_data.emit(ThreadData(evt_signal_data.header, res_order_raw, start_time, self._log_time()))

