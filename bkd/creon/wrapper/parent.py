
from abc import (ABCMeta, abstractmethod, abstractclassmethod)
from datetime import datetime

import numpy as np
import pythoncom
from PyQt5.QtCore import QObject
import win32com.client
import win32event

from alphacrafts.bkd.share.qt import ThreadData


class RequestEventParent(metaclass=ABCMeta):

    """
    Define \n
    def OnReceived(self): \n
        win32event.SetEvent(CpRequestEventReceiverParent.StopEvent) \n
        return \n
    """

    @abstractmethod
    def OnReceived(self):
        pass


class RequestEventReceiverParent:
    """
    Define \n
    StopEvent = win32event.CreateEvent(None, 0, 0, None) \n
    """
    def __init__(self, objEvent, objEventHandler):
        self.obj = objEvent
        handler = win32com.client.WithEvents(self.obj, objEventHandler)

    def message_pump(self, timeout):
        waitables = [self.StopEvent]
        while True:
            rc = win32event.MsgWaitForMultipleObjects(
                waitables,
                0,  # Wait for all = false, so it waits for anyone
                timeout, # (or win32event.INFINITE)
                win32event.QS_ALLEVENTS)  # Accepts all input
    
            if rc == win32event.WAIT_OBJECT_0:
                break
    
            elif rc == win32event.WAIT_OBJECT_0 + len(waitables):
                if pythoncom.PumpWaitingMessages():
                    break

            elif rc == win32event.WAIT_TIMEOUT:
                print('timeout')
                return
            
            else:
                print('exception')
                raise RuntimeError("unexpected win32wait return value")
            

class SubscribeQtParent(QObject):

    """
    Define
    evt_subscribe_data = pyqtSignal(ThreadData)
    """
    
    def __init__(self):
        super().__init__()

    def emit_data(self, data):
        self.evt_subscribe_data.emit(data)


class SubscribeEventParent(metaclass=ABCMeta):

    result_dict = {}

    @classmethod
    def set_qobj(cls, qtobj):
        cls.qtobj = qtobj

    def set_obj(self, obj):
        self.obj = obj

    def set_output(self, output_dict:dict):
        """
        output_dict: {creon_type:name}
        """
        self.output_dict = output_dict

    def OnReceived(self):
        """
        Define
        self.evt_subscribe_data.emit(thread_data)
        """
        res_dict = {}
        for key,value in self.output_dict.items():
            res_dict[value] = self.obj.GetHeaderValue(key)

        self.dict = res_dict

        self.qtobj.emit_data(ThreadData(None, res_dict, None, datetime.now()))


class SubscribeParent(metaclass=ABCMeta):

    subscribe_dict = {}
    """
    obj = win32com.client.Dispatch("CHANGE HERE") 
    """

    def __init__(self):
        pass

    @abstractclassmethod
    def subscribe(cls, key:str, input_dict:dict):
        """
        Define as classmethod \n
        for key,value in input_dict.items(): \n
            cls.obj.SetInputValue(key,value) \n
        handler = win32com.client.WithEvents(cls.obj, "CHANGE HERE") \n
        handler.set_obj(cls.obj) \n
        cls.subscribe_dict[key] = handler \n
        cls.obj.Subscribe() \n
        """
        pass

    @classmethod
    def unsubscribe(cls, subscribe_key, input_dict):
        """
        Define as classmethod
        """
        for key,value in input_dict.items():
            cls.obj.SetInputValue(key,value)
        cls.obj.Unsubscribe()
        del cls.subscribe_dict[subscribe_key]

            
class DiscreteObserverParent(metaclass=ABCMeta):

    """
    Define \n
    def __init__():
        self.obj = win32com.client.Dispatch("Dscbo1.StockMst2") \n
        self.obj_rqrev = __objStockMst2RqRev(self.obj, __objStockMst2RqEvt) \n
        self.__number_of_data = "THIS CHANGES EVERYTIME !!!" \n
        self.__number_of_data_loc = "THIS CHANGES EVERYTIME !!!" \n
     \n
    def set_input(self, input_list:list): \n
        ''' \n
        # input_list: [stock_code] (Max:110) \n
        ''' \n
        submit_code_list = str(input_list)[1:-1].replace("'",'').replace(" ",'') \n
        self.obj.SetInputValue(0, submit_code_list) \n
     \n
    def set_input(self, input_dict:dict): \n
        ''' \n
        # input_dict: {creon_type:value} \n
        ''' \n
        for key,value in input_dict.items(): \n
            cls.obj.SetInputValue(key,value) \n
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def set_input(self):
        pass

    def request(self, timeout=5000):
        self.obj.Request()
        self.obj_rqrev.message_pump(timeout)

    def blockrequest(self):
        self.obj.BlockRequest()

    def get_header(self, output_dict:dict) -> dict:
        """
        output_dict: {creon_type:name}
        """
        res_dict = {}
        for key,value in output_dict.items():
            res_dict[value] = self.obj.GetHeaderValue(key)

        return res_dict

    def get_data(self, output_dict:dict, to_numpy: bool=True) -> dict:
        """
        output_dict: {creon_type:name}
        """
        number_of_data = self.obj.GetHeaderValue(self._number_of_data_loc)

        res_dict = {}
        for key,value in output_dict.items():
            res_dict[value] = []

        for key,value in output_dict.items():
            for i in range(number_of_data):
                res_dict[value].append(self.obj.GetDataValue(key,i))
        
        if to_numpy:
            for key,value in res_dict.items():
                res_dict[key] = np.array(value)

        return res_dict