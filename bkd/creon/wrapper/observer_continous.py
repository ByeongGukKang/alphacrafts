
from PySide2.QtCore import Signal as QSignal
import win32com.client

from alphacrafts.bkd.share.qt import ThreadData
from alphacrafts.bkd.creon.wrapper.parent import (SubscribeQtParent, SubscribeEventParent, SubscribeParent)


class QtObjStockCur(SubscribeQtParent):

    evt_subscribe_data = QSignal(ThreadData)

    def __init__(self):
        super().__init__()

    def emit_data(self, data):
        super().emit_data(data)

class _ObjStockCurEvent(SubscribeEventParent):

    @classmethod
    def set_qtobj(cls, qtobj):
        super().set_qobj(qtobj)

    def set_obj(self, obj):
        super().set_obj(obj)
    
    def set_output(self, output_dict: dict):
        super().set_output(output_dict)
    
    def OnReceived(self):
        super().OnReceived()

class ObjStockCur(SubscribeParent):

    """
    https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=284&seq=16
    """

    obj = win32com.client.Dispatch("DsCbo1.StockCur")

    def __init__(self, qtobj):
        _ObjStockCurEvent.set_qobj(qtobj)

    @classmethod
    def subscribe(cls, subscribe_key: str, input_dict: dict, output_dict:dict):
        """
        subscribe_key: a key to handle subscription manage
        input_dict: {creon_type:value}
        output_dict: {creon_type:value}
        """
        
        for key,value in input_dict.items():
            cls.obj.SetInputValue(key,value)
        handler = win32com.client.WithEvents(cls.obj, _ObjStockCurEvent)
        handler.set_obj(cls.obj)
        handler.set_output(output_dict)
        cls.subscribe_dict[subscribe_key] = handler
        cls.obj.Subscribe()

    @classmethod
    def unsubscribe(cls, input_dict:dict):
        super().unsubscribe(input_dict)
    