
from PySide2.QtCore import Signal as QSignal
import win32com.client

from alphacrafts.bkd.share.qt import ThreadData
from alphacrafts.bkd.creon.wrapper.parent import (SubscribeQtParent, SubscribeEventParent, SubscribeParent)

### CpTrade.CpTdUtil ###

class ObjCpTdUtil:

    """
    https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=284&seq=154&page=1&searchString=CpTdUtil&p=8841&v=8643&m=9505
    """

    obj = win32com.client.Dispatch("CpTrade.CpTdUtil")
    
    @classmethod
    def tradeinit(cls) -> int:
        """
        return status: -1:error, 0:success, 1:key error, 3:cancel
        """
        return cls.obj.TradeInit()
    
    @classmethod
    def get_account_number(cls) -> tuple:
        cls.tradeinit()
        return cls.obj.AccountNumber
    
    @classmethod
    def get_asset_list(cls, filter_num) -> tuple:
        cls.tradeinit()
        return cls.obj.GoodsList(cls.get_account_number()[0], filter_num)
    

### CpTrade.CpTd0311 ###

class ObjCpTd0311:

    """
    https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=284&seq=159
    """

    obj = win32com.client.Dispatch("CpTrade.CpTd0311")

    @classmethod
    def blockrequest(cls):
        cls.obj.BlockRequest()

    @classmethod
    def set_input(cls, input_dict:dict):
        """
        input_dict: {creon_type:value} \n
            0 - (string) 주문종류코드, 1:매도,2:매수 \n
            1 - (string) 계좌번호 \n
            2 - (string) 상품관리구분코드, '1':주식, '2':선물/옵션 \n
            3 - (string) 종목코드 \n
            4 - (long) 주문수량 \n
            5 - (long) 주문단가 \n
            7 - (string) 주문조건구분코드, 0:없음[default], 1:IOC, 2:FOK \n
            8 - (string) 주문호가구분코드, 01:보통[default], 02:임의, 03:시장가, 05:조건부지정가, 12:최유리지정가, 13:최우선지정가 \n
        """
        for key,value in input_dict.items():
            cls.obj.SetInputValue(key,value)

    @classmethod
    def get_header(cls, output_dict:dict) -> dict:
        """
        output_dict: {creon_type:value} \n
            0 - (string) 주문종류코드 \n
            1 - (string) 계좌번호 \n
            2 - (string) 상품관리구분코드 \n
            3 - (string) 종목코드 \n
            4 - (long) 주문수량 \n
            5 - (long) 주문단가 \n
            8 - (long) 주문번호 \n
            9 - (string) 계좌명 \n
            10 - (string) 종목명 \n
            12 - (string) 주문조건구분코드, 0:없음[default], 1:IOC, 2:FOK \n
            13 - (string) 주문호가구분코드, 01:보통[default], 02:임의, 03:시장가, 05:조건부지정가, 12:최유리지정가, 13:최우선지정가 \n
        """
        res_dict = {}
        for key,value in output_dict.items():
            res_dict[value] = cls.obj.GetHeaderValue(key)

        return res_dict


### CpTrade.ObjCpTd0313 ###

class ObjCpTd0313:

    """
    https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=291&seq=161
    """

    obj = win32com.client.Dispatch("CpTrade.CpTd0313")

    @classmethod
    def blockrequest(cls):
        cls.obj.BlockRequest()

    @classmethod
    def set_input(cls, input_dict:dict):
        """
        input_dict: {creon_type:value}
        """
        for key,value in input_dict.items():
            cls.obj.SetInputValue(key,value)

    @classmethod
    def get_header(cls, output_dict:dict) -> dict:
        """
        output_dict: {creon_type:value}
        """
        res_dict = {}
        for key,value in output_dict.items():
            res_dict[value] = cls.obj.GetHeaderValue(key)

        return res_dict


### Dscbo1.CpConclusion ###

class QtObjCpConclusion(SubscribeQtParent):

    evt_subscribe_data = QSignal(ThreadData)

    def __init__(self):
        super().__init__()

    def emit_data(self, data):
        super().emit_data(data)
    
class _ObjCpConclusionEvent(SubscribeEventParent):

    @classmethod
    def set_qtobj(cls, qtobj):
        super().set_qobj(qtobj)

    def set_obj(self, obj):
        super().set_obj(obj)
    
    def set_output(self, output_dict: dict):
        super().set_output(output_dict)
    
    def OnReceived(self):
        super().OnReceived()

class ObjCpConclusion(SubscribeParent):

    """
    https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=291&seq=155
    """

    obj = win32com.client.Dispatch("DsCbo1.CpConclusion")

    def __init__(self, qtobj):
        _ObjCpConclusionEvent.set_qobj(qtobj)

    @classmethod
    def subscribe(cls, subscribe_key:str, output_dict:dict):
        """
        subscribe_key (str): a key to handle subscription manage \n
        output_dict (dict): {creon_type:value} \n
            1 - (string) 계좌명 \n
            2 - (string) 종목명 \n
            3 - (long) 체결수량 \n
            4 - (long) 체결가격 \n
            5 - (long) 주문번호 \n
            6 - (long) 원주문번호 \n
            7 - (string) 계좌번호 \n
            8 - (string) 상품관리구분코드 \n 
            9 - (string) 종목코드 \n 
            12 - (string) 매매구분코드, 1:매도, 2:매수 \n 
            14 - (string) 체결구분코드, 1:체결, 2:확인, 3:거부, 4:접수 \n 
                신규 매수/매도 주문시 접수 or 거부 => 체결 \n
                정정,취소 주문시 정정,취소 확인 => 체결 \n
            15 - (string) 신용대출구분코드 \n 
            16 - (string) 정정취소구분코드, 1:정상, 2:정정, 3:취소 \n
            17 - (string) 현금신용대용구분코드, 1:현금, 2:신용, 3:선물대용, 4:공매도 \n
            18 - (string) 주문호가구분코드 \n
                01:보통, 02:임의, 03:시장가, 05:조건부지정가, 06:희망대량, 09:자사주, 10:스톡옵션자사주 
                11:금전신탁자사주, 12:최유리지정가, 13:최우선지정가, 51:임의시장가, 52:임의조건부지정가
                61:장중대량, 63:장중바스켓, 63:개시전종가, 67:개시전종가대량, 69:개시전시간외바스켓
                71:개시전금전신탁자사주, 72:개시전대량자기, 73:신고대량(전장시가), 77:시간외대량
                79금전신탁종가대량, 80:신고대량(종가) \n
            19 - (string) 주문조건구분코드, 0:없음, 1:IOC, 2:FOK \n
            20 - (string) 대출일 \n
            21 - (long) 장부가 \n
            22 - (long) 매도가능수량 \n
            23 - (long) 체결기준잔고수량 \n       
        """
        handler = win32com.client.WithEvents(cls.obj, _ObjCpConclusionEvent)
        handler.set_obj(cls.obj)
        handler.set_output(output_dict)
        cls.subscribe_dict[subscribe_key] = handler
        cls.obj.Subscribe()

    @classmethod
    def unsubscribe(cls, subscribe_key):
        cls.obj.Unsubscribe()
        del cls.subscribe_dict[subscribe_key]