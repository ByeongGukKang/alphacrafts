
import numpy as np
import win32com.client
import win32event

from alphacrafts.bkd.creon.wrapper.parent import (RequestEventParent, RequestEventReceiverParent, DiscreteObserverParent)


### Dscbo1.StockMstM ###

class _ObjStockMstRqRev(RequestEventReceiverParent):

    StopEvent = win32event.CreateEvent(None, 0, 0, None)

    def __init__(self, objEvent, objEventHandler):
        super().__init__(objEvent, objEventHandler)

    def message_pump(self, timeout):
        return super().message_pump(timeout)
    
class _ObjStockMstRqEvt(RequestEventParent):
    def OnReceived(self):
        win32event.SetEvent(_ObjStockMstRqRev.StopEvent)
        return
    
class ObjStockMst(DiscreteObserverParent):

    """
    https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=288&seq=3&page=3&searchString=&p=&v=&m=
    """

    def __init__(self):
        self.obj = win32com.client.Dispatch("Dscbo1.StockMstM")
        self.obj_rqrev = _ObjStockMstRqRev(self.obj, _ObjStockMstRqEvt)

    def set_input(self, input_value:str):
        """
        input_value: stock_code
        """
        self.obj.SetInputValue(0, input_value)
        self._code = input_value

    def request(self, timeout=5000):
        return super().request(timeout)
    
    def blockrequest(self):
        return super().blockrequest()
    
    def get_header(self, output_dict: dict) -> dict:
        return super().get_header(output_dict)

    def get_data(self, output_value: int=10, to_numpy: bool=True) -> dict:
        """
        output_value: order_book_level (1~10)
        """
        res_dict = {}
        res_dict[self._code] = {
            'ask_price':[],'bid_price':[],
            'ask_size':[],'bid_size':[],
            'ask_relative':[],'bid_relative':[]
        }
        for creon_type, key in enumerate(res_dict.keys()):
            for i in range(output_value):
                res_dict[key].append(self.obj.GetDataValue(creon_type, i))

        if to_numpy:
            for key,value in res_dict.items():
                res_dict[key] = np.array([value])

        return res_dict


### Dscbo1.StockMstM ###

class _ObjStockMstMRqRev(RequestEventReceiverParent):

    StopEvent = win32event.CreateEvent(None, 0, 0, None)

    def __init__(self, objEvent, objEventHandler):
        super().__init__(objEvent, objEventHandler)

    def message_pump(self, timeout):
        return super().message_pump(timeout)
    
class _ObjStockMstMRqEvt(RequestEventParent):
    def OnReceived(self):
        win32event.SetEvent(_ObjStockMstMRqRev.StopEvent)
        return

class ObjStockMstM(DiscreteObserverParent):

    """
    https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=288&seq=14&page=3&searchString=&p=&v=&m=
    """

    _number_of_data_loc = 0

    def __init__(self):
        self.obj = win32com.client.Dispatch("Dscbo1.StockMstM")
        self.obj_rqrev = _ObjStockMstMRqRev(self.obj, _ObjStockMstMRqEvt)

    def set_input(self, input_list:list):
        """
        input_list: [stock_code] (Max:110)
        """
        code_list = str(input_list)[1:-1].replace("'",'').replace(" ",'')
        self.obj.SetInputValue(0, code_list)
    
    def request(self, timeout=5000):
        return super().request(timeout)
    
    def blockrequest(self):
        return super().blockrequest()

    def get_header(self, output_dict: dict) -> dict:
        return super().get_header(output_dict)

    def get_data(self, output_dict: dict, to_numpy: bool=True) -> dict:
        return super().get_data(output_dict, to_numpy)


### Dscbo1.StockMst2 ###

class _ObjStockMst2RqRev(RequestEventReceiverParent):

    StopEvent = win32event.CreateEvent(None, 0, 0, None)

    def __init__(self, objEvent, objEventHandler):
        super().__init__(objEvent, objEventHandler)

    def message_pump(self, timeout):
        return super().message_pump(timeout)

class _ObjStockMst2RqEvt(RequestEventParent):
    def OnReceived(self):
        win32event.SetEvent(_ObjStockMst2RqRev.StopEvent)
        return
    
class ObjStockMst2(DiscreteObserverParent):

    """
    https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=284&seq=15&page=1&searchString=StockMst2&p=&v=&m=
    """

    _number_of_data_loc = 0

    def __init__(self):
        self.obj = win32com.client.Dispatch("Dscbo1.StockMst2")
        self.obj_rqrev = _ObjStockMst2RqRev(self.obj, _ObjStockMst2RqEvt)

    def set_input(self, input_list:list):
        """
        input_list: [stock_code] (Max:110)
        """
        code_list = str(input_list)[1:-1].replace("'",'').replace(" ",'')
        self.obj.SetInputValue(0, code_list)

    def request(self, timeout=5000):
        return super().request(timeout)
    
    def blockrequest(self):
        return super().blockrequest()

    def get_header(self, output_dict: dict) -> dict:
        return super().get_header(output_dict)

    def get_data(self, output_dict: dict, to_numpy: bool=True) -> dict:
        return super().get_data(output_dict, to_numpy)


### CpSysDib.MarketEye ###

class _ObjMarketEyeRqRev(RequestEventReceiverParent):
    
    StopEvent = win32event.CreateEvent(None, 0, 0, None)

    def __init__(self, objEvent, objEventHandler):
        super().__init__(objEvent, objEventHandler)

    def message_pump(self, timeout):
        return super().message_pump(timeout)
    
class _ObjMarketEyeRqEvt(RequestEventParent):
    def OnReceived(self):
        win32event.SetEvent(_ObjMarketEyeRqRev.StopEvent)
        return

class ObjMarketEye(DiscreteObserverParent):

    """
    https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=284&seq=131
    """

    _number_of_data_loc = 2

    def __init__(self):
        self.obj = win32com.client.Dispatch("CpSysDib.MarketEye")
        self.obj_rqrev = _ObjMarketEyeRqRev(self.obj, _ObjMarketEyeRqEvt)

    def set_input(self, input_dict:dict):
        """
        input_dict: {creon_type:value} \n
        \n
        0-(long or long array) 필드 또는 필드배열, 최대 64개의 필드까지 요청가능 \n
            https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=284&seq=131 \n
        1-(string or string array) 종목코드 또는 종목코드배열. 최대 200종목까지 가능 \n
            주의) 해외지수와 환율은 심볼코드를 입력하여야함 예) JP#NI225:니케이지수 \n
        2-(char) 채결비교방식 \n
            '1':체결가비교방식, '2'호가비교방식(default)
        """
        input_dict[1] = list(input_dict[1]) #[1:-2].replace("'",'').replace(" ",'')
        for key,value in input_dict.items():
            self.obj.SetInputValue(key,value)

    def request(self, timeout=5000):
        return super().request(timeout)
    
    def blockrequest(self):
        return super().blockrequest()

    def get_header(self, output_dict: dict) -> dict:
        """
        output_dict: {creon_type:name} \n
        \n
        0-(long) 필드개수 \n
        1-(string array) 필드명의 배열 - 필드는 요청한 필드값의 오름차순으로 정렬되어 있음 \n
        2-(long) 종목개수 \n
        """
        return super().get_header(output_dict)

    # MarkeyEye need different get_data method
    def get_data(self, output_dict:dict, to_numpy: bool=True) -> dict: 
        """
        output_dict: {creon_type:name} \n
        \n
        set_input에서 요청한 {필드값:원하는 dict key} \n
            주의) set_input과 필드값 순서가 동일하여야함
        """
        number_of_data = self.obj.GetHeaderValue(self._number_of_data_loc)

        res_dict = {}
        for key,value in output_dict.items():
            res_dict[value] = []

        for j,value in enumerate(output_dict.values()): # MarketEye의 경우 type이 아니라 type 보낸 index 값 요청함
            for i in range(number_of_data):
                res_dict[value].append(self.obj.GetDataValue(j,i))
        
        if to_numpy:
            for key,value in res_dict.items():
                res_dict[key] = np.array(value)

        return res_dict
    

### CpSysDib.StockChart ###

class _ObjStockChartRqRev(RequestEventReceiverParent):
    
    StopEvent = win32event.CreateEvent(None, 0, 0, None)

    def __init__(self, objEvent, objEventHandler):
        super().__init__(objEvent, objEventHandler)

    def message_pump(self, timeout):
        return super().message_pump(timeout)
    
class _ObjStockChartRqEvt(RequestEventParent):
    def OnReceived(self):
        win32event.SetEvent(_ObjStockChartRqRev.StopEvent)
        return

class ObjStockChart(DiscreteObserverParent):

    """
    https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=284&seq=102
    """

    _number_of_data_loc = 3

    def __init__(self):
        self.obj = win32com.client.Dispatch("CpSysDib.StockChart")
        self.obj_rqrev = _ObjStockChartRqRev(self.obj, _ObjStockChartRqEvt)

    def set_input(self, input_dict:dict):
        """
        # input_dict: {creon_type:value} \n
        \n
        0-(string) 종목코드, 주식(A003540), 업종(U001), ELW(J517016)의 종목코드 \n
        1-(char) 요청구분, '1':기간, '2':개수 \n
            '1':  기간 요청시  주,월,분,틱은 불가 \n
            '2' : 갯수로 요청이고 분,틱 모드인 경우에는 요청 갯수 및 수신 갯수를 누적해서 다음 데이터 요청을 체크해야 함 \n
        2-(ulong) 요청종료일 YYYYMMDD형식으로 데이터의 마지막(가장최근) 날짜 Default(0) - 최근거래날짜 \n
        3-(ulong) 요청시작일 YYYYMMDD형식으로 데이터의 시작(가장오래된) 날짜 \n
        4-(ulong) 요청할 데이터의 개수
        5-(long array) 필드 \n
            https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=284&seq=102 \n
        6-(char) 차트구분 'D':일, 'W':주, 'M':월, 'm':분, 'T':틱 \n
        7-(ushort) 주기 Default(-1) \n
        8-(char) 갭보정여부 '0':갭무보정 [Default], '1':갭보정 \n
        9-(char) 수정주가 '0':무수정주가 [Default], '1':수정주가 \n
        10-(char) 거래량구분 '1':시간외거래량모두포함 [Default], '2'장료시간외거래량만포함, '3':시간외거래량모두제외, '4':장전시간외거래량만포함 \n
        11-(char) 조기적용여부 'Y':8시 45분부터 분차트 주기 계산, 'N':9시 00분부터 분차트 주기 계산 [Default] \n
        """
        for key,value in input_dict.items():
            self.obj.SetInputValue(key,value)

    def request(self, timeout=5000):
        return super().request(timeout)
    
    def blockrequest(self):
        return super().blockrequest()

    def get_header(self, output_dict: dict) -> dict:
        """
        # output_dict: {creon_type:name} \n
        \n 
        
        """
        return super().get_header(output_dict)

    # StockChart need different get_data method
    def get_data(self, output_dict:dict, to_numpy: bool=True) -> dict:
        """
        # output_dict: {creon_type:name} \n
        \n 

        """
        number_of_data = self.obj.GetHeaderValue(self._number_of_data_loc)

        res_dict = {}
        for key,value in output_dict.items():
            res_dict[value] = []

        for j,value in enumerate(output_dict.values()): # StockChart의 경우 type이 아니라 type 보낸 index 값 요청함
            for i in range(number_of_data):
                res_dict[value].append(self.obj.GetDataValue(j,i))
        
        if to_numpy:
            for key,value in res_dict.items():
                res_dict[key] = np.array(value)

        return res_dict
    

### CpSysDib.CssStgFind ###

class _ObjCssStgFindRqRev(RequestEventReceiverParent):
    
    StopEvent = win32event.CreateEvent(None, 0, 0, None)

    def __init__(self, objEvent, objEventHandler):
        super().__init__(objEvent, objEventHandler)

    def message_pump(self, timeout):
        return super().message_pump(timeout)
    
class _ObjCssStgFindRqEvt(RequestEventParent):
    def OnReceived(self):
        win32event.SetEvent(_ObjCssStgFindRqRev.StopEvent)
        return

class ObjCssStgFind(DiscreteObserverParent):

    """
    https://money2.creontrade.com/e5/mboard/ptype_basic/HTS_Plus_Helper/DW_Basic_Read_Page.aspx?boardseq=284&seq=102
    """

    _number_of_data_loc = 0

    def __init__(self):
        self.obj = win32com.client.Dispatch("CpSysDib.CssStgFind")
        self.obj_rqrev = _ObjCssStgFindRqRev(self.obj, _ObjCssStgFindRqEvt)

    def set_input(self, input_dict:dict):
        """
        # input_dict: {creon_type:value} \n
        \n
        0-(string) 전략ID \n
        1-(char) 예제전략여부 ('Y':예제전략) \n
            모의투자 접속시 예제전략 조회시 반드시 'Y'로 세팅해주어야함 \n
        """
        for key,value in input_dict.items():
            self.obj.SetInputValue(key,value)

    def request(self, timeout=5000):
        return super().request(timeout)
    
    def blockrequest(self):
        return super().blockrequest()

    def get_header(self, output_dict: dict) -> dict:
        """
        # output_dict: {creon_type:name} \n
        \n
        0-(long) 검색된 결과 종목 수 \n
        1-(long) 총 검색 종목 수 \n 
        2-(string) 검색시간 \n
        """
        return super().get_header(output_dict)

    # StockChart need different get_data method
    def get_data(self, output_dict:dict, to_numpy: bool=True) -> dict:
        """
        # output_dict: {creon_type:name} \n
        \n
        0-(string) 종목코드 \n
        """
        number_of_data = self.obj.GetHeaderValue(self._number_of_data_loc)

        res_dict = {}
        for key,value in output_dict.items():
            res_dict[value] = []

        for j,value in enumerate(output_dict.values()): # StockChart의 경우 type이 아니라 type 보낸 index 값 요청함
            for i in range(number_of_data):
                res_dict[value].append(self.obj.GetDataValue(j,i))
        
        if to_numpy:
            for key,value in res_dict.items():
                res_dict[key] = np.array(value)

        return res_dict