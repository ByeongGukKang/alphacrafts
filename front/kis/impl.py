import asyncio
from dataclasses import dataclass
import datetime
from enum import Enum
import os
import warnings

from base64 import b64decode
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7

import numpy as np
from orjson import dumps as orjson_dumps
from orjson import loads as orjson_loads
import requests
from picows import WSFrame

from alphacrafts.front.macros import MACRO_ASYNC_SLEEP, MACRO_ASYNC_YIELD, MACRO_DATETIME_NOW, MACRO_TIME_NS
from alphacrafts.front.meta import MetaAccount, MetaAPILimit, MetaWebsocketParser
from alphacrafts.front.net import HttpClient, WebSocketClient
from alphacrafts.front.object import Book, Worker, Scheduler

from alphacrafts.front.kis import wsresp
from alphacrafts.front.kis.wsresp import MetaWSResp


__all__ = [
    # Enums
    'EtOrdType',
    'EtOrdAdjType',
    # Dataclasses
    'OAuth',
    # Classes
    'APILimit',
    'Account',
    'WebSocketParser',
    # Functions
    'newSystem',
    'newSystem2',
    'ws_sub',
    'ws_unsub',
]


@dataclass
class EtOrdType:
    "Enum form KIS order type"
    Limit = '00'
    "00@지정가"
    Market = '01'
    "01@시장가"
    CondLimit = '02'
    "02@조건부지정가"
    BestLimit = '03'
    "03@최유리지정가"
    FirstLimit = '04'
    "04@최우선지정가"
    PreMarket = '05'
    "05@장전 시간외"
    AfterMarket = '06'
    "06@장후 시간외"
    AfterMarketSingle = '07'
    "07@장후 시간외 단일가"
    IOCLimit = '11'
    "11@IOC지정가"
    FOKLimit = '12'
    "12@FOK지정가"
    IOCMarket = '13'
    "13@IOC시장가"
    FOKMarket = '14'
    "14@FOK시장가"
    IOCBestLimit = '15'
    "15@IOC최유리"
    FOKBestLimit = '16'
    "16@FOK최유리"
    MidPrc = '21'
    "21@중간가"
    StopLimit = '22'
    "22@스톱지정가"
    MidPrcIOC = '23'
    "23@중간가IOC"
    MidPrcFOK = '24'
    "24@중간가FOK"

@dataclass
class EtOrdAdjType:
    "Enum form KIS order adjust type"
    Adjust = '01'
    "01@정정"
    Cancel = '02'
    "02@취소"

# @dataclass
# class EtTrKrStOrdAcc:
#     "[국내주식] 주문/계좌 TR"
#     OrdSell = "TTTC0011U"
#     "주식주문(현금)-매도"
#     OrdBuy = "TTTC0012U"
#     "주식주문(현금)-매수"
#     OrdSellMargin = "TTTC0051U"
#     "주식주문(신용)-매도"
#     OrdBuyMargin = "TTTC0052U"
#     "주식주문(신용)-매수"
#     OrdAdjCan = "TTTC0013U"
#     "주식주문(정정취소)"
#     OrdAdjCanLkp = "TTTC0084R"
#     "주식정정취소가능주문조회"
#     DailyOrdExc = "TTTC0081R"
#     "주식일별주문체결조회-3개월이내"
#     DailyOrdExcPast = "CTSC9215R"
#     "주식일별주문체결조회-3개월이전"
#     Balance = "TTTC8434R"
#     "주식잔고"
#     PossibleBuy = "TTTC8908R"
#     "매수가능조회"
#     OrdReserve = "CTSC0008U"
#     "주식예약주문"
#     OrdReserveCan = "CTSC0009U"
#     "주식예약주문취소"
#     OrdReserveAdj = "CTSC0013U"
#     "주식예약주문정정"
#     OrdResereveLkp = "CTSC0004R"
#     "주식예약주문조회"
#     PenBalanceLkp = "TTTC2202R"
#     "퇴직연금 쳬결기준잔고"
#     PenOrdRemain = "TTTC2201R"
#     "퇴직연금 미체결내역"
#     PenPossibleBuy = "TTTC0503R"
#     "퇴직연금 매수가능"
#     PenDeposit = "TTTC0506R"
#     "퇴직연금 예수금"
#     PenBalance = "TTTC2208R"
#     "퇴직연금 잔고"
#     BalanceRealPnl = "TTTC8494R"
#     "주식잔고조회_실현손익"
#     MarginPossibleBuy = "TTTC8909R"
#     "신용매수가능"
#     AccountAssetLkp = "CTRP6548R"
#     "투자계좌자산현황조회"
#     PeriodDailyPnL = "TTTC8708R"
#     "기간별손익일별합산"
#     PeriodTradePnL = "TTTC8715R"
#     "기간별매매손익현황"
#     PossibleSell = "TTTC8408R"
#     "매도가능수량"
#     IntegrateMargin = "TTTC0869R"
#     "주식통합증거금 현황"
#     PeriodRights = "CTRGA011R"
#     "기간별계좌권리현황"    

# @dataclass
# class EtTrKrStPrc:
#     "[국내주식] 기본시세 TR"
#     Prc = "FHKST01010100"
#     "현재가 시세"
#     Exc = "HKST01010300"
#     "현재가 체결"
#     DailyPrc = "HKST01010400"
#     "현재가 일자별"
#     OrdBook = "HKST01010200"
#     "현재가 호가/에상체결"
#     InvestorInfo = "HKST01010900"
#     "투자자"
#     MemberInfo = "FHKST01010600"
#     "회원사"
#     PeriodPrc = "FHKST03010100"
#     "기간별시세"
#     ExcByTime = "HPST01060000"
#     "당일시간대별체결"
#     AfMktExcByTime = "FHPST02310000"
#     "시간외시간대별체결"
#     AfMktDailyPrc = "FHPST02320000"
#     "시간외일자별주가"
#     MinuteBarToday = "HKST03010200"
#     "당일분봉조회"
#     Prc2 = "FHPST01010000"
#     "현재가 시세2"
#     EtfEtnPrc = "FHPST02400000"
#     "ETF/ETN 현재가"
#     Nav = "FHPST02440000"
#     "NAV 비교추이(종목)"
#     MinutelyNav = "FHPST02440100"
#     "NAV 비교추이(분)"
#     DailyNav = "FHPST02440200"
#     "NAV 비교추이(일)"
#     MktClsExpExc = "FHKST117300C0"
#     "장마감 예상체결가"
#     EtfComponentPrc = "FHKST121600C0"
#     "ETF 구성종목시세"
#     AfMktPrc = "FHPST02300000"
#     "시간외 현재가"
#     AfMktOrdBook = "FHPST02300400"
#     "시간외 호가"
#     MinuteBarDaily = "FHKST03010230"
#     "일별분봉조회"

# @dataclass
# class EtTrKrELW:
#     "[국내주식] ELW시세 TR"
#     Prc = "FHKEW15010000 "
#     "현재가 시세"
#     RtnRank = "FHPEW02770000"
#     "상승률순위"
#     TvolRank = "FHPEW02780000"
#     "거래량순위"
#     IndicatorRank = "FHPEW02790000"
#     "지표순위"
#     SensitivityRank = "FHPEW02800000"
#     "민감도 순위"
#     TodayRemark = "FHPEW02870000"
#     "당일급변종목"
#     VolTrendExc = "FHPEW02840100"
#     "변동성추이(체결)"
#     NewlyList = "FHKEW154800C0"
#     "신규상장종목"
#     VolTrendMinutely = "FHPEW02840300"
#     "변동성 추이(분별)"
#     IndicatorTrendExc = "FHPEW02740100"
#     "투자지표추이(체결)"
#     IndicatorTrendMinutely = "FHPEW02740300"
#     "투자지표추이(분별)"
#     SensitivityTrendExc = "FHPEW02830100"
#     "민감도 추이(체결)"
#     VolTrendDaily = "FHPEW02840200"
#     "변동성 추이(일별)"
#     LkpByUnderlying = "FHKEW154101C0"
#     "기초자산별 종목시세"
#     IndicatorTrendDaily = "FHPEW02740200"
#     "투자지표추이(일별)"
#     SensitivityTrendDaily = "FHPEW02830200"
#     "민감도 추이(일별)"
#     VolTrendTick = "FHPEW02840400"
#     "변동성 추이(틱)"
#     LpTrade = "FHPEW03760000"
#     "LP매매추이"
#     ComparsionLkp = "FHKEW151701C0"
#     "비교대상종목조회"
#     Search = "FHKEW15100000"
#     "종목검색"
#     UnderlyingLkp = "FHKEW154100C0"
#     "기초자산 목록조회"
#     ExpirationLkp = "FHKEW154700C0"
#     "만기예정/만기종목"


@dataclass  
class OAuth:
    """
    OAuth token object for KIS.
    
    Args:
        access_token (str): Access token.
        access_token_token_expired (datetime.datetime): Access token expiration time.
        token_type (str): Token type.
        expires_in (int): Expiration time in seconds.

    """
    access_token: str
    access_token_token_expired: datetime.datetime
    token_type: str
    expires_in: int

    def __post_init__(self):
        if isinstance(self.access_token_token_expired, str):
            self.access_token_token_expired = datetime.datetime.strptime(self.access_token_token_expired, "%Y-%m-%d %H:%M:%S")


class APILimit(MetaAPILimit):

    def __init__(self, http_limit: int, socket_limit: int, freq: datetime.timedelta, name: str="APILimit", tolerance: float=0.02):
        """
        API limit object for KIS.

        Args:
            http_limit (int): Http limit.
            socket_limit (int): Socket limit.
            freq (datetime.timedelta): Refresh frequency.
            name (str): Name of the object
                Default is "APILimit"
            tolerance (float): Tolerance for the frequency
                Default is 0.02, if you set 0.02, it will sleep for 0.02*freq seconds.
                If you want to yield not sleep, set 0. It is more accurate but might cause busy waiting.
        """
        if not isinstance(http_limit, int):
            raise TypeError("http_limit must be an integer.")
        if not isinstance(socket_limit, int):
            raise TypeError("socket_limit must be an integer.")
        if not isinstance(freq, datetime.timedelta):
            raise TypeError("freq must be a datetime.timedelta.")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")

        self.__name = name
        self.__sema_http = asyncio.LifoQueue(http_limit)
        self.__sema_sock = asyncio.LifoQueue(socket_limit)
        freq_in_s = freq.total_seconds()

        async def _work():
            nsfreq = freq_in_s * 1_000_000_000 # convert to nanoseconds
            ctime = MACRO_TIME_NS() - nsfreq # Immediately refresh
            sema = self.__sema_http
            maxsize = sema.maxsize
            tsleep = freq_in_s * tolerance # Sleep time in seconds
            if tsleep:
                while True:
                    if (ctime + nsfreq) < MACRO_TIME_NS():
                        for _ in range(maxsize - sema.qsize()): # Since release(n) is not supported
                            sema.put_nowait(None)
                        ctime = MACRO_TIME_NS()
                    await MACRO_ASYNC_SLEEP(tsleep) # Sleep, maximum time difference is (tolerance*100)% of freq
            else:
                while True:
                    if (ctime + nsfreq) < MACRO_TIME_NS():
                        for _ in range(maxsize - sema.qsize()): # Since release(n) is not supported
                            sema.put_nowait(None)
                        ctime = MACRO_TIME_NS()
                    await MACRO_ASYNC_YIELD() # YIELD

        self.__work_http_limit = _work

    @property
    def name(self) -> str:
        """Return name of the object."""
        return self.__name
    
    @property
    def tasks(self) -> list[asyncio.Task]:
        """Return list of all tasks."""
        return [asyncio.create_task(self.__work_http_limit(), name=self.__name+".http_limit")]
    
    @property
    def consume_func(self):
        """Return consume function."""
        return self.__sema_http.get
    
    async def consume(self):
        """Consume http limit."""
        await self.__sema_http.get()

    async def acquire_socket(self):
        """Acquire socket limit."""
        await self.__sema_sock.get()
    
    def release_socket(self):
        """Release socket limit."""
        if not self.__sema_sock.full():
            self.__sema_sock.put_nowait(None)


class Account(MetaAccount):

    def __init__(
        self,
        path:              str,
        accnum:            str,
        assetnum:          str,
        appkey:            str,
        appsecret:         str,
        book:              Book,
        oauth:             OAuth,
        httpclient:        HttpClient,
        # oauth_auto_update: bool,
    ):
        """
        Account object for KIS.
         - Does not recommend to use this object directly. Use newSystem instead.

        Args:
            path (str): Path to store the OAuth token and others.
            accnum (str): Account number.
            assetnum (str): Asset number.
            appkey (str): API key.
            appsecret (str): API secret.
            book (Book): Book.
            oauth (OAuth): OAuth token. If None, it will be refreshed.
            httpclient (HttpClient): Http client object
        
        """
        # MetaAccount init
        # super().__init__() inits self.default_header & self._workers_init
        self.default_header: dict[str:str] = {}
        self._workers_init: list[Worker] = []
        # Account attributes
        self.path:       str = path
        self.accnum:     str = accnum
        self.assetnum:   str = assetnum
        self.appkey:     str = appkey
        self.appsecret:  str = appsecret
        self.book:       Book = book
        self.oauth:      OAuth = oauth
        self.httpclient: HttpClient = httpclient
        # Direct value
        self.access_token: str = None
        if self.oauth is not None:
            self.access_token = self.oauth.access_token
            self.default_header.update({"authorization":"Bearer "+self.access_token, "appkey":self.appkey, "appsecret":self.appsecret})
        self.approval_key: str = None # WebSocket approval key

        ### Init tasks
        ## Get OAuth token
        oauth_refresh = False
        if self.oauth is None:
            oauth_refresh = True
        elif (self.oauth.access_token_token_expired - datetime.timedelta(hours=18)) <= MACRO_DATETIME_NOW():
            oauth_refresh = True
        
        if oauth_refresh:
            resp = requests.post(
                url = f"{self.httpclient.session.base_url}/oauth2/tokenP",
                data = f'{{"grant_type":"client_credentials","appkey":"{self.appkey}","appsecret":"{self.appsecret}"}}'.encode('utf-8'),
                headers = {"content-type":"application/json"},
            )
            print(resp.text)
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to refresh OAuth token, wrong status: {resp.status_code}")
            try:
                data = resp.json()
            except Exception as e:
                raise RuntimeError(f"Failed to refresh OAuth token, failed json parsing: {e}") from e
            self.oauth = OAuth(**data)
            self.access_token = self.oauth.access_token
            with open(os.path.join(self.path, "oauth.json"), "wb") as f:
                f.write(orjson_dumps(data))
            self.default_header.update({"authorization":"Bearer "+self.access_token, "appkey":self.appkey, "appsecret":self.appsecret})

        ## Book initialization TODO
        # # Book init worker
        # worker_book_init = Worker("Account.book")
        # # Work
        # async def _work_book(self: Worker, acc: Account):
        #     httpreq = HttpRequest(
        #         path = b'/uapi/domestic-stock/v1/trading/inquire-balance',
        #         method = "GET",
        #         header = acc.get_header({"tr_id":"TTTC8434R" if acc.httpclient.session.base_url.__str__().endswith("9443") else "VTTC8434R"}),
        #         params = {
        #             "CANO": acc.accnum, "ACNT_PRDT_CD": acc.assetnum, "AFHR_FLPR_YN": "N",
        #             "OFL_YN": "", "INQR_DVSN": "02", "UNPR_DVSN": "01",
        #             "FUND_STTL_ICLD_YN": "N", "FNCG_AMT_AUTO_RDPT_YN": "N",
        #             "PRCS_DVSN": "00", "CTX_AREA_FK100": "", "CTX_AREA_NK100": "",
        #         }
        #     )
        #     while True:
        #         resp = await acc.httpclient.request(httpreq)
        #         ## Error handling
        #         if resp.err is not None:
        #             raise resp.err
        #         if resp.status != 200:
        #             raise ValueError(f"Failed to initialize book, wrong status: {resp.status}")
        #         data, err = resp.jsonb
        #         if err is not None:
        #             raise ValueError(f"Failed to initialize book, failed json parsing: {err}")
        #         ## Update
        #         # positions
        #         for output1 in data["output1"]:
        #             acc.book.add_posit(output1["pdno"], float(output1["hldg_qty"]))
        #             # TODO Add average price
        #         # cash
        #         if (resp.jsonh["tr_cont"] == "D") | (resp.jsonh["tr_cont"] == "E"):
        #             acc.book.add_cash(float(data["output2"][0]["dnca_tot_amt"]))
        #             break
        #         # if more data, update params, request again
        #         httpreq.params["CTX_AREA_FK100"] = data["ctx_area_fk100"]
        #         httpreq.params["CTX_AREA_NK100"] = data["ctx_area_nk100"]

        # # Set work
        # worker_book_init.do_if(Work(_work_book, (self, )), worker_oauth.event, once=True)
        # self._workers_init.append(worker_book_init)

        ## WebSocket approval key
        resp = requests.post(
            url = f"{self.httpclient.session.base_url}/oauth2/Approval",
            data = f'{{"grant_type":"client_credentials","appkey":"{self.appkey}","secretkey":"{self.appsecret}"}}'.encode('utf-8'),
            headers = {"content-type":"application/json"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to get websocket approval_key, wrong status: {resp.status_code}")
        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to get websocket approval_key, failed json parsing: {e}") from e
        self.approval_key = data["approval_key"]

    @property
    def tasks(self) -> list[asyncio.Task]:
        """Return list of all tasks."""
        tasks = []
        for worker in self._workers_init:
            tasks.extend(worker.tasks)
        return tasks
    
    def get_header(self, update: dict[str:str] = None) -> dict[str:str]:
        """
        Get default header for the request.

        Args:
            update (dict[str:str], optional): Additional header. 
                Defaults to None.

        Returns:
            dict[str:str]: Header.
        """
        if self.default_header is None:
            raise ValueError("default_header is None. Please check whether the scheduling is done correctly.")
        header = self.default_header.copy()
        if update is not None:
            header.update(update)
        return header


class WebSocketParser(MetaWebsocketParser):

    def __init__(self, wsclient: WebSocketClient):
        """
        WebSocket parser object for KIS.
        """
        self.__ws_send = wsclient._send
        self._cipher = None
        "AES256 CBC Cipher"
        self._unpadder = PKCS7(algorithms.AES.block_size)
        "PKCS7 Unpadder for AES256 CBC Cipher"

        # WebSocket Response Data Class Map
        self.__RESP_MAP = {}
        for clsname in wsresp.__all__:
            respcls: MetaWSResp = getattr(wsresp, clsname)
            self.__RESP_MAP[respcls.trid(False)] = respcls
            self.__RESP_MAP[respcls.trid(True)] = respcls

    @property
    def RESP_MAP(self) -> dict[str,MetaWSResp]:
        "Response map. key: tr_id, value: MetaWSResp"
        return self.__RESP_MAP

    def _set_resp_map(self, new_map: dict[str,MetaWSResp]):
        """
        Set response map.

        Args:
            new_map (dict[str, any]): New response map.
        """
        self.__RESP_MAP = new_map

    def _update_resp_map(self, new_map: dict[str,MetaWSResp]):
        """
        Update response map.

        Args:
            new_map (dict[str, any]): New response map.
        """
        self.__RESP_MAP.update(new_map)

    def decrypt(self, cipher_bytes: bytes) -> bytes:
        "Decrypt cipher bytes"
        context_decrypt = self._cipher.decryptor()
        context_unpad   = self._unpadder.unpadder()
        return context_unpad.update(context_decrypt.update(b64decode(cipher_bytes)) + context_decrypt.finalize()) + context_unpad.finalize()

    def parse_frame(self, frame: WSFrame) -> tuple[str, list[MetaWSResp]|list[dict]|list[Exception]]:
        try:
            rawbytes: bytes = frame.get_payload_as_bytes() # Copy bytes

            if rawbytes[0] == 48: # b'0'[0] is 48
                _, trid_bytes, cnt_bytes, recv_bytes = rawbytes.split(b'|')
                cbcnt: int = int(cnt_bytes.decode('utf-8')) 
                data:  list[MetaWSResp] = [None] * cbcnt # Pre-allocate
                cbkey: str = trid_bytes.decode('utf-8')
                size:  int = self.__RESP_MAP[cbkey].len()
                recv_bytes: list[bytes] = recv_bytes.split(b'^')

                if cbcnt == 1: # Fast path for single data, Usual case
                    data[0] = self.__RESP_MAP[cbkey](recv_bytes)
                else: # Slow path using loop for multiple data
                    sidx = 0
                    for i, eidx in enumerate(range(size, len(recv_bytes)+1, size)):
                        data[i] = self.__RESP_MAP[cbkey](recv_bytes[sidx:eidx])
                        sidx = eidx
            elif rawbytes[0] == 49: # b'1'[0] is 49, encrypted case
                _, trid_bytes, cnt_bytes, recv_bytes = rawbytes.split(b'|')
                cbcnt: int = int(cnt_bytes.decode('utf-8')) 
                data:  list[MetaWSResp] = [None] * cbcnt # Pre-allocate
                cbkey: str = trid_bytes.decode('utf-8')
                size:  int = self.__RESP_MAP[cbkey].len()
                recv_bytes: bytes = self.decrypt(recv_bytes) # Decrypt
                recv_bytes: list[bytes] = recv_bytes.split(b'^')

                if cbcnt == 1: # Fast path for single data, Usual case
                    data[0] = self.__RESP_MAP[cbkey](recv_bytes)
                else: # This is slow path due to for loop
                    sidx = 0
                    for i, eidx in enumerate(range(size, len(recv_bytes)+1, size)):
                        data[i] = self.__RESP_MAP[cbkey](recv_bytes[sidx:eidx])
                        sidx = eidx
            else:
                jdata: dict[str,any] = orjson_loads(rawbytes)
                if jdata["header"]["tr_id"] == "PINGPONG":
                    # Hard coded PINGPONG, Why KIS doesn't use ping/pong frame???
                    # picows auto pong is not working, so need to manually send pong
                    self.__ws_send(f'{{"header":{{"tr_id":"PINGPONG","datetime":"{jdata["header"]["datetime"]}"}}}}'.encode('utf-8'))
                    return 'pong', (jdata,)
                if jdata["body"]["rt_cd"] == '1': # Error
                    return 'errws', (jdata,)
                if self._cipher is None:
                    output: dict[str,str] = jdata["body"]["output"]
                    if output is not None:
                        self._cipher = Cipher(
                            algorithms.AES(output["key"].encode('utf-8')),
                            modes.CBC(output["iv"].encode('utf-8'))
                        )
                return 'sub', (jdata,)

        except Exception as e:
            return 'err', (e,)
        
        return cbkey, data

    def parse_sub(self, data: bytes|dict) -> bytes:
        if isinstance(data, bytes):
            return data
        elif isinstance(data, dict):
            return orjson_dumps(data)
        else:
            raise TypeError("data for .send() must be bytes or dict.")


def newSystem(
    scheduler: Scheduler,
    path:      str,
    accnum:    str,
    assetnum:  str,
    appkey:    str,
    appsecret: str,
    api_limit: APILimit = APILimit(19, 41, datetime.timedelta(seconds=1)),
    book:      Book     = Book(np.dtype('<U6'), np.dtype('<f8')),
    http_url:  str      = 'https://openapi.koreainvestment.com:9443',
    ws_url:    str      = 'ws://ops.koreainvestment.com:21000',
    oauth:     OAuth    = None,
) -> tuple[Account, HttpClient, WebSocketClient]:
    """
    System initialization for KIS.

    Args:
        scheduler (Scheduler): Task Scheduler.
        path (str): Path to store the OAuth token and others.
        accnum (str): Account number.
        assetnum (str): Asset number.
        appkey (str): API key.
        appsecret (str): API secret.
        api_limit (APILimit): API limit.
            Default is APILimit(19, 41, datetime.timedelta(seconds=1)).
        book (Book): Book.
            Default is Book(np.dtype('<U6'), np.dtype('<f8')).
        http_url (str): http request url. 
            Default is "https://openapi.koreainvestment.com:9443".
        ws_url (str): WebSocket url.
            Default is "ws://ops.koreainvestment.com:21000".
        oauth (OAuth): OAuth token. If None, refreshed. If '', tries to load from the path. If not found, it will be refreshed.
            Default is ''.
    """

    httpclient = HttpClient(api_limit, http_url)
    if oauth == '': # Try to load from file
        if "oauth.json" in os.listdir(path):
            try:
                with open(os.path.join(path, 'oauth.json'), 'rb') as f:
                    oauth = OAuth(**orjson_loads(f.read()))
            except Exception as e:
                warnings.warn(f'Failed to load OAuth token: {e}')
                oauth = None
    acc = Account(path, accnum, assetnum, appkey, appsecret, book, oauth, httpclient)
    wsclient = WebSocketClient(api_limit, WebSocketParser, ws_url)

    # Asynchronous websocket initialization
    async def init_(wsclient):
        await asyncio.wait(wsclient.tasks)
        scheduler.add_task('Runtime', api_limit.tasks, wait=False)
    scheduler.loop_.run_until_complete(init_(wsclient))

    return acc, httpclient, wsclient

def newSystem2(
    scheduler:  Scheduler,
    path:       str,
    accnum:     str,
    assetnum:   str,
    appkey:     str,
    appsecret:  str,
    httpclient: HttpClient = None,
    wsclient:   WebSocketClient = None,
    oauth:      OAuth = None,
) -> tuple[Account, HttpClient, WebSocketClient]:
    """
    System initialization for KIS.

    Args:
        scheduler (Scheduler): Task Scheduler.
        path (str): Path to store the OAuth token and others.
        accnum (str): Account number.
        assetnum (str): Asset number.
        appkey (str): API key.
        appsecret (str): API secret.
        oauth (OAuth): OAuth token. If None, refreshed. If '', tries to load from the path. If not found, it will be refreshed.
            Default is ''.
    """
    if httpclient is None:
        httpclient = HttpClient(
            api_limit = APILimit(19, 41, datetime.timedelta(seconds=1)),
            url = 'https://openapi.koreainvestment.com:9443',
            memsize = 32
        )    

    if oauth == '': # Try to load from file
        if "oauth.json" in os.listdir(path):
            try:
                with open(os.path.join(path, 'oauth.json'), 'rb') as f:
                    oauth = OAuth(**orjson_loads(f.read()))
            except Exception as e:
                warnings.warn(f'Failed to load OAuth token: {e}')
                oauth = None
    acc = Account( path, accnum, assetnum, appkey, appsecret, Book(np.dtype('<U6'), np.dtype('<f8')), oauth, httpclient)

    if wsclient is None:
        wsclient = WebSocketClient(
            httpclient.api_limit, WebSocketParser, 'ws://ops.koreainvestment.com:21000',
            websocket_handshake_timeout = 100, enable_auto_ping = False, enable_auto_pong = False
        )

    # Asynchronous websocket initialization
    async def init_(wsclient):
        await asyncio.wait(wsclient.tasks)
        # Add API limit task
        scheduler.add_task('Runtime', httpclient.api_limit.tasks, wait=False)
    scheduler.loop_.run_until_complete(init_(wsclient))

    return acc, httpclient, wsclient

async def ws_sub(acc: Account, wsclient: WebSocketClient, tr_id: str, tr_key: str, custtype: str="P"):
    """
    KIS WebSocket Subscription

    Args:
        acc (Account): Account instance
        wsclient (WebSocketClient): WebSocketClient instance
        tr_id (str): Transaction ID
        tr_key (str): Transaction Key
        custtype (str, optional): Customer Type. Defaults to "P".
    """
    if tr_id not in wsclient._callback_dict:
        raise ValueError(f"No callback for {tr_id}, please set callback first.")
    await wsclient.send(
        (
            f'{{'
            f'"header":{{"approval_key":"{acc.approval_key}","custtype":"{custtype}","tr_type":"1","content-type":"utf-8"}},'
            f'"body":{{"input":{{"tr_id":"{tr_id}","tr_key":"{tr_key}"}}}}'
            f'}}'
        ).encode("utf-8")
    )

async def ws_unsub(acc: Account, wsclient: WebSocketClient, tr_id: str, tr_key: str, custtype: str="P"):
    """
    KIS WebSocket Unsubscription

    Args:
        acc (Account): Account instance
        wsclient (WebSocketClient): WebSocketClient instance
        tr_id (str): Transaction ID
        tr_key (str): Transaction Key
        custtype (str, optional): Customer Type. Defaults to "P".
    """
    await wsclient.send(
        (
            f'{{'
            f'"header":{{"approval_key":"{acc.approval_key}","custtype":"{custtype}","tr_type":"2","content-type":"utf-8"}},'
            f'"body":{{"input":{{"tr_id":"{tr_id}","tr_key":"{tr_key}"}}}}'
            f'}}'
        ).encode("utf-8")
    )
