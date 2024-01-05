import datetime as dt
import inspect
from threading import Thread
from queue import Queue as ThreadQueue
import pickle

import numpy as np

from PyQt5.QtCore import pyqtSlot, QObject, QThread, pyqtSignal

import zmq

from alphacraft.bkd.share.qt import ThreadData, QtLocalClock
from alphacraft.bkd.creon.wrapper.trade import ObjCpConclusion, _ObjCpConclusionEvent
from alphacraft.bkd.creon.wrapper.trade import QtObjCpConclusion
from grid.parent import GridNodeParent, GridQtNodeParent
from grid.utils import Ledger


# Qtworkers in DataQtNode are synchronized by its local clock
# Clock generate tick --> tick emit signal --> signal trigger worker to request data
# --> worker request data --> worker emit data --> data received by worker_slot (self.update_memory)
# --> if the memory is full --> aggregate the memory --> publish the data
class QtDataNode(GridQtNodeParent):

    _node_type = "DataQtNode"

    ### Commands ###

    def _ncmd_clockoff(self, *args):
        """
        *args[0] (bool): clock off
        """
        self.qt_local_clock.time_stop = args[0]
        self._call_lmsg(20, f"ncmd[MANAGER:clockoff] [success] {args[0]}")

    ### Main ###
    def start(self): # Must Defined
        self.qt_local_clock.start()
        for worker_name in self.workers:
            self.workers[worker_name].start()
            self.workers[worker_name].data.connect(self.workers_slot[worker_name])
        self._call_lmsg(20, f"ncmd[MANAGER:start] [success]")

    def set_start(self, ip, data_pub_port, func_agg_memory):
        self._socket_data_pub = self._context.socket(zmq.PUB)
        self._socket_data_pub.setsockopt(zmq.CONFLATE, True)
        self._socket_data_pub.bind(f"tcp://{ip}:{data_pub_port}")

        self.qt_local_clock = QtLocalClock()  # qt_local_clock
        # qt_workers - emit data
        self.workers = {}  
        # pystSlots - receive data from qt_workers
        self.workers_slot = {}
        # data memory, received data are stored in this dict
        self.workers_memory = {}  # qt_workers memory
        # if the memory is full, then aggregate the memory and publish it
        self._func_agg_memory = func_agg_memory

    def set_clock_frequency(self, frequency):
        self.qt_local_clock.set_freq(frequency)

    def set_worker(self, worker_name, worker):
        """
        worker_name (str): name of worker
        worker : an instance of QtObserver 
        """
        if worker_name == "master":
            raise ValueError("name 'master' is for master thread")
        if not (isinstance(worker, QThread) or isinstance(worker, QObject)):
            raise TypeError("worker must be an instance fro, 'QThread' or 'QObject'")
        self.workers[worker_name] = worker
        self.workers_slot[worker_name] = pyqtSlot(ThreadData)(lambda data: self.update_meory(worker_name, data))

    def update_meory(self, worker_name, data):
        self.workers_memory[worker_name] = data

        if len(self.workers_memory) == len(self.workers):
            self._socket_data_pub.send_pyobj(self._func_agg_memory(self.workers_memory))
            self.workers_memory = {}


class SignalNode(GridNodeParent):

    _node_type = "SignalNode"

    ### Commands ###

    ### Main ###
    def start(self):
        # check settings before start
        super().start(self)

        # start threads
        self._threads["send_order"][0].start()
        self._threads["get_order_result"][0].start()
        self._call_lmsg(20, f"ncmd[MANAGER:start] [success]")

    def set_ready(self):
        self._setlist["set_functions"] = False
        self._setlist["set_data_node"] = False
        self._setlist["set_account_node"] = False
        self._setlist["set_start"] = False
        self._setlist["set_ready"] = True

    def set_start(self, codes):
        # ledger
        self._ledger = Ledger(cash=0, codes=codes)

        # threads
        self._threads["thd_order_send"] = [Thread(target=self._send_order, daemon=True), True]
        self._threads["thd_order_result"] = [Thread(target=self._get_order_result, daemon=True), True]

        self._setlist["set_start"] = True

    def set_functions(self, func_signal, func_order, var_signal={}, var_order={}, args_signal=(), args_order=()):
        # Check Function Parameters
        params = inspect.signature(func_signal).parameters.keys()
        for param in ["time","data","cash","book","var_signal","args_signal"]:
            if param not in params:
                raise ValueError(f"signal_func must have parameters [time, data, cash, book, var_signal, args_signal]")
        params = inspect.signature(func_order).parameters.keys()
        for param in ["time","data","cash","book","signal","var_order", "arg_order"]:
            if param not in params:
                raise ValueError(f"order_func must have parameters [time, data, cash, book, signal, var_order, arg_order]")
            
        self._func_signal = func_signal
        self._func_order = func_order
        self._var_signal = var_signal
        self._var_order = var_order
        self._args_signal = args_signal
        self._args_order = args_order

        self._setlist["set_functions"] = True

    def set_data_node(self, ip, port):
        socket = self._context.socket(zmq.SUB)
        socket.setsockopt(zmq.CONFLATE, True)
        socket.bind(f"tcp://{ip}:{port}")
        self._socket_data_node = socket

        self._setlist["set_data_node"] = True

    def set_account_node(self, ip, router_port, execution_port):
        # Socket for sending/checking order
        self._sockets["account_node_router"] = self._context.socket(zmq.DEALER)
        self._sockets["account_node_router"].setsockopt(zmq.IDENTITY, self._identity.encode("utf-8"))
        self._sockets["account_node_router"].bind(f"tcp://{ip}:{router_port}")

        # Socket for receiving order result
        self._sockets["account_node_execution"] = self._context.socket(zmq.SUB)
        self._sockets["account_node_execution"].setsockopt(zmq.SUBSCRIBE, self._identity.encode("utf-8"))
        self._sockets["account_node_execution"].bind(f"tcp://{ip}:{execution_port}")

    ### Threads ###
    def _send_order(self):
        while True:
            if self._threads["thd_order_send"][1]:
                [topic, data] = self._socket_data_node.recv_multipart()
                # It might be better to put signal function inside of order function parameter to avoid new memory allocation
                # But I remain it as a variable 'signal' for logging purpose
                data = pickle.loads(data)
                
                # Acquire ledger lock
                with self._ledger.withlock():
                    # Get current time, current time is used for signal and order function
                    current_time = dt.datetime.now()
                    
                    # sign_func must return a tuple (signal, var_signal)
                    signal, var_signal = self._func_signal(
                        current_time,
                        data,
                        self._ledger.cash,
                        self._ledger.book, # structured array with (code, order, order_price, position, position_price)
                        self._var_signal, # stored variable for signal
                        *self._args_signal,
                    )
                    self._var_signal = var_signal

                    # We can avoid order function call if signal is None
                    if signal is not None:

                        ##### temporal code for Creon #####
                        # update newest cash
                        # THIS IS ONLY AVAILABLE FOR Single Startegy and Single Account
                        self._sockets["account_node_router"].send_pyobj({"ncmd":"get_cash"})
                        [identity, rev_data] = self._sockets["account_node_router"].recv_multipart()
                        current_cash = pickle.loads(rev_data)
                        self._ledger.update_cash(current_cash)
                        ##### end of temporal code #####

                        # order_func must return a tuple with (structured array with (code, quantity, price), var_order)
                        order, var_order = self._func_order(
                            current_time, # little time lag is ignored
                            data,
                            self._ledger.cash,
                            self._ledger.book, # structured array with (code, order, order_price, position, position_price)
                            signal,
                            self._var_order, # stored variable for order
                            *self._args_order
                        )
                        self._var_order = var_order

                        # Order value must be checked during complie time 
                        # TODO order_func complier
                        # Order is a structured array or dictionary with (code, quantity, price)
                        self._sockets["account_node_router"].send_pyobj(order)
                        [identity, rev_data] = self._sockets["account_node_router"].recv_multipart()

                        # rev_data {"cash":}
                        # 주문 가능 금액을 받아와서 ledger에 반영
                        # rev_data = pickle.loads(rev_data)
                        # self._ledger.update_cash(rev_data["cash"])

    def _get_order_result(self):
        while True:
            if self._threads["thd_order_result"][1]:
                # order_result is a dict with keys: "result_type", "order_type", "code", "delta_order", "order_price", "delta_position", "execute_price"
                # result_type: "execute", "submit/adjust", "cancel/reject"
                [identity, result] = self._sockets["account_node_execution"].recv_multipart()
                result = pickle.loads(result) 

                # Acquire ledger lock
                with self._ledger.withlock():
                    if result["result_type"] == "execute":
                        self._ledger.update_cash(-result['delta_position'] * result["execute_price"])
                        self._ledger.add_position(result['code'], result['delta_position'], result["execute_price"])
                    else:
                        pass
                        # TODO

                    # TODO replace checking system to dictionary?
                    # if result["result_type"] == "execute":
                    #     if result["order_type"] == "sell":
                    #         self.worker.ledger.add_cash(-result['delta_position'] * result["execute_price"])
                    #     self.worker.ledger.add_order(result['code'], result['delta_order'], result["order_price"])
                    #     self.worker.ledger.add_position(result['code'], result['delta_position'], result["execute_price"])

                    # elif result["result_type"] == "submit/adjust":
                    #     if result["order_type"] == "buy":
                    #         # remove old cash(pre-applied cash when sending the order to account_node) and update to new cash
                    #         self.worker.ledger.add_cash(
                    #             result['delta_order'] * (self.worker.ledger.order_price[result['code']] - result["order_price"])
                    #         )
                    #     # remove old order(pre-applied order when sending the order to account_node) and update to new order
                    #     self.worker.ledger.add_order(result['code'], -result['delta_order'], self.worker.ledger.order_price[result['code']])
                    #     # add(update) new old order
                    #     self.worker.ledger.add_order(result['code'], result['delta_order'], result["order_price"])

                    # elif result["result_type"] == "cancel/reject":
                    #     if result["order_type"] == "buy":
                    #         # remove old cash(pre-applied cash when sending the order to account_node) and update to new cash
                    #         self.worker.ledger.add_cash(
                    #             result['delta_order'] * (self.worker.ledger.order_price[result['code']] - result["order_price"])
                    #         )
                    #     # remove old order(pre-applied order when sending the order to account_node) and update to new order
                    #     self.worker.ledger.add_order(result['code'], -result['delta_order'], self.worker.ledger.order_price[result['code']])


class AccountWorkerCreon(QObject):

    def __init__(self, account_number, commodity_number):
        """
        account_numer (str): account number
        commodity_number (str): commodity number, '1':주식, '2':선물/옵션
        """
        # Account and commodity number
        self.accnum = account_number
        self.comnum = commodity_number

        # Strategy dictionary: manages orders per strategy
        self.dict_startegy = {}

        # Order sign converter
        self.sign_converter = {"1":-1, "2":1}

        # Real-time conclusion transaction management
        qtobj_conclusion = QtObjCpConclusion()
        obj_conclusion = ObjCpConclusion(qtobj_conclusion)
        qtobj_conclusion.evt_subscribe_data.connect(self.unpack)
        obj_conclusion.subscribe(
            subscribe_key = "conclusion",
            output_dict = {9:"code",3:"qty",4:"prc",5:"ordnum",6:"origin_ordnum",12:"trdtype",14:"conclutype"}
        )

    def add_strategy(self, identity):
        # wating is not yet submitted orders
        # submitted is submitted orders but not yet executed
        self.dict_startegy[identity] = {"waiting":{},"submitted":{}} # code: order(float)

    def pack(self, identity, order_request):
        packed_orders = []
        for order in order_request:
            if order["code"] not in self.dict_startegy[identity]["waiting"]:
                self.dict_startegy[identity]["waiting"][order["code"]] = order["qty"]
            else:
                self.dict_startegy[identity]["waiting"][order["code"]] += order["qty"]
            packed_order = {
                0: str(np.sign(order["qty"])+1),
                1: self.accnum,
                2: self.comnum,
                3: order["code"],
                4: abs(int(order["qty"])),
                5: int(order["prc"]),
                7: "0",
                8: "03",
            }
            packed_orders.append(packed_order)
        return packed_orders

    @pyqtSlot(ThreadData)
    def unpack(self, order_result, node_self):
        ordres = order_result.data
        node_self._queue_order_result.put_nowait(ordres)
        # {9:"code",3:"qty",4:"prc",5:"ordnum",6:"origin_ordnum",12:"trdtype",14:"conclutype"}
        pass

        """
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

class QtAccountNode(GridQtNodeParent):

    _node_type = "AccountNode"

    ### Commands ###

    ### Main ###
    def start(self):
        # check settings before start
        super().start(self)

        # Order result queue
        self._queue_order_result = ThreadQueue()

        # self._threads["recv_send_order"][0].start()
        # self._threads["pub_order_result"][0].start()
        # self._call_lmsg(20, f"ncmd[start] [Success]")

    def set_ready(self):
        self._setlist["set_signal_node"] = False
        self._setlist["set_worker"] = False
        self._setlist["set_start"] = False
        self._setlist["set_ready"] = True

    def set_start(self, ip, router_port, push_port, pub_port):

        # Router port to receive order request from signal node
        self._socket_router = self._context.socket(zmq.ROUTER)
        self._socket_router.bind(f"tcp://{ip}:{router_port}")

        # Push port to send order request to execution node
        self._socket_push = self._context.socket(zmq.PUSH)
        self._socket_push.bind(f"tcp://{ip}:{push_port}")

        # Pub port to send order result to signal node
        self._socket_pub = self._context.socket(zmq.PUB)
        self._socket_pub.bind(f"tcp://{ip}:{pub_port}")

        self._threads["thd_order_req"] = [Thread(target=self._thd_order_req, daemon=True), True]
        self._threads["thd_order_result"] = [Thread(target=self._thd_order_result, daemon=True), True]

        self._setlist["set_start"] = True

    def set_signal_node(self, ip, port):
        self.pull_socket = self._context.socket(zmq.PULL)
        self.pull_socket.connect(f"tcp://{ip}:{port}")

        self._setlist["set_signal_node"] = True
    
    def set_worker(self, worker):
        # TODO check worker type
        self._worker = worker

    ### Threads ###

    def _thd_order_req(self):
        while self._threads["recv_send_order"][1]:
            [identity, order_request] = self._socket_router.recv_multipart()
            self._socket_router.send_multipart([identity, "success".encode("utf-8")]) # send success message to signal node)
            order_request = pickle.loads(order_request)

            order_request_list = self._worker.pack(identity, order_request) # save order request and pack it
            for order in order_request_list:
                self._socket_push.send_pyobj(order)

    def _thd_order_result(self):
        while self._threads["pub_order_result"][1]:
            # If empty queue then pass
            if self._queue_order_result.empty():
                pass
            else:
                # If not empty queue then send order result to signal node
                ordres = self._queue_order_result.get_nowait()

                if ordres["conclutype"] == ("접수","확인"):
                    strategy_with_order = []
                    for identity, indict in self._worker.dict_startegy.items():
                        if ordres["code"] in indict["waiting"]:
                            strategy_with_order.append(identity)

                    if len(strategy_with_order) == 0:
                        # TODO some error handling
                        pass
                    elif len(strategy_with_order) == 1:
                        # size and sign of conclued order
                        delta_order = (self._worker.sign_converter[ordres["trdtype"]] * ordres["qty"])
                        # handle waiting order dictionary
                        self._worker.dict_startegy[strategy_with_order[0]]["waiting"][ordres["code"]] -= delta_order
                        if self._worker.dict_startegy[strategy_with_order[0]]["waiting"][ordres["code"]] == 0:
                            del self._worker.dict_startegy[strategy_with_order[0]]["waiting"][ordres["code"]]
                        # handle submitted order dictionary
                        if ordres["code"] not in self._worker.dict_startegy[strategy_with_order[0]]["submitted"]:
                            self._worker.dict_startegy[strategy_with_order[0]]["submitted"][ordres["code"]] = delta_order
                        else:
                            self._worker.dict_startegy[strategy_with_order[0]]["submitted"][ordres["code"]] -= delta_order
                    else:
                        # TODO mutiple strategy in a single account
                        # Proper order distrubution algorithmn is needed
                        pass

                elif ordres["conclutype"] == "체결":
                    strategy_with_order = []
                    for identity, indict in self._worker.dict_startegy.items():
                        if ordres["code"] in indict["waiting"]:
                            strategy_with_order.append(identity)
                        

                    pass

                else:
                    if ordres["conclutype"] == "확인":
                        # TODO some error handling
                        pass
                    # TODO some error handling
                    pass
                    
                

    # def _thd_order_result(self):
    #     while self._threads["pub_order_result"][1]:
    #         order_result = self._worker.get_result()
    #         # TODO 
    #         # We need an algorithmn to figure out where this order result should go (which signal node?)
    #         # For FIFO, we need to use queue for order request per stock
    #         # For quantitty based (most easy)
    #         # For fair-share, we need to use some kind of weight (most difficult, consider both time and quantity)

    #         order_result_list = self._worker.unpack(order_result) # Here we unpack order result and allocate it to each signal node
    #         for result in order_result_list: # result is tuple with (identity, order_result)
    #             self._socket_pub.send_multipart([result[0], result[1]])
