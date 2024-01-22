import asyncio
import datetime as dt
import inspect
from time import sleep
from threading import Thread
from queue import Queue as ThreadQueue
import pickle

import numpy as np

from PySide2.QtWidgets import QApplication
from PySide2.QtCore import QObject, QThread
from PySide2.QtCore import Slot as QSlot

import zmq

from alphacrafts.bkd.share.qt import ThreadData, QtLocalClock
from alphacrafts.bkd.creon.wrapper.account import ObjCpCybos 
from alphacrafts.bkd.creon.wrapper.trade import ObjCpTdUtil, ObjCpTd0311
from alphacrafts.bkd.creon.wrapper.trade import ObjCpConclusion, _ObjCpConclusionEvent, QtObjCpConclusion
from alphacrafts.grid.parent import GridNodeParent, GridQtNodeParent
from alphacrafts.grid.utils import Ledger


# Qtworkers in DataQtNode are synchronized by its local clock
# Clock generate tick --> tick emit signal --> signal trigger worker to request data
# --> worker request data --> worker emit data --> data received by worker_slot (self.update_memory)
# --> if the memory is full --> aggregate the memory --> publish the data
class QtDataNode(GridQtNodeParent):

    _node_type = "DataQtNode"


    ### Network ###
    async def _net_pub_data(self, data):
        await self._socket_data_pub.send_pyobj(data)

    ### Commands ###

    def _ncmd_clockoff(self, *args):
        """
        *args[0] (bool): clock off
        """
        self.qt_local_clock.time_stop = args[0]
        self._call_lmsg(20, f"ncmd[MANAGER:clockoff] [success] {args[0]}")

    def _ncmd_shutdown(self):
        """
        Shutdown node
        """
        # Stop all threads
        self._threads["thd_master"][1] = False
        self.qt_local_clock.go_flag = False
        self.qt_local_clock.quit()
        # Disconnect all signals
        for worker_name, worker in self.workers.items():
            self.qt_local_clock.evt_local_time.disconnect(worker.get_data)
            worker.evt_market_data.disconnect(self.workers_slot[worker_name])
        # Closing message
        self._call_lmsg(20, f"ncmd[MANAGER:shutdown] [success]")
        # Close the program
        import sys
        QApplication.quit()
        sys.exit(0)
    
    def _ncmd_set_identity(self, *args):
        super()._ncmd_set_identity(*args)
        
    ### Main ###
    def start(self): # Must Defined
        # check settings before start
        super().start()

        # start threads
        self.qt_local_clock.start()
        # This is for not Qt workers
        for worker_name, worker in self.workers.items():
            self.qt_local_clock.evt_local_time.connect(worker.get_data)
            worker.evt_market_data.connect(self.workers_slot[worker_name])
        self._call_lmsg(20, f"ncmd[MANAGER:start] [success]")

    def set_ready(self):
        self._setlist["set_worker"] = False
        self._setlist["set_clock_frequency"] = False
        self._setlist["set_start"] = False
        self._setlist["set_ready"] = True

    def set_start(self, ip, data_pub_port, func_agg_memory):
        self._socket_data_pub = self._context.socket(zmq.PUB)
        self._socket_data_pub.setsockopt(zmq.CONFLATE, True)
        self._socket_data_pub.bind(f"tcp://{ip}:{data_pub_port}")

        # qt_local_clock - synchronize qt_workers
        self.qt_local_clock = QtLocalClock()  
        # qt_workers - emit data
        self.workers = {}  
        # pystSlots - receive data from qt_workers
        self.workers_slot = {}
        # data memory, received data are stored in this dict
        self.workers_memory = {}  # qt_workers memory
        # if the memory is full, then aggregate the memory and publish it
        self._func_agg_memory = func_agg_memory

        self._setlist["set_start"] = True

    def set_clock_frequency(self, frequency):
        """
        frequency (int): frequency of clock in ms
        """
        if not self._setlist["set_start"]:
            raise Exception("set_state must be called before set_clock_frequency")
        self.qt_local_clock.set_freq(frequency)

        self._setlist["set_clock_frequency"] = True

    def set_worker(self, worker_name, worker):
        """
        worker_name (str): name of worker
        worker : an instance of QtObserver 
        """
        if not (isinstance(worker, QThread) or isinstance(worker, QObject)):
            raise TypeError("worker must be an instance fro, 'QThread' or 'QObject'")
        self.workers[worker_name] = worker
        self.workers_slot[worker_name] = QSlot(ThreadData)(lambda data: self.update_meory(worker_name, data))

        self._setlist["set_worker"] = True

    def update_meory(self, worker_name, data):
        self.workers_memory[worker_name] = data

        if len(self.workers_memory) == len(self.workers):
            asyncio.run(self._net_pub_data(self._func_agg_memory(self.workers_memory)))
            self.workers_memory = {}



class SignalNode(GridNodeParent):

    _node_type = "SignalNode"

    ### Commands ###

    ### Main ###
    def start(self):
        # check settings before start
        super().start()

        # start threads
        self._threads["thd_send_order"][0].start()
        self._threads["thd_get_order_result"][0].start()
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
        self._threads["thd_order_send"] = [Thread(target=self._thd_send_order, daemon=True), True]
        self._threads["thd_order_result"] = [Thread(target=self._thd_get_order_result, daemon=True), True]

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
    def _thd_send_order(self):
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

    def _thd_get_order_result(self):
        while True:
            if self._threads["thd_order_result"][1]:
                # order_result is a dict with keys: "conclutype", "ordtype", "code", "orddelta", "ordprc", "positdelta", "exeprc"
                # result_type: "exe", "sub/adj", "can/rej" ("executed", "submitted"/"adjusted", "cancelled"/"rejected")
                [identity, result] = self._sockets["account_node_execution"].recv_multipart()
                result = pickle.loads(result) 

                # Acquire ledger lock
                with self._ledger.withlock():
                    if result["conclutype"] == "exe":
                        self._ledger.update_cash(-result['positdelta'] * result["exeprc"])
                        self._ledger.add_position(result['code'], result['positdelta'], result["exeprc"])
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
        # Account Node instance
        self.nodeSelf = None

        # Account and commodity number
        self.accnum = account_number
        self.comnum = commodity_number

        # Strategy dictionary: manages orders per strategy
        self.dict_startegy = {}

        # Order sign converter
        self.sign_converter = {"1":-1, "2":1}
        self.result_type_converter = {"체결":"executed", "접수":"accepted", "확인":"confirmed", "거부":"denied"}

        # Real-time conclusion transaction management
        qtobj_conclusion = QtObjCpConclusion()
        obj_conclusion = ObjCpConclusion(qtobj_conclusion)
        qtobj_conclusion.evt_subscribe_data.connect(self.unpack)
        obj_conclusion.subscribe(
            subscribe_key = "conclusion",
            output_dict = {9:"code",3:"ordqty",4:"exeprc",5:"ordnum",6:"origin_ordnum",12:"trdtype",14:"conclutype"}
        )

    def add_strategy(self, identity):
        # wating is not yet submitted orders
        # submitted is submitted orders but not yet executed
        self.dict_startegy[identity] = {"waiting":{},"submitted":{}} # code: order(float)

    def pack(self, identity, order_request):
        packed_orders = []
        for order in order_request:
            if order["code"] not in self.dict_startegy[identity]["waiting"]:
                self.dict_startegy[identity]["waiting"][order["code"]] = order["ordqty"]
            else:
                self.dict_startegy[identity]["waiting"][order["code"]] += order["ordqty"]
            packed_order = {
                0: str(np.sign(order["ordqty"])+1),
                1: self.accnum,
                2: self.comnum,
                3: order["code"],
                4: abs(int(order["ordqty"])),
                5: int(order["ordprc"]),
                7: "0",
                8: "03",
            }
            packed_orders.append(packed_order)
        return packed_orders

    @QSlot(ThreadData)
    def unpack(self, order_result):
        # ordres = {9:"code",3:"ordqty",4:"exeprc",5:"ordnum",6:"origin_ordnum",12:"trdtype",14:"conclutype"}
        ordres = order_result.data
        refined_ordres = {
            "code": ordres["code"],
            "conclutype": self.result_type_converter[ordres["conclutype"]],
            "positdelta": (self.sign_converter[ordres["trdtype"]]*ordres["ordqty"]),
            "exeprc": ordres["exeprc"],
        }
        self.nodeSelf._queue_order_result.put_nowait(refined_ordres)


class QtAccountNode(GridQtNodeParent):

    _node_type = "AccountNode"

    ### Commands ###

    ### Main ###
    def start(self):
        # check settings before start
        super().start()

        # Order result queue
        self._queue_order_result = ThreadQueue()

        self._threads["thd_order_req"][0].start()
        self._threads["thd_order_result"][0].start()
        self._call_lmsg(20, f"ncmd[MANAGER:start] [success]")

    def set_ready(self):
        # self._setlist["set_signal_node"] = False
        self._setlist["set_worker"] = False
        self._setlist["set_start"] = False
        self._setlist["set_ready"] = True

    def set_start(self, ip, router_port, push_port, pub_port):
        """

        ip (str): ip address \n
        router_port (int): port for router socket (receving signals) \n
        push_port (int): port for push socket (sending orders to execution node) \n
        pub_port (int): port for pub socket (sending order results to signal node) \n
        
        """

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
    
    def set_worker(self, worker):
        # TODO check worker type
        self._worker = worker
        self._worker.nodeSelf = self

        self._setlist["set_worker"] = True

    ### Threads ###

    def _thd_order_req(self):
        while self._threads["thd_order_req"][1]:
            [identity, order_request] = self._socket_router.recv_multipart()
            self._socket_router.send_multipart([identity, "success".encode("utf-8")]) # send success message to signal node)
            order_request = pickle.loads(order_request)

            order_request_list = self._worker.pack(identity, order_request) # save order request and pack it
            for order in order_request_list:
                self._socket_push.send_pyobj(order)

    def _thd_order_result(self):
        while self._threads["thd_order_result"][1]:
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
                        delta_order = (self._worker.sign_converter[ordres["trdtype"]] * ordres["ordqty"])
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
                        # We need an algorithmn to figure out where this order result should go (which signal node?)
                        # For FIFO, we need to use queue for order request per stock
                        # For quantitty based (most easy)
                        # For fair-share, we need to use some kind of weight (most difficult, consider both time and quantity)
                        pass

                elif ordres["conclutype"] == "체결":
                    strategy_with_order = []
                    for identity, indict in self._worker.dict_startegy.items():
                        if ordres["code"] in indict["waiting"]:
                            strategy_with_order.append(identity)

                    if len(strategy_with_order) == 0:
                        # TODO some error handling
                        pass
                    elif len(strategy_with_order) == 1:
                        # size and sign of conclued order
                        delta_order = (self._worker.sign_converter[ordres["trdtype"]] * ordres["ordqty"])
                        # handle submitted order dictionary
                        self._worker.dict_startegy[strategy_with_order[0]]["submitted"][ordres["code"]] -= delta_order
                        if self._worker.dict_startegy[strategy_with_order[0]]["submitted"][ordres["code"]] == 0:
                            del self._worker.dict_startegy[strategy_with_order[0]]["submitted"][ordres["code"]]
                        # send order result to signal node
                        self._socket_pub.send_multipart([strategy_with_order[0], pickle.dumps(ordres)])

                    else:
                        # TODO mutiple strategy in a single account
                        # Proper order distrubution algorithmn is needed
                        pass

                else:
                    if ordres["conclutype"] == "거부":
                        # TODO some error handling
                        pass
                    # TODO some error handling
                    pass
                    
                
class ExecutionWorkerCreon:

    def __init__(self):
        self._obj_limit = ObjCpCybos
        self._obj_trade = ObjCpTd0311

        self._ordtype_converter = {'1':"sell",'2':"buy"}

    def check_limit(self):
        remain_count = self._obj_limit.get_remain_count(0)
        if remain_count > 0:
            pass
        else:
            sleep(self._obj_limit.get_refresh_time(0) / 1000)

    def execute_order(self, ordreq):
        ObjCpTdUtil.init()
        self._obj_trade.set_input(ordreq)
        self._obj_trade.blockrequest()
        header = self._obj_trade.get_header(
            {
                0:"ordtype",1:"accnum",2:"comnum",3:"code",4:"ordqty",5:"ordprc",
                8:"ordnum",9:"accname",10:"codename",12:"ordcond",13:"ordcall"
            }
        )


class ExecutionNode(GridNodeParent):

    _node_type = "ExecutionNode"

    ### Commands ###

    ### Main ###
    def start(self):
        # check settings before start
        super().start()

        # start threads
        self._threads["thd_rev_order"][0].start()
        self._call_lmsg(20, f"ncmd[MANAGER:start] [success]")

    def set_ready(self):
        self._setlist["set_worker"] = False
        self._setlist["set_start"] = False
        self._setlist["set_ready"] = True

    def set_start(self, account_node_ip, account_node_push_port):
        # Pull port to receive order request from account node
        self._socket_pull = self._context.socket(zmq.PULL)
        self._socket_pull.bind(f"tcp://{account_node_ip}:{account_node_push_port}")

        # threads
        self._threads["thd_order_execute"] = [Thread(target=self._thd_order_execute, daemon=True), True]

        self._setlist["set_start"] = True

    def set_worker(self, worker):
        self._worker = worker

    ### Threads ###
    def _thd_order_execute(self):
        while True:
            if self._threads["thd_order_execute"][1]:
                self._worker.check_limit()
                order_request = self._socket_pull.recv_pyobj()
                self._worker.execute_order(order_request)

                

    