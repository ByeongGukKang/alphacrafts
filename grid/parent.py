import asyncio
import datetime as dt
from time import sleep
from threading import Thread
import pickle
import sys

from PyQt5.QtWidgets import QMainWindow

import zmq
import zmq.asyncio


class GridNodeParent:

    _node_type = "Parent"

    def __init__(self, identity, master_ip, master_pub_port, master_router_port, logger):
        """

        identity (str): identity of the node \n
        master_ip (str): ip address of the server \n
        master_pub_port (int): PUB port number of master \n
        master_router_port (int): REP port number of master \n
        logger (logging.Logger): logger object \n

        """
        
        ### Commons
    
        # Check settings before start
        self._setlist = {"set_ready":False}

        # logger
        self._logger = logger
        self._logger_match = {10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR", 50: "CRITICAL"}

        # zmq context and threads
        self._context = zmq.asyncio.Context()
        self._threads = {} # {thread_name: (threading.Thread, stopflag)}

        ### Details
        self._identity = identity # name of the node, must be unique

        # Initialize sockets with random identity
        random_identity = self._node_type + dt.datetime.now().strftime("%Y%m%d%H%M%S%f") # Use timestamp as random identity
        self._sockets = {}
        self._sockets["manager_router"] = self._context.socket(zmq.DEALER) # connect to master router socket as dealer
        self._sockets["manager_router"].setsockopt(zmq.IDENTITY, random_identity.encode("utf-8"))
        self._sockets["manager_router"].connect(f"tcp://{master_ip}:{master_router_port}")

        self._sockets["manager_pub"] = self._context.socket(zmq.PUB) # connect to master pub socket as publisher
        self._sockets["manager_pub"].setsockopt(zmq.IDENTITY, random_identity.encode("utf-8"))
        self._sockets["manager_pub"].connect(f"tcp://{master_ip}:{master_pub_port}")
    
        ### Communicate with master ###
        self._threads["thd_master"] = [Thread(target=self._thd_master, daemon=True), True] # (thread, stopflag)
        self._threads["thd_master"][0].start()

        ### Connect to master
        asyncio.run(
            self._router_send({"ncall":"connect","args":(self._identity, self._node_type)}) # node_type is defined in child class
        )
        res = asyncio.run(self._router_recv())
        if res == "success":
            self._sockets["manager_router"].setsockopt(zmq.IDENTITY, self._identity.encode("utf-8")) # Change identity
            self._sockets["manager_pub"].setsockopt(zmq.SUBSCRIBE, self._identity.encode("utf-8"))

    ## Main ###
    def start(self):
        for key, value in self._setlist.items():
            if value == False:
                raise Exception(f"Error: {key} is not set")

    ### Threads ###
    def _thd_master(self):
        while True:
            if self._threads["thd_master"][1]:
                res = asyncio.run(self._pub_recv())
                if "ncmd" in res:
                    try:
                        getattr(self, f"_ncmd_{res['ncmd']}")(res["args"])
                    except Exception as e:
                        self._call_lmsg(40, f"ncmd[USER:{res['ncmd']}] [fail] {e}")
                
    ### Network ###
    async def _pub_recv(self):
        """
        """
        [identity, data] = await self._sockets["manager_pub"].recv_multipart()
        return pickle.loads(data)
        
    async def _router_send(self, data):
        """
        data (python object): data to send\n
        """
        await self._sockets["manager_router"].send_pyobj(data)
        
    async def _router_recv(self):
        """
        timeout (float): timeout in seconds
        """
        [identity, data] = await self._sockets["manager_router"].recv_multipart()
        return pickle.loads(data)
    

    ### Commands ###
    ## call commands 
        
    def _call_msg(self, level, message):
        asyncio.run(self._router_send({"ncall":"msg", "args":(level, message)}))
        res = asyncio.run(self._router_recv())
        if res != "success":
            self._call_log(40, f"ncall[msg] [fail] {res}")

    def _call_log(self, level, message):
        self._logger.log(level, message)

    def _call_lmsg(self, level, message):
        """
        log format: \n
            cmd[USER:START] [success] \n
            ncall[{identity}:{command_and_args['cmd']}] [success] \n
            ncall[USER:{command_and_args['cmd']}] [fail] {e}
        """
        self._call_log(level, message)
        self._call_msg(level, message)

    ## node common commands, cmd commands
    def _ncmd_start(self):
        """
        Start node
        """
        self.start()
        self._call_lmsg(20, f"ncmd[MANAGER:start] [success]")

    def _ncmd_shutdown(self):
        """
        Stop node
        """
        self._threads["thd_master"][1] = False
        self._call_lmsg(20, f"ncmd[MANAGER:shutdown] [success]")
        sys.exit(0)

    def _ncmd_echo(self, *arg):
        """
        arg[0] (str): message \n
        """
        self._call_msg(20, arg[0])

    def _ncmd_set_identity(self, *arg):
        """
        arg[0] (str): new identity \n
        """
        self._identity = arg[0]
        for socket in self._sockets.values():
            socket.setsockopt(zmq.IDENTITY, self._identity.encode("utf-8"))
        self._call_lmsg(20, f"ncmd[MANAGER:setid] [success]")


class GridQtNodeParent(QMainWindow, GridNodeParent):

    def __init__(self, identity, master_ip, master_pub_port, master_router_port, logger):
        QMainWindow.__init__()
        GridNodeParent.__init__(self, identity, master_ip, master_pub_port, master_router_port, logger)

    ## Main ###
    def start(self):
        GridNodeParent.start(self)

    ### Threads ###
    def _thd_master(self):
        GridNodeParent._thd_master(self)

    ### Network ###
    async def _pub_recv(self):
        return GridNodeParent._pub_recv(self)
    
    async def _router_send(self, data):
        GridNodeParent._router_send(self, data)
    
    async def _router_recv(self):
        return GridNodeParent._router_recv(self)
    
    ### Commands ###
    ## call commands

    def _call_msg(self, level, message):
        GridNodeParent._call_msg(self, level, message)
    
    def _call_log(self, level, message):
        GridNodeParent._call_log(self, level, message)

    def _call_lmsg(self, level, message):
        GridNodeParent._call_lmsg(self, level, message)

    ## node common commands, cmd commands
    def _ncmd_start(self):
        GridNodeParent._ncmd_start(self)
    
    def _ncmd_shutdown(self):
        GridNodeParent._ncmd_shutdown(self)
    
    def _ncmd_echo(self, *arg):
        GridNodeParent._ncmd_echo(self, *arg)

    