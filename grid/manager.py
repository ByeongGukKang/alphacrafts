import asyncio
from time import sleep
from threading import Thread
import pickle
import sys

import zmq
import zmq.asyncio


class GridManager:

    def __init__(self, slack_channel, ip, pub_port, router_port, logger, messanger):
        """

        slack_channel (str): name of the slack channel \n
        ip (str): ip address of the server \n
        pub_port (int): port number for the PUB socket \n
        router_port (int): port number for the ROUTER socket \n
        logger (logging.Logger): logger object \n
        messanger (Slack Bot): messanger object \n

        """
        ### Commons
        self._logger = logger
        self._logger_match = {10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR", 50: "CRITICAL"}
        # TODO messager must be improved
        # messanger must have get_channel_id, post_channel_message, get_last_message
        self._messanger = messanger
        self._slack_channel_id = self._messanger.get_channel_id(slack_channel)

        self._context = zmq.asyncio.Context()
        self._threads = {} # {thread_name: (threading.Thread, stopflag)}

        ### Details
        self._pub_socket = self._context.socket(zmq.PUB)
        self._router_socket = self._context.socket(zmq.ROUTER)
        self._pub_socket.bind(f"tcp://{ip}:{pub_port}")
        self._router_socket.bind(f"tcp://{ip}:{router_port}")

        # {identity: {type:, status:,}}
        # type is one in []
        # status is one in [ready, running]
        self._nodes = {} 


    ### Main ###
    def start(self):
        self._messanger.post_channel_message(self._slack_channel_id, "###  Alphacrafts GridManager is Ready ###")
        self._messanger.post_channel_message(self._slack_channel_id, "-->")
        self._threads["slack"] = [Thread(target=self._thd_slack), True] # main thread
        self._threads["zmq"] = [Thread(target=self._thd_zmq, daemon=True), True]
        self._threads["slack"][0].start()
        self._threads["zmq"][0].start()
        self._call_log(20, "call[USER:START] [success]")


    ### Threads ###
        
    # Command Prompt for USER
    # Read slack channel & Execute cmd command
    def _thd_slack(self): 
        lastest_message = None  # lastest message
        while self._threads["slack"][1]:
            sleep(1) # To avoid Slack API limit
            # Get message
            message = self._messanger.get_last_message(self._slack_channel_id)
            # Parse message & Execute command
            if message != lastest_message: # Execute only when new message is arrived
                lastest_message = message # update lastest message
                if message == "--&gt;": # ignore (--> is used for return)
                    continue

                self._call_log(20, f"cmd[USER:INPUT] [received] {message}")
                
                parts = message.split(" ") # parsing
                command_type = parts[0] # cmd or ncmd
                # Check syntax
                isWrongSyntax = False
                if (command_type != "cmd") & (command_type != "ncmd"):
                    isWrongSyntax = True
                elif len(parts) == 1:
                    isWrongSyntax = True
                else:
                    command_and_args = parts[1].split('(')
                    if len(command_and_args) == 1:
                        isWrongSyntax = True
                
                # If syntax is wrong, ignore the command
                if isWrongSyntax:
                    self._call_log(20, f"cmd[USER:INPUT] [ignored] {message}")
                    continue
            
                # Parsed command & arguments
                command = command_and_args[0]
                command_args = command_and_args[1].split(')')[0].split(',')

                # Execute command
                if command_type == "cmd":
                    try:
                        if command_args == ['']:
                            getattr(self, f"_cmd_{command}")()
                        else:
                            getattr(self, f"_cmd_{command}")(*command_args)
                    except Exception as e:
                        self._call_lmsg(30, f"cmd[USER:{command}] [fail] {e}")
                    else:
                        self._call_log(20, f"cmd[USER:{command}] [success]")

                elif command_type == "ncmd":
                    try:
                        if command_args[0] not in self._nodes:
                            self._call_lmsg(30, f"ncmd[USER:{command}] [fail] identity not found")
                        else:
                            getattr(self, f"_ncmd_{command}")(command_args[0], *command_args[1:])
                    except Exception as e:
                        self._call_lmsg(30, f"ncmd[USER:{command}] [fail] {e}")
                    else:
                        self._call_log(20, f"ncmd[USER:{command}] [success]")

    # Read ZMQ router socket & Execute command
    # Accept ncall from nodes
    def _thd_zmq(self):
        while self._threads["zmq"][1]:
            # command_and_args is a dictionary {'cmd': command, 'args': command_args}
            identity, command_and_args = asyncio.run(self._router_recv())
            self._call_log(20, f"ncall[{identity}:INPUT] {command_and_args}")

            # In node, all cmd is called automatically by code, so just trust the syntax and execute command

            # Execute command
            try:
                if command_and_args['args'] == ():
                    getattr(self, f"_ncall_{command_and_args['ncall']}")(identity)
                else:
                    getattr(self, f"_ncall_{command_and_args['ncall']}")(identity, *command_and_args['args'])
            except Exception as e:
                self._call_lmsg(40, f"ncall[{identity}:{command_and_args['ncall']}] [fail] {e}")
            else:
                self._call_log(20, f"ncall[{identity}:{command_and_args['ncall']}] [success]")


    ### Network ###
    # methods for zmq socket
    async def _pub_send(self, identity, data):
        """
        identity (str): identity of node \n
        data (python object): data to send 
        """
        [identity, data] = await self._pub_socket.send_multipart([identity.encode("utf-8"), pickle.dumps(data)])
        identity, data = identity.decode("utf-8"), pickle.loads(data)
        return identity, data
        
    async def _router_send(self, identity, data):
        """
        identity (str): identity of node \n
        data (python object): data to send 
        """
        await self._router_socket.send_multipart([identity.encode("utf-8"), pickle.dumps(data)])
        
    async def _router_recv(self):
        [identity, data] = await self._router_socket.recv_multipart()
        identity, data = identity.decode("utf-8"), pickle.loads(data)
        return identity, data


    ### Commands ###
    # Commands have four types
    # call: called by grid manager itself
    # ncall: called by nodes
    # cmd: called by users
    # ncmd: called by users, but executed by nodes, first argument must be identity of node

    ## call Commands

    def _call_return(self):
        self._messanger.post_channel_message(self._slack_channel_id, "-->")
                
    def _call_msg(self, message):
        """
        message (str): message
        """
        self._messanger.post_channel_message(self._slack_channel_id, message)
        self._messanger.post_channel_message(self._slack_channel_id, "-->")
    
    def _call_log(self, level, message):
        """
        log format: \n
            cmd[USER:START] [success] \n
            ncall[{identity}:{command_and_args['cmd']}] [success] \n
            ncall[USER:{command_and_args['cmd']}] [fail] {e}
        """
        self._logger.log(level, message)

    def _call_lmsg(self, level, message):
        """
        log format: \n
            cmd[USER:START] [success] \n
            ncall[{identity}:{command_and_args['cmd']}] [success] \n
            ncall[USER:{command_and_args['cmd']}] [fail] {e}
        """
        self._logger.log(level, f"call[MANAGER:lmsg] [success] {message}")
        self._messanger.post_channel_message(self._slack_channel_id, f"[{self._logger_match[level]}] {message}")
        self._messanger.post_channel_message(self._slack_channel_id, "-->")

    ## ncall Commands
    # Only called by nodes, not by user
        
    def _ncall_msg(self, identity, *args):
        """
        identity (str): identity of node\n
        args[0] (str): log level
        args[1] (str): message
        """
        self._logger.log(args[0], f"ncall[{identity}:msg] [success] {args[1]}")
        self._messanger.post_channel_message(self._slack_channel_id, f"ncall[{identity}:msg]: {args[1]}")
        self._messanger.post_channel_message(self._slack_channel_id, "-->")

    def _ncall_connect(self, identity, *args):
        """
        identity (str): (temporal) identity of node \n
        args[0] (str): idenity of node \n
        args[1] (str): type of node \n
        """
        if args[0] not in self._nodes:
            self._nodes[identity] = {"type":args[0], "status":"ready"}
            asyncio.run(self._router_send(identity, "success"))
            self._call_log(20, f"ncall[{identity}:connect] [success]")
        else:
            asyncio.run(self._router_send(identity, "fail"))
            self._call_lmsg(40, f"ncall[{identity}:connect] [fail] identity already exist")

    ## cmd Commands
    # cmd commands MUST END with call commands, except for special commands

    def _cmd_help(self, *args):
        if len(args) == 0:
            self._call_msg(        
                """
                + represents optional parameter

                cmd help(+cmd)
                cmd shutdown()
                cmd nodelist()
                cmd nodeinfo(identity)
                cmd echo(message)
                """
            )
        else:
            # TODO wrtie help for each command
            self._call_msg(          
                """
                + represents optional parameter

                cmd help(+cmd)
                cmd shutdown()
                cmd nodelist()
                cmd nodeinfo(identity)
                cmd echo(message)
                """
            )

    # _cmd_shutdown is special cmd command that does not need self._call_return()
    def _cmd_shutdown(self):
        self._messanger.post_channel_message(self._slack_channel_id, "### GridManager is Closed ###")
        self._call_log(20, "call[USER:SHUTDOWN] [success]")
        sys.exit(0)

    def _cmd_nodelist(self):
        self._call_msg(str(self._nodes))

    def _cmd_nodeinfo(self, *args):
        """
        args[0] (str): identity of node
        """
        self._call_msg(str(self._nodes[args[0]]))

    def _cmd_echo(self, *args):
        """
        args[0] (str): echo message
        """
        self._call_msg(" ".join(arg for arg in args))

    ## ncmd Commands
    # ncmd commands MUST END with ncall commands, except for special commands
        
    def _ncmd_help(self, *args):
        if len(args) == 0:
            self._call_msg(        
                """
                + represents optional parameter

                ncmd help(+ncmd)
                ncmd start(identity)
                ncmd shutdown(identity)
                ncmd echo(identity, message)
                """
            )
        else:
            # TODO wrtie help for each command
            self._call_msg(          
                """
                + represents optional parameter

                ncmd help(+ncmd)
                ncmd start(identity)
                ncmd shutdown(identity)
                ncmd echo(identity, message)
                """
            )

    def _ncmd_start(self, identity, *args):
        """
        identity (str): identity of node
        """
        asyncio.run(self._pub_send(identity, {"ncmd": "start", "args":['']}))
        self._call_return()

    def _ncmd_shutdown(self, identity, *args):
        """
        identity (str): identity of node \n
        """
        asyncio.run(self._pub_send(identity, {"ncmd": "shutdown", "args":['']}))
        del(self._nodes[identity])
        self._call_return()

    # _cmd_necho is special cmd command that does not need self._call_return()
    def _ncmd_echo(self, identity, *args):
        """
        identity (str): identity of node\n
        args[0] (str): echo message
        """
        asyncio.run(self._pub_send(identity, {"ncmd": "echo", "args":args[0]}))


