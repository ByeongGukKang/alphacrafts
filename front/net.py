import asyncio
from copy import deepcopy
from dataclasses import dataclass
import inspect
import types
from typing import Union

from blacksheep import URL
from blacksheep.headers import Headers
from blacksheep.client import ClientSession
from blacksheep.contents import Content
from orjson import loads as orjson_loads
from picows import ws_connect, WSFrame, WSListener, WSMsgType, WSTransport

from alphacrafts.cy.share import cyClsMemoryArena
from alphacrafts.front.meta import MetaAPILimit, MetaWebsocketParser


__all__ = [
    "HttpClient",
    "HttpRequest",
    "HttpResponse",
    "WebSocketCallback",
    "WebSocketClient"
]


@dataclass
class HttpRequest:
    """
    Http request object.
    """
    path:     bytes
    """Request path in bytes, use b'url'."""
    method:   str
    """Http method. GET, POST, PUT, DELETE."""
    header:   dict  = None
    params:   dict  = None
    datatype: bytes = None
    data:     bytes = None


@dataclass
class HttpResponse:
    """
    Http response object.
    """
    status:  int            = None
    header:  Headers        = None
    body:    bytes          = None
    err:     Exception|None = None
    """Error as value. if None no error."""

    @property
    def jsonh(self):
        """
        Http response header as a dictionary.
        """
        return {k.decode(): v.decode() for k, v in self.header.items()}

    @property
    def textb(self):
        """
        Http response body as a string.
        """
        return self.body.decode()

    @property
    def jsonb(self) -> tuple[dict, Exception]:
        """
        Http response body as a JSON object.
        """
        try:
            return orjson_loads(self.body), None
        except Exception as e:
            return None, e


class HttpClient:

    def __init__(self, api_limit: MetaAPILimit, url: str, memsize: int=None, timeout: float=5, **kwargs):
        """
        Http client object.

        Args:
            api_limit (MetaAPILimit): rate limiting object.
            url (str): base url.
            timeout (float, optional): request timeout.
                Defaults to 5.
            **kwargs: additional arguments for blacksheep.client.ClientSession.
        """
        self.api_limit = api_limit
        "Rate limiting object."
        self.url = url
        "Base url."

        self.session = ClientSession(
            base_url        = url,
            request_timeout = timeout,
            **kwargs
        )
        "Client session object."
        self.session.delay_before_retry = 0 # Blacksheep retry delay
        
        self.__isidle = asyncio.Event() # Prevent session close while request is in progress
        self._chk_limit = None # Default function for rate limiting
        if api_limit is not None:
            self._chk_limit = api_limit.consume_func # Direct reference to the semaphore(queue)

        if memsize is None:
            pass
        elif isinstance(memsize, int) and (memsize > 0):
            self.__respArena = cyClsMemoryArena(memsize, HttpResponse())
            self.free = self.__respArena.free
        else:
            raise ValueError("memsize must be int > 0 or None")
        
    def free(self, obj: object): # Method hidden by __init__, only used when memsize is set
        """
        Free the object. Use this method to free the response object from .request_unsafe().
        """
        raise NotImplementedError("0 memsize, MemoryArena is not used")

    def set_api_limit(self, api_limit: MetaAPILimit):
        """
        Set rate limiting object.

        Args:
            api_limit (MetaAPILimit): rate limiting object.
        """
        self.api_limit = api_limit
        self._chk_limit = api_limit.consume_func

    async def close(self):
        """
        Close the client session.
        """
        try:
            await self.__isidle.wait() # Wait for the last request to finish
            await self.session.close()
            self.session = None
        except Exception as e:
            return e

    async def request(self, req: HttpRequest) -> HttpResponse:
        """
        Send a http request.

        Args:
            req (HttpRequest): req object.
        """
        self.__isidle.clear() # Busy state, avoid session close while request is in progress 
        resp = HttpResponse() # Response object must be created for every request, not reused
        try:
            if self._chk_limit is not None:
                await self._chk_limit() # API rate limiting
            match req.method:    # Match HttpMethod, Send request
                case 'GET':
                    response = await self.session.get(
                        URL(req.path), req.header, req.params
                    )
                case 'POST':
                    response = await self.session.post(
                        URL(req.path), Content(req.datatype, req.data), req.header, req.params
                    )
                case 'PUT':
                    response = await self.session.put(
                        URL(req.path), Content(req.datatype, req.data), req.header, req.params,
                    )
                case 'DELETE':
                    response = await self.session.delete(
                        URL(req.path), Content(req.datatype, req.data), req.header, req.params,
                    )
                case _:
                    raise ValueError(f'Unsupported HttpMethod: {req.method}')

            resp.status = response.status
            resp.header = response.headers
            resp.body   = await response.read()

        except Exception as e:
            resp.err = e # Error as value

        finally:
            self.__isidle.set()

        return resp

    async def request_unsafe(self, req: HttpRequest) -> HttpResponse:
        """
        Send a http request. memsize MUST be set.

        - Unsafe version, response object is deallocated by the user.

        Args:
            req (HttpRequest): request object.

        Note:
            Requires manual memory management.
            Use free() to free the resposne object.
        """
        resp: HttpResponse = self.__respArena.alloc()
        try:
            await self._chk_limit() # API rate limiting
            match req.method:    # Match HttpMethod, Send request
                case 'GET':
                    response = await self.session.get(
                        URL(req.path), req.header, req.params
                    )
                case 'POST':
                    response = await self.session.post(
                        URL(req.path), Content(req.datatype, req.data), req.header, req.params
                    )
                case 'PUT':
                    response = await self.session.put(
                        URL(req.path), Content(req.datatype, req.data), req.header, req.params,
                    )
                case 'DELETE':
                    response = await self.session.delete(
                        URL(req.path), Content(req.datatype, req.data), req.header, req.params,
                    )
                case _:
                    raise ValueError(f'Unsupported HttpMethod: {req.method}')

            resp.status = response.status
            resp.header = response.headers
            resp.body   = await response.read()
            resp.err    = None # Error as value
            resp._jsonh, resp._textb, resp._jsonb = None, None, None

        except Exception as e:
            resp.err = e # Error as value

        return resp

    async def request_unsafe2(self, req: HttpRequest, resp: HttpResponse) -> HttpResponse:
        """
        Send a http request.
        
        - User must handle both allocation and deallocation of response obect.

        Args:
            req (HttpRequest): request object.
            resp (HttpResponse): response object.
        """
        try:
            await self._chk_limit() # API rate limiting
            match req.method:       # Match HttpMethod, Send request
                case 'GET':
                    response = await self.session.get(
                        URL(req.path), req.header, req.params
                    )
                case 'POST':
                    response = await self.session.post(
                        URL(req.path), Content(req.datatype, req.data), req.header, req.params
                    )
                case 'PUT':
                    response = await self.session.put(
                        URL(req.path), Content(req.datatype, req.data), req.header, req.params,
                    )
                case 'DELETE':
                    response = await self.session.delete(
                        URL(req.path), Content(req.datatype, req.data), req.header, req.params,
                    )
                case _:
                    raise ValueError(f'Unsupported HttpMethod: {req.method}')

            resp.status = response.status
            resp.header = response.headers
            resp.body   = await response.read()
            resp.err    = None

        except Exception as e:
            resp.err = e # Error as value

        return resp


@dataclass
class WebSocketCallback:
    """
    Callback object for websocket response.

    Args:
        f (types.FunctionType): callback function.
            1st parameter must be 'data', data from websocket.
        args (tuple, optional): additional arguments for the callback function.
            Defaults to ( ).
    """
    f: types.FunctionType
    args: tuple = ()

    def __post_init__(self):
        if tuple(inspect.signature(self.f).parameters.keys())[0] != "data":
            raise ValueError("1st parameter of the callback function must be 'data'.")
        if inspect.iscoroutinefunction(self.f):
            raise TypeError("Callback function must be synchronous.")


class WebSocketClient: 

    def __init__(self, api_limit: MetaAPILimit, parser: MetaWebsocketParser, url: str, **kwargs):
        """
        Websocket client object.
            parser should be implemented in impl.py.

        Args:
            api_limit (MetaAPILimit): rate limiting object.
            parser (MetaWebsocketParser): parser object.
            url (str): base url.
            **kwargs: additional arguments for picows.ws_connect.
        """
        self.api_limit = api_limit
        "Rate limiting object."

        parser = parser(self) # Initialize parser
        self._parser = parser 
        "Websocket subscription/frame parser object."
        self._parse_sub = parser.parse_sub
        self._parse_frame = parser.parse_frame
        self._url = url
        "Base url."
        self._kwargs = kwargs
        "Additional arguments for picows.ws_connect."
        self._callback_dict: dict[str, WebSocketCallback] = dict()

        self._trans = None
        self.__isconnected = asyncio.Event()

    @property
    def tasks(self) -> list[asyncio.Task]:
        "Return list of all tasks."
        return [asyncio.create_task(self._connect(), name="WebSocketClient._connect")]
    
    @property
    def cbdict(self) -> dict[str, WebSocketCallback]:
        "Callback dictionary."
        return self._callback_dict
    
    def set_api_limit(self, api_limit: MetaAPILimit):
        """
        Set rate limiting object.

        Args:
            api_limit (MetaAPILimit): rate limiting object.
        """
        self.api_limit = api_limit

    async def close(self):
        """
        Close the websocket connection.
        """
        try:
            self._trans.send_close()
            await self._trans.wait_disconnected()
        except Exception as e:
            return e

    async def _connect(self):
        """
        Connect to the websocket server.
            It is not recommended to call this method directly.
            This method is called automatically when NewSystem() is called.
        """
        class WebSocketHandler(WSListener):

            parser = self._parse_frame
            cbdict = self._callback_dict

            def on_ws_connected(self, trans: WSTransport):
                cb = self.cbdict.get('connected')
                if cb is not None:
                    cb.f(None, *cb.args)

            def on_ws_disconnected(self, trans: WSTransport):
                cb = self.cbdict.get('disconnected')
                if cb is not None:
                    cb.f(None, *cb.args)

            def on_ws_frame(self, _: WSTransport, frame: WSFrame):
                cbkey, data = self.parser(frame)
                cb = self.cbdict.get(cbkey)
                if cb is None:
                    return
                if len(data) == 0:
                    cb.f(data[0], *cb.args)
                    return
                for d in data:
                    cb.f(d, *cb.args)

        self._trans, _ = await ws_connect(WebSocketHandler, self._url, **self._kwargs)
        self.__isconnected.set()

    def _send(self, data: bytes):
        """
        Send raw bytes directly.
            Not recommended to call this method directly.
            Use .send() instead.

        Args:
            data (bytes): data to send.

        """
        # Send raw bytes directly
        self._trans.send(WSMsgType.BINARY, data)

    async def send(self, data: Union[bytes,any]):
        """
        Send data to the websocket server.
            Data is parsed by the parser object.
            parser.parse_sub() is called to parse the data.

        Args:
            data (bytes|any): data to send.
        """
        # Type check for data is done in parser.parse_sub()
        await self.__isconnected.wait() # Wait for connection
        self._trans.send(WSMsgType.BINARY, self._parse_sub(data))

    def set_callback(self, cbkey: str, callback: WebSocketCallback):
        """
        Set callback function for the websocket response.

        Args:
            cbkey (str): callback key.
            callback (WebSocketCallback): callback object.
        """
        if not isinstance(cbkey, str):
            raise TypeError("cbkey must be string.")
        if (callback is not None) & (not isinstance(callback, WebSocketCallback)):
            raise TypeError("callback must be WebSocketCallback object.")
        self._callback_dict[cbkey] = callback
