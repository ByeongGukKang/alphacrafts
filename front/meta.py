from abc import ABCMeta, abstractmethod
import asyncio
from collections.abc import Iterable
from enum import Enum
from typing import Union


from picows import WSFrame

__all__ = [
    "MetaWorker",
    "MetaAccount",
    "MetaWebsocketParser",
    "MetaAPILimit",
]


class MetaWebsocketParser(metaclass=ABCMeta):

    def __init__(self, _):
        pass

    @abstractmethod
    def parse_frame(self, frame: WSFrame) -> tuple[str, Iterable]:
        """
        Parse picows.WSFrame 
            returns a cbkey for callback dictionary key and a list of parsed data.
        """
        pass

    @abstractmethod
    def parse_sub(self, data: Union[bytes,any]) -> bytes:
        """
        Parse user subscription input.
            Must handle two cases, when data is bytes and others.
        """
        pass


class MetaAPILimit(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def consume_func(self):
        pass
    
    @abstractmethod
    async def consume(self):
        pass

    @abstractmethod
    async def acquire_socket(self):
        pass
    
    @abstractmethod
    def release_socket(self):
        pass


class MetaAccount(metaclass=ABCMeta):

    def __init__(self):
        self.default_header: dict[str:str] = {}
        self._workers_init = []

    @property
    @abstractmethod
    def tasks(self) -> list[asyncio.Task]:
        pass

    @abstractmethod
    def get_header(self, header: dict) -> dict:
        pass


class MetaWorker(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, name: str):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def tasks(self) -> list[asyncio.Task]:
        pass

