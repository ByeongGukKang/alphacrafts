from dataclasses import dataclass
from datetime import datetime
import inspect
import types
from warnings import warn

import pandas as pd
import numpy as np
from tqdm import tqdm

from alphacrafts.frt.loopbt.acc64 import (
    Account64Struct, newAccount64,
    Account64Struct_Att_rowlen, Account64Struct_Att_mposit,
    Account64Struct_Func_init_signal, Account64Struct_Func_mark_to_mark, Account64Struct_Func_execute,
)
from alphacrafts.frt.loopbt.btres import BtResult


# For code auto-completion, type hinting
class Account(Account64Struct):
    pass

@dataclass
class _IntpSE:
    start: int
    end: int

@dataclass
class _IntpPacket:
    curridx: int
    intpse: dict[str: _IntpSE]
    mask_univ: np.ndarray

# TODO Make as Numba Struct Class
class Packet: 

    def __init__(self, mainself):
        self.__frame_prc = mainself._frame_prc
        self.__frame_data = mainself._frame_data
        # Data
        self.__pstruct = None
        self.__cache = {}    
        # For masking
        self.__mask_map = {
            '~': slice(None),
            '@': np.array((), dtype=np.intp),
            '#': np.array((), dtype=np.intp),
            '!': None,
        }

    @property
    def mnone(self) -> np.ndarray:
        return self.__mask_map['~']
    
    @property
    def mall(self) -> np.ndarray:
        return self.__mask_map['!']
    
    @property
    def muniv(self) -> np.ndarray:
        return self.__mask_map['@']
    
    @property
    def mposit(self) -> np.ndarray:
        return self.__mask_map['#']

    def _update_packet(self, pstruct: _IntpPacket):
        # Update packet, Clear cache
        self.__pstruct = pstruct
        self.__cache.clear()
        # Mask
        self.__mask_map['@'] = pstruct.mask_univ
        self.__mask_map['!'] = None # Lazy parsing, but need to be changed for numba compatibility

    def _update_mposit(self, mask_posit: np.ndarray):
        self.__mask_map['#'] = mask_posit

    # Lazy Parsing
    def __getitem__(self, key:str) -> np.ndarray:
        """Get data from packet.

        Args:
            key (str): key of data to be retrieved (key of data added to PacketPacker)
        - Special keys:
            _p: current price
            _c: data column
            
        Returns:
            np.ndarray: data
        """
        # Check cache
        if key in self.__cache:
            return self.__cache[key]
        
        # Mask selection
        mkey, dkey = key[0], key[1:]
        try:
            mask = self.__mask_map[mkey]
            # Lazy parsing for '!', muniv | mposit 
            # Since it requires union operation, it is not efficient to calculate it every time
            if mask is None: 
                mask = np.union1d(self.__mask_map['@'], self.__mask_map['#']) # TODO apply Numba
                self.__mask_map[mkey] = mask
        except KeyError as e:
            raise KeyError("No such mask!, mask must be one of '~', '!', '@', '#'") from e

        # Get data
        if dkey.startswith('_'):
            self.__cache[key] = np.ascontiguousarray(self.__frame_prc[self.__pstruct.curridx, mask]) #.ravel('C')
        else:
            intpse = self.__pstruct.intpse[dkey]
            self.__cache[key] = np.ascontiguousarray(self.__frame_data[dkey][intpse.start:intpse.end+1, mask])

        # Return
        return self.__cache[key]


class Pacman: 

    def __init__(self, price_df: pd.DataFrame):
        """
        Args:
            price_df (pd.DataFrame): price data to be used for backtesting
        """
        self._frame_data = {}
        self._frame_data_index = {}
        self._frame_data_column = {}
        self._frame_data_dtype = {}
        self._iscompiled = False

        # TODO price_df의 index 및 column의 dtype 확인
        # Pandas DataFrame check
        if not isinstance(price_df, pd.DataFrame):
            raise TypeError("Data must be pandas DataFrame!")
        # Index data type check
        if not isinstance(price_df.index, pd.DatetimeIndex):
            raise TypeError("Data index must be DatetimeIndex!")
        self._frame_prc = price_df.to_numpy()
        self._frame_prc.flags.writeable = False
        self._frame_prc_index = price_df.index
        self._frame_prc_column = np.array(price_df.columns.to_numpy(), dtype=np.unicode_) # TODO column name type conversion
        
    def add_data(self, key: str, data: pd.DataFrame):
        """
        Args:
            key (str): key of data to be added
            data (pd.DataFrame): data to be added
        """
        # Key check
        if key in self._frame_data:
            raise KeyError(f"Key {key} already exists!")
        elif key[0] in ['~','!','@','#','_']: # Special key check
            raise ValueError("Key cannot start with '~', '!', '@', '#'")
        # Data check
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be pandas DataFrame!")
        if not isinstance(data.index, pd.DatetimeIndex): # Index data type check
            raise TypeError("Data index must be DatetimeIndex!")
        if len(set(data.dtypes)) != 1: # Column data type check
            raise TypeError("dtypes of Data columns are not consistent!")
        
        # TODO support more numberic types...
        if data.dtypes.to_numpy()[0] in [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64, np.bool_]:
            data = data.astype(np.float64)
        else:
            raise TypeError("Data must be numeric type!")
        
        self._frame_data[key] = data.to_numpy()
        self._frame_data[key].flags.writeable = False
        self._frame_data_index[key] = data.index
        self._frame_data_column[key] = np.array(data.columns.to_numpy(), dtype=np.unicode_)
        self._frame_data_dtype[key] = data.dtypes.to_numpy()[0]

    def get_data(self, key: str):
        """
        Args:
            key (str): key of data to be retrieved
        """
        if key not in self._frame_data:
            raise KeyError(f"Key {key} does not exist!")
        return pd.DataFrame(self._frame_data[key], index=self._frame_data_index[key], columns=self._frame_data_column[key])

    def del_data(self, key: str):
        """
        Args:
            key (str): key of data to be deleted
        """
        if key not in self._frame_data:
            raise KeyError(f"Key {key} does not exist!")
        del self._frame_data[key]
        del self._frame_data_index[key]

    # TODO add partial compile
    # add new data, compile only new data

    def compile(
        self,
        window_size: int|dict,
        min_window_size: int|dict = None,
        universe: pd.DataFrame = None,
    ):
        """Compile data into packets.

        Args:
            window_size (int, dict, np.inf): number of obersevations of data for each frame_data to be included in each packet
             if np.inf, expanding window will be used
             dictionary argument should be with key: frame_data key, value: window_size (int)
            min_window_size (int, dict): minimum number of observations for each frame_data to be included in each packet
             if None, window_size will be used
             dictionary argument should be with key: frame_data key, value: min_window_size (int)
            universe (pd.DataFrame): universe data to be used for backtesting. Defaults to None

        """
        # Check window_size
        if isinstance(window_size, int):
            window_size = {key: window_size for key in self._frame_data.keys()}
        elif isinstance(window_size, dict):
            for key, value in window_size.items():
                if key not in self._frame_data.keys():
                    raise KeyError(f"Key {key} does not exist in frame_data!")
                if not isinstance(value, int):
                    raise ValueError("window_size must be integer!")
        elif np.isinf(window_size):
            window_size = {key: np.inf for key in self._frame_data.keys()}
        else:
            raise TypeError("window_size must be int | dict | np.inf !")
        
        # Check min_window_size
        if min_window_size is None:
            min_window_size = window_size.copy()
        elif isinstance(min_window_size, int):
            min_window_size = {key: min_window_size for key in self._frame_data.keys()}
        elif isinstance(min_window_size, dict):
            for key, value in min_window_size.items():
                if key not in self._frame_data:
                    raise KeyError(f"Key '{key}' does not exist in frame_data!")
                if not isinstance(value, int):
                    raise ValueError("min_window_size must be int!")
        else:
            raise TypeError("min_window_size must be integer | dict!")
        
        # Check universe, TODO more efficient way to compile universe (caching)
        if universe is not None:
            if not isinstance(universe, pd.DataFrame):
                raise TypeError("Universe must be pandas DataFrame!")
            elif universe.index[0] > self._frame_prc_index[0]:
                warn("Too short universe data, start date of universe is later than start date of price data!")

            self._universe = pd.DataFrame(
                index = sorted(set(universe.index.tolist() + self._frame_prc_index.tolist())),
                columns = ['universe']
            )
            self._universe.loc[universe.index,:] = universe.to_numpy()
            self._universe.ffill(inplace=True)
            self._universe = self._universe.loc[self._frame_prc_index,:]
            self._universe = np.array([row for row in self._universe.to_numpy()]).ravel('C')
        else:
            self._universe = np.array([self._frame_prc_column for _ in range(self._frame_prc_index.size)]) # All columns are included in universe

        # Sender index
        self._sender_index = []
        # Memory to save complied data
        self._complied_data = {}

        gtime = datetime.now()
        tdict = {'copy':gtime, 'where': gtime, 'savepkt': gtime}

        ### Loop for each date, compile data into packets
        # Caching significantly reduces the time required to find the next available data index
        # Since np.where() iterates through the entire data_index, it is very time consuming
        is_lack_of_data = True
        intp_element_cache = {k:_IntpSE(-1, -1) for k in self._frame_data.keys()}
        for curridx, end_date in enumerate(tqdm(self._frame_prc_index, desc="Compiling data packets...")): # curridx (int), end_date (datetime)
            is_lack_of_data = False

            stime = datetime.now()
            intp_element = intp_element_cache.copy() # Use cache
            tdict['copy'] += (datetime.now() - stime)

            for key, data_index in self._frame_data_index.items():
                # Test if next observed datetime-index exceeds CURRENT_TIME(end_time)
                # If so we can use cached(last) data, because there is no new available data at CURRENT_TIME
                cur_end_idx = intp_element_cache[key].end
                if (data_index[min(cur_end_idx+1, data_index.size-1)] > end_date):
                    continue
                # If not, we need to find available data index
                # There is new available data at CURRENT_TIME
                else:
                    stime = datetime.now()
                    # TODO convert np.where to loop
                    avail_data_index = np.where(data_index < end_date)[0] # array of available data index, before CURRENT_TIME(end_date)
                    tdict['where'] += (datetime.now() - stime)

                    # Check if there is enough data
                    if avail_data_index.size < min_window_size[key]:
                        is_lack_of_data = True
                        break
                    avail_data_index_eidx = avail_data_index.size - 1 # avoid using negative indexing

                    # Get packet index, not datetime idx, but integer idx of data
                    intp_element[key] = _IntpSE(
                        max(avail_data_index[avail_data_index_eidx] - window_size[key] + 1, 0), # start idx
                        avail_data_index[avail_data_index_eidx]                                 # end idx
                    )
            
            # Update cache
            intp_element_cache = intp_element

            # If data size is smaller than min_window_size, skip
            if is_lack_of_data:
                continue

            # save packet, end_date is current time
            stime = datetime.now()
            self._complied_data[end_date] = _IntpPacket(
                curridx,
                intp_element, # dictionary of _IntpSE, contains start & end index of data
                np.ascontiguousarray( 
                    # TODO apply Numba
                    np.where(np.isin(self._frame_prc_column, self._universe[curridx], assume_unique=True))[0],
                    dtype = np.intp
                ) # Universe mask, array of intp
            )
            tdict['savepkt'] += (datetime.now() - stime)

            # append index
            self._sender_index.append(end_date)
        
        # iscomplied flag
        self._iscompiled = True

        for k,v in tdict.items():
            tdict[k] = v - gtime
        print(tdict)

    # Return 
    @property
    def data_index(self):
        # check if data is complied
        if not self._iscompiled:
            raise NotImplementedError("Data is not complied yet!")
        return self._sender_index
    
    @property
    def data_column(self):
        # check if data is complied
        if not self._iscompiled:
            raise NotImplementedError("Data is not complied yet!")
        return self._frame_prc_column

    # TODO Numba can be applied for datafeed generator
    def datafeed(self):
        """ Returns datafeed generator.
        """
        # check if data is complied
        if not self._iscompiled:
            raise NotImplementedError("Data is not complied yet!")

        def packet_generator():
            packet = Packet(mainself=self)
            # packing data
            for _datetime, _packet in self._complied_data.items():

                # update packet
                packet._update_packet(_packet) # packet_tuple

                yield _datetime, packet, self._frame_prc[_packet.curridx,:].ravel('C') # datetime, packet, current_price
        
        return packet_generator()


# Trader Class
class Trader:

    def __init__(
        self,
        pacman: Pacman,
        init_cash: float,
        buy_fee: float = 0.0,
        sell_fee: float = 0.0,
        envname: str = 'TestWorld',
    ):
        """Trader class for backtesting.

        Args:
            pacman (Pacman): compiled Pacman object
            init_cash (float): Initial cash
            buy_fee (float, optional): Buy fee. Defaults to 0.0
            sell_fee (float, optional): Sell fee. Defaults to 0.0
            envname (str, optional): environment name. Defaults to 'TestWorld'

        """
        self.__envname = envname
        self.__pacman = pacman

        self.__init_cash = np.float64(init_cash)
        self.__fee_buy = np.float64(1 + buy_fee)
        self.__fee_sell = np.float64(1 - sell_fee)
    
    @property
    def environment(self):
        return {
            'envname':   self.__envname,
            'init_cash': self.__init_cash,
            'buy_fee':   self.__fee_buy-1,
            'sell_fee':  1-self.__fee_sell,
        }
    
    def run(
        self,
        trade: types.FunctionType,
        signal: dict[str:int] = {},
        name: str = 'mystrategy',
    ) -> BtResult:
        """Run backtesting.

        Args:
            trade (types.FunctionType): trade function
            signal (dict[str:int]): signal dictionary with key: signal name, value: signal size
            name (str, optional): name of the strategy. Defaults to 'mystrategy'

        Returns:
            result (BtResult): BtResult object
        """
        # Check trade function
        if not isinstance(trade, types.FunctionType):
            raise TypeError("trade must be a function")
        if list(inspect.signature(trade).parameters.keys()) != ['time', 'acc', 'pkt']:
            raise SyntaxError("trade must have parameters: time, acc, pkt")
        
        # Check signal
        if not isinstance(signal, dict):
            raise TypeError("signal must be a dictionary")
        if len(signal) != 0:
            if not all([isinstance(k, str) for k in signal.keys()]):
                raise TypeError("signal key must be string")
            if any([k[0] not in ['~','!','@','#'] for k in signal.keys()]):
                raise ValueError("signal key must start with '~', '!', '@', '#'")
            if any([len(k) > 64 for k in signal.keys()]):
                raise ValueError("signal key must be less than 64 characters")
            if not all([isinstance(v, int) for v in signal.values()]):
                raise TypeError("signal size must be integer")

        gstart = datetime.now()
        tdict = {'packet':gstart, 'mtm':gstart, 'trade':gstart, 'execution':gstart}

        # Call to stack, TODO or just pass to the account?
        fee_buy, fee_sell  = self.__fee_buy, self.__fee_sell

        # Initialize Account
        acc = newAccount64(
            rowlen = len(self.__pacman.data_index),
            collen = len(self.__pacman.data_column),
            initial_cash = self.__init_cash,
        )

        # Initialize Signal Dictionary
        Account64Struct_Func_init_signal(
            acc,
            np.array(tuple(signal.keys()), dtype=np.dtype('U64')),
            np.array(tuple(signal.values()), dtype=np.int64)
        )

        # Get datafeed generator
        pacman = self.__pacman.datafeed()

        # Run backtesting
        for _ in tqdm(range(Account64Struct_Att_rowlen(acc)-1), desc=f'Backtesting [{self.__envname}|{name}]'):

            # Get data packet
            stime = datetime.now()
            time, pkt, price = next(pacman)
            pkt._update_mposit(Account64Struct_Att_mposit(acc)) # Update packet.mPosit
            tdict['packet'] += (datetime.now() - stime)

            # Mark to Market
            stime = datetime.now()
            Account64Struct_Func_mark_to_mark(acc, price, pkt.muniv)
            tdict['mtm'] += (datetime.now() - stime)

            # Generate signals
            stime = datetime.now()
            trade(time, acc, pkt)
            tdict['trade'] += (datetime.now() - stime)
            
            # Execution
            stime = datetime.now()
            Account64Struct_Func_execute(acc, fee_buy, fee_sell)
            tdict['execution'] += (datetime.now() - stime)

            # TODO Calculate margin(leverage or shorting) fee
        
        # Backtest done
        _, pkt, price = next(pacman)
        Account64Struct_Func_mark_to_mark(acc, price, pkt.muniv)

        for k,v in tdict.items():
            tdict[k] = v - gstart
        print(tdict)

        return BtResult(
            env    = self.environment,
            name   = name,
            prc_df = self.__pacman._frame_prc[-Account64Struct_Att_rowlen(acc):],
            acc    = acc,
            data_index  = self.__pacman.data_index,
            data_column = self.__pacman.data_column
        )
    
