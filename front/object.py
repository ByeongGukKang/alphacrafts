import asyncio
from dataclasses import dataclass
import datetime
import inspect
import types
import sys
import warnings

import numpy as np

from .macros import MACRO_AS_ASYNC, MACRO_ASYNC_SLEEP, MACRO_ASYNC_YIELD, MACRO_TIME_NS
from .meta import MetaWorker
from .net import HttpClient, WebSocketClient


__all__ = [
    "Book", "Scheduler", "Work", "Worker",
    "TickGenerator", "Refresher",
]


# Since there is no await or yield in data access, no need for Lock
class Book:
    
    def __init__(self, code_type: np.dtype=np.dtype('<U32'), posit_type: np.dtype=np.dtype('f8')):
        """
        Book object for account management.

        Args:
            code_type (np.dtype): Code data type.
                default: np.dtype('<U32'), recommend to adjust the length.
            posit_type (np.dtype): Position data type.
                default: np.dtype('f8')
        
        """
        self.__cash = 0
        self.__position = np.empty(0, dtype=[('code', code_type), ('posit', posit_type)])
        self.__position_code_map = dict() # For quick existence check & access key: code, value: index

    def get_cash(self) -> float:
        """Return cash balance."""
        return self.__cash

    def add_cash(self, cash: int|float) -> None:
        """
        Add cash balance.
        
        Args:
            cash (int|float): Cash to add.
        """
        self.__cash += cash

    def get_posit(self, code: str) -> float:
        """
        Get position of the code.

        Args:
            code (str): Code.
        """
        index = self.__position_code_map.get(code)
        if index is None:
            return None
        return self.__position[index]['posit']

    def add_posit(self, code: str, posit: int|float) -> None:
        """
        Add position of the code.

        Args:
            code (str): Code.
            posit (int|float): Position to add.
        """
        index = self.__position_code_map.get(code)
        if index is None:
            self.__position = np.append(self.__position, np.array(((code, posit)), dtype=self.__position.dtype))
            self.__position_code_map[code] = len(self.__position) - 1
            self.__position.sort(order='code', kind='mergesort') # Likely already sorted
            for i, row in enumerate(self.__position):
                self.__position_code_map[row['code']] = i
        self.__position[index]['posit'] += posit

    @property
    def arr_code(self) -> np.ndarray:
        return self.__position['code']

    @property
    def arr_posit(self) -> np.ndarray:
        return self.__position['posit']


@dataclass
class Work:
    """
    Work object for worker tasks.

    Args:
        f (types.FunctionType): Work function.
            1st parameter must be 'self', worker itself.
            If you want to receive queue signal, 2nd parameter must be 'qval', a value from the queue.
        args (tuple, optional): Work function arguments.
            Default: ()
    """
    f: types.FunctionType
    args: tuple = ()

    def __post_init__(self):
        if tuple(inspect.signature(self.f).parameters.keys())[0] != "self":
            raise ValueError("1st parameter of the function must be 'self', worker itself.")
        if not inspect.iscoroutinefunction(self.f):
            self.f = MACRO_AS_ASYNC(self.f)


class Worker(MetaWorker):

    def __init__(self, name: str, qsize: int=0):
        """
        Worker object for system tasks.
        
        Args:
            name (str): Worker name.
            qsize (int, optional): Queue size. Default: 0
        
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        self.__name       = name
        self.__condition  = asyncio.Condition()
        self.__event      = asyncio.Event()
        self.__lock       = asyncio.Lock()
        self.__queue      = asyncio.Queue(qsize)
        self.__work_do    = None
        self.__work_do_if = None

    @property
    def name(self) -> str:
        """Name of the worker."""
        return self.__name
    
    @property
    def tasks(self) -> list[asyncio.Task]:
        """Return list of all tasks."""
        todos = []
        if self.__work_do:
            todos.append(asyncio.create_task(self.__work_do, name=f"{self.__name}_do"))
        if self.__work_do_if:
            todos.append(asyncio.create_task(self.__work_do_if, name=f"{self.__name}_do_if"))
        if len(todos) == 0:
            warnings.warn(f"No work is assigned to Worker.{self.__name}.")
        return todos

    @property
    def condition(self) -> asyncio.Condition:
        """Condition signal of the worker."""
        return self.__condition

    @property
    def event(self) -> asyncio.Event:
        """Event signal of the worker."""
        return self.__event
    
    @property
    def lock(self) -> asyncio.Lock:
        """Lock signal of the worker."""
        return self.__lock

    @property
    def queue(self) -> asyncio.Queue:
        """Queue signal of the worker."""
        return self.__queue
    
    @property
    def task_do(self) -> asyncio.Task:
        """Return do task."""
        if self.__work_do is None:
            raise ValueError(f"No work is assigned to Worker.{self.__name}_do.")
        return asyncio.create_task(self.__work_do, name=f"{self.__name}_do")
    
    @property
    def task_do_if(self) -> asyncio.Task:
        """Return do_if task."""
        if self.__work_do_if is None:
            raise ValueError(f"No work is assigned to Worker.{self.__name}_do.")
        return asyncio.create_task(self.__work_do_if, name=f"{self.__name}_do_if")

    async def condition_wake_all(self):
        """Wake all tasks waiting on the condition signal."""
        async with self.__condition:
            self.__condition.notify_all()
    
    async def condition_wake_one(self):
        """Wake one task waiting on the condition signal."""
        async with self.__condition:
            self.__condition.notify()

    def event_clear(self):
        """Clear the event signal."""
        self.__event.clear()

    def event_set(self):
        """Set the event signal."""
        self.__event.set()

    async def lock_acquire(self):
        """Acquire the lock."""
        await self.__lock.acquire()

    def lock_release(self):
        """Release the lock."""
        self.__lock.release()

    async def queue_put(self, item):
        """Put an item into the queue."""
        await self.__queue.put(item)

    def queue_put_nowait(self, item):
        """Put an item into the queue."""
        self.__queue.put_nowait(item)

    def do(self, work: Work, once: bool):
        """
        Assign a work to the worker.

        Args:
            work (Work): work object.
            once (bool): If True, the work is executed only once.

        """
        if not isinstance(work, Work):
            raise TypeError("work must be an instance of Work.")
        if not isinstance(once, bool):
            raise TypeError("once must be a boolean.")
        if once:
            async def _work():
                await work.f(self, *work.args)
        else:
            async def _work():
                while True:
                    await work.f(self, *work.args)
                    await MACRO_ASYNC_YIELD() # YIELD
        self.__work_do = _work()

    def do_if(self, work: Work, signal: asyncio.Condition|asyncio.Event|asyncio.Lock|asyncio.Queue, once: bool):
        """
        Assign a work to the worker with a signal.

        Args:
            work (Work): work object.
            signal (asyncio.Condition|asyncio.Event|asyncio.Lock|asyncio.Queue): signal object from another worker or something.
            once (bool): If True, the work is executed only once.
        
        """
        if not isinstance(work, Work):
            raise TypeError("work must be an instance of Work.")
        if not isinstance(once, bool):
            raise TypeError("once must be a boolean.")

        if isinstance(signal, asyncio.Condition):
            if signal == self.condition:
                raise ValueError("Recursive signal assignment! using self.signal.")
            if once:
                async def _work():
                    async with signal:
                        await signal.wait()
                    await work.f(self, *work.args)
            else:
                async def _work():
                    while True:
                        async with signal:
                            await signal.wait()
                        await work.f(self, *work.args)
        elif isinstance(signal, asyncio.Event):
            if signal == self.event:
                raise ValueError("Recursive signal assignment! using self.signal.")
            if once:
                async def _work():
                    await signal.wait()
                    await work.f(self, *work.args)
            else:
                async def _work():
                    while True:
                        await signal.wait()
                        await work.f(self, *work.args)
        elif isinstance(signal, asyncio.Lock):
            if signal == self.lock:
                raise ValueError("Recursive signal assignment! using self.lock.")
            if once:
                async def _work():
                    async with signal:
                        await work.f(self, *work.args)
            else:
                async def _work():
                    while True:
                        async with signal:
                            await work.f(self, *work.args)
        elif isinstance(signal, asyncio.Queue):
            if signal == self.queue:
                raise ValueError("Recursive signal assignment! using self.queue.")
            if tuple(inspect.signature(work.f).parameters.keys())[1] != "qval":
                raise ValueError("2nd parameter of the work function must be 'qval', a value from the queue.")
            if once:
                async def _work():
                    qval = await signal.get()
                    await work.f(self, qval, *work.args)
            else:
                async def _work():
                    while True:
                        qval = await signal.get()
                        await work.f(self, qval, *work.args)
        else:
            raise TypeError("signal must be an one of instance from asyncio.Condition|asyncio.Event|asyncio.Lock|asyncio.Queue.")
        self.__work_do_if = _work()
    

class TickGenerator(MetaWorker):

    def __init__(self, name: str, freq: datetime.timedelta, tolerance: float=0.02):
        """
        Tick generating worker object. Wake all tasks waiting on the signal every tick.

        Args:
            name (str): Worker name.
            freq (datetime.timedelta): Tick frequency.
            tolerance (float, optional): Tolerance for frequency.
                Default is 0.02, if you set 0.02, it will sleep for 0.02*freq seconds.
                If you want to yield not sleep, set 0. It is more accurate but might cause busy waiting.

        """
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if not isinstance(freq, datetime.timedelta):
            raise TypeError("freq must be a datetime.timedelta.")
        self.__name = name
        self.__condition = asyncio.Condition()
        freq_in_s = freq.total_seconds()

        async def _work():
            signal = self.__condition
            nsfreq = freq_in_s * 1_000_000_000 # convert to nanoseconds
            ctime = MACRO_TIME_NS() - nsfreq # Immediately refresh
            tsleep = freq_in_s * tolerance # Sleep time in seconds
            if tsleep:
                while True:
                    if (ctime + nsfreq) < MACRO_TIME_NS():
                        async with signal:
                            signal.notify_all()
                        ctime = MACRO_TIME_NS()
                    await MACRO_ASYNC_SLEEP(tsleep) # Sleep, maximum time difference is (tolerance*100)% of freq
            else:
                while True:
                    if (ctime + nsfreq) < MACRO_TIME_NS():
                        async with signal:
                            signal.notify_all()
                        ctime = MACRO_TIME_NS()
                    await MACRO_ASYNC_YIELD() # YIELD
        self.__work_clock = _work()

    @property
    def name(self) -> str:
        """Name of the worker."""
        return self.__name

    @property
    def tasks(self) -> list[asyncio.Task]:
        """Return list of all tasks."""
        return [asyncio.create_task(self.__work_clock, name=f"{self.__name}_clock")]

    @property
    def condition(self) -> asyncio.Condition:
        """Condition signal of the worker."""
        return self.__condition


class Refresher(MetaWorker):

    def __init__(self, name: str, limit: int, freq: datetime.timedelta, tolerance: float=0.02):
        """
        Refresher worker object. Refresh the semaphore every tick to the maxsize.
            Use .consume() to consume the semaphore. 

        Args:
            name (str): Worker name.
            limit (int): Semaphore limit.
            freq (datetime.timedelta): Tick frequency.
            tolerance (float, optional): Tolerance for frequency.
                Default is 0.02, if you set 0.02, it will sleep for 0.02*freq seconds.
                If you want to yield not sleep, set 0. It is more accurate but might cause busy waiting.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if not isinstance(limit, int):
            raise TypeError("limit must be an integer.")
        if not isinstance(freq, datetime.timedelta):
            raise TypeError("freq must be a datetime.timedelta.")
        self.__name = name
        self.__sema = asyncio.LifoQueue(limit) # LIFO queue semaphore

        freq_in_s = freq.total_seconds()

        async def _work():
            nsfreq = freq_in_s * 1_000_000_000 # convert to nanoseconds
            ctime = MACRO_TIME_NS() - nsfreq # Immediately refresh
            sema = self.__sema
            maxsize = sema.maxsize
            tsleep = freq_in_s * tolerance # Sleep time in seconds
            if tsleep:
                while True:
                    if (ctime + nsfreq) < MACRO_TIME_NS():
                        for _ in range(maxsize - sema.qsize()):
                            sema.put_nowait(None)
                        ctime = MACRO_TIME_NS()
                    await MACRO_ASYNC_SLEEP(tsleep) # Sleep, maximum time difference is (tolerance*100)% of freq
            else:
                while True:
                    if (ctime + nsfreq) < MACRO_TIME_NS():
                        for _ in range(maxsize - sema.qsize()):
                            sema.put_nowait(None)
                        ctime = MACRO_TIME_NS()
                    await MACRO_ASYNC_YIELD() # YIELD

        self.__work_refresh = _work()

    @property
    def name(self) -> str:
        """Name of the worker."""
        return self.__name
    
    @property
    def tasks(self) -> list[asyncio.Task]:
        """Return list of all tasks."""
        return [asyncio.create_task(self.__work_refresh, name=f"{self.__name}_refresh")]
    
    @property
    def consume_func(self):
        """Return consume function."""
        return self.__sema.get

    async def consume(self):
        """Consume the semaphore."""
        await self.__sema.get()


# TODO
# @dataclass
# class Task:
#     gname:   str
#     name:    str
#     status:  str
#     iswait:  bool
#     task:    asyncio.Task
#     errlist: list[Exception]


@dataclass
class TaskGroup:
    name:            str
    priority:        int
    status:          str
    tasks_wait:      list[asyncio.Task]
    tasks_nowait:    list[asyncio.Task]
    tasks_sche_done: set[asyncio.Task]
    tasks_sche_wait: set[asyncio.Task]
    tasks_sche_err:  set[asyncio.Task]


class Scheduler:

    def __init__(self, debug: bool=False):
        """
        Task(group) scheduler.

        Args:
            debug (bool, optional): Debug mode. Default: False
            - If True, print error messages. Else, raise error.

        Note:
            - Scheduler has default task group, 'Runtime' with priority of 100.
            - Lower priority is executed first.
            - If possible, automatically use uvloop for faster performance.
        """
        if sys.platform not in ('win32', 'cygwin', 'cli'):
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        self.loop_: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        "Event loop"
        self.__task_group: dict[str,TaskGroup] = {
            'Runtime': TaskGroup('Runtime', 100, 'idle', [], [], set(), set(), set()),
        }
        self.__sche_status:  dict[str,str] = {}

        self.debug_flag: bool = debug

    @property
    def task_group(self) -> dict:
        """Return task group dictionary."""
        return {
            tg.name: {
                'priority': tg.priority,
                'status': tg.status,
                'tasks_wait': [t.get_name() for t in tg.tasks_wait],
                'tasks_nowait': [t.get_name() for t in tg.tasks_nowait],
                'tasks_sche_done': [t.get_name() for t in tg.tasks_sche_done],
                'tasks_sche_wait': [t.get_name() for t in tg.tasks_sche_wait],
                'tasks_sche_err': [t.get_name() for t in tg.tasks_sche_err],
            } for tg in sorted(self.__task_group.values(), key=lambda x: x.priority)
        }

    def status(self, name: str, tname: str=None) -> str:
        """
        Return status of the task group.

        Args:
            name (str): Task group name.
            tname (str, optional): Task name.

        Note:
            - idle: waiting for initialization.
            - running: task group is running.
            - done: task group is done.
            - error: task group is terminated due to error.
            - NoneType: name or tname is not found.
        """
        status = self.__task_group.get(name)
        if tname & (status == 'running'):
            return self.__sche_status.get(tname)
        return status
    
    def set_priority(self, name: str, priority: int):
        """
        Set priority of the task group.

        Args:
            name (str): Task group name.
            priority (int): Task group priority.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if name not in self.__task_group:
            raise ValueError(f"Task group {name} does not exist.")
        if not isinstance(priority, int):
            raise TypeError("priority must be an integer.")
        if priority < 0:
            raise ValueError("priority must be greater than or equal to 0.")
        if priority in [v.priority for v in self.__task_group.values()]:
            raise ValueError(f"Priority {priority} already exists.")
        self.__task_group[name].priority = priority

    def add_task_group(self, name: str, priority: int):
        """
        Create a new task group.

        Args:
            name (str): Task group name.
            priority (int): Task group priority.

        Note:
            - Priority is used to determine the order of execution
            - Lower priority is executed first.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if name in self.__task_group:
            raise ValueError(f"Task group {name} already exists.")
        if not isinstance(priority, int):
            raise TypeError("priority must be an integer.")
        if priority < 0:
            raise ValueError("priority must be greater than or equal to 0.")
        if priority in [v.priority for v in self.__task_group.values()]:
            raise ValueError(f"Priority {priority} already exists.")
        self.__task_group[name] = TaskGroup(name, priority, "idle", [], [], set(), set(), set())

    def add_task(self, name: str, tasks: asyncio.Task|list[asyncio.Task], wait: bool):
        """
        Add a task to the task group.

        Args:
            name (str): Task group name.
            task (asyncio.Task): Task or list of tasks.
            wait (bool): If True, the task is added to the waiting list.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if name not in self.__task_group:
            raise ValueError(f"Task group {name} does not exist.")
        if isinstance(tasks, asyncio.Task):
            tasks = [tasks]
        elif not isinstance(tasks, list):
            raise TypeError("task must be an instance of asyncio.Task or a list of asyncio.Task.")
        if not isinstance(wait, bool):
            raise TypeError("wait must be a boolean.")
        
        tnames = [t.get_name() for t in self.__task_group[name].tasks_wait + self.__task_group[name].tasks_nowait]
        for task in tasks:
            if not isinstance(task, asyncio.Task):
                raise TypeError("task must be an instance of asyncio.Task.")
            if task.get_name() in tnames:
                raise ValueError("Task name already exists.")

        if wait:
            self.__task_group[name].tasks_wait.extend(tasks)
        else:
            self.__task_group[name].tasks_nowait.extend(tasks)

    async def _sche(self):
        """
        scheduler loop.

        Note:
            - This method is called automatically when .run() is called.
            - Order all task groups and run tasks.
        """
        errs: list[tuple[str,Exception]] = []
        try:
            for tg in sorted(self.__task_group.values(), key=lambda x: x.priority):
                tg.status = 'running'

                tg.tasks_sche_wait, sche_nowait = set(tg.tasks_wait), set(tg.tasks_nowait)
                if len(tg.tasks_sche_wait) == 0:
                    raise RuntimeError(f"Task group {tg.name} has no task to wait.")
                if len(sche_nowait) == 0: # If empty, add a dummy task (yield only)
                    sche_nowait.add(asyncio.create_task(MACRO_ASYNC_SLEEP(0)))
                sched_wait = asyncio.create_task(asyncio.wait(tg.tasks_sche_wait, return_when=asyncio.FIRST_EXCEPTION))
                sched_nowait = asyncio.create_task(asyncio.wait(sche_nowait, return_when=asyncio.FIRST_EXCEPTION))

                self.__sche_status = {t.get_name(): 'running' for t in tg.tasks_sche_wait | sche_nowait}

                while tg.tasks_sche_wait:
                    done_scheds, _ = await asyncio.wait({sched_wait, sched_nowait}, return_when=asyncio.FIRST_COMPLETED)
                    for done_sched in done_scheds:
                        done_tasks, _ = await done_sched
                        for task in done_tasks:
                            tg.tasks_sche_wait.discard(task)
                            try:
                                await task
                            except Exception as e:
                                tg.tasks_sche_err.add(task)
                                self.__sche_status[task.get_name()] = 'error'
                                if not self.debug_flag:
                                    raise e
                                print(f'TG[{tg.name}] TASK[{task.get_name()}] ERROR: {e}')
                                # TODO Error handling
                            else:
                                tg.tasks_sche_done.add(task)
                                self.__sche_status[task.get_name()] = 'done'

                tg.status = 'done'
                self.__sche_status = {nm: 'done' for nm in self.__sche_status if nm != 'error'}

        except Exception as e:
            errs.append((tg.name, task.get_name(), e))

        finally:
            if len(errs) > 0:
                print('Errors:')
                for tg_nm, task_nm, err in errs:
                    print(f'    TaskGroup[{tg_nm}] Task[{task_nm}] Err[{err}]')
                print()
                raise RuntimeError("Scheduler loop has been terminated due to errors.")

    def run(self, main: types.CoroutineType):
        """ Run the scheduler.

        Args:
            main (types.CoroutineType): Main coroutine to run.
            - Your main logics must be in this coroutine.
        """
        if not asyncio.iscoroutine(main):
            raise TypeError("coro must be a coroutine.")
        async def _run():
            await main
            await self._sche()
        self.loop_.run_until_complete(_run())
