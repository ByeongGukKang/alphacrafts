
import datetime as dt

import numpy as np

from alphacrafts.frt.types import NbDate

def to_nb_datetime(str_datetime):
    np_datetime = np.datetime64(str_datetime, 's')
    dt_datetime = dt.datetime.strptime(np.datetime_as_string(np_datetime), '%Y-%m-%dT%H:%M:%S')
    return NbDate(np_datetime, dt_datetime.year, dt_datetime.month, dt_datetime.day, dt_datetime.weekday(), dt_datetime.hour, dt_datetime.minute, dt_datetime.second)

class Singleton:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance