
import datetime as dt

import numpy as np
import pandas as pd

from alphacrafts.frt.types import nbDatetime


def to_nb_datetime(object):

    if isinstance(object, pd.Timestamp) or isinstance(object, dt.datetime):
        np_datetime = np.datetime64(object, 'ms')
        nb_datetime = nbDatetime(
            np_datetime, str(np_datetime), object.year, object.month, object.day,
            object.weekday(), object.hour, object.minute, object.second, 
            int(object.microsecond/1000)
        )
    else:
        raise ValueError("date_object must be a pandas Timestamp or a datetime object")
    
    return nb_datetime

# def to_nb_timedelta(object):

