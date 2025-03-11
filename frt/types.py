
import numba as nb
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


class Singleton:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

@nb.experimental.jitclass([
    ('datetime', nb.types.NPDatetime('ms')),
    ('__str', nb.types.unicode_type),
    ('year', nb.uint16),
    ('month', nb.uint8),
    ('day', nb.uint8),
    ('weekday', nb.uint8),
    ('hour', nb.uint8),
    ('minute', nb.uint8),
    ('second', nb.uint8),
    ('millisecond', nb.uint16)
])
class nbDatetime:

    def __init__(self, np_datetime, str_datetime, year, month, day, weekday, hour, minute, second, millisecond):
        self.datetime = np_datetime
        self.__str = str_datetime
        self.year = year
        self.month = month
        self.day = day
        self.weekday = weekday
        self.hour = hour
        self.minute = minute
        self.second = second
        self.millisecond = millisecond

    def __str__(self):
        return self.__str

    def __eq__(self, other):
        if isinstance(other, nbDatetime):
            return self.datetime == other.datetime
        elif "datetime64" in str(type(other)):
            return self.datetime == other
        else:
            raise ValueError("other must be a nbDatetime or a np.datetime64 object")

    def __ne__(self, other):
        if isinstance(other, nbDatetime):
            return self.datetime != other.datetime
        elif "datetime64" in str(type(other)):
            return self.datetime != other
        else:
            raise ValueError("other must be a nbDatetime or a np.datetime64 object")
    
    def __lt__(self, other):
        if isinstance(other, nbDatetime):
            return self.datetime < other.datetime
        elif isinstance(other, np.datetime64):
            return self.datetime < other
        else:
            raise ValueError("other must be a nbDatetime or a np.datetime64 object")
    
    def __le__(self, other):
        if isinstance(other, nbDatetime):
            return self.datetime <= other.datetime
        elif isinstance(other, np.datetime64):
            return self.datetime <= other
        else:
            raise ValueError("other must be a nbDatetime or a np.datetime64 object")
    
    def __gt__(self, other):
        if isinstance(other, nbDatetime):
            return self.datetime > other.datetime
        elif isinstance(other, np.datetime64):
            return self.datetime > other
        else:
            raise ValueError("other must be a nbDatetime or a np.datetime64 object")
    
    def __ge__(self, other):
        if isinstance(other, nbDatetime):
            return self.datetime >= other.datetime
        elif isinstance(other, np.datetime64):
            return self.datetime >= other
        else:
            raise ValueError("other must be a nbDatetime or a np.datetime64 object")
    
    def __add__(self, other):
        if isinstance(other, nbTimeDelta):
            return self.datetime + other
        else:
            raise ValueError("other must be a np.timedelta64 object")
    
    def __sub__(self, other):
        if isinstance(other, nbDatetime):
            return self.datetime - other.datetime
        elif "timedelta64" in str(type(other)):
            return self.datetime - other
        else:
            raise ValueError("other must be a nbDatetime or a np.timedelta64 object")
    
    def __radd__(self, other):
        if "timedelta64" in str(type(other)):
            return other + self.datetime
        else:
            raise ValueError("other must be a np.timedelta64 object")
    
    def __rsub__(self, other):
        if isinstance(other, nbDatetime):
            return  other.datetime - self.datetime
        elif "timedelta64" in str(type(other)):
            return other - self.datetime
        else:
            raise ValueError("other must be a nbDatetime or a np.datetime64 object")
    
# @nb.experimental.jitclass([
#     ('M', nb.types.NPTimedelta('M')),
#     ('W', nb.types.NPTimedelta('W')),
#     ('D', nb.types.NPTimedelta('D')),
#     ('h', nb.types.NPTimedelta('h')),
#     ('m', nb.types.NPTimedelta('m')),
#     ('s', nb.types.NPTimedelta('s')),
#     ('ms', nb.types.NPTimedelta('ms')),
#     ('unit', nb.types.unicode_type)
# ])
# class nbTimeDelta:

#     def __init__(self, unit):
#         self.M = np.timedelta64(0, 'M')
#         self.W = np.timedelta64(0, 'W')
#         self.D = np.timedelta64(0, 'D')
#         self.h = np.timedelta64(0, 'h')
#         self.m = np.timedelta64(0, 'm')
#         self.s = np.timedelta64(0, 's')
#         self.ms = np.timedelta64(0, 'ms')
#         self.unit = unit
    
#     def set(self, value):
#         getattr(self, self.unit) = value

# @nb.experimental.jitclass([
#     ('timedelta', nb.types.NPTimedelta('W')),
# ])
# class nbTimedeltaWeek:

#     def __init__(self, np_timedelta):
#         self.timedelta = np_timedelta

# @nb.experimental.jitclass([
#     ('timedelta', nb.types.NPTimedelta('D')),
# ])
# class nbTimedeltaDay:

#     def __init__(self, np_timedelta):
#         self.timedelta = np_timedelta

# @nb.experimental.jitclass([
#     ('timedelta', nb.types.NPTimedelta('h')),
# ])
# class nbTimedeltaHour:

#     def __init__(self, np_timedelta):
#         self.timedelta = np_timedelta

# @nb.experimental.jitclass([
#     ('timedelta', nb.types.NPTimedelta('m')),
# ])
# class nbTimedeltaMinute:

#     def __init__(self, np_timedelta):
#         self.timedelta = np_timedelta

# @nb.experimental.jitclass([
#     ('timedelta', nb.types.NPTimedelta('s')),
# ])
# class nbTimedeltaSecond:

#     def __init__(self, np_timedelta):
#         self.timedelta = np_timedelta

# @nb.experimental.jitclass([
#     ('timedelta', nb.types.NPTimedelta('ms')),
# ])
# class nbTimedeltaMillisecond:

#     def __init__(self, np_timedelta):
#         self.timedelta = np_timedelta



class Performance:

    def __init__(self, trader, vol_days):
        self.pf_value = pd.DataFrame(trader.pf_value, index=[idx.date for idx in trader.price_index], columns=[f'{trader.name} pf_value'])
        self.returns = self.pf_value.pct_change(fill_method=None)
        high_water_mark = self.pf_value.cummax()
        self.drawdown = (self.pf_value - high_water_mark) / high_water_mark
        self.drawdown.columns = [f'{trader.name} drawdown']
        self.cash_balance = pd.DataFrame(trader.cash_balance, index=[idx.date for idx in trader.price_index], columns=[f'{trader.name} cash_balance'])
        self.positions = pd.DataFrame(trader.positions, index=[idx.date for idx in trader.price_index], columns=trader.symbols)
        self.cashflow = pd.DataFrame(trader.cashflow, index=[idx.date for idx in trader.price_index], columns=trader.symbols)
        self.signals = pd.DataFrame(trader.signals, index=[idx.date for idx in trader.price_index], columns=trader.symbols)
        self.orders = pd.DataFrame(trader.orders, index=[idx.date for idx in trader.price_index], columns=trader.symbols)

        self.daily_pf_value = self.pf_value.resample('D', convention='end').last().dropna(axis=0)
        self.daily_returns = self.daily_pf_value.pct_change(fill_method=None)
        high_water_mark = self.daily_pf_value.cummax()
        self.daily_drawdown = (self.daily_pf_value - high_water_mark) / high_water_mark

        self.vol_days = vol_days

    def plot_overall(self, use_daily=True, figsize=(18,18)):
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=figsize)
        
        axs[0].set_title('Portfolio Value')
        axs[1].set_title('Drawdown')
        axs[2].set_title('Returns')

        if use_daily:
            axs[0].plot(self.daily_pf_value)
            axs[1].plot(self.daily_drawdown)
            axs[2].plot(self.daily_returns)
        else:
            axs[0].plot(self.pf_value)
            axs[1].plot(self.drawdown)
            axs[2].plot(self.returns)

        fig.tight_layout()
        plt.show()

    def values(self, to_df=False):
        res = {}
        res['CAGR'] = ((self.daily_pf_value.values[-1]/self.daily_pf_value.values[0])**(self.vol_days/len(self.daily_pf_value)) -1)[0]
        res['Maximum Drawdown'] = self.daily_drawdown.min()[0]
        res['Annual Std'] = self.daily_returns.std()[0] * np.sqrt(self.vol_days)
        res['Downside Std'] = self.daily_returns[self.daily_returns<0].std()[0] * np.sqrt(self.vol_days)
        res['Sharpe Ratio'] = res['CAGR']/res['Annual Std']
        res['Sortino Ratio'] = res['CAGR']/res['Downside Std']
    
        if to_df:
            res = pd.DataFrame(list(res.values()), index=(res.keys()), columns=['values'])

        return res