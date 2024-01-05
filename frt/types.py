
import numba as nb
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

@nb.experimental.jitclass([
    ('date', nb.types.NPDatetime('s')),
    ('year', nb.types.int16),
    ('month', nb.types.int8),
    ('day', nb.types.int8),
    ('weekday', nb.types.int8),
    ('hour', nb.types.int8),
    ('minute', nb.types.int8),
    ('second', nb.types.int8),
])
class NbDate:
    def __init__(self, np_datetime, year, month, day, weekday, hour, minute, second):
        self.date = np_datetime
        self.year = year
        self.month = month
        self.day = day
        self.weekday = weekday
        self.hour = hour
        self.minute = minute
        self.second = second

@nb.experimental.jitclass([
    ('symbol', nb.types.unicode_type),
    ('price', nb.types.float64),
    ('volume', nb.types.float64),
    ('order_type', nb.types.int8),
])
class Order:

    def __init__(
        self,
        symbol:nb.types.unicode_type,
        price:nb.types.float64,
        volume:nb.types.int64,
        order_type:nb.types.int8 = 0
    ):
        self.symbol = symbol # 종목 코드
        self.price = price # 가격
        self.volume = volume # 수량
        self.order_type = order_type # 시장가, FOK 등등

@nb.experimental.jitclass([
    ('symbol', nb.types.unicode_type),
    ('price', nb.types.float64),
    ('volume', nb.types.float64),
    ('result_type', nb.types.int8),
])
class OrderResult:

    def __init__(self, symbol, price, volume, result_type):
        self.symbol = symbol
        self.price = price
        self.volume = volume
        self.result_type = result_type


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