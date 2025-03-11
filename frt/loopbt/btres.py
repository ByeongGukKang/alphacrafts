import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from alphacrafts.frt.loopbt.acc64 import Account64Struct

# Performance Class
class BtResult:

    def __init__(self, env: dict, name: str, prc_df, acc: Account64Struct, data_index, data_column):
        self._env = env
        self._name = name
        self._prc_df = pd.DataFrame(prc_df, index=data_index, columns=data_column)
        self._acc = acc
        self._data_index = data_index
        self._data_column = data_column
        
        self._calculated = {
            'daily_returns': None,
        }

        # Basic Performance Calculation
        pnl = pd.DataFrame(self._acc.value_hist, index=self._data_index, columns=['pnl'])
        pnl = pnl.resample('1D').last().dropna()
        self._calculated['daily_returns'] = pnl.pct_change(fill_method=None).fillna(0)
    
    @property
    def info(self):
        return {'env':self._env, 'name':self._name}
    
    @property
    def val_pnl(self):
        return pd.DataFrame(self._acc.value_hist, index=self._data_index, columns=['pnl'])

    @property
    def val_position(self):
        return pd.DataFrame(self._acc.posit_hist, index=self._data_index, columns=self._data_column)
    
    @property
    def val_weight(self):
        tmp = self._prc_df * self.val_position
        tmp = tmp.apply(lambda x: x / x.sum(), axis=1)
        tmp.fillna(0,inplace=True)
        return tmp
    
    def analyze_liquidity(self, tval, tvol, tval_deno=100000000):
        weight = self.val_weight
        weight = weight.loc[weight.sum(1)[weight.sum(1)!=0].index]
        weight = weight.abs()

        dup_index = [idx for idx in weight.index if ((idx in tval.index) and (idx in tvol.index))]
        tmp_tval = pd.DataFrame(index=weight.index, columns=weight.columns)
        tmp_tval.loc[dup_index] = tval.loc[dup_index]
        tmp_tvol = pd.DataFrame(index=weight.index, columns=weight.columns)
        tmp_tvol.loc[dup_index] = tvol.loc[dup_index]

        tmp_tval = tmp_tval.astype(float).bfill()
        tmp_tvol = tmp_tvol.astype(float).bfill()
        
        weighted_tval = (weight * tmp_tval).sum(axis=1)/tval_deno
        weighted_tvol = (weight * tmp_tvol).sum(axis=1)

        print(
            f"Liquidity Analysis - TVAL_denominator: {tval_deno/1_000_000_000:.1f} Bilion\n"
            f"Weighted TVAL[mean]: {weighted_tval.mean():.0f}\n"
            f"Weighted TVOL[mean]: {weighted_tvol.mean():.0f}\n"
            f"Weighted TVAL[1%,5%,95%]: {weighted_tval.quantile(0.01):.0f}, {weighted_tval.quantile(0.05):.0f}, {weighted_tval.quantile(0.95):.0f}\n"
            f"Weighted TVOL[1%,5%,95%]: {weighted_tvol.quantile(0.01):.0f}, {weighted_tvol.quantile(0.05):.0f}, {weighted_tvol.quantile(0.95):.0f}\n"
            f"Weighted TVAL[min]: {weighted_tval.min():.0f}\n"
            f"Weighted TVOL[min]: {weighted_tvol.min():.0f}\n"
        )

        weighted_tval = weighted_tval[-250:]
        weighted_tvol = weighted_tvol[-250:]
        print(
            f"Liquidity Analysis (Recent 1-Year) - TVAL_denominator: {tval_deno/1_000_000_000:.1f} Bilion\n"
            f"Weighted TVAL[mean]: {weighted_tval.mean():.0f}\n"
            f"Weighted TVOL[mean]: {weighted_tvol.mean():.0f}\n"
            f"Weighted TVAL[1%,5%,95%]: {weighted_tval.quantile(0.01):.0f}, {weighted_tval.quantile(0.05):.0f}, {weighted_tval.quantile(0.95):.0f}\n"
            f"Weighted TVOL[1%,5%,95%]: {weighted_tvol.quantile(0.01):.0f}, {weighted_tvol.quantile(0.05):.0f}, {weighted_tvol.quantile(0.95):.0f}\n"
            f"Weighted TVAL[min]: {weighted_tval.min():.0f}\n"
            f"Weighted TVOL[min]: {weighted_tvol.min():.0f}\n"
        )

    def position_at(self, date):
        tmp = self.val_position.loc[date,:]
        return tmp[tmp != 0.0].dropna(axis=1, how='all')

    def summary(self, sdate=None, edate=None, daycount=250):
        if sdate is None:
            sdate = self._data_index[0]
        if edate is None:
            edate = self._data_index[-1]
        rtns = self._calculated['daily_returns'].loc[sdate:edate,:]

        res = {}
        res['mean'] = rtns.mean().iat[0] * daycount
        res['std'] = rtns.std().iat[0] * np.sqrt(daycount)
        res['sharpe'] = res['mean'] / res['std']
        res['sortino'] = res['mean'] / rtns[rtns<0].std().iat[0]
        cum_pnl = (rtns + 1).cumprod()
        res['mdd'] = ((cum_pnl - cum_pnl.cummax()) / cum_pnl.cummax()).min().iat[0]
        res['pnl'] = cum_pnl.to_numpy()[-1][0]
        res['win_rate'] = (rtns > 0).sum().iat[0] / len(rtns[rtns != 0])
        res['avg_win'] = rtns[rtns > 0].mean().iat[0]
        res['avg_loss'] = rtns[rtns < 0].mean().iat[0]
        res['return[95%]'] = (res['mean'] - 1.96*res['std'], res['mean'] + 1.96*res['std'])

        for key, value in res.items():
            if isinstance(value, tuple):
                res['return[95%]'] = tuple("%.4f" % x for x in res['return[95%]'])
            elif isinstance(value, float):
                res[key] = f"{value:.4f}"

        return res
    
    def summary_plot(self, sdate=None, edate=None, figsize=(16,4), log=False):
        if sdate is None:
            sdate = self._data_index[0]
        if edate is None:
            edate = self._data_index[-1]
        rtns = self._calculated['daily_returns'].loc[sdate:edate,:]

        cum_pnl = (rtns + 1).cumprod()
        drawdown = (cum_pnl - cum_pnl.cummax()) / cum_pnl.cummax()
        plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)
        if log:
            plt.plot(np.log(cum_pnl), label='Cumulative PnL (Log)')
        else:
            plt.plot(cum_pnl, label='Cumulative PnL')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(drawdown, label='Drawdown')
        plt.legend()

        plt.show()

    def summary_byperiod(self, freq='1Y', daycount=250):
        sdate = self._data_index[0]
        final_date = self._data_index[-1]
        if freq.endswith('Y'):
            offset = pd.DateOffset(years=int(freq[:-1]))
        elif freq.endswith('M'):
            offset = pd.DateOffset(months=int(freq[:-1]))
        elif freq.endswith('W'):
            offset = pd.DateOffset(weeks=int(freq[:-1]))
        else:
            raise ValueError("Invalid Frequency")
        # min_date = pd.Timedelta(days=2)

        res = {}
        while True:
            edate  = min(sdate + offset, final_date)
            try:
                res[f"{sdate.strftime('%Y-%m-%d')} ~ {edate.strftime('%Y-%m-%d')}"] = self.summary(daycount=daycount, sdate=sdate, edate=edate)
            except:
                pass
            if edate == final_date:
                break
            sdate = edate
        
        return pd.DataFrame(res).T
    
    def chart_rolling(self, sdate=None, edate=None, figsize=(16,8), win_size=250, daycount=250):
        if sdate is None:
            sdate = self._data_index[0]
        if edate is None:
            edate = self._data_index[-1]

        roll_mean = self._calculated['daily_returns'].rolling(window=win_size).mean() * daycount
        roll_std = self._calculated['daily_returns'].rolling(window=win_size).std() * np.sqrt(daycount)

        roll_mean = roll_mean.loc[sdate:edate,:].dropna()
        roll_std = roll_std.loc[sdate:edate,:].dropna()
        roll_sharpe = roll_mean / roll_std
        roll_mean.columns = ['Rolling Mean']
        roll_std.columns = ['Rolling Std']
        roll_sharpe.columns = ['Rolling Sharpe']

        expand_mean = self._calculated['daily_returns'].expanding(min_periods=win_size).mean() * daycount
        expand_std = self._calculated['daily_returns'].expanding(min_periods=win_size).std() * np.sqrt(daycount)

        expand_mean = expand_mean.loc[sdate:edate,:].dropna()
        expand_std = expand_std.loc[sdate:edate,:].dropna()
        expand_sharpe = expand_mean / expand_std
        expand_mean.columns = ['Expanding Mean']
        expand_std.columns = ['Expanding Std']
        expand_sharpe.columns = ['Expanding Sharpe']
        
        plt.figure(figsize=figsize)

        plt.subplot(3, 2, 1)
        plt.plot(roll_mean, label='Rolling Mean')
        plt.plot(roll_mean.index, np.zeros(len(roll_mean))+roll_mean.mean().iloc[0])
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(expand_mean, label='Expanding Mean')
        plt.plot(expand_mean.index, np.zeros(len(expand_mean))+expand_mean.mean().iloc[0])
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(roll_std, label='Rolling Std')
        plt.plot(roll_std.index, np.zeros(len(roll_std))+roll_std.mean().iloc[0])
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(expand_std, label='Expanding Std')
        plt.plot(expand_std.index, np.zeros(len(expand_std))+expand_std.mean().iloc[0])
        plt.legend()

        plt.subplot(3, 2, 5)
        plt.plot(roll_sharpe, label='Rolling Sharpe')
        plt.plot(roll_sharpe.index, np.zeros(len(roll_sharpe))+roll_sharpe.mean().iloc[0])
        plt.legend()

        plt.subplot(3, 2, 6)
        plt.plot(expand_sharpe, label='Expanding Sharpe')
        plt.plot(expand_sharpe.index, np.zeros(len(expand_sharpe))+expand_sharpe.mean().iloc[0])
        plt.legend()

        plt.show()

    def chart_return_hist(self, sdate=None, edate=None, figsize=(8,6), bins=30):
        if sdate is None:
            sdate = self._data_index[0]
        if edate is None:
            edate = self._data_index[-1]
        rtns = self._calculated['daily_returns'].loc[sdate:edate,:]

        plt.figure(figsize=figsize)
        plt.hist(rtns.to_numpy(), bins=bins)
        plt.show()

    def chart_num_position(self, sdate=None, edate=None, figsize=(6,3)):
        if sdate is None:
            sdate = self._data_index[0]
        if edate is None:
            edate = self._data_index[-1]
        pos = self.val_position.loc[sdate:edate,:]

        plt.figure(figsize=figsize)
        plt.plot(pos.replace(0, np.nan).count(1))
        plt.legend('Number of Holding Position')
        plt.show()

    def outlier_return(self, threshold=0.5):
        upper = np.nanpercentile(self._calculated['daily_returns'].to_numpy(), 100-threshold)
        lower = np.nanpercentile(self._calculated['daily_returns'].to_numpy(), threshold)
        return {
            'upper_mean': self._calculated['daily_returns'][self._calculated['daily_returns'] > upper].mean().iat[0],
            'lower_mean': self._calculated['daily_returns'][self._calculated['daily_returns'] < lower].mean().iat[0],
            'upper_days': self._calculated['daily_returns'][self._calculated['daily_returns'] > upper].dropna().index,
            'lower_days': self._calculated['daily_returns'][self._calculated['daily_returns'] < lower].dropna().index,
        }