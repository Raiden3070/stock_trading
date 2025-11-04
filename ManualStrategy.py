import numpy as np
import pandas as pd
from util import get_data
import random as rand
from indicators import *
import ManualStrategy as manstr
import datetime as datetime
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt


class ManualStrategy:


    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def add_evidence(self, symbol='IBM', sd=datetime.datetime(2008, 1, 1, 0, 0), ed=datetime.datetime(2009, 1, 1, 0, 0), sv=100000):
        pass

    def testPolicy(self, symbol='IBM', sd=datetime.datetime(2009, 1, 1, 0, 0), ed=datetime.datetime(2010, 1, 1, 0, 0), sv=100000, price_series=None):

        sym = symbol
        if price_series is None:
            data_df = get_data([sym], pd.date_range(sd, ed))
            price_df = data_df[[sym]].ffill().bfill()
        else:
            s = pd.Series(price_series).loc[sd:ed].ffill().bfill()
            price_df = pd.DataFrame({sym: s})

        dates = price_df.index
        trades_df = price_df.copy()
        trades_df[:] = 0
        position = 0


        P_to_sma = compute_price_sma(sd, ed, sym, window_size=20, plot=False, price_series=price_df[sym])
        bb_percent = compute_bb_percentage(sd, ed, sym, window_size=20, plot=False, price_series=price_df[sym])
        momentum_ind = compute_momentum(sd, ed, sym, N=20, plot=False, price_series=price_df[sym])
        macd_ind = compute_macd(sd, ed, sym, delta=26, plot=False, price_series=price_df[sym])
        ppo_ind = compute_ppo(sd, ed, sym, short_window=12, long_window=26, signal_window=9, plot=False, price_series=price_df[sym])
        """
        print("sma length:", len(P_to_sma))
        print("bb length:", len(bb_percent))
        print("momen length:", len(momentum_ind))
        print("macd length:", len(macd_ind))
        print("ppo length:", len(ppo_ind))

        print("sma:", P_to_sma) 
        print("bb:",bb_percent) 
        print("momentum:",momentum_ind) 
        print("macd:",macd_ind) 
        print("ppo:",ppo_ind) 
        """
        action = 0
        for i in range(len(dates)):
            date = dates[i]

            sma = P_to_sma[date]
            if sma <= 0.9:
                sma_score = 1
            elif sma >= 1.1:
                sma_score = -1
            else:
                sma_score = 0

            bb = bb_percent[date]
            if bb <= 15:
                bb_score = 1
            elif bb >= 90:
                bb_score = -1
            else:
                bb_score = 0

            momen = momentum_ind[date]
            if momen >= 0.8:
                momen_score = 1
            elif momen <= -0.8:
                momen_score = -1
            else:
                momen_score = 0

            macd = macd_ind[date]
            if macd >= 0.6:
                macd_score = 1
            elif macd <= -0.6:
                macd_score = -1
            else:
                macd_score = 0

            ppo = ppo_ind[date]
            if ppo >= 3.8:
                ppo_score = 1
            elif ppo <= -3.8:
                ppo_score = -1
            else:
                ppo_score = 0

            total = sma_score + bb_score + macd_score + momen_score + ppo_score
            if total >= 2:
                action = 1000 - position
            elif total <= -2:
                action = - 1000 - position
            else:
                action = 0

            # Avoid chained assignment for pandas 3.0+ compatibility
            trades_df.loc[date, symbol] = action
            position += action

        return trades_df

    def author(self):
        return 'jkim3070'


def calculate_benchmark(sym,sd,ed,sv):
    data_df = get_data([sym], pd.date_range(sd, ed))
    price_df = data_df[[sym]]
    price_df = price_df[[sym]].ffill().bfill()
    trades_df = price_df.copy()
    trades_df[:] = 0
    trades_df.loc[trades_df.index[0]] = 1000
    orders_df = trades_to_orders(trades_df)
    portvals = compute_portvals(orders_df, sv, commission=9.95, impact=0.005, sd=sd, ed=ed)
    return portvals

def trades_to_orders(trades_df, sym="JPM"):
    orders_df = pd.DataFrame(index=trades_df.index, columns=["Symbol", "Order", "Shares"])
    trades_df_abs = trades_df.abs()
    orders_df["Symbol"] = sym
    orders_df["Order"] = trades_df.where(trades_df < 0, "BUY").where(trades_df > 0, "SELL")
    orders_df["Shares"] = trades_df_abs

    return orders_df

def calculate_metrics(portvals,sd,ed):
    daily_rets = (portvals / portvals.shift(1)) - 1
    daily_rets = daily_rets[1:]
    cr = portvals[-1] / portvals[0] - 1
    adr = daily_rets.mean()
    # sddr = statistics.stdev(daily_rets)
    sddr = daily_rets.std()
    sr = (adr / sddr) * (252 ** 0.5)

    print(f"Date Range: {sd} to {ed}")
    print()
    print(f"Sharpe Ratio of Fund: {sr}")
    print()
    print(f"Cumulative Return of Fund: {cr}")
    print()
    print(f"Standard Deviation of Fund: {sddr}")
    print()
    print(f"Average Daily Return of Fund: {adr}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


def generate_plot(benchmark,portvals, trades_df, test_method):

    portvals_normalized = portvals / portvals.iloc[0]
    benchmark_normalized = benchmark / benchmark.iloc[0]

    plt.figure(figsize=(10, 6))
    portvals_normalized.plot(label='Manual Strategy', color='red')
    benchmark_normalized.plot(label='Benchmark', color='purple')

    l_entry_long = trades_df.index[
                       (trades_df['JPM'].diff() == 2000) | (trades_df['JPM'].diff() == 1000)] + pd.Timedelta(days=1)
    s_entry_short = trades_df.index[
                        (trades_df['JPM'].diff() == -2000) | (trades_df['JPM'].diff() == -1000)] + pd.Timedelta(days=1)

    long_line = plt.axvline(color='blue', linestyle='-', label='Long Entry')
    short_line = plt.axvline(color='black', linestyle='-', label='Short Entry')
    for entry_point in l_entry_long:
        plt.axvline(x=entry_point, color='blue', linestyle='-')

    for entry_point in s_entry_short:
        plt.axvline(x=entry_point, color='black', linestyle='-')

    plt.title(f'Performance of Manual Strategy Compared with Benchmark- {test_method}')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"images/manual_strategy-{test_method}.png")
    #plt.show()
def test_code():
    man_strat = ManualStrategy(verbose=False, impact=0.0, commission=0.0)
    in_dates_sd = datetime.datetime(2008, 1, 1)
    in_dates_ed = datetime.datetime(2009, 12, 31)
    out_dates_sd = datetime.datetime(2010, 1, 1)
    out_dates_ed = datetime.datetime(2011, 12, 31)
    sv =100000
    sym = "JPM"

    man_strat_trades_df_in = man_strat.testPolicy(symbol="JPM", sd=in_dates_sd,ed=in_dates_ed, sv=sv)
    man_strat_orders_df_in = trades_to_orders(man_strat_trades_df_in, sym)

    man_strat_portvals_in = compute_portvals(man_strat_orders_df_in, sv, commission=9.95, impact=0.005,sd=in_dates_sd,ed=in_dates_ed)
    benchmark_portvals_in = calculate_benchmark(sym="JPM", sd=in_dates_sd,ed=in_dates_ed, sv=sv)

    man_strat_trades_df_out = man_strat.testPolicy(symbol = "JPM", sd = out_dates_sd, ed = out_dates_ed, sv=sv)
    man_strat_orders_df_out = trades_to_orders(man_strat_trades_df_out, sym)

    man_strat_portvals_out = compute_portvals(man_strat_orders_df_out, sv, commission=9.95, impact=0.005, sd=out_dates_sd, ed=out_dates_ed)
    benchmark_portvals_out = calculate_benchmark(sym = "JPM", sd = out_dates_sd, ed=out_dates_ed, sv=sv)
    """
    print("Manual Strategy in-sample")
    calculate_metrics(man_strat_portvals_in,in_dates_sd,in_dates_ed)
    print("Benchmark in-sample")
    calculate_metrics(benchmark_portvals_in,in_dates_sd,in_dates_ed)
    print("Manual Strategy out-of-sample")
    calculate_metrics(man_strat_portvals_out,out_dates_sd,out_dates_ed)
    print("Benchmark out-of-sample")
    calculate_metrics(benchmark_portvals_out,out_dates_sd,out_dates_ed)
    """

    generate_plot(benchmark_portvals_in,man_strat_portvals_in, man_strat_trades_df_in,'In-Sample')
    generate_plot(benchmark_portvals_out,man_strat_portvals_out,man_strat_trades_df_out,'Out-of-Sample')


if __name__ == "__main__":
    test_code()



