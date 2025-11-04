import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data


def author():
    return 'jkim3070'


def compute_price_sma(sd, ed, symbol, window_size=20, plot=False, price_series=None):
    extended_sd = sd - pd.Timedelta(2 * window_size, 'D')

    if price_series is None:
        df = get_data([symbol], pd.date_range(extended_sd, ed))
        s_all = df[[symbol]].ffill().bfill()[symbol]
    else:
        s_all = pd.Series(price_series).ffill().bfill()

    sma = s_all.rolling(window=window_size).mean()

    price_sma_ratio = (s_all / sma).loc[sd:ed]

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle('Price and SMA for {}'.format(symbol))

        ax1.plot(s_all.index, s_all, label='Price', color='black')
        ax1.plot(sma.index, sma, label='SMA ({} days)'.format(window_size), color='red')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax2.plot(price_sma_ratio.index, price_sma_ratio, label='Price/SMA Ratio', color='green')
        ax2.axhline(1, linestyle='--', color='gray')
        ax2.set_ylabel('Price/SMA Ratio')
        ax2.legend(loc='upper left')
        ax2.grid(True)

        plt.xlabel('Date')
        plt.savefig("images/price_sma.png")
        plt.close()

    return price_sma_ratio


def compute_bb_percentage(sd, ed, symbol, window_size=20, plot=False, price_series=None):
    extended_sd = sd - pd.Timedelta(2 * window_size, 'D')
    if price_series is None:
        s_all = get_data([symbol], pd.date_range(extended_sd, ed))[symbol].ffill().bfill()
    else:
        s_all = pd.Series(price_series).ffill().bfill()

    sma = s_all.rolling(window=window_size).mean()
    stdev = s_all.rolling(window=window_size).std()

    upper_band = sma + 2 * stdev
    lower_band = sma - 2 * stdev

    bb_percentage = ((s_all - lower_band) / (upper_band - lower_band)) * 100
    bb_percentage = bb_percentage.loc[sd:ed]

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        plt.suptitle('Bollinger Bands Percentage Indicator (%B) for {}'.format(symbol))

        ax1.plot(s_all.index, s_all, label=symbol, color='blue')
        ax1.plot(upper_band.index, upper_band, label='Upper Bollinger Band', color='red')
        ax1.plot(lower_band.index, lower_band, label='Lower Bollinger Band', color='green')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend(loc='lower left')

        ax2.plot(bb_percentage.index, bb_percentage, label='%B', color='purple')
        ax2.axhline(100, linestyle='--', color='gray')
        ax2.axhline(0, linestyle='--', color='gray')
        ax2.axhline(-100, linestyle='--', color='gray')
        ax2.set_ylabel('%B')
        ax2.grid(True)
        ax2.legend(loc='upper right')

        plt.xlabel('Date')
        plt.savefig("images/bb_percentage.png")

    return bb_percentage

def compute_momentum(sd, ed, symbol, N=20, plot=False, price_series=None):
    extended_sd = sd - pd.Timedelta(2 * N, 'D')
    if price_series is None:
        s_all = get_data([symbol], pd.date_range(extended_sd, ed))[symbol].ffill().bfill()
    else:
        s_all = pd.Series(price_series).ffill().bfill()

    momentum = (s_all / s_all.shift(N)) - 1
    momentum = momentum.loc[sd:ed]
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        plt.suptitle('Price and Momentum for {}'.format(symbol))

        ax1.plot(s_all.index, s_all, label='Price', color='black')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        plt.xlabel('Date')
        ax1.grid(True)

        ax2.plot(momentum.index, momentum, label='Momentum', color='blue')
        ax2.axhline(0, linestyle='--', color='gray')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Momentum')
        ax2.legend(loc='upper left')
        ax2.grid(True)

        plt.savefig("images/momentum.png")

    return momentum


def compute_macd(sd, ed, symbol, delta=26, plot=False, price_series=None):

    extended_sd = sd - pd.Timedelta(2 * delta, 'D')

    if price_series is None:
        df_price = get_data([symbol], pd.date_range(extended_sd, ed))[symbol]
        df_price = df_price.ffill().bfill()
    else:
        df_price = pd.Series(price_series).ffill().bfill()

    ema_12 = df_price.ewm(span=12, adjust=False).mean()
    ema_26 = df_price.ewm(span=26, adjust=False).mean()
    macd_raw = ema_12 - ema_26
    macd_signal = macd_raw.ewm(span=9, adjust=False).mean()

    macd_raw = macd_raw.truncate(before=sd)
    macd_signal = macd_signal.truncate(before=sd)
    macd_histogram = macd_raw-macd_signal

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        plt.suptitle('MACD and MACD Signal for {}'.format(symbol))

        ax1 = plt.subplot(211)
        ax1.plot(df_price.index, df_price, label='Price', color='blue')
        ax1.legend()
        ax1.set_ylabel('Price')
        ax1.grid(True)

        ax2 = plt.subplot(212)
        ax2.plot(macd_raw.index, macd_raw, label='MACD', color='orange')
        ax2.plot(macd_signal.index, macd_signal, label='MACD Signal', color='red')
        ax2.legend()
        ax2.set_xlabel('Date')
        ax2.set_ylabel('MACD')
        ax2.grid(True)

        plt.savefig("images/macd_combined.png", bbox_inches='tight')

    return macd_histogram


def compute_ppo(sd, ed, symbol, short_window=12, long_window=26, signal_window=9, plot=False, price_series=None):

    extended_sd = sd - pd.Timedelta(2 * long_window, 'D')

    if price_series is None:
        df_price = get_data([symbol], pd.date_range(extended_sd, ed))[symbol]
        df_price = df_price.ffill().bfill()
    else:
        df_price = pd.Series(price_series).ffill().bfill()

    short_ema = df_price.ewm(span=short_window, adjust=False).mean()
    long_ema = df_price.ewm(span=long_window, adjust=False).mean()
    ppo = ((short_ema - long_ema) / long_ema) * 100
    ppo_signal = ppo.ewm(span=signal_window, adjust=False).mean()

    ppo = ppo.truncate(before=sd)
    ppo_signal = ppo_signal.truncate(before=sd)
    ppo_histogram = ppo-ppo_signal
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        plt.suptitle('Price, PPO, and PPO Signal for {}'.format(symbol))

        ax1.plot(df_price.index, df_price, label='Price', color='black')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True)


        ax2.plot(ppo.index, ppo, label='PPO', color='green')
        ax2.plot(ppo_signal.index, ppo_signal, label='PPO Signal', color='blue')
        ax2.set_ylabel('PPO / PPO Signal')
        ax2.legend(loc='upper left')
        ax2.grid(True)

        plt.xlabel('Date')
        plt.savefig("images/ppo_with_price.png")

    return ppo_histogram


def test_code():
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    sma_df = compute_price_sma(sd, ed, symbol, window_size=20, plot=True)

    bb_percentage_df = compute_bb_percentage(sd, ed, symbol, window_size=20, plot=True)

    momentum_df = compute_momentum(sd, ed, symbol, N=10, plot=True)

    macd_df = compute_macd(sd, ed, symbol, delta=26, plot=True)

    ppo_df = compute_ppo(sd, ed, symbol, short_window=12, long_window=26, signal_window=9, plot=True)


if __name__ == "__main__":
    test_code()
