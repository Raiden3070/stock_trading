import datetime as datetime
import pandas as pd
from util import get_data



def author():
    return 'jkim3070'

def compute_portvals(orders_df, start_val=1000000, commission=9.95, impact=0.005,
                     sd=datetime.datetime(2008, 1, 1), ed=datetime.datetime(2009, 12, 31),
                     prices_df_override: pd.DataFrame | None = None):
    orders_df = orders_df.sort_index()
    symbols = orders_df['Symbol'].unique()
    dates = orders_df.index

    if prices_df_override is not None:
        prices = prices_df_override.copy()
    else:
        prices = get_data(symbols, pd.date_range(dates[0], dates[-1])).drop(columns='SPY')

    if 'SPY' in symbols:
        symbols = symbols.drop('SPY')
        prices = prices.drop(columns='SPY')
    if '$SPX' in symbols:
        symbols = symbols.drop('$SPX')
        prices = prices.drop(columns='$SPX')

    # Ensure float dtypes to avoid incompatible dtype warnings on arithmetic
    prices_df = pd.DataFrame(prices).astype(float)
    prices_df['cash'] = 1.0
    trades_df = prices_df.copy()
    trades_df.loc[:, :] = 0.0

    holdings_df = trades_df.copy()

    comms = {d: 0.0 for d in dates}

    for index, row in orders_df.iterrows():
        sym = row['Symbol']
        share = row['Shares']
        if row['Order'] == 'SELL':
            trades_df.loc[index, sym] -= share
        else:
            trades_df.loc[index, sym] += share

        if share != 0:
            comms[index] -= commission + (share * prices_df.loc[index, sym] * impact)

    for index in dates:
        trades_df.loc[index, 'cash'] += -1 * (trades_df.loc[index, symbols] *
                                              prices_df.loc[index, symbols]).sum() + comms[index]

    holdings_df.iloc[0, :-1] = trades_df.iloc[0, :-1]
    holdings_df.loc[dates[0], 'cash'] = start_val + trades_df.loc[dates[0], 'cash']

    for i in range(1, holdings_df.shape[0]):
        holdings_df.iloc[i, :] = trades_df.iloc[i, :] + holdings_df.iloc[i - 1, :]

    values_df = holdings_df * prices_df
    portvals = values_df.sum(axis=1)

    # Use position-based indexing explicitly for pandas 3.0+ compatibility
    portvals_z = portvals.iloc[0]
    portvals = portvals.shift(1)
    portvals = portvals.loc[sd:ed]
    portvals.iloc[0] = portvals_z

    return portvals

