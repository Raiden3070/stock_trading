""""""  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	   			  		 			     			  	 
All Rights Reserved  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	   			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   			  		 			     			  	 
or edited.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	   			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	   			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   			  		 			     			  	 
GT honor code violation.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Student Name: Raiden Kim (replace with your name)  		  	   		 	   			  		 			     			  	 
GT User ID: jkim3070 (replace with your User ID)  		  	   		 	   			  		 			     			  	 
GT ID: 903072988 (replace with your GT ID)  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import datetime as dt
import random
import pandas as pd
from util import get_data
import RTLearner as rtl
import BagLearner as bl
import indicators as ind
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
class StrategyLearner(object):  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		 	   			  		 			     			  	 
    :type verbose: bool  		  	   		 	   			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	   			  		 			     			  	 
    :type impact: float  		  	   		 	   			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	   			  		 			     			  	 
    :type commission: float  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    # constructor  		  	   		 	   			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Constructor method  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        self.verbose = verbose  		  	   		 	   			  		 			     			  	 
        self.impact = impact  		  	   		 	   			  		 			     			  	 
        self.commission = commission
        self.learner = bl.BagLearner(learner=rtl.RTLearner, bags=50, kwargs={"verbose": False, "leaf_size": 5}, boost=False, verbose=False)
  		  	   		 	   			  		 			     			  	 
    # this method should create a QLearner, and train it for trading  		  	   		 	   			  		 			     			  	 
    def add_evidence(   
        self,  		  	   		 	   			  		 			     			  	 
        symbol="IBM",  		  	   		 	   			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		 	   			  		 			     			  	 
        ed=dt.datetime(2009, 1, 1),  		  	   		 	   			  		 			     			  	 
        sv=100000,
        price_series=None,
    ):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		 	   			  		 			     			  	 
        :type symbol: str  		  	   		 	   			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   			  		 			     			  	 
        :type sd: datetime  		  	   		 	   			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   			  		 			     			  	 
        :type ed: datetime  		  	   		 	   			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	   			  		 			     			  	 
        :type sv: int  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        # add your code to do learning here  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        # Prefer injected series to avoid filesystem reads (works on Streamlit Cloud)
        if price_series is not None:
            prices = pd.Series(price_series).loc[sd:ed].ffill().bfill()
        else:
            prices = get_data([symbol], pd.date_range(sd, ed))[symbol]
        if self.verbose:
            print("Training on prices for:", symbol)
            print(prices)

        # Compute indicators (use provided series if any)
        sma = ind.compute_price_sma(sd, ed, symbol, price_series=price_series)
        bb = ind.compute_bb_percentage(sd, ed, symbol, price_series=price_series)
        macd = ind.compute_macd(sd, ed, symbol, price_series=price_series)
        momen = ind.compute_momentum(sd, ed, symbol, price_series=price_series)
        ppo = ind.compute_ppo(sd, ed, symbol, price_series=price_series)

        N = 19
        # Use forward N-day return for labeling; simulator handles impact/commission
        returns = (prices.shift(-N) / prices) - 1.0

        YBUY = 0.06
        YSELL = -0.18
        # Build target as integer Series from the start to avoid downcast warnings
        Y = pd.Series(0, index=returns.index, dtype=int, name='Y')
        Y.loc[returns > YBUY] = 1
        Y.loc[returns < YSELL] = -1

        dataset = pd.concat([sma, bb, macd, ppo, momen, Y], axis=1).fillna(0)

        X = dataset.iloc[:, :-1].values
        Y = dataset.iloc[:, -1].values
        self.learner.add_evidence(X, Y)
    # this method should use the existing policy and test it against new data  		  	   		 	   			  		 			     			  	 
    def testPolicy(   
        self,  		  	   		 	   			  		 			     			  	 
        symbol="IBM",  		  	   		 	   			  		 			     			  	 
        sd=dt.datetime(2009, 1, 1),  		  	   		 	   			  		 			     			  	 
        ed=dt.datetime(2010, 1, 1),  		  	   		 	   			  		 			     			  	 
        sv=100000,
        price_series=None,
    ):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		 	   			  		 			     			  	 
        :type symbol: str  		  	   		 	   			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   			  		 			     			  	 
        :type sd: datetime  		  	   		 	   			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   			  		 			     			  	 
        :type ed: datetime  		  	   		 	   			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	   			  		 			     			  	 
        :type sv: int  		  	   		 	   			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	   			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	   			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	   			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	   			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		 	   			  		 			     			  	 
        """

        if price_series is not None:
            prices = pd.Series(price_series).loc[sd:ed].ffill().bfill()
        else:
            prices = get_data([symbol], pd.date_range(sd, ed))[symbol]

        sma = ind.compute_price_sma(sd, ed, symbol, price_series=prices)
        bb = ind.compute_bb_percentage(sd, ed, symbol, price_series=prices)
        macd = ind.compute_macd(sd, ed, symbol, price_series=prices)
        momen = ind.compute_momentum(sd, ed, symbol, price_series=prices)
        ppo = ind.compute_ppo(sd, ed, symbol, price_series=prices)

        indicators = pd.concat([sma, bb, macd, ppo, momen], axis=1).fillna(0)

        dataX = indicators.values
        dataY = self.learner.query(dataX)
        if self.verbose:
            print("DATA Y")
            print(dataY)

        trades = pd.DataFrame(index=prices.index, columns=[symbol])
        holding = 0
        for i in range(len(prices)):
            if dataY[i] == 1:  # LONG
                trades.iloc[i] = 1000 - holding
            elif dataY[i] == -1:  # SHORT
                trades.iloc[i] = -1000 - holding
            else:  # CASH
                trades.iloc[i] = -holding
            holding += trades.iloc[i]

        if self.verbose:
            print("This is trades type")
            print(type(trades))
            print("This is trades")
            print(trades)
            print("This is prices for testing policy period")
            print(prices)
        return trades

    def author(self):
        return 'jkim3070'
  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	   			  		 			     			  	 
    print("One does not simply think up a strategy")  		  	   		 	   			  		 			     			  	 
