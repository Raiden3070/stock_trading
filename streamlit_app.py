import os
import datetime as dt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from alpha_data import fetch_daily_close
from StrategyLearner import StrategyLearner
from ManualStrategy import trades_to_orders
from marketsimcode import compute_portvals
import indicators as ind


st.set_page_config(page_title="ML for Stock Decisions", page_icon="📈", layout="wide")


def get_api_key() -> str | None:
    # Prefer environment variable; fall back to Streamlit secrets if available.
    key = os.getenv("ALPHAVANTAGE_API_KEY")
    if key:
        return key
    try:
        # Accessing st.secrets may raise if no secrets file exists; guard with try/except
        return st.secrets["ALPHAVANTAGE_API_KEY"]
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def cached_fetch(symbol: str, sd_str: str, ed_str: str, outputsize: str = "full") -> pd.Series:
    # Cache wrapper that accepts strings (hashable) for dates
    api_key = get_api_key()
    sd = dt.datetime.fromisoformat(sd_str)
    ed = dt.datetime.fromisoformat(ed_str)
    return fetch_daily_close(symbol, sd, ed, api_key=api_key, outputsize=outputsize)


def compute_latest_signal(sl: StrategyLearner, symbol: str, prices: pd.Series) -> tuple[int, pd.Timestamp]:
    # Build indicators aligned to prices
    sd = prices.index.min().to_pydatetime()
    ed = prices.index.max().to_pydatetime()
    sma = ind.compute_price_sma(sd, ed, symbol, price_series=prices)
    bb = ind.compute_bb_percentage(sd, ed, symbol, price_series=prices)
    macd = ind.compute_macd(sd, ed, symbol, price_series=prices)
    momen = ind.compute_momentum(sd, ed, symbol, price_series=prices)
    ppo = ind.compute_ppo(sd, ed, symbol, price_series=prices)

    features = pd.concat([sma, bb, macd, ppo, momen], axis=1).fillna(0)
    y_pred = sl.learner.query(features.values)
    return int(y_pred[-1]), features.index[-1]


def main():
    st.title("Machine Learning for Stock Purchase Decisions")
    st.caption("Fetch live prices, train on a window, and get BUY/SELL/HOLD decisions.")

    api_key = get_api_key()
    if not api_key:
        st.warning("Set ALPHAVANTAGE_API_KEY in your environment or Streamlit secrets to use live data.")

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        symbol = st.text_input("Symbol", value="GOOG", help="Ticker symbol, e.g., AAPL, MSFT, GOOG").upper().strip()
        train_start = st.date_input(
            "Train start",
            value=dt.date(2023, 1, 1),
            help="Start of training window. For realistic results, the training period should PRECEDE the test/prediction window."
        )
        train_end = st.date_input(
            "Train end",
            value=dt.date(2023, 12, 31),
            help="End of training window. Avoid overlapping with the test window to prevent look-ahead bias."
        )
        test_end = st.date_input(
            "Test up to",
            value=dt.date.today(),
            help="The last date in the test window (prediction window ends here)."
        )
        test_months = st.slider(
            "Test window (months)",
            min_value=3,
            max_value=24,
            value=6,
            step=1,
            help="Length of the test window counted backwards from 'Test up to'."
        )
        impact = st.slider(
            "Impact",
            0.0,
            0.02,
            0.0,
            0.001,
            help="Simulated market impact (slippage) as a fractional price penalty per trade. Example: 0.001 = 0.1%."
        )
        commission = st.slider(
            "Commission ($)",
            0.0,
            19.95,
            0.0,
            0.05,
            help="Flat commission charged per order in dollars (applies to each BUY or SELL)."
        )

        with st.popover("Tips for better evaluation"):
            st.markdown(
                "- Train on an earlier window and test on a later, non-overlapping window.\n"
                "- Keep 'Train end' BEFORE the test window to avoid look-ahead bias.\n"
                "- Increase the test window months for more robust evaluation."
            )

        # Inform users about the bundled GOOG dataset (offline, not live)
        try:
            data_dir_info = os.path.join(os.path.dirname(__file__), "data")
            goog_full_csv = os.path.join(data_dir_info, "GOOG_TIME_SERIES_DAILY_full.csv")
            if os.path.isfile(goog_full_csv):
                _df_info = pd.read_csv(goog_full_csv, index_col=0, parse_dates=True)
                last_dt = _df_info.index.max().date() if not _df_info.empty else None
                if symbol.upper() == "GOOG":
                    st.info(f"GOOG is preloaded (offline). You can train/test without API limits up to {last_dt or 'the bundled end date'}. Data is not live.")
                else:
                    st.caption(f"Tip: GOOG is preloaded offline up to {last_dt or 'the bundled end date'} for unlimited demo runs (not live).")
        except Exception:
            pass


        run_btn = st.button("Train + Predict")

    if not run_btn:
        st.info("Select your parameters in the sidebar and click 'Train + Predict'.")
        return

    # Fetch data
    with st.spinner("Fetching data…"):
        tr_sd = dt.datetime.combine(train_start, dt.time())
        tr_ed = dt.datetime.combine(train_end, dt.time())
        te_ed = dt.datetime.combine(test_end, dt.time())
        te_sd = te_ed - dt.timedelta(days=int(test_months * 30))

        # Single API call: fetch a superset window, then slice into train/test
        combined_sd = min(tr_sd, te_sd)
        combined_ed = max(tr_ed, te_ed)
        span_days = (combined_ed - combined_sd).days
        outputsize = "compact" if span_days <= 100 else "full"

        try:
            full_series = cached_fetch(symbol, combined_sd.isoformat(), combined_ed.isoformat(), outputsize=outputsize)
            prices_train = full_series.loc[tr_sd:tr_ed]
            prices_test = full_series.loc[te_sd:te_ed]
        except ValueError as ve:
            # No API key: try bundled GOOG full CSV fallback
            data_dir = os.path.join(os.path.dirname(__file__), "data")
            fallback_csv = os.path.join(data_dir, "GOOG_TIME_SERIES_DAILY_full.csv")
            if symbol.upper() == "GOOG" and os.path.isfile(fallback_csv):
                df_fb = pd.read_csv(fallback_csv, index_col=0, parse_dates=True)
                if "GOOG" in df_fb.columns:
                    full_series = df_fb["GOOG"].astype(float)
                    prices_train = full_series.loc[tr_sd:tr_ed]
                    prices_test = full_series.loc[te_sd:te_ed]
                    st.warning("Using bundled GOOG dataset due to missing API key.")
                else:
                    st.error("Bundled GOOG fallback file found but missing 'GOOG' column.")
                    st.stop()
            else:
                st.error("Alpha Vantage API key not found. On Streamlit Cloud, add it via Settings → Secrets as ALPHAVANTAGE_API_KEY. For local runs, set the environment variable or .streamlit/secrets.toml.")
                st.stop()
        except RuntimeError as re:
            # Likely rate limit or API-side error; try bundled GOOG full CSV fallback
            data_dir = os.path.join(os.path.dirname(__file__), "data")
            fallback_csv = os.path.join(data_dir, "GOOG_TIME_SERIES_DAILY_full.csv")
            if symbol.upper() == "GOOG" and os.path.isfile(fallback_csv):
                df_fb = pd.read_csv(fallback_csv, index_col=0, parse_dates=True)
                if "GOOG" in df_fb.columns:
                    full_series = df_fb["GOOG"].astype(float)
                    prices_train = full_series.loc[tr_sd:tr_ed]
                    prices_test = full_series.loc[te_sd:te_ed]
                    st.warning("Using bundled GOOG dataset due to API rate limit.")
                else:
                    st.error("Bundled GOOG fallback file found but missing 'GOOG' column.")
                    st.stop()
            else:
                st.error(f"Data fetch failed: {re}")
                st.info("Tip: Alpha Vantage free tier allows up to 5 requests/min and up to 25 requests/day. Wait a minute and try again.")
                st.stop()

    if prices_train.empty:
        st.error("No training data returned. Check symbol or API limits.")
        return
    if prices_test.empty:
        st.error("No test data returned. Try expanding the test window.")
        return

    # Train
    with st.spinner("Training StrategyLearner…"):
        sl = StrategyLearner(verbose=False, impact=impact, commission=commission)
        sl.add_evidence(symbol=symbol, sd=tr_sd, ed=tr_ed, sv=100000, price_series=prices_train)

    # Predict latest signal
    signal, last_ts = compute_latest_signal(sl, symbol, prices_test)
    label = {1: "BUY", 0: "HOLD", -1: "SELL"}.get(signal, "UNKNOWN")

    # Show latest signal in the sidebar explicitly
    with st.sidebar:
        st.subheader("Latest signal")
        st.metric(label=f"{symbol} – {last_ts.date()}", value=label)

    # Compute learner trades for the test window once (used for chart + portfolios)
    sl_trades = sl.testPolicy(symbol=symbol, sd=te_sd, ed=te_ed, sv=100000, price_series=prices_test)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"{symbol} price (test window)")
        # Single chart with learner BUY/SELL markers
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(prices_test.index, prices_test.values, label=f"{symbol} price", color="tab:blue")
            if symbol in sl_trades.columns:
                buys = sl_trades[sl_trades[symbol] > 0]
                sells = sl_trades[sl_trades[symbol] < 0]
                if not buys.empty:
                    ax.scatter(buys.index, prices_test.loc[buys.index], marker="^", color="green", s=70, label="BUY")
                if not sells.empty:
                    ax.scatter(sells.index, prices_test.loc[sells.index], marker="v", color="red", s=70, label="SELL")
            ax.set_ylabel("Price")
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.info(f"Could not render trade markers: {e}")
    with col2:
        st.subheader("Latest signal")
        st.metric(label=f"{symbol} – {last_ts.date()}", value=label)

    # Optional: simulate portfolios for context
    with st.expander("Show normalized simulated portfolios (benchmark vs learner)"):
        # Use previously computed sl_trades
        sl_orders = trades_to_orders(sl_trades, symbol)
        sl_portvals = compute_portvals(sl_orders, start_val=100000, commission=commission, impact=impact,
                                       sd=te_sd, ed=te_ed,
                                       prices_df_override=pd.DataFrame({symbol: prices_test}))

        bench = pd.Series(0, index=prices_test.index, name=symbol).to_frame()
        if not bench.empty:
            bench.iloc[0] = 1000
        bench_orders = trades_to_orders(bench, symbol)
        bench_portvals = compute_portvals(bench_orders, start_val=100000, commission=commission, impact=impact,
                                          sd=te_sd, ed=te_ed,
                                          prices_df_override=pd.DataFrame({symbol: prices_test}))

        df = pd.DataFrame({
            "Learner": sl_portvals.squeeze(),
            "Benchmark": bench_portvals.squeeze(),
        })
        df_norm = df / df.iloc[0]
        st.line_chart(df_norm)


if __name__ == "__main__":
    main()
