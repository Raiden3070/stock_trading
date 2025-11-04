import os
import requests
import pandas as pd
from typing import Optional

API_URL = "https://www.alphavantage.co/query"


def _cache_path(symbol: str, outputsize: str, cache_dir: str) -> str:
    filename = f"{symbol}_TIME_SERIES_DAILY_{outputsize}.csv"
    return os.path.join(cache_dir, filename)


def fetch_daily_close(symbol: str,
                      sd: pd.Timestamp,
                      ed: pd.Timestamp,
                      api_key: Optional[str] = None,
                      outputsize: str = "full",
                      use_cache: bool = True,
                      cache_dir: Optional[str] = None,
                      force_refresh: bool = False) -> pd.Series:
    """Fetch unadjusted daily close prices from Alpha Vantage TIME_SERIES_DAILY.

    Requirements:
    - Free API key (set ALPHAVANTAGE_API_KEY env var or pass api_key explicitly)
    - Uses '4. close' (unadjusted close) per user request

    Returns a pandas Series indexed by date (ascending) named with the symbol.
    """
    # Resolve cache directory (default: alongside this file in .av_cache)
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), ".av_cache")
    if use_cache and not os.path.isdir(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    cache_file = _cache_path(symbol, outputsize, cache_dir)

    # Try cache
    if use_cache and not force_refresh and os.path.isfile(cache_file):
        try:
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if symbol in cached.columns:
                series = cached[symbol].astype(float)
                series.name = symbol
                return series.loc[sd:ed]
        except Exception:
            # Cache read failure -> fall through to fetch
            pass

    # If we couldn't serve from cache, ensure API key is available for fetch
    if api_key is None:
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Alpha Vantage API key not provided. Set ALPHAVANTAGE_API_KEY or pass api_key.")

    # Fetch from API
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": outputsize,
        "datatype": "json",
    }
    resp = requests.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    key = "Time Series (Daily)"
    if key not in data:
        # Alpha Vantage sometimes returns a note on throttling or error message
        raise RuntimeError(f"Alpha Vantage response missing daily series. Response keys: {list(data.keys())}")

    ts = pd.DataFrame.from_dict(data[key], orient="index")
    # Columns are like '1. open','2. high','3. low','4. close','5. volume'
    ts.index = pd.to_datetime(ts.index)
    ts = ts.sort_index()
    close = ts['4. close'].astype(float)
    close.name = symbol

    # Write/refresh cache (store the full series for reuse, then slice for return)
    if use_cache:
        try:
            close.to_frame().to_csv(cache_file)
        except Exception:
            pass

    return close.loc[sd:ed]
