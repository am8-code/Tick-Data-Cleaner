import pandas as pd
import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

API_KEY    = "PKKWBQQC4HCQV4OEDPKGZ2XQVW"
SECRET_KEY = "Fbk4Xoy6pgoFxN21znv86oXyP7XyZGHPg2tBK4F5Kert"

def fetch_data(symbol, start, end):
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    request = StockBarsRequest(
        symbol_or_symbols = symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc = TRUE)
    df['timestamp'] = df['timestamp'].dt.tz_convert('US/Eastern')
    df = df.sort_values('timestamp').reset_index(drop = TRUE)
    return df

def completeness_ratio(df):
    market_open = pd.Timestamp('09:30').time()
    market_close = pd.Timestamp('16:00').time()

    trading_df = df[
        (df['timestamp'].dt.time >= market_open) &
        (df['timestamp'].dt.time <= market_close)
    ]

    expected_minutes = 390 
    actual_minutes = len(trading_df)
    ratio = round(actual_minutes / expected_minutes * 100, 2)

    return ratio, actual_minutes, expected_minutes

def gap_severity(df):
    trading_df = df[
        (df['timestamp'].dt.time >= pd.Timestamp('09:30').time()) &
        (df['timestamp'].dt.time <= pd.Timestamp('16:00').time())
    ].copy()

    trading_df['delta'] = trading_df['timestamp'].diff()
    expected_gap = pd.Timedelta(minutes = 1)

    gaps = trading_df[trading_df['delta'] > expected_gap * 2]

    total_gap_minutes = gaps['delta'].sum().total_seconds() / 60
    severity = round(total_gap_minutes / 390 * 100, 2)
    return severity, len(gaps), round(total_gap_minutes, 1)

def rolling_zscore_outliers(df, window = 20, threshold=3.0):
    df = df.copy
    rolling_mean = df['close'].rolling(window=window).mean
    rolling_std = df['close'].rolling(window=window).std
    df['zscore'] = (df['close'] - rolling_mean) / rolling_std
    outliers = df[df['zscore'].abs() > threshold]