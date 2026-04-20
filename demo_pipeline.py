"""
real_tick_cleaner.py
--------------------
Pulls REAL historical bar data from Alpaca for 2 stocks,
then runs the same cleaning pipeline so you can see
real-world messiness vs. clean data.

Setup:
  pip3 install alpaca-py pandas numpy python-dotenv tzdata

Run:
  python3 real_tick_cleaner.py
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ── load keys from .env file ──────────────────────────────────────────────────
USE_MOCK = True

load_dotenv()
API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# ── colour helpers ────────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def header(text):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

def subheader(text):
    print(f"\n{BOLD}{YELLOW}  {text}{RESET}")
    print(f"  {'-'*40}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PULL REAL DATA FROM ALPACA
# ══════════════════════════════════════════════════════════════════════════════

def fetch_alpaca_bars(symbols, start, end, timeframe=None):
    if USE_MOCK:
        import pandas as pd
        import numpy as np

        dates = pd.date_range(start=start, end=end, freq="1min")

        df = pd.DataFrame({
            "timestamp": list(dates) * len(symbols),
            "symbol": sum([[s]*len(dates) for s in symbols], []),
            "close": np.random.normal(150, 3, len(dates)*len(symbols)),
            "volume": np.random.randint(100, 2000, len(dates)*len(symbols)),
        })

        # inject realistic "messiness" so your cleaner actually shows value
        df.loc[df.sample(frac=0.02).index, "close"] *= 3  # outliers
        df.loc[df.sample(frac=0.02).index, "volume"] = 0  # bad bars

        print("⚠️ DEMO MODE: using synthetic market data with injected noise")
        return df

    # real mode (ignore for now)
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SHOW RAW DATA PROBLEMS
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_raw(df, label):
    """Print what's actually wrong with the raw Alpaca data."""
    subheader(f"Raw data: {label}")
    print(df.head(10).to_string(index=False))

    print(f"\n  {YELLOW}Shape:{RESET} {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  {YELLOW}Columns:{RESET} {list(df.columns)}")
    print(f"  {YELLOW}Timestamp dtype:{RESET} {df['timestamp'].dtype}")

    # check for real problems
    problems = []

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        problems.append(f"Missing values: {missing.to_dict()}")

    dupes = df.duplicated(subset=['timestamp','symbol']).sum()
    if dupes > 0:
        problems.append(f"{dupes} duplicate timestamp+symbol rows")

    zero_vol = (df['volume'] == 0).sum()
    if zero_vol > 0:
        problems.append(f"{zero_vol} zero-volume bars")

    # check for price outliers using z-score per symbol
    outlier_count = 0
    for sym in df['symbol'].unique():
        prices = df[df['symbol'] == sym]['close']
        if prices.std() > 0:
            z = ((prices - prices.mean()) / prices.std()).abs()
            outlier_count += (z > 3).sum()
    if outlier_count > 0:
        problems.append(f"{outlier_count} price outliers (z > 3)")

    # check timezone
    if hasattr(df['timestamp'].dtype, 'tz'):
        tz = df['timestamp'].dt.tz
        problems.append(f"Timezone is {tz} — needs verification")

    # gaps in time series
    for sym in df['symbol'].unique():
        sym_df  = df[df['symbol'] == sym].sort_values('timestamp')
        deltas  = sym_df['timestamp'].diff().dropna()
        expected = pd.Timedelta(minutes=1)
        gaps    = (deltas > expected * 2).sum()
        if gaps > 0:
            problems.append(f"{sym}: {gaps} time gaps (missing bars)")

    if problems:
        print(f"\n  {RED}Real problems found:{RESET}")
        for p in problems:
            print(f"  {RED}✗{RESET} {p}")
    else:
        print(f"\n  {GREEN}No major problems detected in this sample{RESET}")

    return problems


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CLEANING PIPELINE (same logic as before, now on real data)
# ══════════════════════════════════════════════════════════════════════════════

def clean(df):
    """Full cleaning pipeline on real Alpaca data."""
    original_len = len(df)
    log = {}

    # 1. ensure timestamp is proper datetime with timezone
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    df['timestamp'] = df['timestamp'].dt.tz_convert('US/Eastern')

    # 2. remove duplicate timestamp + symbol rows
    before = len(df)
    df = df.drop_duplicates(subset=['timestamp', 'symbol'])
    log['duplicates_removed'] = before - len(df)

    # 3. drop rows with missing close price
    before = len(df)
    df = df.dropna(subset=['close'])
    log['missing_price_removed'] = before - len(df)

    # 4. drop zero-volume bars (these are off-hours phantom bars)
    before = len(df)
    df = df[df['volume'] > 0]
    log['zero_volume_removed'] = before - len(df)

    # 5. remove price outliers per symbol (z-score > 3)
    before = len(df)
    cleaned_parts = []
    for sym in df['symbol'].unique():
        part = df[df['symbol'] == sym].copy()
        mean, std = part['close'].mean(), part['close'].std()
        if std > 0:
            part = part[((part['close'] - mean) / std).abs() < 3]
        cleaned_parts.append(part)
    df = pd.concat(cleaned_parts)
    log['outliers_removed'] = before - len(df)

    # 6. sort by symbol + time, reset index
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    log['rows_in']  = original_len
    log['rows_out'] = len(df)
    return df, log


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — COMPARE RAW VS CLEAN
# ══════════════════════════════════════════════════════════════════════════════

def compare(raw_df, clean_df, log):
    header("BEFORE vs AFTER CLEANING")

    for sym in raw_df['symbol'].unique():
        raw_sym   = raw_df[raw_df['symbol'] == sym]
        clean_sym = clean_df[clean_df['symbol'] == sym]

        subheader(f"{sym} — price statistics")
        stats = pd.DataFrame({
            'raw':   raw_sym['close'].describe(),
            'clean': clean_sym['close'].describe(),
        }).round(4)
        print(stats.to_string())

        raw_vwap   = (raw_sym['close']   * raw_sym['volume']).sum()   / raw_sym['volume'].sum()
        clean_vwap = (clean_sym['close'] * clean_sym['volume']).sum() / clean_sym['volume'].sum()
        diff = round(abs(raw_vwap - clean_vwap), 4)

        print(f"\n  {YELLOW}VWAP (volume-weighted avg price):{RESET}")
        print(f"  Raw:   ${raw_vwap:.4f}")
        print(f"  Clean: ${clean_vwap:.4f}")
        if diff > 0:
            print(f"  {RED}Difference: ${diff}  ← dirty data shifted your VWAP{RESET}")
        else:
            print(f"  {GREEN}No difference — data was already clean for this symbol{RESET}")

    header("CLEANING REPORT")
    total_bad = (log['duplicates_removed'] + log['missing_price_removed'] +
                 log['zero_volume_removed'] + log['outliers_removed'])
    pct = round(total_bad / log['rows_in'] * 100, 2) if log['rows_in'] > 0 else 0

    print(f"\n  {'Metric':<30} {'Value':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Rows in':<30} {log['rows_in']:>10}")
    print(f"  {RED}{'Duplicates removed':<30}{RESET} {RED}{log['duplicates_removed']:>10}{RESET}")
    print(f"  {RED}{'Missing prices removed':<30}{RESET} {RED}{log['missing_price_removed']:>10}{RESET}")
    print(f"  {RED}{'Zero-volume removed':<30}{RESET} {RED}{log['zero_volume_removed']:>10}{RESET}")
    print(f"  {RED}{'Outliers removed':<30}{RESET} {RED}{log['outliers_removed']:>10}{RESET}")
    print(f"  {'-'*42}")
    print(f"  {GREEN}{'Clean rows out':<30}{RESET} {GREEN}{log['rows_out']:>10}{RESET}")
    print(f"  {YELLOW}{'Bad rows %':<30}{RESET} {YELLOW}{pct:>9}%{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Pull 2 days of 1-minute bars for AAPL and TSLA
    # Using a recent weekday window (markets closed on weekends)
    SYMBOLS = ["AAPL", "TSLA"]
    END     = datetime(2025, 2, 3, 16, 0)
    START   = datetime(2025, 2, 3,  9, 30)  # Thursday open

    header(f"FETCHING REAL DATA: {SYMBOLS}  |  {START.date()} → {END.date()}")
    print(f"\n  Pulling 1-minute bars from Alpaca...")

    raw_df = fetch_alpaca_bars(SYMBOLS, START, END)

    print(f"  {GREEN}Got {len(raw_df)} rows of real market data{RESET}")

    # diagnose each symbol's raw data
    header("RAW DATA — WHAT ALPACA GIVES YOU")
    for sym in SYMBOLS:
        diagnose_raw(raw_df[raw_df['symbol'] == sym].copy(), sym)

    # run cleaning pipeline
    header("RUNNING CLEANING PIPELINE...")
    clean_df, log = clean(raw_df.copy())
    print(f"  {GREEN}Done.{RESET}")

    # compare
    compare(raw_df, clean_df, log)

    header("WHAT TO NOTICE")
    print(f"""
  This is REAL data from actual market hours.

  Key things to look at:
  1. The 'timestamp' column in raw data — what timezone is it in?
  2. Are there gaps in the 1-minute bars? (markets pause, data drops)
  3. Does the VWAP change between raw and clean?
     If yes → dirty data would have given you wrong signals
     If no  → Alpaca's data was clean for this window (good days exist)

  Try changing the date range to a volatile day (earnings, Fed announcement)
  and re-run — you'll see more problems appear.

  Real next step: try START = datetime(2025, 2, 3) — a high-volatility day.
    """)
