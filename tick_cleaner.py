"""
tick_cleaner.py
---------------
A hands-on intro to the tick data problem.

What this does:
  1. Simulates raw tick data from 3 brokers (Alpaca, IB, Polygon)
     — each with realistic, different messiness
  2. Cleans the data (removes duplicates, fills gaps, normalises timezones + field names)
  3. Prints a clear before/after comparison so you can SEE the problem

No API keys needed. Run with:  python tick_cleaner.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ── colour helpers for terminal output ────────────────────────────────────────
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
# SECTION 1 — SIMULATE RAW BROKER DATA
# Each broker returns the same underlying trades but with different problems.
# ══════════════════════════════════════════════════════════════════════════════

def generate_base_ticks(n=40):
    """True underlying trades — what actually happened in the market."""
    base_time = datetime(2024, 3, 15, 9, 30, 0)   # 9:30 AM market open
    times = [base_time + timedelta(seconds=i*3) for i in range(n)]
    prices = [150.00]
    for _ in range(n - 1):
        change = round(random.uniform(-0.15, 0.15), 2)
        prices.append(round(prices[-1] + change, 2))
    volumes = [random.randint(100, 1000) for _ in range(n)]
    return list(zip(times, prices, volumes))


def simulate_alpaca(base_ticks):
    """
    Alpaca quirks (real):
    - Field names: timestamp, price, size  (not 'volume')
    - Timestamps in UTC
    - ~10% duplicate ticks (sent twice due to network retry)
    - Occasional None price on cancelled orders
    """
    rows = []
    for ts, price, vol in base_ticks:
        ts_utc = ts + timedelta(hours=5)   # convert EST → UTC
        row = {
            "timestamp": ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price":     price,
            "size":      vol,               # <-- called 'size', not 'volume'
            "exchange":  "NASDAQ"
        }
        rows.append(row)
        # inject ~10% duplicates
        if random.random() < 0.10:
            rows.append(dict(row))          # exact copy

    # inject 2 rows with missing price (cancelled order artefacts)
    for idx in random.sample(range(len(rows)), 2):
        rows[idx]["price"] = None

    return pd.DataFrame(rows)


def simulate_interactive_brokers(base_ticks):
    """
    Interactive Brokers quirks (real):
    - Field names: time, last, volume  ('last' instead of 'price')
    - Timestamps in US/Eastern with AM/PM format
    - ~5% duplicate ticks
    - Some ticks have volume=0 (quote updates, not real trades)
    - Timestamps have occasional millisecond noise making them look unique
    """
    rows = []
    for ts, price, vol in base_ticks:
        row = {
            "time":     ts.strftime("%m/%d/%Y %I:%M:%S %p"),  # MM/DD/YYYY 12hr
            "last":     price,              # <-- called 'last', not 'price'
            "volume":   vol,
            "exchange": "NYSE"
        }
        rows.append(row)
        # inject ~5% duplicates (slightly different ms timestamp — looks unique!)
        if random.random() < 0.05:
            dupe = dict(row)
            # add 1ms — this is what fools naive dedup on timestamp alone
            ts_fake = ts + timedelta(milliseconds=1)
            dupe["time"] = ts_fake.strftime("%m/%d/%Y %I:%M:%S %p")
            rows.append(dupe)

    # inject volume=0 rows (quote ticks, not real trades)
    for idx in random.sample(range(len(rows)), 3):
        rows[idx]["volume"] = 0

    return pd.DataFrame(rows)


def simulate_polygon(base_ticks):
    """
    Polygon.io quirks (real):
    - Field names: t (unix ms), p (price), s (size)  — very terse
    - Timestamps as Unix milliseconds
    - ~15% duplicates (their free tier deduplication is poor)
    - A few extreme outlier prices (bad tick / fat finger data)
    """
    rows = []
    epoch = datetime(1970, 1, 1)
    for ts, price, vol in base_ticks:
        ms = int((ts - epoch).total_seconds() * 1000)
        row = {
            "t": ms,                        # <-- unix ms, not a readable time
            "p": price,                     # <-- single letter field names
            "s": vol,
            "x": "CBOE"                     # exchange code, also different
        }
        rows.append(row)
        # inject ~15% duplicates
        if random.random() < 0.15:
            rows.append(dict(row))

    # inject 2 extreme outlier prices (bad ticks — real thing)
    for idx in random.sample(range(len(rows)), 2):
        rows[idx]["p"] = round(random.choice([0.01, 9999.99]), 2)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CLEANING FUNCTIONS
# Each function solves one specific real problem.
# ══════════════════════════════════════════════════════════════════════════════

def normalise_alpaca(df):
    """Step 1: Make Alpaca data look like our standard schema."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern")  # → local time
    df = df.rename(columns={"size": "volume"})                      # fix field name
    df["source"] = "alpaca"
    return df[["timestamp", "price", "volume", "source"]]


def normalise_ib(df):
    """Step 1: Make IB data look like our standard schema."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p")
    df["timestamp"] = df["timestamp"].dt.tz_localize("US/Eastern")
    df = df.rename(columns={"last": "price"})                       # fix field name
    df["source"] = "ib"
    return df[["timestamp", "price", "volume", "source"]]


def normalise_polygon(df):
    """Step 1: Make Polygon data look like our standard schema."""
    df = df.copy()
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    df["timestamp"] = epoch + pd.to_timedelta(df["t"], unit="ms")   # unix ms → datetime
    df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern")
    df = df.rename(columns={"p": "price", "s": "volume"})           # fix field names
    df["source"] = "polygon"
    return df[["timestamp", "price", "volume", "source"]]


def remove_duplicates(df):
    """
    Duplicates are ticks with the same timestamp + price + volume.
    We keep the first occurrence and drop the rest.
    Note: IB's ms-offset duplicates are caught because we round to seconds.
    """
    df = df.copy()
    df["ts_rounded"] = df["timestamp"].dt.floor("s")   # round to second
    before = len(df)
    df = df.drop_duplicates(subset=["ts_rounded", "price", "volume"])
    df = df.drop(columns=["ts_rounded"])
    after = len(df)
    return df, before - after


def remove_missing_prices(df):
    """Drop rows where price is NaN/None — these are order artefacts."""
    before = len(df)
    df = df.dropna(subset=["price"])
    return df, before - len(df)


def remove_zero_volume(df):
    """Drop volume=0 rows — these are quote updates, not real trades."""
    before = len(df)
    df = df[df["volume"] > 0]
    return df, before - len(df)


def remove_outliers(df, z_thresh=3.0):
    """
    Remove price outliers using z-score.
    A tick with price 0.01 or 9999.99 on a $150 stock is clearly bad data.
    z_thresh=3 means: flag anything more than 3 std deviations from the mean.
    """
    before = len(df)
    mean  = df["price"].mean()
    std   = df["price"].std()
    df = df[((df["price"] - mean) / std).abs() < z_thresh]
    return df, before - len(df)


def sort_and_index(df):
    """Final step: sort by time and reset index."""
    return df.sort_values("timestamp").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def clean_pipeline(alpaca_raw, ib_raw, polygon_raw):
    """Run all cleaning steps and return the clean combined dataframe."""

    # Step 1 — normalise each broker to standard schema
    alpaca_norm  = normalise_alpaca(alpaca_raw)
    ib_norm      = normalise_ib(ib_raw)
    polygon_norm = normalise_polygon(polygon_raw)

    # Step 2 — combine all brokers
    combined = pd.concat([alpaca_norm, ib_norm, polygon_norm], ignore_index=True)

    total_rows_in = len(combined)

    # Step 3 — apply cleaning steps
    combined, dupes_removed    = remove_duplicates(combined)
    combined, missing_removed  = remove_missing_prices(combined)
    combined, zerovol_removed  = remove_zero_volume(combined)
    combined, outliers_removed = remove_outliers(combined)

    # Step 4 — sort and index
    combined = sort_and_index(combined)

    stats = {
        "rows_in":          total_rows_in,
        "rows_out":         len(combined),
        "dupes":            dupes_removed,
        "missing_prices":   missing_removed,
        "zero_volume":      zerovol_removed,
        "outliers":         outliers_removed,
    }

    return combined, stats


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PRINT COMPARISON REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(alpaca_raw, ib_raw, polygon_raw, clean_df, stats):

    header("RAW DATA — WHAT EACH BROKER GIVES YOU")

    subheader("Alpaca  (UTC timestamps, 'size' not 'volume', ~10% dupes)")
    print(alpaca_raw.head(8).to_string(index=False))
    print(f"\n  {RED}Problems visible:{RESET}")
    print(f"  - Timestamps in UTC (need conversion to Eastern)")
    print(f"  - Field called 'size' instead of 'volume'")
    print(f"  - Some 'price' values are None  ← order artefacts")
    dupes_a = len(alpaca_raw) - alpaca_raw.drop_duplicates().shape[0]
    print(f"  - {dupes_a} duplicate rows in this sample")

    subheader("Interactive Brokers  (12hr EST format, 'last' not 'price', vol=0 rows)")
    print(ib_raw.head(8).to_string(index=False))
    print(f"\n  {RED}Problems visible:{RESET}")
    print(f"  - Timestamps as '03/15/2024 09:30:01 AM' (non-standard)")
    print(f"  - Field called 'last' instead of 'price'")
    print(f"  - Some volume=0 rows  ← quote updates, not real trades")

    subheader("Polygon  (unix ms timestamps, single-letter fields, ~15% dupes)")
    print(polygon_raw.head(8).to_string(index=False))
    print(f"\n  {RED}Problems visible:{RESET}")
    print(f"  - Timestamps as Unix milliseconds (e.g. 1710491400000)")
    print(f"  - Fields named 't', 'p', 's' — not human readable")
    print(f"  - Some extreme outlier prices  ← bad ticks / fat finger")

    # ── what would happen if you naively combined raw data ──
    naive_combined = pd.concat([
        alpaca_raw.rename(columns={"size": "volume"})[["timestamp","price","volume"]],
        ib_raw.rename(columns={"last": "price", "time": "timestamp"})[["timestamp","price","volume"]],
        polygon_raw.rename(columns={"p": "price", "s": "volume", "t": "timestamp"})[["timestamp","price","volume"]]
    ], ignore_index=True)

    header("NAIVE COMBINATION — What happens if you just pd.concat() everything")
    print(naive_combined.head(12).to_string(index=False))
    print(f"\n  {RED}This dataframe is broken:{RESET}")
    print(f"  - 'timestamp' column has 3 different formats mixed together")
    print(f"  - You can't sort by time (comparing UTC strings to Unix ints)")
    print(f"  - Duplicates tripling up the same trades")
    print(f"  - None prices and zero volumes polluting any calculation")
    print(f"  - Total rows: {len(naive_combined)}  ← many are garbage")

    header("CLEANED DATA — After running the pipeline")
    print(clean_df.head(12).to_string(index=False))

    header("CLEANING REPORT")
    print(f"\n  {'Metric':<30} {'Value':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Rows in (all 3 brokers)':<30} {stats['rows_in']:>10}")
    print(f"  {RED}{'Duplicates removed':<30}{RESET} {RED}{stats['dupes']:>10}{RESET}")
    print(f"  {RED}{'Missing prices removed':<30}{RESET} {RED}{stats['missing_prices']:>10}{RESET}")
    print(f"  {RED}{'Zero-volume rows removed':<30}{RESET} {RED}{stats['zero_volume']:>10}{RESET}")
    print(f"  {RED}{'Outlier prices removed':<30}{RESET} {RED}{stats['outliers']:>10}{RESET}")
    total_bad = stats['dupes'] + stats['missing_prices'] + stats['zero_volume'] + stats['outliers']
    pct = round(total_bad / stats['rows_in'] * 100, 1)
    print(f"  {'-'*42}")
    print(f"  {GREEN}{'Clean rows out':<30}{RESET} {GREEN}{stats['rows_out']:>10}{RESET}")
    print(f"  {YELLOW}{'Bad rows (% of total)':<30}{RESET} {YELLOW}{pct:>9}%{RESET}")

    header("WHAT YOU LEARNED")
    print(f"""
  The same 40 underlying trades came in from 3 brokers.
  After combining: {stats['rows_in']} rows.
  After cleaning:  {stats['rows_out']} rows.

  {pct}% of your data was garbage.

  If you had run a mean reversion strategy on the raw data:
  - Outlier prices ($0.01, $9999.99) would trigger false signals
  - Duplicate ticks would make you think volume was 2–3x higher
  - Zero-volume rows would distort your volume-weighted calcs
  - Timezone mismatches would cause you to compare prices at wrong times

  This is why traders spend 3–5 hours/week on data cleaning.
  This is the problem you're solving.
    """)

    header("NEXT STEPS TO EXPLORE")
    print("""
  1. Try changing the duplicate rate (line: if random.random() < 0.10)
     → see how it affects the cleaning report numbers

  2. Add a new broker by writing a new simulate_X() and normalise_X()
     → you'll immediately feel how annoying inconsistent schemas are

  3. Try the strategy below on raw vs clean data and compare results:
       raw_mean  = alpaca_raw['price'].dropna().mean()
       clean_mean = clean_df['price'].mean()
       print(raw_mean, clean_mean)   # they should be different

  4. Real next step: swap simulate_alpaca() for actual Alpaca API call
     (free account at alpaca.markets, no trading required)
    """)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    random.seed(42)   # reproducible output

    # Generate base trades (what actually happened)
    base = generate_base_ticks(n=40)

    # Simulate each broker's messy version
    alpaca_raw  = simulate_alpaca(base)
    ib_raw      = simulate_interactive_brokers(base)
    polygon_raw = simulate_polygon(base)

    # Run the cleaning pipeline
    clean_df, stats = clean_pipeline(alpaca_raw, ib_raw, polygon_raw)

    # Print the full comparison report
    print_report(alpaca_raw, ib_raw, polygon_raw, clean_df, stats)
