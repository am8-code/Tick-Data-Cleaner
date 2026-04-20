# Tick-Data-Cleaner

A pipeline that detects and corrects structural inconsistencies in high-frequency market data, improving reliability of quantitative trading signals.

## Problem
Broker-provided market data often contains issues such as:
- Outliers (erroneous price spikes)
- Missing or zero-volume bars
- Timestamp inconsistencies

These errors can distort trading metrics and lead to incorrect quantitative decisions.

## Solution
This system:
- Ingests raw market data
- Detects anomalies (outliers, zero-volume, gaps)
- Cleans and normalizes the dataset
- Quantifies impact on trading signals (e.g., VWAP shift)

## Key Insight
Even small data quality issues can significantly distort trading signals like VWAP, impacting downstream strategy decisions.

## How to Run
```bash
python demo_pipeline.py
