import argparse
import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# -----------------------------
# Argument Parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Yahoo Finance Data Pipeline")

    parser.add_argument("--tickers_file", type=str, required=True,
                        help="File containing tickers (one per line)")
    parser.add_argument("--output", type=str, default="data/data.parquet",
                        help="Output file path")
    parser.add_argument("--period", type=str, default="5d",
                        help="Yahoo period (e.g. 1d, 5d, 1mo)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (YYYY-MM-DD)")

    # Filtering
    parser.add_argument("--min_volume", type=int, default=None)
    parser.add_argument("--min_return", type=float, default=None)

    return parser.parse_args()


# -----------------------------
# Load tickers
# -----------------------------
def load_tickers(path):
    with open(path) as f:
        tickers = [line.strip() for line in f if line.strip()]
    return tickers


# -----------------------------
# Fetch data
# -----------------------------
def fetch_data(tickers, period=None, start=None, end=None):
    print(f"📥 Fetching data for {len(tickers)} tickers...")

    df = yf.download(
        tickers,
        period=period if start is None else None,
        start=start,
        end=end,
        group_by="ticker",
        progress=False
    )

    return df


# -----------------------------
# Transform data
# -----------------------------
def transform_data(df):
    # Convert multi-index → flat format
    df = df.stack(level=0).rename_axis(["Date", "Ticker"]).reset_index()
    return df


# -----------------------------
# Feature Engineering
# -----------------------------
def compute_features(df):
    df["Return"] = df.groupby("Ticker")["Close"].pct_change()
    return df


# -----------------------------
# Filtering
# -----------------------------
def filter_data(df, min_volume=None, min_return=None):
    if min_volume is not None:
        df = df[df["Volume"] >= min_volume]

    if min_return is not None:
        df = df[df["Return"] >= min_return]

    return df


# -----------------------------
# Save / Update data
# -----------------------------
def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        old = pd.read_parquet(output_path)
        df = pd.concat([old, df])
        df = df.drop_duplicates(subset=["Date", "Ticker"], keep="last")

    df.to_parquet(output_path, index=False)
    print(f"✅ Saved data → {output_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    tickers = load_tickers(args.tickers_file)

    raw = fetch_data(
        tickers,
        period=args.period,
        start=args.start,
        end=args.end
    )

    df = transform_data(raw)
    df = compute_features(df)
    df = filter_data(df, args.min_volume, args.min_return)

    save_data(df, args.output)


if __name__ == "__main__":
    main()
