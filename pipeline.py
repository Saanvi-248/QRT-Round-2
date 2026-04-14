import argparse
import yfinance as yf
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------
# Config
# -----------------------------
MAX_WORKERS = 8
BATCH_SIZE = 50
RETRIES = 2


# -----------------------------
# Argument Parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Yahoo Finance Data Pipeline")

    parser.add_argument("--tickers_file", type=str, required=True,
                        help="File with tickers (one per line)")
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/")
    parser.add_argument("--format", choices=["csv", "parquet"], default="parquet")

    # Filtering args
    parser.add_argument("--min_volume", type=int, default=None)
    parser.add_argument("--min_return", type=float, default=None)

    return parser.parse_args()


# -----------------------------
# Load tickers
# -----------------------------
def load_tickers(path):
    with open(path, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    return tickers


# -----------------------------
# Fetch data (with retry)
# -----------------------------
def fetch_one(ticker, start, end):
    for attempt in range(RETRIES + 1):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)

            if df.empty:
                return None

            df.reset_index(inplace=True)
            df["Ticker"] = ticker
            return df

        except Exception as e:
            if attempt < RETRIES:
                time.sleep(1)
            else:
                print(f"Failed {ticker}: {e}")
                return None


# -----------------------------
# Feature Engineering
# -----------------------------
def compute_features(df):
    df["Return"] = df["Adj Close"].pct_change()
    df["MA_10"] = df["Adj Close"].rolling(10).mean()
    df["Volatility"] = df["Return"].rolling(10).std()
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
# Process batch (parallel)
# -----------------------------
def process_batch(tickers, start, end, min_volume, min_return):
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_one, t, start, end): t for t in tickers
        }

        for future in as_completed(futures):
            ticker = futures[future]
            df = future.result()

            if df is not None:
                df = compute_features(df)
                df = filter_data(df, min_volume, min_return)
                results.append(df)

    return results


# -----------------------------
# Save batch
# -----------------------------
def save_batch(dfs, output_dir, batch_id, fmt):
    if not dfs:
        return

    combined = pd.concat(dfs, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)

    if fmt == "parquet":
        path = os.path.join(output_dir, f"batch_{batch_id}.parquet")
        combined.to_parquet(path, index=False)
    else:
        path = os.path.join(output_dir, f"batch_{batch_id}.csv")
        combined.to_csv(path, index=False)

    print(f"✅ Saved batch {batch_id} -> {path}")


# -----------------------------
# Main Pipeline
# -----------------------------
def run_pipeline(args):
    tickers = load_tickers(args.tickers_file)
    total = len(tickers)

    print(f"🚀 Processing {total} tickers...")

    for i in range(0, total, BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        batch_id = i // BATCH_SIZE

        print(f"\n📦 Batch {batch_id} ({len(batch)} tickers)")

        results = process_batch(
            batch,
            args.start,
            args.end,
            args.min_volume,
            args.min_return
        )

        save_batch(results, args.output_dir, batch_id, args.format)

        # avoid rate limiting
        time.sleep(2)


# -----------------------------
# Entry point
# -----------------------------
def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
