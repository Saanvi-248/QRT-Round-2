# Install dependencies
!pip install yfinance pyarrow

import pandas as pd
import yfinance as yf
import os
import time

# -----------------------------
# STEP 1: Create tickers.txt (run once)
# -----------------------------
if not os.path.exists("tickers.txt"):
    print("📥 Creating tickers.txt...")

    url = "https://github.com/yszanwar/phase2_qrt_challenge/releases/download/price_data/all_prices_5000_tickers.parquet"
    df = pd.read_parquet(url)

    tickers = df.columns.get_level_values("Ticker").unique().tolist()

    def normalize(t):
        return str(t).replace(".", "-").replace("/", "-")

    tickers = [normalize(t) for t in tickers]

    with open("tickers.txt", "w") as f:
        for t in tickers:
            f.write(t + "\n")

    print(f"✅ Created tickers.txt ({len(tickers)} tickers)")


# -----------------------------
# STEP 2: Load tickers
# -----------------------------
def load_tickers(file_path):
    with open(file_path) as f:
        return [line.strip() for line in f if line.strip()]

tickers = load_tickers("tickers.txt")
print(f"📊 Loaded {len(tickers)} tickers")

# 👉 TEST FIRST (remove later)
tickers = tickers[:200]


# -----------------------------
# STEP 3: Fetch OHLCV data
# -----------------------------
def fetch_data(tickers, batch_size=20):
    all_data = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"\n📡 Batch {i//batch_size + 1}")

        try:
            data = yf.download(
                batch,
                period="5d",
                group_by="ticker",
                progress=False,
                auto_adjust=False,
                threads=False
            )

            if data.empty:
                continue

            data = data.stack(level=0, future_stack=True).rename_axis(["Date", "Ticker"]).reset_index()

            all_data.append(data)

            time.sleep(1)

        except Exception as e:
            print(f"❌ Error: {e}")

    return pd.concat(all_data, ignore_index=True)


data = fetch_data(tickers)

print(f"\n📦 Data shape: {data.shape}")


# -----------------------------
# STEP 4: Compute return
# -----------------------------
data["return"] = data.groupby("Ticker")["Close"].pct_change(fill_method=None)


# -----------------------------
# STEP 5: Fetch metadata (sector + marketCap)
# -----------------------------
def fetch_metadata(tickers):

    if os.path.exists("metadata.parquet"):
        print("📂 Loading cached metadata...")
        return pd.read_parquet("metadata.parquet")

    print("📊 Fetching metadata...")

    meta = []

    for i, t in enumerate(tickers):
        try:
            info = yf.Ticker(t).info

            meta.append({
                "symbol": t,
                "sector": info.get("sector"),
                "marketCap": info.get("marketCap")
            })

            if i % 50 == 0:
                print(f"Processed {i}")

        except:
            continue

    meta_df = pd.DataFrame(meta)
    meta_df.to_parquet("metadata.parquet", index=False)

    return meta_df


meta = fetch_metadata(tickers)


# -----------------------------
# STEP 6: FINAL TRANSFORM
# -----------------------------
def transform(data, meta):

    df = data.rename(columns={
        "Ticker": "symbol",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    df = df.merge(meta, on="symbol", how="left")

    df = df[[
        "Date", "symbol",
        "open", "high", "low", "close",
        "volume", "return",
        "sector", "marketCap"
    ]]

    df = df.sort_values(["symbol", "Date"]).reset_index(drop=True)

    return df


final_data = transform(data, meta)


# -----------------------------
# STEP 7: Save
# -----------------------------
final_data.to_csv("final_dataset.csv", index=False)

print("✅ Saved as final_dataset.csv")


# -----------------------------
# STEP 8: Preview
# -----------------------------
pd.set_option('display.max_columns', None)
print("\n📊 Final sample:")
print(final_data.head())
