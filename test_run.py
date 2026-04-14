import pandas as pd
import subprocess

# -----------------------------
# STEP 1: Load dataset
# -----------------------------
url = "https://github.com/yszanwar/phase2_qrt_challenge/releases/download/price_data/all_prices_5000_tickers.parquet"

print("Loading dataset...")
df = pd.read_parquet(url)

# -----------------------------
# STEP 2: Extract first 200 tickers
# -----------------------------
tickers = df["Ticker"].unique().tolist()[:200]

# Fix Yahoo format
tickers = [t.replace(".", "-") for t in tickers]

# -----------------------------
# STEP 3: Save to file
# -----------------------------
with open("tickers_200.txt", "w") as f:
    for t in tickers:
        f.write(t + "\n")

print("Saved tickers_200.txt")

# -----------------------------
# STEP 4: Call your pipeline
# -----------------------------
print("Running pipeline...")

subprocess.run([
    "python", "pipeline.py",
    "--tickers_file", "tickers_200.txt",
    "--period", "5d",
    "--output", "test_output.parquet"
])
