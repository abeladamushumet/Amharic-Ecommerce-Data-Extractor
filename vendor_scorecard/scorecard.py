# scorecard.py

import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Paths ---
METRICS_FILE = "/content/drive/MyDrive/Colab Notebooks/Amharic-Ecommerce-Data-Extractor/data/processed/vendor_metrics/vendor_summary.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(METRICS_FILE), "vendor_scorecard")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "vendor_scorecard.csv")
WEIGHTS_FILE = "/content/drive/MyDrive/Colab Notebooks/Amharic-Ecommerce-Data-Extractor/configs/scorecard_weights.yaml"

# --- Load vendor metrics ---
logger.info("🔹 Loading vendor metrics...")
try:
    df = pd.read_csv(METRICS_FILE)
    if df.empty:
        raise ValueError("Vendor metrics file is empty.")
    logger.info(f"✅ Loaded {len(df)} vendors")
except Exception as e:
    logger.error(f"❌ Failed to load vendor metrics: {e}")
    raise

# --- Preprocess metrics ---
df.fillna(0, inplace=True)

# Log-transform avg_price, handle 0 gracefully
if "avg_price" in df.columns:
    df['avg_price_log'] = df['avg_price'].apply(lambda x: np.log1p(x) if x > 0 else 0)
else:
    df['avg_price_log'] = 0

# --- Load weights from YAML ---
try:
    with open(WEIGHTS_FILE, "r") as f:
        weights = yaml.safe_load(f)
    logger.info(f"✅ Loaded weights: {weights}")
except Exception as e:
    logger.warning(f"⚠️ Failed to load weights file, using defaults. Error: {e}")
    weights = {
        'num_products': 0.3,
        'num_prices': 0.2,
        'num_locations': 0.2,
        'num_contacts': 0.1,
        'avg_price_log': 0.2
    }

# --- Normalize metrics (0-1) ---
features_to_scale = [f for f in weights.keys() if f in df.columns]
if features_to_scale:
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    logger.info("🔹 Metrics normalized (0-1)")
else:
    logger.error("❌ None of the weight features exist in vendor metrics!")
    raise ValueError("Weights and vendor metrics do not match.")

# --- Compute overall vendor score ---
df_scaled['score'] = sum(df_scaled[col] * w for col, w in weights.items() if col in df_scaled.columns)

# --- Rank vendors ---
df_scaled['rank'] = df_scaled['score'].rank(ascending=False, method="dense").astype(int)

# --- Save scorecard ---
try:
    df_scaled.sort_values('score', ascending=False).to_csv(OUTPUT_FILE, index=False)
    logger.info(f"✅ Vendor scorecard saved to {OUTPUT_FILE}")
except Exception as e:
    logger.error(f"❌ Failed to save scorecard: {e}")
    raise

# --- Optional: Display top 5 vendors ---
logger.info("🔹 Top 5 vendors by score:")
logger.info(df_scaled.sort_values('score', ascending=False).head().to_string())
