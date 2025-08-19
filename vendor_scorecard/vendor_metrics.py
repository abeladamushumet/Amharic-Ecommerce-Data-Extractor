# vendor_metrics.py

import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Paths ---
DATA_PATH = "/content/drive/MyDrive/Colab Notebooks/Amharic-Ecommerce-Data-Extractor/data/processed/ner_output.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_PATH), "vendor_metrics")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "vendor_summary.csv")

# --- Load NER Data ---
logger.info("🔹 Loading NER output...")
try:
    df = pd.read_csv(DATA_PATH)
    logger.info(f"✅ Loaded {len(df)} rows")

    # If no vendor column, assign dummy vendors per sentence
    if 'vendor' not in df.columns:
        df['vendor'] = np.nan
        df['sentence_idx'] = df.index  # Use row index as sentence
        df['vendor'] = "Vendor_" + (df['sentence_idx'] + 1).astype(str)  # Vendor_1, Vendor_2, ...
    logger.info("✅ Vendor column prepared")
except Exception as e:
    logger.error(f"Failed to load NER output: {e}")
    raise

# --- Compute Vendor Metrics ---
logger.info("🔹 Computing vendor metrics...")

vendor_summary = []

try:
    for vendor, group in df.groupby('vendor'):
        num_products = group[group['entity'].str.contains("PRODUCT")].shape[0] if 'entity' in group.columns else 0
        num_prices = group[group['entity'].str.contains("PRICE")].shape[0] if 'entity' in group.columns else 0
        num_locations = group[group['entity'].str.contains("LOC")].shape[0] if 'entity' in group.columns else 0
        num_contacts = group[group['entity'].str.contains("CONTACT")].shape[0] if 'entity' in group.columns else 0

        # Average price calculation
        prices = pd.to_numeric(group.loc[group['entity'].str.contains("PRICE"), 'word'], errors='coerce') if 'entity' in group.columns else pd.Series(dtype=float)
        avg_price = prices.mean() if not prices.empty else np.nan
        min_price = prices.min() if not prices.empty else np.nan
        max_price = prices.max() if not prices.empty else np.nan

        vendor_summary.append({
            "vendor": vendor,
            "num_products": num_products,
            "num_prices": num_prices,
            "num_locations": num_locations,
            "num_contacts": num_contacts,
            "avg_price": avg_price,
            "min_price": min_price,
            "max_price": max_price
        })
except Exception as e:
    logger.error(f"Error computing metrics: {e}")
    raise

vendor_df = pd.DataFrame(vendor_summary)

# --- Save Metrics ---
try:
    vendor_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"✅ Vendor metrics saved to {OUTPUT_FILE}")
except Exception as e:
    logger.error(f"Failed to save vendor metrics: {e}")
    raise

# --- Optional: Display top 5 vendors ---
logger.info("🔹 Sample vendor metrics:")
logger.info(vendor_df.head().to_string())
